from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class H2OConfig:
    """H2O 的固定 KV Cache 预算。"""

    heavy_hitter_size: int = 1024
    recent_size: int = 1024

    @property
    def cache_size(self) -> int:
        return self.heavy_hitter_size + self.recent_size

    def validate(self) -> None:
        if self.heavy_hitter_size <= 0:
            raise ValueError("h2o_heavy_hitter_size 必须大于 0")
        if self.recent_size <= 0:
            raise ValueError("h2o_recent_size 必须大于 0")


class H2OController:
    """按累计 attention score 保留 heavy hitters 与最近 token。"""

    def __init__(self, config: H2OConfig) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._scores: dict[int, torch.Tensor] = {}
        self._eviction_steps = 0
        self._evicted_tokens = 0
        self._max_cache_length = 0
        self._observed_layers: set[int] = set()

    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        past_key_values: Any,
        attention_weights: torch.Tensor | None,
    ) -> None:
        """累计当前层真实 attention 权重，并将该层 cache 淘汰到固定预算。"""
        if past_key_values is None or attention_weights is None:
            return
        if not hasattr(past_key_values, "layers") or layer_idx >= len(past_key_values.layers):
            return

        layer_cache = past_key_values.layers[layer_idx]
        key_states = getattr(layer_cache, "keys", None)
        value_states = getattr(layer_cache, "values", None)
        if key_states is None or value_states is None or key_states.dim() != 4:
            return
        if attention_weights.dim() != 4:
            raise ValueError("H2O 需要形状为 [batch, heads, query, key] 的 attention 权重")

        batch_size, num_kv_heads, sequence_length, head_dim = key_states.shape
        if attention_weights.shape[0] != batch_size or attention_weights.shape[-1] != sequence_length:
            raise ValueError("attention 权重与 KV Cache 的 batch 或序列长度不一致")

        num_attention_heads = attention_weights.shape[1]
        if num_attention_heads % num_kv_heads != 0:
            raise ValueError("attention head 数必须能被 KV head 数整除")

        groups = num_attention_heads // num_kv_heads
        current_scores = attention_weights.detach().to(torch.float32).reshape(
            batch_size,
            num_kv_heads,
            groups,
            attention_weights.shape[-2],
            sequence_length,
        ).sum(dim=(2, 3))

        previous_scores = self._scores.get(layer_idx)
        if previous_scores is None or previous_scores.shape[:2] != current_scores.shape[:2]:
            accumulated_scores = torch.zeros_like(current_scores)
        elif previous_scores.shape[-1] <= sequence_length:
            new_token_count = sequence_length - previous_scores.shape[-1]
            accumulated_scores = torch.cat(
                [
                    previous_scores.to(current_scores.device),
                    torch.zeros(
                        batch_size,
                        num_kv_heads,
                        new_token_count,
                        dtype=torch.float32,
                        device=current_scores.device,
                    ),
                ],
                dim=-1,
            )
        else:
            raise ValueError("H2O 分数长度超过当前 KV Cache 长度，请在新请求前调用 reset()")

        accumulated_scores = accumulated_scores + current_scores
        self._observed_layers.add(layer_idx)
        self._max_cache_length = max(self._max_cache_length, sequence_length)

        if sequence_length <= self.config.cache_size:
            self._scores[layer_idx] = accumulated_scores
            return

        recent_size = self.config.recent_size
        old_length = sequence_length - recent_size
        heavy_hitter_size = min(self.config.heavy_hitter_size, old_length)

        _, heavy_indices = torch.topk(
            accumulated_scores[..., :old_length],
            k=heavy_hitter_size,
            dim=-1,
        )
        heavy_indices = heavy_indices.sort(dim=-1).values
        recent_indices = torch.arange(
            old_length,
            sequence_length,
            device=key_states.device,
        ).view(1, 1, -1).expand(batch_size, num_kv_heads, -1)
        keep_indices = torch.cat([heavy_indices, recent_indices], dim=-1)

        expanded_indices = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        layer_cache.keys = torch.gather(key_states, 2, expanded_indices)
        layer_cache.values = torch.gather(value_states, 2, expanded_indices)
        self._scores[layer_idx] = torch.gather(accumulated_scores, 2, keep_indices)

        self._eviction_steps += 1
        self._evicted_tokens += sequence_length - self.config.cache_size

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": "h2o_attention_score",
            "heavy_hitter_size": self.config.heavy_hitter_size,
            "recent_size": self.config.recent_size,
            "cache_size": self.config.cache_size,
            "observed_layers": len(self._observed_layers),
            "eviction_steps": self._eviction_steps,
            "layer_token_evictions": self._evicted_tokens,
            "max_cache_length": self._max_cache_length,
        }
