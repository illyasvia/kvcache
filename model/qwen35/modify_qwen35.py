import math
import warnings
from typing import Any, List, Optional, Tuple, Union
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging, TransformersKwargs,
)
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .v451_modeling_qwen35 import (
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen3_5RMSNorm,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    _flash_attention_forward = None

logger = logging.get_logger(__name__)


class H2OKVCache_LayerWise:
    _shared_hh_score = None  # 类级别共享 hh_score，用于跨层传递

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        hh_layer=27,
    ):
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        if self.layer_idx == 0:
            print(f"H2O KVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio} {hh_layer}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.seq_len = None
        self.hh_layer = hh_layer

    def __call__(self, past_key_values, attn_score_cache):
        return self.call_2(past_key_values, attn_score_cache)

    def call_2(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio) + 15
            self.cache_size = self.hh_size + self.recent_size

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        # Aggregate attention scores by kv groups to match num_kv_heads
        attn_score_cache = attn_score_cache.view(bsz, num_kv_heads, num_kv_groups, q_len, seq_len)
        attn_score_cache = attn_score_cache.sum(dim=2)  # [bsz, num_kv_heads, q_len, seq_len]

        if self.layer_idx > self.hh_layer and H2OKVCache_LayerWise._shared_hh_score is not None:
            self.hh_score = H2OKVCache_LayerWise._shared_hh_score
        else:
            self.hh_score = attn_score_cache[:, :, :self.recent_size, :].sum(0).sum(1)
            H2OKVCache_LayerWise._shared_hh_score = self.hh_score

        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        self.hh_score = self.hh_score * value_L2

        seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values.layers[self.layer_idx].keys.device).repeat(num_heads, 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values.layers[self.layer_idx].keys.device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values.layers[self.layer_idx].keys.squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values.layers[self.layer_idx].values.squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        past_key_values.layers[self.layer_idx].keys = k_hh_recent
        past_key_values.layers[self.layer_idx].values = v_hh_recent
        return past_key_values

    def _update_hh_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]
        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        H2OKVCache_LayerWise._shared_hh_score = None

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class H2OQwen3_5Attention_drop(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with H2O KV cache eviction."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # H2O KV cache: evict tokens with lowest attn_weights, keep hh_ratio proportion
        self.kv_cache = H2OKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            hh_layer=config.hh_layer,
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Explicit attention computation to obtain attn_weights for H2O eviction
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # H2O eviction: after prefill (q_len != 1), evict tokens with lowest attn_weights
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, attn_weights


class StreamingLLMKVCache_LayerWise:
    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"StreamingLLMKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None

    def __call__(self, past_key_values, attn_weights=None):
        if self.hh_ratio is not None:
            self.cache_size = int(past_key_values.layers[self.layer_idx].keys.shape[2] * (self.hh_ratio + self.recent_ratio))
        self.hh_size = self.cache_size // 2
        self.recent_size = self.cache_size - self.hh_size

        if past_key_values is None:
            return None

        seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        bsz, num_heads, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape

        keep_sink = torch.arange(0, self.hh_size, device=past_key_values.layers[self.layer_idx].keys.device).repeat(num_heads, 1)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values.layers[self.layer_idx].keys.device).repeat(num_heads, 1)
        keep_idx = torch.cat([keep_sink, keep_recent], dim=-1)

        mask = torch.zeros(num_heads, seq_len, dtype=torch.bool).to(past_key_values.layers[self.layer_idx].keys.device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values.layers[self.layer_idx].keys.squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values.layers[self.layer_idx].values.squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        past_key_values.layers[self.layer_idx].keys = k_hh_recent
        past_key_values.layers[self.layer_idx].values = v_hh_recent
        return past_key_values

    def _clean_scores(self):
        self.hh_score = None


class StreamingLLMQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = StreamingLLMKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None


class LookMKVCache_LayerWise:
    """
    Look-M KV Cache 驱逐策略 (Look-Once Optimization in KV Cache)。

    核心思想：与 NeedleLLM 类似使用最后一个 query token 的注意力分布识别重要 token，
    但关键创新在于被驱逐的 token 不是简单丢弃，而是通过 Token Merging 将其 KV 信息
    合并到保留的 token 中，从而保留更多上下文信息。

    合并策略：
    - pivot: 被驱逐 token 找到注意力最高的保留 token (pivot)，将 KV 加权合并到 pivot 上
    - mean:  被驱逐 token 的 KV 平均合并到最近的保留 token
    - weighted: 按注意力分数加权合并到最近的保留 token

    保留策略：
    - Attention Sink: 保留前 sink_size 个 token（位置编码稳定性）
    - Important tokens: 按注意力分数选择 top-k（关键信息）
    - Recent tokens: 保留最近的 recent_size 个 token（时序连贯性）

    跨层共享机制：
    - 当 layer_idx <= needle_layer 时，计算当前层的 score 并保存到类级别共享变量
    - 当 layer_idx > needle_layer 时，直接复用共享的 score
    """

    _shared_lookm_score = None  # 类级别共享 score，用于跨层传递

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        needle_layer=27,
        merge_strategy="pivot",  # "pivot", "mean", "weighted"
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"LookMKVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio}, "
                  f"needle_layer={needle_layer}, merge_strategy={merge_strategy}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_size = 4  # attention sink: 保留前 4 个 token
        self.needle_layer = needle_layer
        self.merge_strategy = merge_strategy

    def __call__(self, past_key_values, attn_score_cache):
        """
        基于 Look-M 策略进行 KV cache 驱逐与合并。

        Args:
            past_key_values: DynamicCache 对象
            attn_score_cache: [bsz, num_heads, q_len, seq_len] 注意力权重
        """
        if attn_score_cache is None:
            return past_key_values

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape

        # 动态计算保留大小
        if self.hh_ratio is not None:
            self.hh_size = int(seq_len * self.hh_ratio)
            self.recent_size = int(seq_len * self.recent_ratio) + 15
        cache_size = self.sink_size + self.hh_size + self.recent_size

        # 获取 KV heads 数量并聚合
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        kv_seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)
        if kv_seq_len <= cache_size:
            return past_key_values

        # 跨层共享机制
        if self.layer_idx > self.needle_layer and LookMKVCache_LayerWise._shared_lookm_score is not None:
            lookm_scores = LookMKVCache_LayerWise._shared_lookm_score
        else:
            # 使用最后一个 query token 的注意力分布 (Look-Once)
            lookm_scores = attn_score_cache[:, :, -1, :]

            # 聚合到 KV heads 维度
            lookm_scores = lookm_scores.view(bsz, num_kv_heads, num_kv_groups, seq_len)
            lookm_scores = lookm_scores.sum(dim=2)  # [bsz, num_kv_heads, seq_len]
            lookm_scores = lookm_scores.squeeze(0)  # [num_kv_heads, seq_len]

            # 保存到类级别共享变量
            LookMKVCache_LayerWise._shared_lookm_score = lookm_scores

        # 结合 value 的 L2 norm
        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        importance_scores = lookm_scores * value_L2

        # 构建保留索引
        _, num_kv_h, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape
        device = past_key_values.layers[self.layer_idx].keys.device

        # 1. Attention sink: 前 sink_size 个 token
        sink_idx = torch.arange(0, self.sink_size, device=device).unsqueeze(0).expand(num_kv_h, -1)

        # 2. Important tokens: 在中间区域选择 attention 最高的 top-k
        middle_start = self.sink_size
        middle_end = kv_seq_len - self.recent_size
        if middle_end > middle_start and self.hh_size > 0:
            middle_scores = importance_scores[:, middle_start:middle_end]
            actual_hh_size = min(self.hh_size, middle_end - middle_start)
            _, topk_idx = torch.topk(middle_scores, actual_hh_size, dim=-1)
            topk_idx = (topk_idx + middle_start).sort(dim=-1).values
        else:
            topk_idx = torch.empty(num_kv_h, 0, dtype=torch.long, device=device)

        # 3. Recent tokens: 最近 recent_size 个 token
        recent_idx = torch.arange(
            kv_seq_len - self.recent_size, kv_seq_len, device=device
        ).unsqueeze(0).expand(num_kv_h, -1)

        # 合并所有保留索引
        keep_idx = torch.cat([sink_idx, topk_idx, recent_idx], dim=-1)

        # 构建 keep mask
        keep_mask = torch.zeros(num_kv_h, kv_seq_len, dtype=torch.bool, device=device)
        keep_mask.scatter_(-1, keep_idx, True)

        # ============ Token Merging: 将被驱逐 token 的 KV 合并到保留 token ============
        keys = past_key_values.layers[self.layer_idx].keys    # [bsz, num_kv_h, kv_seq_len, head_dim]
        values = past_key_values.layers[self.layer_idx].values  # [bsz, num_kv_h, kv_seq_len, head_dim]

        if self.merge_strategy == "pivot":
            keys, values = self._pivot_merge(
                keys, values, keep_mask, lookm_scores, num_kv_h, kv_seq_len, head_dim, device
            )
        elif self.merge_strategy == "mean":
            keys, values = self._mean_merge(
                keys, values, keep_mask, num_kv_h, kv_seq_len, head_dim, device
            )
        elif self.merge_strategy == "weighted":
            keys, values = self._weighted_merge(
                keys, values, keep_mask, lookm_scores, num_kv_h, kv_seq_len, head_dim, device
            )

        # 使用 mask 选择保留的 KV
        k_kept = keys.squeeze(0)[keep_mask].view(1, num_kv_h, -1, head_dim)
        v_kept = values.squeeze(0)[keep_mask].view(1, num_kv_h, -1, head_dim)
        past_key_values.layers[self.layer_idx].keys = k_kept
        past_key_values.layers[self.layer_idx].values = v_kept

        return past_key_values

    def _pivot_merge(self, keys, values, keep_mask, scores, num_kv_h, kv_seq_len, head_dim, device):
        """
        Pivot Merging: 为每个被驱逐 token 找到注意力最高的保留 token (pivot)，
        将被驱逐 token 的 KV 按注意力比例加权合并到 pivot 上。
        """
        evict_mask = ~keep_mask  # [num_kv_h, kv_seq_len]

        for h in range(num_kv_h):
            evict_positions = evict_mask[h].nonzero(as_tuple=True)[0]
            if len(evict_positions) == 0:
                continue

            keep_positions = keep_mask[h].nonzero(as_tuple=True)[0]

            # 每个被驱逐 token 的注意力分数
            evict_scores = scores[h, evict_positions]  # [num_evict]

            # 找到每个被驱逐 token 在保留 token 中注意力最高的 pivot
            # 使用被驱逐 token 的 key 与保留 token 的 key 的相似度来找 pivot
            evict_keys = keys[0, h, evict_positions, :]   # [num_evict, head_dim]
            keep_keys = keys[0, h, keep_positions, :]     # [num_keep, head_dim]

            # 计算余弦相似度找 pivot
            similarity = torch.matmul(evict_keys, keep_keys.transpose(0, 1))  # [num_evict, num_keep]
            pivot_indices = similarity.argmax(dim=-1)  # [num_evict] -> index in keep_positions

            # 按 pivot 分组合并
            for i, evict_pos in enumerate(evict_positions):
                pivot_keep_idx = keep_positions[pivot_indices[i]]
                weight = evict_scores[i] / (evict_scores[i] + scores[h, pivot_keep_idx] + 1e-8)

                # 加权合并到 pivot
                keys[0, h, pivot_keep_idx] = (1 - weight) * keys[0, h, pivot_keep_idx] + weight * keys[0, h, evict_pos]
                values[0, h, pivot_keep_idx] = (1 - weight) * values[0, h, pivot_keep_idx] + weight * values[0, h, evict_pos]

        return keys, values

    def _mean_merge(self, keys, values, keep_mask, num_kv_h, kv_seq_len, head_dim, device):
        """
        Mean Merging: 将被驱逐 token 的 KV 平均合并到最近的保留 token。
        """
        for h in range(num_kv_h):
            evict_positions = (~keep_mask[h]).nonzero(as_tuple=True)[0]
            if len(evict_positions) == 0:
                continue

            keep_positions = keep_mask[h].nonzero(as_tuple=True)[0]

            # 为每个被驱逐 token 找最近的保留 token
            for evict_pos in evict_positions:
                distances = (keep_positions - evict_pos).abs()
                nearest_keep_idx = keep_positions[distances.argmin()]

                # 均值合并
                keys[0, h, nearest_keep_idx] = (keys[0, h, nearest_keep_idx] + keys[0, h, evict_pos]) / 2
                values[0, h, nearest_keep_idx] = (values[0, h, nearest_keep_idx] + values[0, h, evict_pos]) / 2

        return keys, values

    def _weighted_merge(self, keys, values, keep_mask, scores, num_kv_h, kv_seq_len, head_dim, device):
        """
        Weighted Merging: 按注意力分数加权合并被驱逐 token 到最近的保留 token。
        """
        for h in range(num_kv_h):
            evict_positions = (~keep_mask[h]).nonzero(as_tuple=True)[0]
            if len(evict_positions) == 0:
                continue

            keep_positions = keep_mask[h].nonzero(as_tuple=True)[0]

            for evict_pos in evict_positions:
                distances = (keep_positions - evict_pos).abs()
                nearest_keep_idx = keep_positions[distances.argmin()]

                # 注意力加权合并
                evict_score = scores[h, evict_pos]
                keep_score = scores[h, nearest_keep_idx]
                weight = evict_score / (evict_score + keep_score + 1e-8)

                keys[0, h, nearest_keep_idx] = (1 - weight) * keys[0, h, nearest_keep_idx] + weight * keys[0, h, evict_pos]
                values[0, h, nearest_keep_idx] = (1 - weight) * values[0, h, nearest_keep_idx] + weight * values[0, h, evict_pos]

        return keys, values

    def _clean_scores(self):
        self.hh_score = None
        LookMKVCache_LayerWise._shared_lookm_score = None


class LookMQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    """
    Look-M Attention: 基于 Look-Once + Token Merging 的 KV Cache 优化。

    与 NeedleLLM 的区别：被驱逐的 token 通过 pivot/mean/weighted merging 将信息
    合并到保留的 token 中，而非简单丢弃，从而保留更多上下文信息。
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = LookMKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            needle_layer=getattr(config, 'hh_layer', 27),
            merge_strategy=getattr(config, 'merge_strategy', 'pivot'),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # Look-M eviction + merging: 使用最后 query token 的注意力，驱逐 + 合并
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None


class MadaKVCache_LayerWise:
    """
    MadaKV KV Cache 驱逐策略 (Adaptive Modality-Perception KV Cache Eviction)。

    核心思想：基于注意力分数分布的集中程度自适应地调整每层的 KV cache 保留预算。
    注意力集中的层（少数 token 吸引大部分注意力）可以更激进地驱逐，
    注意力分散的层则需要保留更多 token。

    自适应机制：
    - 使用基尼系数衡量每层注意力分数的集中程度
    - 基尼系数高 → 注意力集中 → 适合激进驱逐 → 减小 cache 预算
    - 基尼系数低 → 注意力分散 → 需要更多 token → 增大 cache 预算
    - 通过自适应系数调整 hh_size，recent_size 保持不变

    保留策略：
    - Attention Sink: 保留前 sink_size 个 token（位置编码稳定性）
    - Important tokens: 按注意力分数选择 top-k（自适应调整 k）
    - Recent tokens: 保留最近的 recent_size 个 token（时序连贯性）
    """

    _shared_mada_score = None  # 类级别共享 score，用于跨层传递

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        needle_layer=27,
        adaptive_min=0.5,   # 自适应系数下限
        adaptive_max=1.5,   # 自适应系数上限
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"MadaKVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio}, "
                  f"needle_layer={needle_layer}, adaptive_range=[{adaptive_min}, {adaptive_max}]")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_size = 4
        self.needle_layer = needle_layer
        self.adaptive_min = adaptive_min
        self.adaptive_max = adaptive_max

    def _compute_gini(self, scores):
        """
        计算基尼系数，衡量注意力分数的集中程度。

        Args:
            scores: [num_heads, seq_len] 注意力分数

        Returns:
            gini: 标量，基尼系数 (0=完全均匀, 1=完全集中)
        """
        # 对每个 head 计算基尼系数，然后取平均
        sorted_scores, _ = scores.sort(dim=-1)
        n = sorted_scores.shape[-1]
        if n <= 1:
            return torch.tensor(0.0, device=scores.device)
        cumsum = sorted_scores.cumsum(dim=-1)
        total = sorted_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Gini = 1 - 2 * (area under Lorenz curve)
        lorenz = cumsum / total
        indices = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
        gini_per_head = 1.0 - 2.0 * lorenz.sum(dim=-1) / n + 1.0 / n
        return gini_per_head.mean()

    def __call__(self, past_key_values, attn_score_cache):
        """
        基于自适应预算分配进行 KV cache 驱逐。

        Args:
            past_key_values: DynamicCache 对象
            attn_score_cache: [bsz, num_heads, q_len, seq_len] 注意力权重
        """
        if attn_score_cache is None:
            return past_key_values

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape

        # 获取 KV heads 数量并聚合
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        # 跨层共享机制
        if self.layer_idx > self.needle_layer and MadaKVCache_LayerWise._shared_mada_score is not None:
            mada_scores = MadaKVCache_LayerWise._shared_mada_score
        else:
            # 使用最后一个 query token 的注意力分布
            mada_scores = attn_score_cache[:, :, -1, :]
            mada_scores = mada_scores.view(bsz, num_kv_heads, num_kv_groups, seq_len)
            mada_scores = mada_scores.sum(dim=2)
            mada_scores = mada_scores.squeeze(0)  # [num_kv_heads, seq_len]
            MadaKVCache_LayerWise._shared_mada_score = mada_scores

        # ============ 自适应预算计算 ============
        # 计算基尼系数
        gini = self._compute_gini(mada_scores)

        # 基尼系数高 → 注意力集中 → 需要保留的重要 token 少 → 减小 hh_ratio
        # 基尼系数低 → 注意力分散 → 需要保留更多 token → 增大 hh_ratio
        # adaptive_factor: gini 高时 factor 小，gini 低时 factor 大
        adaptive_factor = self.adaptive_max - (self.adaptive_max - self.adaptive_min) * gini.item()
        adaptive_factor = max(self.adaptive_min, min(self.adaptive_max, adaptive_factor))

        # 动态计算保留大小
        if self.hh_ratio is not None:
            adaptive_hh_size = int(seq_len * self.hh_ratio * adaptive_factor)
            self.recent_size = int(seq_len * self.recent_ratio) + 15
        else:
            adaptive_hh_size = int(self.hh_size * adaptive_factor)

        cache_size = self.sink_size + adaptive_hh_size + self.recent_size

        kv_seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)
        if kv_seq_len <= cache_size:
            return past_key_values

        # 结合 value 的 L2 norm
        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        importance_scores = mada_scores * value_L2

        # 构建保留索引
        _, num_kv_h, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape
        device = past_key_values.layers[self.layer_idx].keys.device

        # 1. Attention sink
        sink_idx = torch.arange(0, self.sink_size, device=device).unsqueeze(0).expand(num_kv_h, -1)

        # 2. Important tokens: 自适应调整后的 top-k
        middle_start = self.sink_size
        middle_end = kv_seq_len - self.recent_size
        if middle_end > middle_start and adaptive_hh_size > 0:
            middle_scores = importance_scores[:, middle_start:middle_end]
            actual_hh_size = min(adaptive_hh_size, middle_end - middle_start)
            _, topk_idx = torch.topk(middle_scores, actual_hh_size, dim=-1)
            topk_idx = (topk_idx + middle_start).sort(dim=-1).values
        else:
            topk_idx = torch.empty(num_kv_h, 0, dtype=torch.long, device=device)

        # 3. Recent tokens
        recent_idx = torch.arange(
            kv_seq_len - self.recent_size, kv_seq_len, device=device
        ).unsqueeze(0).expand(num_kv_h, -1)

        # 合并保留索引
        keep_idx = torch.cat([sink_idx, topk_idx, recent_idx], dim=-1)

        mask = torch.zeros(num_kv_h, kv_seq_len, dtype=torch.bool, device=device)
        mask.scatter_(-1, keep_idx, True)

        k_kept = past_key_values.layers[self.layer_idx].keys.squeeze(0)[mask].view(
            1, num_kv_h, -1, head_dim
        )
        v_kept = past_key_values.layers[self.layer_idx].values.squeeze(0)[mask].view(
            1, num_kv_h, -1, head_dim
        )
        past_key_values.layers[self.layer_idx].keys = k_kept
        past_key_values.layers[self.layer_idx].values = v_kept

        return past_key_values

    def _clean_scores(self):
        self.hh_score = None
        MadaKVCache_LayerWise._shared_mada_score = None


class MadaKVQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    """
    MadaKV Attention: 基于注意力分布集中度自适应调整每层 KV cache 预算。

    与 NeedleLLM 的区别：不使用固定的 hh_ratio，而是根据每层注意力分数的基尼系数
    动态调整保留预算。注意力集中的层保留更少 token，注意力分散的层保留更多 token。
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = MadaKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            needle_layer=getattr(config, 'hh_layer', 27),
            adaptive_min=getattr(config, 'adaptive_min', 0.5),
            adaptive_max=getattr(config, 'adaptive_max', 1.5),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # MadaKV eviction: 自适应预算 + 基于注意力分数的驱逐
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None


class RazorAttentionKVCache_LayerWise:
    """
    RazorAttention KV Cache 驱逐策略 (Efficient KV Cache Compression Through Retrieval Heads)。

    核心思想：发现只有少数 "retrieval heads" 需要完整的 KV cache 来检索远处信息，
    而大多数 "non-retrieval heads" 只需要局部信息（最近的 token）。通过在 prefill
    阶段识别 retrieval heads，可以为不同的 head 分配不同的 cache 预算。

    识别机制：
    - 在 prefill 阶段，观察每个 head 的注意力分布
    - 计算每个 head 对 "远处 token"（超出 local window）的累积注意力比例
    - 如果远处注意力比例超过阈值 retrieval_threshold，则标记为 retrieval head
    - Retrieval heads 保留完整的 KV cache（sink + important + recent）
    - Non-retrieval heads 只保留 sink + recent tokens

    保留策略：
    - Retrieval heads: sink + top-k important tokens + recent tokens（完整预算）
    - Non-retrieval heads: sink + recent tokens（精简预算）

    跨层共享机制：
    - 当 layer_idx <= needle_layer 时，计算当前层的 retrieval head mask 并共享
    - 当 layer_idx > needle_layer 时，复用共享的 retrieval head mask
    """

    _shared_retrieval_mask = None  # 类级别共享 retrieval head mask
    _shared_razor_score = None    # 类级别共享 attention score

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        needle_layer=27,
        retrieval_threshold=0.1,  # 远处注意力比例阈值，超过则为 retrieval head
        local_window_ratio=0.3,   # 定义 "局部窗口" 占总序列长度的比例
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"RazorAttentionKVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio}, "
                  f"needle_layer={needle_layer}, retrieval_threshold={retrieval_threshold}, "
                  f"local_window_ratio={local_window_ratio}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_size = 4
        self.needle_layer = needle_layer
        self.retrieval_threshold = retrieval_threshold
        self.local_window_ratio = local_window_ratio

    def _identify_retrieval_heads(self, attn_scores, num_kv_heads, seq_len):
        """
        识别 retrieval heads：计算每个 head 对远处 token 的注意力比例。

        Args:
            attn_scores: [num_kv_heads, seq_len] 最后 query token 的注意力分布
            num_kv_heads: KV head 数量
            seq_len: 序列长度

        Returns:
            retrieval_mask: [num_kv_heads] bool tensor, True 表示该 head 是 retrieval head
        """
        # 定义 "局部窗口"：最后 local_window_ratio 比例的 token
        local_window_size = max(int(seq_len * self.local_window_ratio), self.recent_size)
        local_window_size = min(local_window_size, seq_len)

        # 计算远处 token（不在局部窗口内）的注意力总和
        # 远处区域: [sink_size, seq_len - local_window_size]
        remote_start = self.sink_size
        remote_end = seq_len - local_window_size

        if remote_end <= remote_start:
            # 序列太短，没有远处区域，所有 head 都不是 retrieval head
            return torch.zeros(num_kv_heads, dtype=torch.bool, device=attn_scores.device)

        # 计算远处区域的注意力比例
        remote_attention = attn_scores[:, remote_start:remote_end].sum(dim=-1)  # [num_kv_heads]
        total_attention = attn_scores.sum(dim=-1).clamp(min=1e-8)  # [num_kv_heads]
        remote_ratio = remote_attention / total_attention  # [num_kv_heads]

        # 超过阈值的 head 标记为 retrieval head
        retrieval_mask = remote_ratio > self.retrieval_threshold

        return retrieval_mask

    def __call__(self, past_key_values, attn_score_cache):
        """
        基于 RazorAttention 策略进行 KV cache 驱逐。

        Args:
            past_key_values: DynamicCache 对象
            attn_score_cache: [bsz, num_heads, q_len, seq_len] 注意力权重
        """
        if attn_score_cache is None:
            return past_key_values

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape

        # 动态计算保留大小
        if self.hh_ratio is not None:
            self.hh_size = int(seq_len * self.hh_ratio)
            self.recent_size = int(seq_len * self.recent_ratio) + 15

        # 获取 KV heads 数量并聚合
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        kv_seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)

        # Non-retrieval heads 的 cache 大小：sink + recent
        non_retrieval_cache_size = self.sink_size + self.recent_size
        # Retrieval heads 的 cache 大小：sink + hh_size + recent
        retrieval_cache_size = self.sink_size + self.hh_size + self.recent_size

        if kv_seq_len <= non_retrieval_cache_size:
            return past_key_values

        # 跨层共享机制
        if self.layer_idx > self.needle_layer and RazorAttentionKVCache_LayerWise._shared_retrieval_mask is not None:
            retrieval_mask = RazorAttentionKVCache_LayerWise._shared_retrieval_mask
            razor_scores = RazorAttentionKVCache_LayerWise._shared_razor_score
        else:
            # 使用最后一个 query token 的注意力分布
            last_attn = attn_score_cache[:, :, -1, :]  # [bsz, num_heads, seq_len]
            # 聚合到 KV heads 维度
            last_attn = last_attn.view(bsz, num_kv_heads, num_kv_groups, seq_len)
            last_attn = last_attn.sum(dim=2)  # [bsz, num_kv_heads, seq_len]
            razor_scores = last_attn.squeeze(0)  # [num_kv_heads, seq_len]

            # 识别 retrieval heads
            retrieval_mask = self._identify_retrieval_heads(razor_scores, num_kv_heads, seq_len)

            # 保存到类级别共享变量
            RazorAttentionKVCache_LayerWise._shared_retrieval_mask = retrieval_mask
            RazorAttentionKVCache_LayerWise._shared_razor_score = razor_scores

        # 结合 value 的 L2 norm
        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        importance_scores = razor_scores * value_L2

        # 构建保留索引 - 每个 head 根据是否为 retrieval head 分配不同预算
        _, num_kv_h, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape
        device = past_key_values.layers[self.layer_idx].keys.device

        # 统一的 cache 大小（取 retrieval head 的预算，确保所有 head 输出相同长度）
        # 为了保持 tensor 规整，我们对所有 head 使用相同的输出长度
        # Retrieval head: sink + topk_important + recent
        # Non-retrieval head: sink + recent (用 recent 填满多余空间)

        # 计算实际保留大小
        middle_start = self.sink_size
        middle_end = kv_seq_len - self.recent_size

        # 对于统一的输出，使用 retrieval_cache_size 作为目标
        target_cache_size = retrieval_cache_size
        if kv_seq_len <= target_cache_size:
            return past_key_values

        # 为每个 head 构建 mask
        mask = torch.zeros(num_kv_h, kv_seq_len, dtype=torch.bool, device=device)

        # 1. 所有 head 保留 sink tokens
        mask[:, :self.sink_size] = True

        # 2. 所有 head 保留 recent tokens
        mask[:, kv_seq_len - self.recent_size:] = True

        # 3. Retrieval heads: 在中间区域保留 top-k important tokens
        if middle_end > middle_start and self.hh_size > 0:
            # 获取 retrieval heads 的索引
            retrieval_head_indices = retrieval_mask.nonzero(as_tuple=True)[0]

            if len(retrieval_head_indices) > 0:
                middle_scores = importance_scores[retrieval_head_indices, middle_start:middle_end]
                actual_hh_size = min(self.hh_size, middle_end - middle_start)
                _, topk_idx = torch.topk(middle_scores, actual_hh_size, dim=-1)
                topk_idx = topk_idx + middle_start  # 转换为全局索引

                # 为 retrieval heads 设置 important token mask
                for i, head_idx in enumerate(retrieval_head_indices):
                    mask[head_idx].scatter_(0, topk_idx[i], True)

            # Non-retrieval heads: 不保留中间区域的 token（只有 sink + recent）
            # 但为了保持输出长度一致，non-retrieval heads 扩展 recent window
            non_retrieval_head_indices = (~retrieval_mask).nonzero(as_tuple=True)[0]
            if len(non_retrieval_head_indices) > 0:
                # 扩展 recent window 来填满预算
                extended_recent_size = self.recent_size + self.hh_size
                extended_recent_start = max(self.sink_size, kv_seq_len - extended_recent_size)
                mask[non_retrieval_head_indices, extended_recent_start:] = True

        # 确保所有 head 保留相同数量的 token（通过 padding）
        # 计算每个 head 保留的 token 数
        tokens_per_head = mask.sum(dim=-1)  # [num_kv_h]
        max_tokens = tokens_per_head.max().item()

        # 使用统一的保留策略：按 mask 选择并 pad
        # 为简化实现，对每个 head 独立处理
        keys = past_key_values.layers[self.layer_idx].keys   # [bsz, num_kv_h, kv_seq_len, head_dim]
        values = past_key_values.layers[self.layer_idx].values

        # 收集所有 head 保留的 KV
        new_keys_list = []
        new_values_list = []

        for h in range(num_kv_h):
            keep_positions = mask[h].nonzero(as_tuple=True)[0]
            k_h = keys[0, h, keep_positions, :]  # [num_kept, head_dim]
            v_h = values[0, h, keep_positions, :]

            # 如果保留数量少于 max_tokens，从 recent 端补齐
            if k_h.shape[0] < max_tokens:
                deficit = max_tokens - k_h.shape[0]
                # 从紧邻 recent window 之前的位置补充
                extra_start = max(self.sink_size, kv_seq_len - self.recent_size - deficit)
                extra_end = kv_seq_len - self.recent_size
                extra_positions = torch.arange(extra_start, extra_end, device=device)
                # 过滤已保留的位置
                extra_mask = ~mask[h, extra_start:extra_end]
                extra_positions = extra_positions[extra_mask[:len(extra_positions)]][:deficit]
                if len(extra_positions) > 0:
                    extra_k = keys[0, h, extra_positions, :]
                    extra_v = values[0, h, extra_positions, :]
                    k_h = torch.cat([k_h, extra_k], dim=0)
                    v_h = torch.cat([v_h, extra_v], dim=0)

            # 截断到 max_tokens
            k_h = k_h[:max_tokens]
            v_h = v_h[:max_tokens]

            new_keys_list.append(k_h)
            new_values_list.append(v_h)

        new_keys = torch.stack(new_keys_list, dim=0).unsqueeze(0)  # [1, num_kv_h, max_tokens, head_dim]
        new_values = torch.stack(new_values_list, dim=0).unsqueeze(0)

        past_key_values.layers[self.layer_idx].keys = new_keys
        past_key_values.layers[self.layer_idx].values = new_values

        return past_key_values

    def _clean_scores(self):
        self.hh_score = None
        RazorAttentionKVCache_LayerWise._shared_retrieval_mask = None
        RazorAttentionKVCache_LayerWise._shared_razor_score = None


class RazorAttentionQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    """
    RazorAttention: 基于 Retrieval Head 识别的异构 KV Cache 压缩。

    与其他方法的区别：不同 head 获得不同的 cache 预算。Retrieval heads 保留完整的
    KV cache（包括远处的 important tokens），Non-retrieval heads 只保留局部窗口。
    通过识别哪些 head 真正需要远处信息来实现高效压缩。
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = RazorAttentionKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            needle_layer=getattr(config, 'hh_layer', 27),
            retrieval_threshold=getattr(config, 'retrieval_threshold', 0.1),
            local_window_ratio=getattr(config, 'local_window_ratio', 0.3),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # RazorAttention eviction: 识别 retrieval heads，异构压缩
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None


class NeedleLLMKVCache_LayerWise:
    """
    NeedleLLM KV Cache 驱逐策略。

    核心思想：利用最后一个 query token 的注意力分布来识别 "needle"（关键信息）token。
    与 H2O 使用前 N 个 token 的累积注意力不同，NeedleLLM 聚焦于最后一个 query token
    的注意力模式，因为最后一个 token（通常是问题的结尾或生成提示）的注意力分布
    最能反映哪些上下文 token 对回答问题真正重要。

    保留策略：
    - Attention Sink: 保留前 sink_size 个 token（位置编码稳定性）
    - Needle tokens: 按最后 query 的注意力分数选择 top-k（关键信息）
    - Recent tokens: 保留最近的 recent_size 个 token（时序连贯性）

    跨层共享机制：
    - 当 layer_idx <= needle_layer 时，计算当前层的 needle score 并保存到类级别共享变量
    - 当 layer_idx > needle_layer 时，直接复用共享的 needle score，避免重复计算
    - 这基于深层注意力模式与浅层相似的观察，可减少计算开销
    """

    _shared_needle_score = None  # 类级别共享 needle_score，用于跨层传递

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        needle_layer=27,
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"NeedleLLMKVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio}, needle_layer={needle_layer}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_size = 4  # attention sink: 保留前 4 个 token
        self.needle_layer = needle_layer

    def __call__(self, past_key_values, attn_score_cache):
        """
        基于最后一个 query token 的注意力分布进行 KV cache 驱逐。

        Args:
            past_key_values: DynamicCache 对象
            attn_score_cache: [bsz, num_heads, q_len, seq_len] 注意力权重
        """
        if attn_score_cache is None:
            return past_key_values

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape

        # 动态计算保留大小
        if self.hh_ratio is not None:
            self.hh_size = int(seq_len * self.hh_ratio)
            self.recent_size = int(seq_len * self.recent_ratio) + 15
        cache_size = self.sink_size + self.hh_size + self.recent_size

        # 获取 KV heads 数量并聚合
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        kv_seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)
        if kv_seq_len <= cache_size:
            return past_key_values

        # 跨层共享机制：深层复用浅层的 needle score
        if self.layer_idx > self.needle_layer and NeedleLLMKVCache_LayerWise._shared_needle_score is not None:
            needle_scores = NeedleLLMKVCache_LayerWise._shared_needle_score
        else:
            # 使用最后一个 query token 的注意力分布作为 needle 检测信号
            # attn_score_cache: [bsz, num_heads, q_len, seq_len]
            # 取最后一个 query 的注意力: [bsz, num_heads, seq_len]
            needle_scores = attn_score_cache[:, :, -1, :]

            # 聚合到 KV heads 维度
            needle_scores = needle_scores.view(bsz, num_kv_heads, num_kv_groups, seq_len)
            needle_scores = needle_scores.sum(dim=2)  # [bsz, num_kv_heads, seq_len]
            needle_scores = needle_scores.squeeze(0)  # [num_kv_heads, seq_len]

            # 保存到类级别共享变量，供后续层复用
            NeedleLLMKVCache_LayerWise._shared_needle_score = needle_scores

        # 结合 value 的 L2 norm 作为额外信号（每层的 value 不同，所以始终用当前层的）
        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        needle_scores = needle_scores * value_L2

        # 构建保留索引
        _, num_kv_h, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape
        device = past_key_values.layers[self.layer_idx].keys.device

        # 1. Attention sink: 前 sink_size 个 token
        sink_idx = torch.arange(0, self.sink_size, device=device).unsqueeze(0).expand(num_kv_h, -1)

        # 2. Needle tokens: 在中间区域（排除 sink 和 recent）选择 attention 最高的 top-k
        middle_start = self.sink_size
        middle_end = kv_seq_len - self.recent_size
        if middle_end > middle_start and self.hh_size > 0:
            middle_scores = needle_scores[:, middle_start:middle_end]
            actual_hh_size = min(self.hh_size, middle_end - middle_start)
            _, topk_idx = torch.topk(middle_scores, actual_hh_size, dim=-1)
            topk_idx = (topk_idx + middle_start).sort(dim=-1).values
        else:
            topk_idx = torch.empty(num_kv_h, 0, dtype=torch.long, device=device)

        # 3. Recent tokens: 最近 recent_size 个 token
        recent_idx = torch.arange(
            kv_seq_len - self.recent_size, kv_seq_len, device=device
        ).unsqueeze(0).expand(num_kv_h, -1)

        # 合并所有保留索引
        keep_idx = torch.cat([sink_idx, topk_idx, recent_idx], dim=-1)

        # 使用 mask 选择保留的 KV
        mask = torch.zeros(num_kv_h, kv_seq_len, dtype=torch.bool, device=device)
        mask.scatter_(-1, keep_idx, True)

        k_kept = past_key_values.layers[self.layer_idx].keys.squeeze(0)[mask].view(
            1, num_kv_h, -1, head_dim
        )
        v_kept = past_key_values.layers[self.layer_idx].values.squeeze(0)[mask].view(
            1, num_kv_h, -1, head_dim
        )
        past_key_values.layers[self.layer_idx].keys = k_kept
        past_key_values.layers[self.layer_idx].values = v_kept

        return past_key_values

    def _clean_scores(self):
        self.hh_score = None
        NeedleLLMKVCache_LayerWise._shared_needle_score = None


class NeedleLLMQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = NeedleLLMKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            needle_layer=getattr(config, 'hh_layer', 27),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # NeedleLLM eviction: 使用最后 query token 的注意力分布识别 needle token
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None


class HybridKVCache_LayerWise:
    """
    HybridKV KV Cache 压缩策略 (Hybrid KV Cache Compression for Efficient MLLM Inference)。

    核心思想：对视觉 token 和文本 token 采用不同的 KV cache 压缩策略，利用两种模态
    在注意力分布上的不同特性来实现更高效的压缩。

    关键观察：
    - 视觉 token 的注意力分布更均匀（高熵），不适合用驱逐策略（会丢失分散的重要信息）
    - 文本 token 的注意力分布更集中（低熵），适合用驱逐策略（只保留高注意力的 token）
    - 视觉 token 对量化更鲁棒

    压缩策略：
    - 视觉 token: 全部保留，不参与驱逐（论文中使用低比特量化，此处简化为全保留）
    - 文本 token: 基于注意力分数的 top-k 驱逐

    视觉 token 识别：
    - 方式1（推荐）: 通过类变量 _visual_token_mask 外部设置（多模态模型可通过 input_ids 解析）
    - 方式2（自动）: 通过注意力分布的熵自动检测（高熵连续区域标记为视觉 token）

    保留策略：
    - Attention Sink: 保留前 sink_size 个 token（位置编码稳定性）
    - Visual tokens: 全部保留（模态保护）
    - Important text tokens: 按注意力分数选择 top-k（关键信息）
    - Recent tokens: 保留最近的 recent_size 个 token（时序连贯性）

    跨层共享机制：
    - 当 layer_idx <= hh_layer 时，计算当前层的 score 和视觉 token 检测结果并共享
    - 当 layer_idx > hh_layer 时，复用共享的 score 和视觉 token mask
    """

    _shared_hybrid_score = None       # 类级别共享 attention score
    _shared_visual_token_mask = None  # 类级别共享视觉 token mask（自动检测结果）
    _visual_token_mask = None         # 外部设置的视觉 token mask（优先级最高）

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=None,
        recent_ratio=None,
        hh_layer=27,
        visual_token_budget_ratio=1.0,  # 视觉 token 保留比例 (1.0=全保留)
        entropy_threshold_percentile=0.7,  # 自动检测时的熵阈值百分位
        min_visual_block_size=16,  # 自动检测时最小视觉 token 块大小
    ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"HybridKVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio}, "
                  f"hh_layer={hh_layer}, visual_budget={visual_token_budget_ratio}, "
                  f"entropy_pct={entropy_threshold_percentile}, min_block={min_visual_block_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.sink_size = 4
        self.hh_layer = hh_layer
        self.visual_token_budget_ratio = visual_token_budget_ratio
        self.entropy_threshold_percentile = entropy_threshold_percentile
        self.min_visual_block_size = min_visual_block_size

    def _detect_visual_tokens(self, attn_scores, seq_len, device):
        """
        通过注意力分布的熵自动检测视觉 token。

        视觉 token 特征：
        1. 作为 query 时，注意力分布更均匀（高熵）
        2. 通常在序列中形成连续的大块

        检测方法：
        1. 计算每个 token（作为 query）的注意力分布熵
        2. 使用阈值区分高熵（视觉）和低熵（文本）token
        3. 通过连续性约束过滤噪声（只保留足够大的连续块）

        Args:
            attn_scores: [num_kv_heads, seq_len] 聚合后的注意力分数
            seq_len: 序列长度
            device: 设备

        Returns:
            visual_mask: [seq_len] bool tensor, True 表示视觉 token
        """
        # 使用注意力分数的分布特征来区分
        # 视觉 token 收到的注意力更均匀分布在多个 query 上
        # 文本 token 收到的注意力更集中在少数几个 query 上

        # 计算每个 key token 收到的注意力分数的变异系数 (CV = std/mean)
        # 低 CV = 均匀分布 → 视觉 token
        # 高 CV = 集中分布 → 文本 token
        mean_scores = attn_scores.mean(dim=0)  # [seq_len]
        std_scores = attn_scores.std(dim=0)    # [seq_len]
        cv = std_scores / (mean_scores + 1e-8)  # [seq_len]

        # 使用百分位阈值
        threshold = torch.quantile(cv, self.entropy_threshold_percentile)
        raw_visual_mask = cv < threshold  # 低变异系数 = 视觉 token

        # 连续性约束：只保留足够大的连续块
        visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # 找到连续的 True 区间
        if raw_visual_mask.any():
            changes = torch.diff(raw_visual_mask.int(), prepend=torch.tensor([0], device=device),
                                 append=torch.tensor([0], device=device))
            starts = (changes == 1).nonzero(as_tuple=True)[0]
            ends = (changes == -1).nonzero(as_tuple=True)[0]

            for s, e in zip(starts, ends):
                block_size = e - s
                if block_size >= self.min_visual_block_size:
                    visual_mask[s:e] = True

        return visual_mask

    def __call__(self, past_key_values, attn_score_cache):
        """
        基于 HybridKV 策略进行模态感知的 KV cache 驱逐。

        Args:
            past_key_values: DynamicCache 对象
            attn_score_cache: [bsz, num_heads, q_len, seq_len] 注意力权重
        """
        if attn_score_cache is None:
            return past_key_values

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape

        # 获取 KV heads 数量并聚合
        num_kv_heads = past_key_values.layers[self.layer_idx].keys.shape[1]
        num_kv_groups = num_heads // num_kv_heads

        kv_seq_len = past_key_values.layers[self.layer_idx].keys.size(self.k_seq_dim)
        device = past_key_values.layers[self.layer_idx].keys.device

        # ============ 1. 获取注意力分数 ============
        if self.layer_idx > self.hh_layer and HybridKVCache_LayerWise._shared_hybrid_score is not None:
            hybrid_scores = HybridKVCache_LayerWise._shared_hybrid_score
        else:
            # 使用最后一个 query token 的注意力分布
            hybrid_scores = attn_score_cache[:, :, -1, :]
            hybrid_scores = hybrid_scores.view(bsz, num_kv_heads, num_kv_groups, seq_len)
            hybrid_scores = hybrid_scores.sum(dim=2)
            hybrid_scores = hybrid_scores.squeeze(0)  # [num_kv_heads, seq_len]
            HybridKVCache_LayerWise._shared_hybrid_score = hybrid_scores

        # ============ 2. 获取视觉 token mask ============
        if HybridKVCache_LayerWise._visual_token_mask is not None:
            # 优先使用外部设置的视觉 token mask
            visual_mask = HybridKVCache_LayerWise._visual_token_mask
            if visual_mask.shape[0] != kv_seq_len:
                # 如果 mask 长度与 KV 序列不匹配（可能由于之前的驱逐），截断或扩展
                if visual_mask.shape[0] > kv_seq_len:
                    visual_mask = visual_mask[:kv_seq_len]
                else:
                    visual_mask = F.pad(visual_mask, (0, kv_seq_len - visual_mask.shape[0]), value=False)
            visual_mask = visual_mask.to(device)
        elif self.layer_idx > self.hh_layer and HybridKVCache_LayerWise._shared_visual_token_mask is not None:
            visual_mask = HybridKVCache_LayerWise._shared_visual_token_mask
        else:
            # 自动检测视觉 token
            visual_mask = self._detect_visual_tokens(hybrid_scores, seq_len, device)
            HybridKVCache_LayerWise._shared_visual_token_mask = visual_mask

        num_visual_tokens = visual_mask.sum().item()

        # ============ 3. 计算预算 ============
        if self.hh_ratio is not None:
            # 文本 token 的 hh_size 需要扣除视觉 token 占用的预算
            total_budget = int(seq_len * (self.hh_ratio + self.recent_ratio))
            self.recent_size = int(seq_len * self.recent_ratio) + 15
            # 视觉 token 保留预算
            visual_keep = int(num_visual_tokens * self.visual_token_budget_ratio)
            # 文本 token 的 hh 预算 = 总预算 - 视觉保留 - recent - sink
            text_hh_budget = max(0, total_budget - visual_keep - self.recent_size - self.sink_size)
        else:
            visual_keep = int(num_visual_tokens * self.visual_token_budget_ratio)
            text_hh_budget = max(0, self.hh_size - visual_keep)
            self.recent_size = self.recent_size

        cache_size = self.sink_size + visual_keep + text_hh_budget + self.recent_size

        if kv_seq_len <= cache_size:
            return past_key_values

        # 结合 value 的 L2 norm
        value_L2 = torch.norm(past_key_values.layers[self.layer_idx].values[0], p=2, dim=-1)
        importance_scores = hybrid_scores * value_L2

        # ============ 4. 构建保留索引 ============
        _, num_kv_h, _, head_dim = past_key_values.layers[self.layer_idx].keys.shape

        # 构建保留 mask
        keep_mask = torch.zeros(num_kv_h, kv_seq_len, dtype=torch.bool, device=device)

        # 4a. Attention sink: 前 sink_size 个 token
        keep_mask[:, :self.sink_size] = True

        # 4b. 视觉 token: 全部保留（模态保护）
        # visual_mask 扩展到所有 head
        visual_positions = visual_mask.nonzero(as_tuple=True)[0]
        if len(visual_positions) > 0:
            if self.visual_token_budget_ratio >= 1.0:
                # 全部保留
                keep_mask[:, visual_positions] = True
            else:
                # 按预算保留（基于 importance 选择最重要的视觉 token）
                visual_importance = importance_scores[:, visual_positions]  # [num_kv_h, num_visual]
                actual_visual_keep = min(visual_keep, len(visual_positions))
                if actual_visual_keep > 0:
                    _, visual_topk = torch.topk(visual_importance, actual_visual_keep, dim=-1)
                    for h in range(num_kv_h):
                        keep_mask[h, visual_positions[visual_topk[h]]] = True

        # 4c. Recent tokens: 最近 recent_size 个 token
        recent_start = max(0, kv_seq_len - self.recent_size)
        keep_mask[:, recent_start:] = True

        # 4d. Important text tokens: 在中间区域的文本 token 中选择 top-k
        # 中间区域：排除 sink、recent 以及视觉 token
        middle_start = self.sink_size
        middle_end = kv_seq_len - self.recent_size

        if middle_end > middle_start and text_hh_budget > 0:
            # 构建文本 token 的候选 mask（排除视觉 token 和已保留的 token）
            text_candidate_mask = torch.ones(kv_seq_len, dtype=torch.bool, device=device)
            text_candidate_mask[:middle_start] = False  # 排除 sink
            text_candidate_mask[middle_end:] = False    # 排除 recent
            text_candidate_mask[visual_mask] = False    # 排除视觉 token

            text_positions = text_candidate_mask.nonzero(as_tuple=True)[0]

            if len(text_positions) > 0:
                text_importance = importance_scores[:, text_positions]  # [num_kv_h, num_text_candidates]
                actual_text_hh = min(text_hh_budget, len(text_positions))
                _, text_topk = torch.topk(text_importance, actual_text_hh, dim=-1)
                text_topk_positions = text_positions[text_topk]  # [num_kv_h, actual_text_hh]
                keep_mask.scatter_(1, text_topk_positions, True)

        # ============ 5. 应用驱逐 ============
        k_kept = past_key_values.layers[self.layer_idx].keys.squeeze(0)[keep_mask].view(
            1, num_kv_h, -1, head_dim
        )
        v_kept = past_key_values.layers[self.layer_idx].values.squeeze(0)[keep_mask].view(
            1, num_kv_h, -1, head_dim
        )
        past_key_values.layers[self.layer_idx].keys = k_kept
        past_key_values.layers[self.layer_idx].values = v_kept

        return past_key_values

    def _clean_scores(self):
        self.hh_score = None
        HybridKVCache_LayerWise._shared_hybrid_score = None
        HybridKVCache_LayerWise._shared_visual_token_mask = None
        # 注意：不清理 _visual_token_mask，因为它由外部设置

    @classmethod
    def set_visual_token_mask(cls, mask: torch.Tensor):
        """
        外部设置视觉 token 的位置 mask。

        在多模态模型中，可以通过分析 input_ids 中的 image_token_id / video_token_id
        来构建这个 mask，然后在 prefill 前设置。

        Args:
            mask: [seq_len] bool tensor, True 表示视觉 token
        """
        cls._visual_token_mask = mask

    @classmethod
    def clear_visual_token_mask(cls):
        """清除外部设置的视觉 token mask。"""
        cls._visual_token_mask = None


class HybridKVQwen3_5Attention_drop(H2OQwen3_5Attention_drop):
    """
    HybridKV Attention: 基于模态感知的混合 KV Cache 压缩。

    核心区别：区分视觉 token 和文本 token，对视觉 token 保护不驱逐（论文中使用
    低比特量化），对文本 token 使用基于注意力分数的 top-k 驱逐。

    适用于多模态大语言模型（MLLM），通过利用视觉 token 注意力分布均匀、文本 token
    注意力分布集中的特性，实现比统一驱逐策略更好的压缩效果。

    参考论文: HybridKV: Hybrid KV Cache Compression for Efficient Multimodal
              Large Language Model Inference
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = HybridKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            hh_layer=getattr(config, 'hh_layer', 27),
            visual_token_budget_ratio=getattr(config, 'visual_token_budget_ratio', 1.0),
            entropy_threshold_percentile=getattr(config, 'entropy_threshold_percentile', 0.7),
            min_visual_block_size=getattr(config, 'min_visual_block_size', 16),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        bsz, num_heads, q_len, head_dim = query_states.shape

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)

        # HybridKV eviction: 模态感知的混合压缩（保护视觉 token，驱逐文本 token）
        if q_len != 1:
            self.kv_cache(past_key_values, attn_weights)

        return attn_output, None