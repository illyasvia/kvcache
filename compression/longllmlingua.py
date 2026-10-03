import time
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .base import CompressionResult
from .needle_prompt import parse_needle_prompt


def _to_dynamic_cache(past_key_values: Any, model: Any) -> Any:
    """将 LLMLingua 使用的旧式 KV 列表转换为 Transformers 5 Cache。"""
    if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    return DynamicCache(
        [(key, value) for key, value in past_key_values],
        config=model.config,
    )


def _to_legacy_cache(past_key_values: Any) -> Any:
    """将 Transformers 5 Cache 转成 LLMLingua 0.2.2 可切片的 KV 列表。"""
    if not hasattr(past_key_values, "layers"):
        return past_key_values
    return [
        [layer.keys, layer.values]
        for layer in past_key_values.layers
        if getattr(layer, "keys", None) is not None
    ]


def _transformers5_get_ppl(
    backend: Any,
    text: str,
    granularity: str = "sentence",
    input_ids: Any = None,
    attention_mask: Any = None,
    past_key_values: Any = None,
    return_kv: bool = False,
    end: int | None = None,
    condition_mode: str = "none",
    condition_pos_id: int = 0,
) -> Any:
    """兼容 Transformers 5 Cache API 的 PromptCompressor.get_ppl。"""
    if input_ids is None:
        tokenized_text = backend.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(backend.device)
        attention_mask = tokenized_text["attention_mask"].to(backend.device)

    legacy_cache = past_key_values
    past_length = legacy_cache[0][0].shape[2] if legacy_cache is not None else 0
    if end is None:
        end = input_ids.shape[1]
    end = min(end, past_length + backend.max_position_embeddings)

    model_cache = _to_dynamic_cache(legacy_cache, backend.model)
    with torch.no_grad():
        response = backend.model(
            input_ids[:, past_length:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=model_cache,
            use_cache=True,
        )

    shift_logits = response.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., past_length + 1:end].contiguous()
    active = (attention_mask[:, past_length:end] == 1)[..., :-1].reshape(-1)
    active_logits = shift_logits.reshape(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.reshape(-1)[active]
    loss = torch.nn.functional.cross_entropy(
        active_logits,
        active_labels,
        reduction="none",
    )
    if condition_mode == "before":
        loss = loss[:condition_pos_id]
    elif condition_mode == "after":
        loss = loss[condition_pos_id:]

    result = loss.mean() if granularity == "sentence" else loss
    cache = _to_legacy_cache(response.past_key_values)
    return (result, cache) if return_kv else result


@dataclass(frozen=True)
class LongLLMLinguaConfig:
    model_name: str = "microsoft/phi-2"
    device: str = "auto"
    rate: float = 0.5
    chunk_tokens: int = 512
    iterative_size: int = 200
    reorder_context: str = "sort"
    dynamic_context_compression_ratio: float = 0.3
    context_budget_offset: int = 100
    enable_recovery: bool = True

    def validate(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValueError("compression rate 必须位于 (0, 1] 区间")
        if self.chunk_tokens <= 0:
            raise ValueError("compression chunk_tokens 必须大于 0")
        if self.iterative_size <= 0:
            raise ValueError("compression iterative_size 必须大于 0")
        if self.reorder_context not in {"original", "sort", "two_stage"}:
            raise ValueError("reorder_context 必须是 original、sort 或 two_stage")
        if not 0 <= self.dynamic_context_compression_ratio <= 1:
            raise ValueError("dynamic_context_compression_ratio 必须位于 [0, 1] 区间")


class LongLLMLinguaCompressor:
    """官方 LLMLingua `PromptCompressor` 的 Needle benchmark 适配器。"""

    def __init__(
        self,
        config: LongLLMLinguaConfig,
        backend: Any = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = self._resolve_device(config.device)
        if backend is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise RuntimeError(
                    "启用 LongLLMLingua 需要安装 llmlingua："
                    "pip install llmlingua==0.2.2"
                ) from exc
            backend = PromptCompressor(
                model_name=config.model_name,
                device_map=self.device,
            )
            # llmlingua 0.2.2 直接遍历旧式 (key, value) tuples；项目依赖的
            # Transformers 5 返回 DynamicCache，因此在边界处做双向转换。
            backend.get_ppl = MethodType(_transformers5_get_ppl, backend)
        self.backend = backend

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.backend.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return list(input_ids)

    def split_context(self, context: str) -> list[str]:
        """按压缩模型 tokenizer 切块，使 coarse-grained ranking 可用于单篇长文档。"""
        token_ids = self._token_ids(context)
        chunks = []
        for start in range(0, len(token_ids), self.config.chunk_tokens):
            chunk_ids = token_ids[start:start + self.config.chunk_tokens]
            chunk = self.backend.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if chunk.strip():
                chunks.append(chunk)
        return chunks or [context]

    def compress(self, prompt: str) -> CompressionResult:
        parsed = parse_needle_prompt(prompt)
        contexts = self.split_context(parsed.context)

        started_at = time.perf_counter()
        result = self.backend.compress_prompt(
            contexts,
            instruction=parsed.instruction,
            question=parsed.question,
            rate=self.config.rate,
            iterative_size=self.config.iterative_size,
            use_sentence_level_filter=False,
            use_context_level_filter=len(contexts) > 1,
            use_token_level_filter=True,
            keep_split=False,
            condition_in_question="after_condition",
            reorder_context=self.config.reorder_context,
            dynamic_context_compression_ratio=self.config.dynamic_context_compression_ratio,
            condition_compare=True,
            context_budget=f"{self.config.context_budget_offset:+d}",
            rank_method="longllmlingua",
        )
        latency_seconds = time.perf_counter() - started_at

        return CompressionResult(
            original_prompt=prompt,
            compressed_prompt=result["compressed_prompt"],
            origin_tokens=int(result["origin_tokens"]),
            compressed_tokens=int(result["compressed_tokens"]),
            latency_seconds=latency_seconds,
            metadata={
                "method": "longllmlingua",
                "model_name": self.config.model_name,
                "device": self.device,
                "target_rate": self.config.rate,
                "chunk_tokens": self.config.chunk_tokens,
                "iterative_size": self.config.iterative_size,
                "context_chunks": len(contexts),
                "reorder_context": self.config.reorder_context,
                "dynamic_context_compression_ratio": (
                    self.config.dynamic_context_compression_ratio
                ),
                "recovery_enabled": self.config.enable_recovery,
            },
        )

    def recover(self, result: CompressionResult, response: str) -> str:
        if not self.config.enable_recovery:
            return response
        return self.backend.recover(
            result.original_prompt,
            result.compressed_prompt,
            response,
        )
