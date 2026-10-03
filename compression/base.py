from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompressionResult:
    """输入压缩结果及可序列化的评测元数据。"""

    original_prompt: str
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        if self.compressed_tokens <= 0:
            return 1.0
        return self.origin_tokens / self.compressed_tokens

    @property
    def info(self) -> dict[str, Any]:
        return {
            "origin_tokens": self.origin_tokens,
            "compressed_tokens": self.compressed_tokens,
            "ratio": self.ratio,
            "latency_seconds": self.latency_seconds,
            **self.metadata,
        }


class InputCompressor(Protocol):
    """模型封装层使用的文本输入压缩器协议。"""

    def compress(self, prompt: str) -> CompressionResult:
        ...

    def recover(self, result: CompressionResult, response: str) -> str:
        ...
