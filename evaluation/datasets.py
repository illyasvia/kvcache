"""Hugging Face 多模态基准数据集适配器。"""

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.scoring import score_prediction


@dataclass(frozen=True)
class BenchmarkSample:
    """模型推理所需的统一样本表示。"""

    sample_id: str
    prompt: str
    images: tuple[Any, ...]
    answers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetAdapter:
    """描述数据源、字段转换和评分方式。"""

    name: str
    repository: str
    local_path: str
    split: str
    scorer: str
    config: str | None = None

    def to_sample(self, record: dict[str, Any], index: int) -> BenchmarkSample:
        builders = {
            "blink": _build_blink_sample,
            "docvqa": _build_docvqa_sample,
            "mathvista": _build_mathvista_sample,
            "mmmu": _build_mmmu_sample,
            "mmstar": _build_mmstar_sample,
            "textvqa": _build_textvqa_sample,
        }
        return builders[self.name](record, index)

    def score(self, prediction: str, answers: tuple[str, ...]) -> float:
        return score_prediction(prediction, answers, self.scorer)


def _answers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _sample_id(record: dict[str, Any], index: int) -> str:
    for key in ("questionId", "question_id", "pid", "id", "idx"):
        if record.get(key) is not None:
            return f"{index}:{record[key]}"
    return str(index)


def _metadata(record: dict[str, Any], index: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_index": index}
    for key in (
        "questionId", "question_id", "pid", "id", "idx", "category", "task",
        "sub_task", "question_type",
    ):
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def _require_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"样本缺少文本字段，候选字段: {', '.join(keys)}")


def _require_images(record: dict[str, Any], *keys: str) -> tuple[Any, ...]:
    images: list[Any] = []
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            images.extend(image for image in value if image is not None)
        else:
            images.append(value)
    if not images:
        raise ValueError(f"样本缺少图像字段，候选字段: {', '.join(keys)}")
    return tuple(images)


def _parse_options(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            matches = re.findall(r"(?:^|\n)\s*\(?([A-Z])\)?[.):]\s*(.+)", value)
            return [(label, text.strip()) for label, text in matches]
    if isinstance(value, dict):
        return [(str(label), str(text)) for label, text in value.items()]
    if isinstance(value, (list, tuple)):
        return [(chr(ord("A") + index), str(text)) for index, text in enumerate(value)]
    return []


def _with_options(question: str, record: dict[str, Any]) -> str:
    options = _parse_options(record.get("options", record.get("choices")))
    if not options:
        return question
    rendered = "\n".join(f"({label}) {text}" for label, text in options)
    return f"{question}\n\nOptions:\n{rendered}\n\nAnswer with the option letter only."


def _build_docvqa_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=(
            f"{_require_text(record, 'question')}\n"
            "Answer using only the text visible in the document image."
        ),
        images=_require_images(record, "image"),
        answers=_answers(record.get("answers", record.get("answer"))),
        metadata=_metadata(record, index),
    )


def _build_textvqa_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=(
            f"{_require_text(record, 'question')}\n"
            "Answer briefly using the text visible in the image."
        ),
        images=_require_images(record, "image"),
        answers=_answers(record.get("answers", record.get("answer"))),
        metadata=_metadata(record, index),
    )


def _build_mathvista_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    question = _with_options(_require_text(record, "query", "question"), record)
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=f"{question}\nGive only the final answer.",
        images=_require_images(record, "decoded_image", "image"),
        answers=_answers(record.get("answer")),
        metadata=_metadata(record, index),
    )


def _build_blink_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    image_keys = tuple(f"image_{number}" for number in range(1, 9)) + ("image",)
    question = _with_options(_require_text(record, "question"), record)
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=f"{question}\nGive only the final answer.",
        images=_require_images(record, *image_keys),
        answers=_answers(record.get("answer", record.get("answers"))),
        metadata=_metadata(record, index),
    )


def _build_mmmu_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    image_keys = tuple(f"image_{number}" for number in range(1, 8))
    question = _with_options(_require_text(record, "question"), record)
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=f"{question}\nGive only the final answer.",
        images=_require_images(record, *image_keys),
        answers=_answers(record.get("answer")),
        metadata=_metadata(record, index),
    )


def _build_mmstar_sample(record: dict[str, Any], index: int) -> BenchmarkSample:
    question = _with_options(_require_text(record, "question"), record)
    return BenchmarkSample(
        sample_id=_sample_id(record, index),
        prompt=f"{question}\nGive only the option letter.",
        images=_require_images(record, "image"),
        answers=_answers(record.get("answer")),
        metadata=_metadata(record, index),
    )


_ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        DatasetAdapter("blink", "BLINK-Benchmark/BLINK", "BLINK-Benchmark", "val", "multiple_choice"),
        DatasetAdapter("docvqa", "lmms-lab/DocVQA", "lmms-lab/DocVQA", "validation", "anls", "DocVQA"),
        DatasetAdapter("mathvista", "AI4Math/MathVista", "AI4Math/MathVista", "testmini", "exact_match"),
        DatasetAdapter("mmmu", "lmms-lab/MMMU", "lmms-lab/MMMU", "validation", "multiple_choice"),
        DatasetAdapter("mmstar", "Lin-Chen/MMStar", "Lin-Chen/MMStar", "val", "multiple_choice"),
        DatasetAdapter("textvqa", "lmms-lab/TextVQA", "lmms-lab/TextVQA", "validation", "vqa"),
    )
}


def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_adapter(name: str) -> DatasetAdapter:
    normalized_name = name.lower()
    if normalized_name not in _ADAPTERS:
        raise ValueError(
            f"未支持的数据集: {name}；可选值: {', '.join(supported_datasets())}"
        )
    return _ADAPTERS[normalized_name]


def resolve_dataset_source(
    adapter: DatasetAdapter,
    dataset_path: str | None = None,
    dataset_root: str | None = None,
) -> str:
    if dataset_path:
        return dataset_path
    root = dataset_root or os.environ.get("HF_DATASET_ROOT")
    if root:
        return str(Path(root).expanduser() / adapter.local_path)
    return adapter.repository


def load_dataset_split(
    adapter: DatasetAdapter,
    dataset_path: str | None = None,
    dataset_root: str | None = None,
    config: str | None = None,
    split: str | None = None,
):
    """按适配器配置加载数据；未设置本地根目录时从 Hugging Face 获取。"""
    from datasets import load_dataset

    source = resolve_dataset_source(adapter, dataset_path, dataset_root)
    selected_config = adapter.config if config is None else config
    try:
        dataset = load_dataset(source, selected_config) if selected_config else load_dataset(source)
    except ValueError as error:
        if adapter.name == "blink" and selected_config is None:
            raise ValueError(
                "BLINK 包含多个子任务，请通过 --dataset_config 指定配置，"
                "例如 Counting 或 Relative_Depth"
            ) from error
        raise

    selected_split = split or adapter.split
    split_aliases = {"val": "validation", "validation": "val"}
    if selected_split not in dataset and split_aliases.get(selected_split) in dataset:
        selected_split = split_aliases[selected_split]
    if selected_split not in dataset:
        raise ValueError(
            f"数据集 {adapter.name} 不包含 split={selected_split}；"
            f"可选值: {', '.join(dataset.keys())}"
        )
    return dataset[selected_split]
