"""统一多模态推理循环、断点续跑和结果汇总。"""

import json
import os
import tempfile
import time
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.datasets import BenchmarkSample, DatasetAdapter


def normalize_response(response: Any) -> str:
    if isinstance(response, (list, tuple)):
        response = response[0] if response else None
    return "" if response is None else str(response).strip()


def predict_sample(model, sample: BenchmarkSample, max_new_tokens: int) -> str:
    """根据图像数量分发到模型的文本、单图或多图接口。"""
    if not sample.images:
        response = model.process(sample.prompt, max_new_tokens=max_new_tokens)
    elif len(sample.images) == 1:
        response = model.process_multimodel(
            sample.prompt,
            sample.images[0],
            max_new_tokens=max_new_tokens,
        )
    else:
        response = model.process_multiple_multimodel(
            sample.prompt,
            list(sample.images),
            max_new_tokens=max_new_tokens,
        )
    return normalize_response(response)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _new_result(adapter: DatasetAdapter, split: str, model_name: str) -> dict[str, Any]:
    return {
        "_meta": {
            "dataset": adapter.name,
            "split": split,
            "scorer": adapter.scorer,
            "model": model_name,
        },
        "samples": {},
        "_summary": {},
    }


def _load_result(
    output_path: Path,
    adapter: DatasetAdapter,
    split: str,
    model_name: str,
    resume: bool,
) -> dict[str, Any]:
    if not resume or not output_path.exists():
        return _new_result(adapter, split, model_name)
    with output_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    metadata = result.get("_meta", {})
    expected_metadata = {
        "dataset": adapter.name,
        "split": split,
        "scorer": adapter.scorer,
        "model": model_name,
    }
    mismatched = [
        key for key, value in expected_metadata.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(f"续跑文件配置不一致: {', '.join(mismatched)}")
    result.setdefault("samples", {})
    return result


def _planned_count(records: Iterable, start: int, limit: int | None) -> int | None:
    if not isinstance(records, Sized):
        return limit
    available = max(len(records) - start, 0)
    return min(available, limit) if limit is not None else available


def _update_summary(result: dict[str, Any], planned_samples: int | None) -> None:
    samples = list(result["samples"].values())
    scored = [sample["score"] for sample in samples if sample.get("score") is not None]
    result["_summary"] = {
        "completed_samples": len(samples),
        "planned_samples": planned_samples,
        "scored_samples": len(scored),
        "average_score": sum(scored) / len(scored) if scored else 0.0,
        "total_inference_seconds": sum(
            sample.get("inference_seconds", 0.0) for sample in samples
        ),
        "input_tokens_total": sum(sample.get("input_tokens", 0) for sample in samples),
        "output_tokens_total": sum(sample.get("output_tokens", 0) for sample in samples),
    }


def run_benchmark(
    model,
    adapter: DatasetAdapter,
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    split: str | None = None,
    start: int = 0,
    limit: int | None = None,
    max_new_tokens: int = 128,
    resume: bool = False,
) -> dict[str, Any]:
    """执行评测并在每个样本后原子写入检查点。"""
    if start < 0:
        raise ValueError("start 不能小于 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")

    selected_split = split or adapter.split
    model_name = str(getattr(model, "model_path", type(model).__name__))
    output_path = Path(output_path)
    result = _load_result(output_path, adapter, selected_split, model_name, resume)
    planned_samples = _planned_count(records, start, limit)
    processed_in_this_run = 0

    progress = tqdm(
        enumerate(records),
        total=len(records) if isinstance(records, Sized) else None,
        desc=f"评测 {adapter.name}",
        leave=True,
        ncols=80,
    )
    for index, record in progress:
        if index < start:
            continue
        if limit is not None and index >= start + limit:
            break

        sample = adapter.to_sample(record, index)
        if resume and sample.sample_id in result["samples"]:
            continue

        started_at = time.perf_counter()
        prediction = predict_sample(model, sample, max_new_tokens)
        elapsed = time.perf_counter() - started_at
        score = adapter.score(prediction, sample.answers) if sample.answers else None
        result["samples"][sample.sample_id] = {
            "prompt": sample.prompt,
            "answers": list(sample.answers),
            "prediction": prediction,
            "score": score,
            "inference_seconds": elapsed,
            "input_tokens": int(getattr(model, "last_input_tokens", 0)),
            "output_tokens": int(getattr(model, "last_output_tokens", 0)),
            "metadata": sample.metadata,
        }
        processed_in_this_run += 1
        _update_summary(result, planned_samples)
        _write_json_atomic(output_path, result)

    _update_summary(result, planned_samples)
    result["_summary"]["processed_in_this_run"] = processed_in_this_run
    _write_json_atomic(output_path, result)
    return result
