"""统一多模态数据集评测组件。"""

from evaluation.datasets import (
    BenchmarkSample,
    DatasetAdapter,
    get_adapter,
    load_dataset_split,
    supported_datasets,
)
from evaluation.runner import predict_sample, run_benchmark

__all__ = [
    "BenchmarkSample",
    "DatasetAdapter",
    "get_adapter",
    "load_dataset_split",
    "predict_sample",
    "run_benchmark",
    "supported_datasets",
]
