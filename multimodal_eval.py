"""DocVQA、TextVQA、MathVista 等多模态数据集的统一评测入口。"""

import argparse
import os
from pathlib import Path

from evaluation import get_adapter, load_dataset_split, run_benchmark, supported_datasets

BASE_DIR = Path(__file__).resolve().parent
MODEL_DEFAULT_PATHS = {
    "qwen3vl": BASE_DIR / "weight/Qwen/Qwen3-VL-4B-Instruct",
    "qwen35": BASE_DIR / "weight/Qwen/Qwen3.5-2B",
}


def build_model(model_type: str, model_path: str | None = None):
    selected_path = str(Path(model_path).expanduser()) if model_path else str(
        MODEL_DEFAULT_PATHS[model_type]
    )
    if model_type == "qwen3vl":
        from model.qwen3vl.Qwen3_vl import Qwen3vl

        return Qwen3vl(model_path=selected_path)
    if model_type == "qwen35":
        from model.qwen35.Qwen35 import Qwen35

        return Qwen35(model_path=selected_path, kv_mode="origin")
    raise ValueError(f"未支持的模型类型: {model_type}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="统一运行 DocVQA、TextVQA、BLINK、MathVista、MMMU 和 MMStar 评测"
    )
    parser.add_argument("--dataset", required=True, choices=supported_datasets())
    parser.add_argument("--model_type", required=True, choices=sorted(MODEL_DEFAULT_PATHS))
    parser.add_argument("--model_path", default=None, help="模型权重目录")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="数据集本地目录或 Hugging Face 仓库名；优先于 HF_DATASET_ROOT",
    )
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get("HF_DATASET_ROOT"),
        help="本地数据集根目录，缺省读取 HF_DATASET_ROOT",
    )
    parser.add_argument("--dataset_config", default=None, help="覆盖数据集 config")
    parser.add_argument("--split", default=None, help="覆盖默认 split")
    parser.add_argument("--start", type=int, default=0, help="起始样本下标")
    parser.add_argument("--limit", type=int, default=None, help="最多评测的样本数")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="从已有结果文件断点续跑")
    parser.add_argument(
        "--output_dir",
        default=str(BASE_DIR / "output/multimodal"),
        help="评测结果目录",
    )
    parser.add_argument("--output_file", default=None, help="覆盖自动生成的结果文件名")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    adapter = get_adapter(args.dataset)
    selected_split = args.split or adapter.split
    records = load_dataset_split(
        adapter,
        dataset_path=args.dataset_path,
        dataset_root=args.dataset_root,
        config=args.dataset_config,
        split=selected_split,
    )
    model = build_model(args.model_type, args.model_path)

    output_path = Path(args.output_file) if args.output_file else (
        Path(args.output_dir) / f"{adapter.name}_{selected_split}_{args.model_type}.json"
    )
    result = run_benchmark(
        model,
        adapter,
        records,
        output_path,
        split=selected_split,
        start=args.start,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
    )
    summary = result["_summary"]
    print(f"结果已保存至: {output_path}")
    print(
        f"完成 {summary['completed_samples']} 个样本，"
        f"平均分 {summary['average_score']:.2f}"
    )
    return result


if __name__ == "__main__":
    main()
