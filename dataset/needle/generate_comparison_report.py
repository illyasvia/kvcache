"""汇总 NeedleBench V2 1000k 三模式输出并生成 Markdown 实验报告。"""

import argparse
import datetime
import json
import platform
from pathlib import Path

import torch


MODE_CONFIGS = {
    "origin": {
        "name": "全量 KV Cache",
        "output_dir": "needlebench_v2_1000k_origin",
        "configuration": "kv_mode=origin",
    },
    "h2o": {
        "name": "H2O",
        "output_dir": "needlebench_v2_1000k_h2o",
        "configuration": "heavy=512, recent=512, chunk=512",
    },
    "longllmlingua": {
        "name": "LongLLMLingua",
        "output_dir": "needlebench_v2_1000k_longllmlingua",
        "configuration": "rate=0.5, chunk=512, reorder=sort",
    },
}
DEPTHS = (0, 50, 100)


def numeric_samples(data: dict) -> list[dict]:
    return [data[key] for key in sorted(data, key=lambda value: int(value) if value.isdigit() else -1) if key.isdigit()]


def score_sample(sample: dict) -> float:
    keyword = sample["gold"].rsplit("*", 1)[-1].strip()
    return 100.0 if keyword in sample.get("pre", "") else 0.0


def load_mode(output_root: Path, mode: str) -> dict:
    config = MODE_CONFIGS[mode]
    mode_dir = output_root / config["output_dir"]
    result = {
        "mode": config["name"],
        "configuration": config["configuration"],
        "completed": 0,
        "expected": len(DEPTHS) * 10,
        "scores": [],
        "latencies": [],
        "input_tokens": [],
        "peak_memory_gb": 0.0,
        "peak_rss_gb": 0.0,
        "compression_origin_tokens": 0,
        "compression_output_tokens": 0,
        "compression_seconds": 0.0,
        "h2o_evictions": 0,
        "files": [],
    }
    for depth in DEPTHS:
        path = mode_dir / f"Length1000000Depth{depth}.json"
        if not path.exists():
            continue
        result["files"].append(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        completed = [sample for sample in numeric_samples(data) if "pre" in sample]
        result["completed"] += len(completed)
        result["scores"].extend(score_sample(sample) for sample in completed)
        result["latencies"].extend(sample.get("inference_seconds", 0.0) for sample in completed)
        result["input_tokens"].extend(
            sample["input_tokens"] for sample in completed if sample.get("input_tokens")
        )
        memory = data.get("_memory_stats", {})
        result["peak_memory_gb"] = max(
            result["peak_memory_gb"], memory.get("peak_memory_gb", 0.0)
        )
        result["peak_rss_gb"] = max(
            result["peak_rss_gb"], memory.get("peak_process_rss_gb", 0.0)
        )
        compression = data.get("_compression_stats", {})
        result["compression_origin_tokens"] += compression.get("origin_tokens", 0)
        result["compression_output_tokens"] += compression.get("compressed_tokens", 0)
        result["compression_seconds"] += compression.get("total_latency_seconds", 0.0)
        result["h2o_evictions"] += data.get("_h2o_stats", {}).get(
            "layer_token_evictions", 0
        )
    return result


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_float(value: float) -> str:
    return f"{value:,.2f}"


def build_report(output_root: Path, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modes = [load_mode(output_root, mode) for mode in MODE_CONFIGS]
    source_info = {}
    for depth in DEPTHS:
        source_path = manifest_path.parent / f"Length1000000Depth{depth}_origin_en_1000k.json"
        if source_path.exists():
            source_data = json.loads(source_path.read_text(encoding="utf-8"))
            source_info = source_data.get("_dataset_info", {})
            break

    lines = [
        "# NeedleBench V2 1000k 三模式性能对比报告",
        "",
        f"生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 实验设置",
        "",
        "- 主模型：Qwen3.5-2B",
        "- 数据集：opencompass/NeedleBench，Single-Needle Retrieval（English）",
        f"- 数据版本：`{manifest.get('revision', 'unknown')}`",
        "- 名义上下文长度：1,000,000 GPT-4 tokens",
        "- 深度与样本：0/50/100，每个深度 10 个，共 30 个",
        "- 生成上限：128 tokens；评分：完整关键词大小写敏感命中（0/100）",
        f"- 实际上下文长度：{source_info.get('actual_context_tokens_min', 'unknown'):,}–{source_info.get('actual_context_tokens_max', 'unknown'):,} GPT-4 tokens",
        f"- 运行环境：{platform.platform()}；PyTorch {torch.__version__}；MPS={torch.backends.mps.is_available()}；CUDA={torch.cuda.is_available()}",
        "",
        "> 官方英文 Paul Graham 语料不足以填满 1000k；本实验忠实采用官方生成逻辑，不循环复制语料。",
        "",
        "## 汇总结果",
        "",
        "| 模式 | 进度 | 准确率 | 平均端到端耗时/样本 | 平均模型输入 tokens | 峰值设备内存 | 峰值进程 RSS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        lines.append(
            f"| {mode['mode']} | {mode['completed']}/{mode['expected']} | "
            f"{format_float(average(mode['scores']))}% | "
            f"{format_float(average(mode['latencies']))} s | "
            f"{average(mode['input_tokens']):,.0f} | "
            f"{format_float(mode['peak_memory_gb'])} GiB | "
            f"{format_float(mode['peak_rss_gb'])} GiB |"
        )

    lines.extend(["", "## 模式参数与专项指标", ""])
    for mode in modes:
        lines.extend([
            f"### {mode['mode']}",
            "",
            f"- 参数：`{mode['configuration']}`",
            f"- 已完成：{mode['completed']}/{mode['expected']} 样本",
        ])
        if mode["compression_output_tokens"]:
            ratio = mode["compression_origin_tokens"] / mode["compression_output_tokens"]
            lines.extend([
                f"- LLMLingua token 压缩倍数：{ratio:.2f}×",
                f"- 压缩累计耗时：{mode['compression_seconds']:.2f} 秒",
            ])
        if mode["h2o_evictions"]:
            lines.append(f"- H2O 层级 token 淘汰总数：{mode['h2o_evictions']:,}")
        lines.append("")

    complete = all(mode["completed"] == mode["expected"] for mode in modes)
    lines.extend([
        "## 结论",
        "",
        (
            "三种模式均已完成全量评测，可依据上表比较检索准确率、端到端耗时和内存占用。"
            if complete else
            "实验仍在运行；本报告为自动更新的阶段性结果，待三种模式各完成 30 个样本后形成最终结论。"
        ),
        "",
        "## 可复现性",
        "",
        "运行入口：`run/qwen35_needlebench_v2_1000k_compare.sh`。各模式原始预测、逐样本耗时、token 数、内存和压缩统计均保存在对应输出目录。",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="生成 NeedleBench 三模式实验报告")
    parser.add_argument("--output_root", type=Path, default=project_root / "output")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "dataset" / "needlebench_v2" / "manifest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "output" / "needlebench_v2_1000k_comparison_report.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(args.output_root, args.manifest), encoding="utf-8")
    print(f"实验报告已生成：{args.report}")


if __name__ == "__main__":
    main()
