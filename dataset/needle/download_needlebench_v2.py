"""下载官方 NeedleBench V2 数据，并生成当前评测器可读取的单针数据。"""

import argparse
import json
import random
from pathlib import Path

import tiktoken
from huggingface_hub import HfApi, snapshot_download


REPO_ID = "opencompass/NeedleBench"
DEFAULT_DEPTHS = [0, 50, 100]
LANGUAGE_CONFIGS = {
    "en": {
        "language": "English",
        "haystack_file": "PaulGrahamEssays.jsonl",
        "length_buffer": 3000,
    },
    "zh": {
        "language": "Chinese",
        "haystack_file": "zh_finance.jsonl",
        "length_buffer": 200,
    },
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_prompt(context: str, question: str, language: str) -> str:
    if language == "English":
        return (
            "This is a test of long-text capability. You need to first read the long "
            "document below, and then answer the final question based on the information "
            "in the document.\nThe content of the long document is as follows\n\n"
            f"<Document>\n{context}\n</Document>\n\n"
            f"Based on the information in the document, now please answer: {question}\n"
        )
    return (
        "这是一个长文本能力的测试，你需要首先阅读下面的长文档，然后根据文档中的信息回答最后的问题。\n"
        "长文档的内容如下\n\n"
        f"<文档>\n{context}\n</文档>\n\n"
        f"根据文档中的信息，现在请问：{question}\n"
    )


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    context_length: int,
    depth: int,
    repeats: int,
    language_key: str,
) -> Path:
    config = LANGUAGE_CONFIGS[language_key]
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    haystack_rows = load_jsonl(source_dir / config["haystack_file"])
    needles = [
        row for row in load_jsonl(source_dir / "needles.jsonl")
        if row["language"] == config["language"]
    ]
    tokenized_rows = [tokenizer.encode(row["text"]) for row in haystack_rows]
    shuffled_rows = tokenized_rows.copy()
    target_context_length = context_length - config["length_buffer"]
    dataset = {}
    actual_context_lengths = []

    for counter in range(repeats):
        random.Random(counter).shuffle(shuffled_rows)
        needle = random.Random(counter).choice(needles)
        needle_text = "\n" + needle["needle"] + "\n"
        needle_tokens = tokenizer.encode(needle_text)
        target_haystack_length = max(target_context_length - len(needle_tokens), 0)

        accumulated_tokens = []
        for row_tokens in shuffled_rows:
            accumulated_tokens.extend(row_tokens)
            if len(accumulated_tokens) >= target_haystack_length:
                break
        accumulated_tokens = accumulated_tokens[:target_haystack_length]
        insertion_point = int(len(accumulated_tokens) * (depth / 100))
        context_tokens = (
            accumulated_tokens[:insertion_point]
            + needle_tokens
            + accumulated_tokens[insertion_point:]
        )
        actual_context_lengths.append(len(context_tokens))
        context = tokenizer.decode(context_tokens)
        prompt = build_prompt(context, needle["retrieval_question"], config["language"])
        dataset[str(counter)] = {
            "origin_prompt": [{"role": "HUMAN", "prompt": prompt}],
            "prediction": "",
            "gold": needle_text + "*" + needle["arg2"],
        }

    dataset["_dataset_info"] = {
        "version": "NeedleBench V2",
        "nominal_context_tokens": context_length,
        "actual_context_tokens_min": min(actual_context_lengths),
        "actual_context_tokens_max": max(actual_context_lengths),
        "depth": depth,
        "tokenizer_model": "gpt-4",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"Length{context_length}Depth{depth}_origin_{language_key}_1000k.json"
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
    return output_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="下载并生成官方 NeedleBench V2 单针数据")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=project_root / "dataset" / "NeedleBench_v2_source",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "dataset" / "needlebench_v2",
    )
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[1_000_000])
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--lang", choices=sorted(LANGUAGE_CONFIGS), default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LANGUAGE_CONFIGS[args.lang]
    allow_patterns = ["README.md", "needles.jsonl", config["haystack_file"]]
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.source_dir,
        allow_patterns=allow_patterns,
    )
    revision = HfApi().dataset_info(REPO_ID).sha

    generated_files = []
    for context_length in args.context_lengths:
        if context_length <= 0:
            raise ValueError("context_lengths 必须全部大于 0")
        for depth in args.depths:
            if not 0 <= depth <= 100:
                raise ValueError("depths 必须位于 [0, 100] 区间")
            output_path = generate_dataset(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                context_length=context_length,
                depth=depth,
                repeats=args.num_repeats,
                language_key=args.lang,
            )
            generated_files.append(str(output_path))
            print(f"已生成：{output_path}")

    manifest = {
        "dataset": REPO_ID,
        "revision": revision,
        "version": "NeedleBench V2",
        "task": "single-needle retrieval",
        "tokenizer_model": "gpt-4",
        "context_lengths": args.context_lengths,
        "depths": args.depths,
        "num_repeats_per_file": args.num_repeats,
        "language": config["language"],
        "length_buffer": config["length_buffer"],
        "files": generated_files,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"数据版本：{revision}")


if __name__ == "__main__":
    main()
