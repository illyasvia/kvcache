"""
拉取单个 HuggingFace 仓库到本地。

鉴权（切勿把 token 写进代码）：
    export HF_TOKEN=hf_xxxxxxxx        # 或 hf auth login

用法：
    python dataset/download.py                                  # 默认 Qwen/Qwen3.5-2B
    python dataset/download.py --repo-id Qwen/Qwen3-VL-2B-Instruct
    python dataset/download.py --repo-id xxx --local-dir /path/to/dir
"""

import argparse
import os

from huggingface_hub import snapshot_download

# 仓库根目录，用于把默认下载目录固定在 weight/ 下，避免依赖运行时的当前路径
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace 仓库快照")
    parser.add_argument("--repo-id", default="Qwen/Qwen3.5-2B",
                        help="HuggingFace 仓库 id，默认 Qwen/Qwen3.5-2B")
    parser.add_argument("--local-dir", default=None,
                        help="下载目标目录，缺省为 weight/<repo_id>")
    args = parser.parse_args()

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "weight", args.repo_id)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        print("未检测到 HF_TOKEN，将以匿名方式下载（公开仓库通常可行）。")

    snapshot_download(
        repo_id=args.repo_id,
        token=token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"完成：{args.repo_id} -> {local_dir}")


if __name__ == "__main__":
    main()
