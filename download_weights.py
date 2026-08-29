"""
按模型系列与规模下载权重到本地 weight/ 目录。

结构：系列(family) × 规模(size) × 变体(variant)

查看所有可选组合：
    python download_weights.py --list
    python download_weights.py --check                      # 本地已有哪些

下载：
    python download_weights.py --family qwen35  --size 2B
    python download_weights.py --family qwen35  --size 2B --variant Base
    python download_weights.py --family qwen3vl --size 4B                # 默认 Instruct
    python download_weights.py --family qwen3vl --size 8B --variant Thinking
    python download_weights.py --family qwen35  --size 2B,4B             # 多个规模
    python download_weights.py --family qwen3vl --tier small             # 整个规模档
    python download_weights.py --family qwen25omni --size 7B

鉴权（切勿把 token 写进代码）：
    export HF_TOKEN=hf_xxxxxxxx        # 或 hf auth login

国内网络可走镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_DISABLE_XET=1

    注意：新版 huggingface_hub 会带 hf-xet，默认从 cas-server.xethub.hf.co
    取文件内容。镜像站只代理元数据、不代理 Xet 字节流，此时小文件能下、
    大的 safetensors 会报 401 Unauthorized。走镜像时务必设 HF_HUB_DISABLE_XET=1
    退回普通 HTTP 通道。

依赖：
    pip install huggingface_hub
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

# 单次下载体积估算超过此值时，需要显式加 --yes 才继续，避免误拉超大模型
CONFIRM_THRESHOLD_GB = 100.0

# 只下载推理需要的文件
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py"]
IGNORE_PATTERNS = ["*.gguf", "*.onnx", "*.onnx_data", "*.pth", "*.bin", "original/*"]


@dataclass(frozen=True)
class Family:
    """一个模型系列，及其支持的规模与变体。"""
    key: str
    display: str
    repo_prefix: str              # 如 "Qwen/Qwen3.5"，实际 repo = prefix-<size><variant_suffix>
    tiers: dict                   # 规模档 -> [size, ...]，用于分类展示
    variants: dict                # 变体名 -> repo 名后缀
    default_size: str
    default_variant: str
    loader: str = ""              # 对应 main.py 的 --model_type 取值
    note: str = ""

    @property
    def sizes(self):
        """按档位顺序展开的全部规模。"""
        out = []
        for members in self.tiers.values():
            out.extend(members)
        return out

    def tier_of(self, size):
        for tier, members in self.tiers.items():
            if size in members:
                return tier
        return "?"

    def repo_id(self, size, variant):
        if variant not in self.variants:
            raise KeyError(variant)
        return f"{self.repo_prefix}-{size}{self.variants[variant]}"


FAMILIES = {
    "qwen35": Family(
        key="qwen35",
        display="Qwen3.5（纯文本）",
        repo_prefix="Qwen/Qwen3.5",
        tiers={
            "small": ["0.8B", "2B", "4B", "9B"],
            "medium": ["27B", "35B-A3B", "122B-A10B"],
            "large": ["397B-A17B"],
        },
        # 后训练版无后缀；Base 为仅预训练版；FP8 为量化版（并非每个规模都发布）
        variants={"Instruct": "", "Base": "-Base", "FP8": "-FP8"},
        default_size="2B",
        default_variant="Instruct",
        loader="qwen35",
        note="Qwen35 类使用；A3B/A10B/A17B 为 MoE，磁盘按总参数量计",
    ),
    "qwen3vl": Family(
        key="qwen3vl",
        display="Qwen3-VL（多模态）",
        repo_prefix="Qwen/Qwen3-VL",
        tiers={
            "small": ["2B", "4B", "8B"],
            "medium": ["30B-A3B", "32B"],
            "large": ["235B-A22B"],
        },
        variants={
            "Instruct": "-Instruct",
            "Thinking": "-Thinking",
            "Instruct-FP8": "-Instruct-FP8",
        },
        default_size="4B",
        default_variant="Instruct",
        loader="qwen3vl",
        note="Qwen3vl 类使用，架构匹配 Qwen3VLForConditionalGeneration",
    ),
    "qwen25omni": Family(
        key="qwen25omni",
        display="Qwen2.5-Omni（全模态）",
        repo_prefix="Qwen/Qwen2.5-Omni",
        tiers={"small": ["3B", "7B"]},
        variants={"Instruct": ""},
        default_size="7B",
        default_variant="Instruct",
        loader="",
        note="Qwen3vl 类无法加载它，需另写 loader",
    ),
}


def estimate_gb(size_label, variant):
    """
    按规模标签粗略估算磁盘占用（GB）。

    取标签中的总参数量（MoE 如 35B-A3B 取 35B，因为权重需全部落盘），
    bf16 约 2 字节/参数，FP8 约 1 字节/参数。仅供体积预警，非精确值。
    """
    m = re.match(r"^([\d.]+)B", size_label)
    if not m:
        return None
    params_b = float(m.group(1))
    bytes_per_param = 1 if "FP8" in variant else 2
    return params_b * bytes_per_param


def local_dir_for(repo_id):
    """统一落盘布局：weight/<repo_id>，如 weight/Qwen/Qwen3.5-2B。"""
    return os.path.join("weight", *repo_id.split("/"))


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def dir_size(path):
    """递归统计目录占用，返回 (总字节数, 文件数)。"""
    total = 0
    count = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
                count += 1
    return total, count


def has_weights(local_dir):
    """目录中存在 safetensors 分片即视为已下载。"""
    if not os.path.isdir(local_dir):
        return False
    for _, _, files in os.walk(local_dir):
        if any(f.endswith(".safetensors") for f in files):
            return True
    return False


def cmd_list():
    """按系列与规模档打印所有可选组合。"""
    for fam in FAMILIES.values():
        print(f"\n=== {fam.key}  {fam.display} ===")
        if fam.note:
            print(f"    {fam.note}")
        print(f"    变体: {', '.join(fam.variants)}   默认: {fam.default_size} / {fam.default_variant}")
        for tier, members in fam.tiers.items():
            print(f"    [{tier}]")
            for size in members:
                est = estimate_gb(size, fam.default_variant)
                est_s = f"~{est:.0f}GB" if est else "?"
                repo = fam.repo_id(size, fam.default_variant)
                mark = " (默认)" if size == fam.default_size else ""
                print(f"       {size:<12} {est_s:<9} {repo}{mark}")
    print("\n体积为 bf16 估算值，仅供参考。")


def cmd_check():
    """扫描 weight/ 下已存在的权重。"""
    rows = []
    for fam in FAMILIES.values():
        for size in fam.sizes:
            for variant in fam.variants:
                repo = fam.repo_id(size, variant)
                d = local_dir_for(repo)
                if has_weights(d):
                    size_bytes, count = dir_size(d)
                    rows.append((fam.key, size, variant, human_size(size_bytes), count, d))
    if not rows:
        print("weight/ 下暂无已下载的权重。用 --list 查看可选项。")
        return
    print(f"{'family':<12} {'size':<12} {'variant':<14} {'大小':<10} {'文件数':<7} 目录")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:<12} {r[2]:<14} {r[3]:<10} {r[4]:<7} {r[5]}")


def verify_repo(repo_id, token):
    """下载前确认仓库可访问，返回 (是否可用, 说明)。"""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
    try:
        HfApi().model_info(repo_id, token=token)
        return True, ""
    except RepositoryNotFoundError:
        return False, "仓库不存在或无权访问（该规模/变体组合可能未发布；私有仓库需 HF_TOKEN）"
    except GatedRepoError:
        return False, f"需先在网页同意授权：https://huggingface.co/{repo_id}"
    except HfHubHTTPError as e:
        return False, f"HTTP 错误：{e}"


def download_one(repo_id, token, revision=None, max_workers=8, no_filter=False):
    """下载单个仓库，返回是否成功。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    local_dir = local_dir_for(repo_id)
    print(f"\n=== {repo_id} -> {local_dir} ===")

    if has_weights(local_dir):
        size_bytes, _ = dir_size(local_dir)
        print(f"    已存在权重（{human_size(size_bytes)}），跳过。删除该目录可重新下载。")
        return True

    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        max_workers=max_workers,
        token=token,
    )
    if not no_filter:
        kwargs["allow_patterns"] = ALLOW_PATTERNS
        kwargs["ignore_patterns"] = IGNORE_PATTERNS

    try:
        # snapshot_download 自带断点续传，中断后重跑即可继续
        snapshot_download(**kwargs)
    except GatedRepoError:
        print(f"    [失败] 需先在网页同意授权：https://huggingface.co/{repo_id}")
        return False
    except RepositoryNotFoundError:
        print(f"    [失败] 仓库不存在或无权访问：{repo_id}")
        return False
    except HfHubHTTPError as e:
        print(f"    [失败] HTTP 错误：{e}")
        return False
    except KeyboardInterrupt:
        print("\n    已中断。重新运行本脚本会从断点继续。")
        raise

    size_bytes, count = dir_size(local_dir)
    print(f"    [完成] {count} 个文件，共 {human_size(size_bytes)}")
    return True


def resolve_sizes(fam, size_arg, tier_arg):
    """把 --size / --tier 解析成具体的规模列表。"""
    if tier_arg:
        if tier_arg not in fam.tiers:
            return None, f"系列 {fam.key} 没有规模档 {tier_arg!r}；可用: {', '.join(fam.tiers)}"
        return list(fam.tiers[tier_arg]), None
    if not size_arg:
        return [fam.default_size], None
    if size_arg == "all":
        return list(fam.sizes), None
    picked = [s.strip() for s in size_arg.split(",") if s.strip()]
    unknown = [s for s in picked if s not in fam.sizes]
    if unknown:
        return None, (f"系列 {fam.key} 不支持规模 {', '.join(unknown)}；"
                      f"可用: {', '.join(fam.sizes)}")
    return picked, None


def main():
    parser = argparse.ArgumentParser(
        description="按系列与规模下载模型权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="可用系列: " + ", ".join(FAMILIES) + "\n用 --list 查看每个系列支持的规模",
    )
    parser.add_argument("--family", choices=list(FAMILIES),
                        help="模型系列")
    parser.add_argument("--size", default=None,
                        help="规模，可逗号分隔多个，或 all；缺省用该系列默认规模")
    parser.add_argument("--tier", default=None,
                        help="按规模档选择：small / medium / large")
    parser.add_argument("--variant", default=None,
                        help="变体，如 Instruct / Base / Thinking；缺省用系列默认")
    parser.add_argument("--revision", default=None,
                        help="锁定 commit / tag / 分支，便于复现")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="并行下载线程数，默认 8")
    parser.add_argument("--no-filter", action="store_true",
                        help="不过滤文件，下载仓库全部内容")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过下载前的仓库可访问性检查")
    parser.add_argument("--yes", action="store_true",
                        help=f"预估体积超过 {CONFIRM_THRESHOLD_GB:.0f}GB 时免确认")
    parser.add_argument("--list", action="store_true",
                        help="列出所有系列与规模")
    parser.add_argument("--check", action="store_true",
                        help="只检查本地已下载的权重")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return 0
    if args.check:
        cmd_check()
        return 0
    if args.family is None:
        parser.print_help()
        print()
        cmd_list()
        return 0

    fam = FAMILIES[args.family]

    # 先做纯参数校验，参数写错时立刻给出准确提示
    if args.size and args.tier:
        print("--size 与 --tier 不能同时使用")
        return 1

    sizes, err = resolve_sizes(fam, args.size, args.tier)
    if err:
        print(err)
        return 1

    variant = args.variant or fam.default_variant
    if variant not in fam.variants:
        print(f"系列 {fam.key} 不支持变体 {variant!r}；可用: {', '.join(fam.variants)}")
        return 1

    targets = [fam.repo_id(s, variant) for s in sizes]

    # 体积预警
    total_est = 0.0
    print(f"系列 {fam.key} - {fam.display}，变体 {variant}，共 {len(targets)} 个目标：")
    for s, repo in zip(sizes, targets):
        est = estimate_gb(s, variant)
        total_est += est or 0
        print(f"  {s:<12} ~{est:.0f}GB" if est else f"  {s:<12} ?", end="")
        print(f"   {repo}")
    print(f"预估合计 ~{total_est:.0f}GB（bf16 估算，仅供参考）")

    if total_est > CONFIRM_THRESHOLD_GB and not args.yes:
        print(f"\n预估体积超过 {CONFIRM_THRESHOLD_GB:.0f}GB。确认无误请加 --yes 重新运行。")
        return 1

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("缺少依赖，请先执行：pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        print("\n未检测到 HF_TOKEN，将以匿名方式下载（公开仓库通常可行）。")
    if os.environ.get("HF_ENDPOINT"):
        print(f"使用镜像: {os.environ['HF_ENDPOINT']}")

    failed = []
    for repo in targets:
        if not args.no_verify:
            ok, why = verify_repo(repo, token)
            if not ok:
                print(f"\n=== {repo} ===\n    [跳过] {why}")
                failed.append(repo)
                continue
        if not download_one(repo, token,
                            revision=args.revision,
                            max_workers=args.max_workers,
                            no_filter=args.no_filter):
            failed.append(repo)

    print()
    if failed:
        print("以下目标未成功：")
        for r in failed:
            print(f"  {r}")
        return 1

    print("全部完成。")
    if fam.loader:
        print(f"\n提示：main.py 的 {fam.loader} 分支需把 model_path 指向对应目录，例如")
        print(f"  model_path = \"{local_dir_for(targets[0])}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
