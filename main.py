import argparse
import datetime
import json
import os
import resource
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from tqdm import tqdm

MODEL_DEFAULT_PATHS = {
    'qwen3vl': "weight/Qwen/Qwen3-VL-4B-Instruct",
    'qwen35': "weight/Qwen/Qwen3.5-2B",
}


def needle_score(prediction, reference, generated=True):
    """按关键词命中率给单个样本打分，满分 100。"""
    if generated:
        keyword = ''
        lines = reference.strip().splitlines()
        # 参考答案按行交替存放：偶数行是答案正文，奇数行是关键词
        for i in range(len(lines)):
            if i % 2 == 1:
                keyword = keyword + lines[i]
        keyword = keyword.replace('*', ' ')
    else:
        _, keyword = reference.split('*')
    keywords = keyword.lower().split()
    prediction = prediction.lower()

    keyword_score = 100 / len(keywords) if keywords else 0

    matched_keywords = sum(1 for kword in keywords if kword in prediction)
    score = matched_keywords * keyword_score

    return score, matched_keywords


def sample_keys(data):
    """只取数字键的样本（跳过 _memory_stats 等元数据），并按数值排序。"""
    return sorted((k for k in data.keys() if str(k).isdigit()), key=int)


def normalize_response(response):
    """模型返回值可能是 str / list / None，统一成非空字符串。"""
    if isinstance(response, (list, tuple)):
        response = response[0] if response else None
    if response is None:
        return '0'
    return str(response)


def evaluate(save_json_path, scoring='legacy'):
    with open(save_json_path, "r", encoding='utf-8') as f:
        data = json.loads(f.read())

    keys = sample_keys(data)
    total_score = 0
    for k in tqdm(keys, total=None, desc="评测", leave=True, ncols=75, mininterval=0.1):
        pre = data[k]['pre']
        gt = data[k]['gold']

        if scoring == 'needlebench_v2':
            keyword = gt.rsplit('*', 1)[-1].strip()
            matched_keywords = int(keyword in pre)
            score = 100.0 * matched_keywords
        else:
            score, matched_keywords = needle_score(pre, gt)
        data[k]['score'] = score
        data[k]['matched_keywords'] = matched_keywords
        total_score += score

    average_score = total_score / len(keys) if keys else 0
    data['_eval_stats'] = {
        'num_samples': len(keys),
        'average_score': average_score,
    }

    with open(save_json_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return average_score


def process_peak_memory_bytes():
    """返回当前进程峰值常驻内存；macOS 为 bytes，Linux 为 KiB。"""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024)


def current_mps_memory_bytes():
    if not torch.backends.mps.is_available():
        return 0
    return int(torch.mps.current_allocated_memory())


def write_json_atomic(path, data):
    """原子写入长时间评测检查点，避免中断留下损坏 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temporary_path, path)


def update_runtime_stats(data, peak_cuda_bytes, peak_mps_bytes, peak_rss_bytes):
    keys = sample_keys(data)
    completed = [data[key] for key in keys if 'pre' in data[key]]
    compression = [sample['compression'] for sample in completed if 'compression' in sample]
    h2o = [sample['h2o'] for sample in completed if 'h2o' in sample]
    inference_seconds = sum(sample.get('inference_seconds', 0.0) for sample in completed)
    input_tokens = [sample['input_tokens'] for sample in completed if 'input_tokens' in sample]
    output_tokens = [sample['output_tokens'] for sample in completed if 'output_tokens' in sample]

    data['_run_stats'] = {
        'completed_samples': len(completed),
        'total_samples': len(keys),
        'input_tokens_total': sum(input_tokens),
        'input_tokens_average': sum(input_tokens) / len(input_tokens) if input_tokens else 0,
        'output_tokens_total': sum(output_tokens),
    }
    data['_memory_stats'] = {
        'peak_memory_gb': max(peak_cuda_bytes, peak_mps_bytes) / (1024 ** 3),
        'peak_cuda_allocated_gb': peak_cuda_bytes / (1024 ** 3),
        'peak_mps_allocated_gb': peak_mps_bytes / (1024 ** 3),
        'peak_process_rss_gb': peak_rss_bytes / (1024 ** 3),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
    }
    data['_timing_stats'] = {
        'total_inference_seconds': inference_seconds,
        'average_inference_seconds': inference_seconds / len(completed) if completed else 0,
    }
    compression_origin_tokens = sum(info['origin_tokens'] for info in compression)
    compression_output_tokens = sum(info['compressed_tokens'] for info in compression)
    data['_compression_stats'] = {
        'enabled': bool(compression),
        'num_samples': len(compression),
        'origin_tokens': compression_origin_tokens,
        'compressed_tokens': compression_output_tokens,
        'ratio': (
            compression_origin_tokens / compression_output_tokens
            if compression_output_tokens else 1.0
        ),
        'total_latency_seconds': sum(info['latency_seconds'] for info in compression),
    }
    data['_h2o_stats'] = {
        'enabled': bool(h2o),
        'num_samples': len(h2o),
        'eviction_steps': sum(info['eviction_steps'] for info in h2o),
        'layer_token_evictions': sum(info['layer_token_evictions'] for info in h2o),
    }


def inference(model, json_path, save_json_path, max_new_tokens=None, resume=False):
    with open(json_path, "r", encoding='utf-8') as file:
        data = json.load(file)

    if resume and os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding='utf-8') as file:
            saved_data = json.load(file)
        for key in sample_keys(data):
            if key in saved_data and 'pre' in saved_data[key]:
                data[key].update(saved_data[key])

    previous_memory = data.get('_memory_stats', {})
    peak_cuda_bytes = int(previous_memory.get('peak_cuda_allocated_gb', 0) * (1024 ** 3))
    peak_mps_bytes = int(previous_memory.get('peak_mps_allocated_gb', 0) * (1024 ** 3))
    peak_rss_bytes = max(
        int(previous_memory.get('peak_process_rss_gb', 0) * (1024 ** 3)),
        process_peak_memory_bytes(),
    )
    keys = sample_keys(data)
    pending_keys = [key for key in keys if not (resume and 'pre' in data[key])]

    for key in tqdm(pending_keys, total=None, desc="推理", leave=True, ncols=75, mininterval=0.1):
        prompt = data[key]['origin_prompt'][-1]['prompt']
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started_at = time.perf_counter()
        if max_new_tokens is None:
            response = model.process(prompt)
        else:
            response = model.process(prompt, max_new_tokens=max_new_tokens)
        sample_seconds = time.perf_counter() - started_at
        data[key]['inference_seconds'] = sample_seconds
        data[key]['input_tokens'] = getattr(model, 'last_input_tokens', 0)
        data[key]['output_tokens'] = getattr(model, 'last_output_tokens', 0)

        compression_info = getattr(model, 'last_compression_info', None)
        if compression_info is not None:
            data[key]['compression'] = compression_info

        h2o_info = getattr(model, 'last_h2o_info', None)
        if h2o_info is not None:
            data[key]['h2o'] = h2o_info

        sampled_device_peak = getattr(model, 'last_peak_device_memory_bytes', 0)
        if torch.cuda.is_available():
            peak_cuda_bytes = max(
                peak_cuda_bytes,
                torch.cuda.max_memory_allocated(),
                sampled_device_peak,
            )
        peak_mps_bytes = max(
            peak_mps_bytes,
            current_mps_memory_bytes(),
            sampled_device_peak if torch.backends.mps.is_available() else 0,
        )
        peak_rss_bytes = max(peak_rss_bytes, process_peak_memory_bytes())

        data[key]['prediction'] = response
        data[key]['pre'] = normalize_response(response)
        update_runtime_stats(data, peak_cuda_bytes, peak_mps_bytes, peak_rss_bytes)
        write_json_atomic(save_json_path, data)

    update_runtime_stats(data, peak_cuda_bytes, peak_mps_bytes, peak_rss_bytes)
    write_json_atomic(save_json_path, data)


def build_input_compressor(args):
    if args.prompt_compression == 'none':
        return None
    if args.model_type != 'qwen35':
        raise ValueError("LongLLMLingua 当前仅支持 qwen35 文本推理")

    from compression.longllmlingua import LongLLMLinguaCompressor, LongLLMLinguaConfig

    config = LongLLMLinguaConfig(
        model_name=args.compressor_model_path,
        device=args.compressor_device,
        rate=args.compression_rate,
        chunk_tokens=args.compression_chunk_tokens,
        iterative_size=args.compression_iterative_size,
        reorder_context=args.compression_reorder,
        dynamic_context_compression_ratio=args.compression_dynamic_ratio,
        context_budget_offset=args.compression_context_budget_offset,
        enable_recovery=args.compression_recovery,
    )
    return LongLLMLinguaCompressor(config)


def build_model(args):
    model_path = args.model_path or os.path.join(BASE_DIR, MODEL_DEFAULT_PATHS[args.model_type])
    input_compressor = build_input_compressor(args)

    if args.model_type == 'qwen3vl':
        if args.kv_mode != 'origin':
            raise ValueError("H2O 当前仅支持 qwen35")
        from model.qwen3vl.Qwen3_vl import Qwen3vl
        return Qwen3vl(model_path=model_path)

    if args.model_type == 'qwen35':
        from model.qwen35.Qwen35 import Qwen35
        return Qwen35(
            model_path=model_path,
            kv_mode=args.kv_mode,
            h2o_heavy_hitter_size=args.h2o_heavy_hitter_size,
            h2o_recent_size=args.h2o_recent_size,
            h2o_chunk_size=args.h2o_chunk_size,
            prefill_chunk_size=args.prefill_chunk_size,
            input_compressor=input_compressor,
        )

    raise ValueError(f"未支持的 model_type: {args.model_type}")


def main(model, context_lengths, depths_list, args):
    results = {}
    progress_path = os.path.join(args.output_dir, "progress_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    for length in context_lengths:
        for depth in depths_list:
            tag = f"Length{length}Depth{depth}"
            json_path = os.path.join(
                args.dataset_dir, f"{tag}_origin_{args.lang}_{args.dataset_suffix}.json")
            save_json_path = os.path.join(args.output_dir, f"{tag}.json")

            if not os.path.exists(json_path):
                print(f"[跳过] 数据集不存在: {json_path}")
                continue

            print(f"\n===== {tag} =====")
            inference(
                model,
                json_path,
                save_json_path,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
            )
            results[tag] = evaluate(save_json_path, scoring=args.scoring)
            print(f"{tag} 平均分: {results[tag]:.2f}")

            # 每完成一组就落盘，长时间评测中断后可查看已完成部分
            with open(progress_path, "w", encoding='utf-8') as pf:
                json.dump({
                    'model_type': args.model_type,
                    'model_path': model.model_path,
                    'kv_mode': args.kv_mode,
                    'h2o_heavy_hitter_size': args.h2o_heavy_hitter_size,
                    'h2o_recent_size': args.h2o_recent_size,
                    'h2o_chunk_size': args.h2o_chunk_size,
                    'prefill_chunk_size': args.prefill_chunk_size,
                    'prompt_compression': args.prompt_compression,
                    'compression_rate': args.compression_rate,
                    'lang': args.lang,
                    'dataset_suffix': args.dataset_suffix,
                    'scoring': args.scoring,
                    'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'scores': results,
                }, pf, indent=4, ensure_ascii=False)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack 长文本评测")
    parser.add_argument('--model_type', required=True, choices=sorted(MODEL_DEFAULT_PATHS.keys()),
                        help="使用的模型类型")
    parser.add_argument('--model_path', default=None,
                        help="模型权重路径，缺省时使用该 model_type 的默认路径")
    parser.add_argument('--kv_mode', default='origin', choices=['origin', 'h2o'],
                        help="KV Cache 模式，仅 qwen35 支持（origin 或 h2o）")
    parser.add_argument('--h2o_heavy_hitter_size', type=int, default=1024,
                        help="H2O 每个 KV head 保留的累计高 attention token 数")
    parser.add_argument('--h2o_recent_size', type=int, default=1024,
                        help="H2O 每个 KV head 保留的最近 token 数")
    parser.add_argument('--h2o_chunk_size', type=int, default=1024,
                        help="H2O 分块 prefill 的 chunk token 数")
    parser.add_argument('--prefill_chunk_size', type=int, default=512,
                        help="origin/LongLLMLingua 长输入分块 prefill 的 token 数")
    parser.add_argument('--context_lengths', type=int, nargs='+',
                        default=[32000, 48000, 100000, 300000],
                        help="待评测的上下文长度列表")
    parser.add_argument('--depths', type=int, nargs='+', default=[0, 52, 100],
                        help="needle 插入深度列表")
    parser.add_argument('--lang', default='en', choices=['en', 'zh'], help="数据集语言")
    parser.add_argument('--dataset_dir', default=os.path.join(BASE_DIR, 'dataset/needle'),
                        help="数据集目录")
    parser.add_argument('--dataset_suffix', default='200k',
                        help="数据文件名中的版本后缀，例如 200k 或 1000k")
    parser.add_argument('--scoring', choices=['legacy', 'needlebench_v2'], default='legacy',
                        help="评测打分规则；NeedleBench V2 使用完整关键词命中率")
    parser.add_argument('--output_dir', default=os.path.join(BASE_DIR, 'output/needle'),
                        help="结果输出目录")
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help="最大生成 token 数，缺省时使用模型自身默认值")
    parser.add_argument('--resume', action='store_true',
                        help="跳过输出文件中已有结果的样本，并逐样本保存检查点")
    parser.add_argument('--prompt_compression', default='none',
                        choices=['none', 'longllmlingua'], help="输入 prompt 压缩方式")
    parser.add_argument('--compression_rate', type=float, default=0.5,
                        help="压缩后/原始 token 比例，范围 (0, 1]")
    parser.add_argument('--compressor_model_path', default='microsoft/phi-2',
                        help="LongLLMLingua 使用的小型 causal LM 名称或本地路径")
    parser.add_argument('--compressor_device', default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'], help="压缩模型运行设备")
    parser.add_argument('--compression_chunk_tokens', type=int, default=512,
                        help="将 Needle 长文档切成用于粗粒度排序的 token 块大小")
    parser.add_argument('--compression_iterative_size', type=int, default=200,
                        help="LongLLMLingua 细粒度迭代窗口大小")
    parser.add_argument('--compression_reorder', default='sort',
                        choices=['original', 'sort', 'two_stage'], help="压缩后上下文重排策略")
    parser.add_argument('--compression_dynamic_ratio', type=float, default=0.3,
                        help="文档动态压缩率偏移，范围 [0, 1]")
    parser.add_argument('--compression_context_budget_offset', type=int, default=100,
                        help="粗粒度上下文预算相对目标 token 数的偏移")
    parser.add_argument('--compression_recovery', action=argparse.BooleanOptionalAction,
                        default=True, help="是否启用生成结果的子序列恢复")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = build_model(args)
    results = main(model, args.context_lengths, args.depths, args)

    print("\n===== 汇总 =====")
    for tag, score in results.items():
        print(f"{tag}: {score:.2f}")
    if results:
        print(f"总平均分: {sum(results.values()) / len(results):.2f}")
    print('end')
