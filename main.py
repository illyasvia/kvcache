import argparse
import datetime
import json
import os
import sys

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


def evaluate(save_json_path):
    with open(save_json_path, "r", encoding='utf-8') as f:
        data = json.loads(f.read())

    keys = sample_keys(data)
    total_score = 0
    for k in tqdm(keys, total=None, desc="评测", leave=True, ncols=75, mininterval=0.1):
        pre = data[k]['pre']
        gt = data[k]['gold']

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


def inference(model, json_path, save_json_path, max_new_tokens=None):
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.loads(f.read())

    peak_memory_bytes = 0
    keys = sample_keys(data)
    for k in tqdm(keys, total=None, desc="推理", leave=True, ncols=75, mininterval=0.1):
        prompt = data[k]['origin_prompt'][-1]['prompt']
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if max_new_tokens is None:
            response = model.process(prompt)
        else:
            response = model.process(prompt, max_new_tokens=max_new_tokens)

        if torch.cuda.is_available():
            peak_memory_bytes = max(peak_memory_bytes, torch.cuda.max_memory_allocated())

        pre = normalize_response(response)
        data[k]['prediction'] = response
        data[k]['pre'] = pre

    data['_memory_stats'] = {
        'peak_memory_gb': peak_memory_bytes / (1024 ** 3),
        'cuda_available': torch.cuda.is_available(),
    }

    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_model(args):
    model_path = args.model_path or os.path.join(BASE_DIR, MODEL_DEFAULT_PATHS[args.model_type])

    if args.model_type == 'qwen3vl':
        from model.qwen3vl.Qwen3_vl import Qwen3vl
        return Qwen3vl(model_path=model_path)

    if args.model_type == 'qwen35':
        from model.qwen35.Qwen35 import Qwen35
        return Qwen35(model_path=model_path, kv_mode=args.kv_mode)

    raise ValueError(f"未支持的 model_type: {args.model_type}")


def main(model, context_lengths, depths_list, args):
    results = {}
    progress_path = os.path.join(args.output_dir, "progress_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    for length in context_lengths:
        for depth in depths_list:
            tag = f"Length{length}Depth{depth}"
            json_path = os.path.join(
                args.dataset_dir, f"{tag}_origin_{args.lang}_200k.json")
            save_json_path = os.path.join(args.output_dir, f"{tag}.json")

            if not os.path.exists(json_path):
                print(f"[跳过] 数据集不存在: {json_path}")
                continue

            print(f"\n===== {tag} =====")
            inference(model, json_path, save_json_path, max_new_tokens=args.max_new_tokens)
            results[tag] = evaluate(save_json_path)
            print(f"{tag} 平均分: {results[tag]:.2f}")

            # 每完成一组就落盘，长时间评测中断后可查看已完成部分
            with open(progress_path, "w", encoding='utf-8') as pf:
                json.dump({
                    'model_type': args.model_type,
                    'model_path': model.model_path,
                    'kv_mode': args.kv_mode,
                    'lang': args.lang,
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
    parser.add_argument('--kv_mode', default='origin',
                        help="KV Cache 模式，仅 qwen35 支持（origin 表示不做压缩）")
    parser.add_argument('--context_lengths', type=int, nargs='+',
                        default=[32000, 48000, 100000, 300000],
                        help="待评测的上下文长度列表")
    parser.add_argument('--depths', type=int, nargs='+', default=[0, 52, 100],
                        help="needle 插入深度列表")
    parser.add_argument('--lang', default='en', choices=['en', 'zh'], help="数据集语言")
    parser.add_argument('--dataset_dir', default=os.path.join(BASE_DIR, 'dataset/needle'),
                        help="数据集目录")
    parser.add_argument('--output_dir', default=os.path.join(BASE_DIR, 'output/needle'),
                        help="结果输出目录")
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help="最大生成 token 数，缺省时使用模型自身默认值")
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
