import sys

from model.qwen35.v451_modeling_qwen35 import Qwen3_5ForConditionalGeneration

sys.path.append("../")

from transformers import AutoProcessor, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.utils import is_flash_attn_2_available
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     
)


def default_attn_implementation():
    """flash_attention_2 只在装了 flash-attn 的 CUDA 环境可用，其余环境退回 sdpa。"""
    return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def default_device():
    """无 CUDA 时优先用 MPS（Apple Silicon），最后退回 CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Qwen35:
    def __init__(self, model_path, kv_mode='origin', hh_ratio=0.1, recent_ratio=0.1, hh_layer=1,
                 input_keep_ratio=0.4, input_block_size=64, input_initial_tokens=64,
                 input_query_tokens=256, input_relevance_weight=0.7,
                 attn_implementation=None):
        self.model_path = model_path
        # accelerate 的 device_map="auto" 分发只在 CUDA 环境可靠，MPS 上会段错误，
        # 因此非 CUDA 环境先加载到 CPU，再整体搬到目标设备。
        device = default_device()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype='auto',
            device_map="auto" if device == "cuda" else None,
            # quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation=attn_implementation or default_attn_implementation(),
        ).eval()
        if device != "cuda":
            self.model = self.model.to(device)
        self.kv_mode = kv_mode
        # 输入压缩器尚未接入，置空表示 process() 走不压缩输入的常规流程
        self.input_compressor = None
        self.last_compression_info = None
        self.input_compress_config = {
            'keep_ratio': input_keep_ratio,
            'block_size': input_block_size,
            'initial_tokens': input_initial_tokens,
            'query_tokens': input_query_tokens,
            'relevance_weight': input_relevance_weight,
        }
        self.model.generation_config.do_sample = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def process(self, prompt, max_new_tokens=32768, chunk_size=None):
        """
        处理纯文本推理请求。当输入过长时自动使用分块 prefill 以避免 OOM。

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            chunk_size: 分块大小。None 表示自动选择（短输入直接推理，长输入分块）。
                        设置为具体数字（如 4096、8192）则强制分块。
        """
        conversation = [
            {"role": "system", "content": "You are an efficient assistant. For any question posed by the user, do not perform or display any form of chain-of-thought reasoning or process. Provide only the final answer directly, without including <thought> tags or any intermediate steps. The output format should contain only the answer itself."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", max_length=1000000, truncation=True, enable_thinking=False)
        inputs = inputs.to(self.model.device)
        token_count = inputs.input_ids.shape[1]
        print(f"输入文本token数量: {token_count}")

        # input_compression 模式：在 prefill/decoding 之前压缩输入 token 序列，
        # 剪短输入模型的序列长度，只保留关键信息，随后走正常 generate 流程
        if self.input_compressor is not None:
            result = self.input_compressor.compress(inputs.input_ids)
            inputs["input_ids"] = result.pruned_input_ids
            inputs["attention_mask"] = torch.ones_like(result.pruned_input_ids)
            token_count = inputs.input_ids.shape[1]
            self.last_compression_info = dict(result.info)
            print(f"压缩后token数量: {token_count}")
            # 压缩后走下方通用流程：短序列直接 generate，长序列分块 prefill。
            # input_compression 模式下 _process_chunked 不会再做任何 KV 驱逐
            # （见 _process_chunked 中仅对 origin 调用 _compress_kv_cache），
            # 因此分块只用于降低峰值显存，结果与一次性 prefill 等价。

        # 自动选择是否分块：使用 KV 压缩模式（非 origin）且输入较长时使用分块 prefill 以避免 OOM
        use_chunked = False
        if chunk_size is not None:
            use_chunked = True
        elif self.kv_mode != 'origin' and token_count > 8192:
            use_chunked = True
            chunk_size = 4096

        if use_chunked:
            return self._process_chunked(inputs, max_new_tokens, chunk_size)
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return text

    def process_multimodel(self, prompt, image, instruction=None, role=None, max_new_tokens=256):
        """
        处理单张图片的多模态推理请求。
        使用 processor 将图像和文本一起处理，通过视觉编码器提取图像特征。
        返回值格式与 Qwen3vl.process_multimodel 保持一致（返回列表）。
        """
        if role is None:
            role = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": [{"type": "text", "text": role}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        if instruction is not None:
            messages[1]["content"].append({"type": "text", "text": instruction})

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    def process_multiple_multimodel(self, prompt, images, instruction=None, role=None, max_new_tokens=256):
        """
        处理多张图片的多模态推理请求。
        使用 processor 将多张图像和文本一起处理，通过视觉编码器提取图像特征。
        返回值格式与 Qwen3vl.process_multiple_multimodel 保持一致（返回列表）。
        """
        if role is None:
            role = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": [{"type": "text", "text": role}]},
            {"role": "user", "content": []},
        ]
        for img in images:
            messages[1]["content"].append({"type": "image", "image": img})
        messages[1]["content"].append({"type": "text", "text": prompt})
        if instruction is not None:
            messages[1]["content"].append({"type": "text", "text": instruction})

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    def _compress_kv_cache(self, past_key_values, hh_ratio=0.2, recent_ratio=0.2):
        """
        压缩 KV Cache：保留最近 recent_ratio 比例的 token + key L2 norm 最高的 hh_ratio 比例的 token。
        key 的 L2 norm 作为 attention weight 的近似重要性指标。

        Args:
            past_key_values: DynamicCache 对象
            hh_ratio: heavy-hitter 比例（按 key norm 选择最重要的 token）
            recent_ratio: 最近 token 保留比例
        """
        num_layers = len(past_key_values.layers)

        for layer_idx in range(num_layers):
            layer_cache = past_key_values.layers[layer_idx]

            # 跳过 linear attention 层（没有 keys/values 属性）
            if not hasattr(layer_cache, 'keys') or layer_cache.keys is None:
                continue

            key_states = layer_cache.keys  # [bsz, num_heads, seq_len, head_dim]
            value_states = layer_cache.values

            # 跳过非标准形状的 cache
            if key_states.dim() != 4:
                continue

            bsz, num_heads, seq_len, head_dim = key_states.shape
            hh_size = int(seq_len * hh_ratio)
            recent_size = int(seq_len * recent_ratio)
            cache_size = hh_size + recent_size

            if seq_len <= cache_size:
                continue

            # 使用 key 的 L2 norm 作为重要性分数（近似 attention weight）
            # [bsz, num_heads, seq_len]
            key_norm = torch.norm(key_states, p=2, dim=-1)

            # 在非 recent 区域选择 top-k（按 key norm）
            select_scores = key_norm[:, :, :seq_len - recent_size]  # [bsz, num_heads, seq_len - recent_size]
            _, keep_topk = torch.topk(select_scores, hh_size, dim=-1)  # [bsz, num_heads, hh_size]
            keep_topk = keep_topk.sort(dim=-1).values

            # recent token 的索引
            keep_recent = torch.arange(
                seq_len - recent_size, seq_len,
                device=key_states.device
            ).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)  # [bsz, num_heads, recent_size]

            # 合并索引
            keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)  # [bsz, num_heads, cache_size]

            # 使用 gather 选择保留的 token
            keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, cache_size, head_dim]
            new_key_states = torch.gather(key_states, 2, keep_idx_expanded)
            new_value_states = torch.gather(value_states, 2, keep_idx_expanded)

            past_key_values.layers[layer_idx].keys = new_key_states
            past_key_values.layers[layer_idx].values = new_value_states

        return past_key_values

    @torch.no_grad()
    def _process_chunked(self, inputs, max_new_tokens=32768, chunk_size=8192):
        """
        分块 Prefill + 自回归生成。
        将长输入分成多个 chunk 逐步处理，每个 chunk 后进行 KV Cache 压缩，
        从而将内存使用量控制在可接受范围内。

        压缩策略：每个 chunk 后保留最近 20% + key norm 最高的 20%（共 40%）。

        对于 1M token 输入：
        - 注意力矩阵从 [1, heads, 1M, 1M]（不可能）
          变为 [1, heads, chunk_size, cache_size]（可控）
        """
        input_ids = inputs["input_ids"]  # [1, seq_len]
        attention_mask = inputs.get("attention_mask", None)  # [1, seq_len]
        seq_len = input_ids.shape[1]

        print(f"使用分块 Prefill: chunk_size={chunk_size}, 总 token={seq_len}, "
              f"共 {(seq_len + chunk_size - 1) // chunk_size} 个 chunk")

        # 初始化 KV Cache（传 None 让模型自行创建带 config 的 DynamicCache，
        # 以正确支持 Qwen3.5 的混合 linear attention + standard attention 架构）
        past_key_values = None

        # 分块 prefill
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)

            chunk_input_ids = input_ids[:, start:end]

            # 构建 attention_mask：需要包含之前 cache 中的 token
            cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            chunk_len = end - start
            # attention_mask: [1, cache_len + chunk_len]，全为 1（causal mask 由模型内部处理）
            chunk_attention_mask = torch.ones(
                1, cache_len + chunk_len,
                dtype=torch.long, device=input_ids.device
            )

            # position_ids：基于已处理的 token 总数（包含被压缩掉的）
            # 注意：使用实际的绝对位置，而不是 cache 中的位置
            chunk_position_ids = torch.arange(
                start, end, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

            # Forward pass for this chunk
            outputs = self.model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                position_ids=chunk_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            # 当使用 H2O/StreamingLLM 等模式时，层内 eviction 已自动在 forward 中执行，
            # 无需额外压缩；仅在 origin 模式的分块 prefill 中使用通用压缩
            if self.kv_mode == 'origin':
                past_key_values = self._compress_kv_cache(past_key_values, hh_ratio=0.2, recent_ratio=0.2)

            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                current_cache_len = past_key_values.get_seq_length()
                print(f"  Chunk {i+1}/{num_chunks} 完成, 当前 KV Cache 长度: {current_cache_len}")

        # 自回归生成
        print(f"Prefill 完成, 开始生成... (KV Cache 长度: {past_key_values.get_seq_length()})")

        generated_ids = []
        # 获取最后一个 token 的 logits 来开始生成
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_ids = set(eos_token_id)
        else:
            eos_token_ids = {eos_token_id}

        for step in range(max_new_tokens):
            generated_ids.append(next_token_id.item())

            # 检查是否生成了结束符
            if next_token_id.item() in eos_token_ids:
                break

            # 准备下一步的输入
            cache_len = past_key_values.get_seq_length()
            step_attention_mask = torch.ones(
                1, cache_len + 1,
                dtype=torch.long, device=input_ids.device
            )
            # position_id = 原始 seq_len + 已生成的 step 数
            step_position_ids = torch.tensor(
                [[seq_len + step]], dtype=torch.long, device=input_ids.device
            )

            outputs = self.model(
                input_ids=next_token_id,
                attention_mask=step_attention_mask,
                position_ids=step_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 解码生成的 token
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"生成完成, 共生成 {len(generated_ids)} 个 token")
        return text