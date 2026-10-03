import sys

from compression.h2o import H2OConfig, H2OController
from model.qwen35.v451_modeling_qwen35 import (
    Qwen3_5Attention,
    Qwen3_5ForConditionalGeneration,
)

sys.path.append("../")

from transformers import AutoProcessor, AutoTokenizer
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


def device_allocated_memory():
    """获取当前设备分配内存，用于在分块循环内采样峰值。"""
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    if torch.backends.mps.is_available():
        return int(torch.mps.driver_allocated_memory())
    return 0


class Qwen35:
    def __init__(self, model_path, kv_mode='origin', h2o_heavy_hitter_size=1024,
                 h2o_recent_size=1024, h2o_chunk_size=1024, prefill_chunk_size=512,
                 input_keep_ratio=0.4, input_block_size=64, input_initial_tokens=64,
                 input_query_tokens=256, input_relevance_weight=0.7,
                 attn_implementation=None, input_compressor=None):
        if kv_mode not in {'origin', 'h2o'}:
            raise ValueError("kv_mode 必须是 origin 或 h2o")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size 必须大于 0")
        if kv_mode == 'h2o':
            if h2o_chunk_size <= 0:
                raise ValueError("h2o_chunk_size 必须大于 0")
            if attn_implementation not in {None, 'eager'}:
                raise ValueError("H2O 需要 eager attention 以获取真实 attention score")
            h2o_config = H2OConfig(
                heavy_hitter_size=h2o_heavy_hitter_size,
                recent_size=h2o_recent_size,
            )
            h2o_config.validate()
        else:
            h2o_config = None

        self.model_path = model_path
        self.kv_mode = kv_mode
        self.h2o_chunk_size = h2o_chunk_size
        self.prefill_chunk_size = prefill_chunk_size

        # accelerate 的 device_map="auto" 分发只在 CUDA 环境可靠，MPS 上会段错误，
        # 因此非 CUDA 环境先加载到 CPU，再整体搬到目标设备。
        device = default_device()
        selected_attention = 'eager' if kv_mode == 'h2o' else (
            attn_implementation or default_attn_implementation()
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype='auto',
            device_map="auto" if device == "cuda" else None,
            # quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation=selected_attention,
        ).eval()
        if device != "cuda":
            self.model = self.model.to(device)

        self.h2o_controller = H2OController(h2o_config) if h2o_config is not None else None
        if self.h2o_controller is not None:
            for module in self.model.modules():
                if isinstance(module, Qwen3_5Attention):
                    module.h2o_controller = self.h2o_controller

        # 输入压缩在 chat template 之前执行，以保留 system/user/control token。
        self.input_compressor = input_compressor
        self.last_compression_info = None
        self.last_h2o_info = None
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_peak_device_memory_bytes = 0
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
        self.last_compression_info = None
        self.last_h2o_info = None
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_peak_device_memory_bytes = device_allocated_memory()
        if self.h2o_controller is not None:
            self.h2o_controller.reset()

        compression_result = None
        if self.input_compressor is not None:
            compression_result = self.input_compressor.compress(prompt)
            prompt = compression_result.compressed_prompt
            self.last_compression_info = compression_result.info
            print(
                "LongLLMLingua 输入压缩: "
                f"{compression_result.origin_tokens} -> "
                f"{compression_result.compressed_tokens} tokens "
                f"({compression_result.ratio:.2f}x)"
            )

        conversation = [
            {"role": "system", "content": "You are an efficient assistant. For any question posed by the user, do not perform or display any form of chain-of-thought reasoning or process. Provide only the final answer directly, without including <thought> tags or any intermediate steps. The output format should contain only the answer itself."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", max_length=1000000, truncation=True, enable_thinking=False)
        inputs = inputs.to(self.model.device)
        token_count = inputs.input_ids.shape[1]
        self.last_input_tokens = token_count
        print(f"输入模型的token数量: {token_count}")

        # 长输入统一走显式分块 prefill，避免标准 SDPA 构造超大注意力矩阵。
        # H2O 另用自己的 chunk 配置，并在每次 forward 后淘汰 KV cache。
        use_chunked = (
            self.kv_mode == 'h2o'
            or chunk_size is not None
            or token_count > self.prefill_chunk_size
        )
        if chunk_size is None:
            chunk_size = (
                self.h2o_chunk_size if self.kv_mode == 'h2o'
                else self.prefill_chunk_size
            )

        if use_chunked:
            text = self._process_chunked(inputs, max_new_tokens, chunk_size)
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            self.last_output_tokens = len(generated_ids_trimmed[0])
            text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if compression_result is not None:
            recovered_text = self.input_compressor.recover(compression_result, text)
            self.last_compression_info["response_recovered"] = recovered_text != text
            text = recovered_text
        if self.h2o_controller is not None:
            self.last_h2o_info = self.h2o_controller.stats
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

    @torch.no_grad()
    def _process_chunked(self, inputs, max_new_tokens=32768, chunk_size=8192):
        """
        分块 Prefill + 自回归生成。
        H2O 模式在每个 full-attention 层累计真实 attention score，
        并在 prefill 与 decoding 的每次 forward 后执行固定预算淘汰。

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
            self.last_peak_device_memory_bytes = max(
                self.last_peak_device_memory_bytes,
                device_allocated_memory(),
            )

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

            # 已生成结束符或达到预算时，不再执行多余的下一 token forward。
            if next_token_id.item() in eos_token_ids or step == max_new_tokens - 1:
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
            self.last_peak_device_memory_bytes = max(
                self.last_peak_device_memory_bytes,
                device_allocated_memory(),
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 解码生成的 token
        self.last_output_tokens = len(generated_ids)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"生成完成, 共生成 {len(generated_ids)} 个 token")
        return text