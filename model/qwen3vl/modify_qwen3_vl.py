import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
from transformers.utils import (  # add_start_docstrings,
    # add_start_docstrings_to_model_forward,
    is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging,  # replace_return_docstrings,
)
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

# from transformers.models.llama.modeling_llama import repeat_kv

from .v451_modeling_qwen3_vl import (Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb, repeat_kv, Qwen3VLTextRMSNorm, eager_attention_forward, )

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    _flash_attention_forward = None
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)

def kl_loss(a: torch.Tensor, b: torch.Tensor, dim=-1) -> torch.Tensor:
    a = F.softmax(a,dim=dim).clamp(1e-6,1-1e-6)
    b = F.softmax(b,dim=dim).clamp(1e-6,1-1e-6)
    log_a = a.log()
    log_b = b.log()
    return(a*(log_a-log_b)).sum(dim=dim).mean()

def attention_entropy(attention_weights, base=2):
    probs = np.array(attention_weights)
    probs = probs / np.sum(probs)  # 防止数值误差
    entropy = -sum(p * np.log(p) / np.log(base) for p in probs if p > 0)
    return entropy

# the same as the original code
def _make_causal_mask(bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)  # torch.Size([1436, 1436])
    mask_cond = torch.arange(mask.size(-1), device=device)  # torch.Size([1436])
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)  # torch.Size([1436, 1436])
    mask = mask.to(dtype)

    if past_key_values_length > 0:  # 1436
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class H2OKVCache_LayerWise:
    def __init__(self, layer_idx: Optional[int] = None, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, hh_ratio=None, recent_ratio=None, hh_layer=27, ):
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                                "when creating this class.")
        if self.layer_idx == 0:
            print(f"H2O KVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio} {hh_layer}")  # 200, 200
        self.hh_size = hh_size  # 200
        self.recent_size = recent_size  # 200
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size  # 400
        self.k_seq_dim = k_seq_dim  # 2
        self.v_seq_dim = v_seq_dim  # 2
        self.hh_ratio = hh_ratio  # 0.1
        self.recent_ratio = recent_ratio  # 0.1
        self.hh_score = None
        self.seq_len = None
        self.hh_layer = hh_layer
    
    def get_result(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:  # 0.1
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio) + 15
            self.cache_size = self.hh_size + self.recent_size
        bsz, num_heads, q_len, seq_len = attn_score_cache.shape  # [1, 79913, 32, 8]
        num_kv_heads = past_key_values[self.layer_idx][0].shape[1]
        num_kv_groups = num_heads // num_kv_heads
        attn_score_cache = attn_score_cache.view(bsz, num_kv_heads, num_kv_groups, q_len, seq_len)
        attn_score_cache = attn_score_cache.sum(dim=2)  # [bsz, num_kv_heads, q_len, seq_len]
        
        if self.layer_idx > self.hh_layer:
            if past_key_values.hh_score is None:
                past_key_values.hh_score = attn_score_cache[:, :, :self.recent_size, :].sum(0).sum(1)
            self.hh_score = past_key_values.hh_score
        else:
            self.hh_score = attn_score_cache[:, :, :self.recent_size, :].sum(0).sum(1)
            past_key_values.hh_score = self.hh_score
        value_L2 = torch.norm(past_key_values[self.layer_idx][1][0], p=2, dim=-1)  # 后面的0是第0个数据（batch）
        self.hh_score = self.hh_score * value_L2
        seq_len = past_key_values[self.layer_idx][0].size(self.k_seq_dim)  # 1436
        if seq_len <= self.cache_size:  # 1436  286
            return past_key_values
        bsz, num_heads, _, head_dim = past_key_values[self.layer_idx][0].shape  # torch.Size([1, 4, 1436, 128])
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)  # 每个头，查找最大注意力分数的token的序号 torch.Size([4, 143])
        keep_topk = keep_topk.sort().values  # 排序 torch.Size([4, 143])
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values[self.layer_idx][0].device).repeat(num_heads, 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)  # 保留的token torch.Size([4, 286])

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[self.layer_idx][0].device)  # torch.Size([4, 1436]):全0
        mask = mask.scatter(-1, keep_idx, 1)  # torch.Size([4, 1436])

        k_hh_recent = past_key_values[self.layer_idx][0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)  # torch.Size([1, 4, 286, 128])
        v_hh_recent = past_key_values[self.layer_idx][1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)  # torch.Size([1, 4, 286, 128])

        past_key_values.layers[self.layer_idx].keys = k_hh_recent
        past_key_values.layers[self.layer_idx].values = v_hh_recent
        return past_key_values

    def __call__(self, past_key_values, attn_score_cache):
        return self.call_2(past_key_values, attn_score_cache)

    def call_2(self, past_key_values, attn_score_cache):
        if self.hh_ratio is not None:  # 0.1
            self.hh_size = int(attn_score_cache.shape[-1] * self.hh_ratio)
            self.recent_size = int(attn_score_cache.shape[-1] * self.recent_ratio) + 15
            self.cache_size = self.hh_size + self.recent_size

        bsz, num_heads, q_len, seq_len = attn_score_cache.shape  # [1, num_heads, q_len, seq_len]
        num_kv_heads = past_key_values[self.layer_idx][0].shape[1]
        num_kv_groups = num_heads // num_kv_heads

        attn_score_cache = attn_score_cache.view(bsz, num_kv_heads, num_kv_groups, q_len, seq_len)
        attn_score_cache = attn_score_cache.sum(dim=2)  # [bsz, num_kv_heads, q_len, seq_len]

        if self.layer_idx > self.hh_layer:
            if not hasattr(past_key_values  , 'hh_score'):
                past_key_values.hh_score = attn_score_cache[:, :, :self.recent_size, :].sum(0).sum(1)
            self.hh_score = past_key_values.hh_score
        else:
            self.hh_score = attn_score_cache[:, :, :self.recent_size, :].sum(0).sum(1)
            past_key_values.hh_score = self.hh_score
        value_L2 = torch.norm(past_key_values[self.layer_idx][1][0], p=2, dim=-1)  # 后面的0是第0个数据（batch）
        # import ipdb; ipdb.set_trace()
        self.hh_score = self.hh_score * value_L2

        seq_len = past_key_values[self.layer_idx][0].size(self.k_seq_dim)  # 1436

        if seq_len <= self.cache_size:  # 1436  286
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[self.layer_idx][0].shape  # torch.Size([1, 4, 1436, 128])

        ##################before-code################################
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)  # 每个头，查找最大注意力分数的token的序号 torch.Size([4, 143])
        keep_topk = keep_topk.sort().values  # 排序 torch.Size([4, 143])

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values[self.layer_idx][0].device).repeat(num_heads, 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)  # 保留的token torch.Size([4, 286])

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[self.layer_idx][0].device)  # torch.Size([4, 1436]):全0
        mask = mask.scatter(-1, keep_idx, 1)  # torch.Size([4, 1436])

        k_hh_recent = past_key_values[self.layer_idx][0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)  # torch.Size([1, 4, 286, 128])
        v_hh_recent = past_key_values[self.layer_idx][1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)  # torch.Size([1, 4, 286, 128])

        past_key_values.layers[self.layer_idx].keys = k_hh_recent
        past_key_values.layers[self.layer_idx].values = v_hh_recent
        # print(f"k_hh_recent={k_hh_recent.shape},v_hh_recent={v_hh_recent.shape}")
        return past_key_values

    def _update_hh_score(self, attn_score_cache):
        ############## stop here and find the bug ##############
        num_new_tokens = attn_score_cache.shape[2]  # 1436

        # attn_score_cache = attn_score_cache[:, :, 2186:, :]
        if self.hh_score is None:  # --
            self.hh_score = attn_score_cache.sum(0).sum(1)  # torch.Size([4, 1436])
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)  # torch.Size([4, 1437])
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score  # 18
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None

class H2OQwen2_5OmniAttention_drop(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.kv_cache = H2OKVCache_LayerWise(layer_idx=self.layer_idx, hh_size=config.hh_size,  # 200
            recent_size=config.
            recent_size,  # 200
            k_seq_dim=2,  # k的长度的维度
            v_seq_dim=2, hh_ratio=config.hh_ratio,  # 0.1
            recent_ratio=config.recent_ratio,  # 0.1
            hh_layer=config.hh_layer)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_values: Optional[Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs: Unpack[FlashAttentionKwargs], ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs, )
        # print(f"attn_weights={attn_weights.shape},hidden_states={hidden_states.shape},query_states={query_states.shape}")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        is_kvcache = True
        if q_len != 1 and is_kvcache:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)  # [1, 32, 31676, 128]
            query_states = query_states[:, :, -16:, :]  # [1, 32, 32, 128]
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attention_mask = torch.full([1, query_states.shape[-2], key_states.shape[-2]], torch.finfo(query_states.dtype).min, device=query_states.device, dtype=query_states.dtype, )
            for i in range(0, query_states.shape[-2]):
                attention_mask[..., i, 0: i + key_states.shape[-2] - query_states.shape[-2] + 1] = 0
            causal_mask = attention_mask[:, -query_states.shape[-2]:, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [1, 32, 32, 31676]
            self.kv_cache(past_key_values, attn_weights)  # past_key_values = [1, 8, 31676, 128]
        return attn_output, attn_weights

    def _clean_cache(self):
        self.kv_cache._clean_scores()


class HybridKVQwen2_5OmniAttention_drop(H2OQwen2_5OmniAttention_drop):
    """
    HybridKV Attention for Qwen3VL: 基于模态感知的混合 KV Cache 压缩。

    核心区别：区分视觉 token 和文本 token，对视觉 token 保护不驱逐，
    对文本 token 使用基于注意力分数的 top-k 驱逐。

    参考论文: HybridKV: Hybrid KV Cache Compression for Efficient Multimodal
              Large Language Model Inference
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        from model.qwen35.modify_qwen35 import HybridKVCache_LayerWise
        self.kv_cache = HybridKVCache_LayerWise(
            layer_idx=self.layer_idx,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            hh_layer=getattr(config, 'hh_layer', 27),
            visual_token_budget_ratio=getattr(config, 'visual_token_budget_ratio', 1.0),
            entropy_threshold_percentile=getattr(config, 'entropy_threshold_percentile', 0.7),
            min_visual_block_size=getattr(config, 'min_visual_block_size', 16),
        )

class H2OQwen3Vl_Chunk_Attention_drop(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, block_size: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.block_size = block_size
        self.mask = None

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.kv_cache = H2OKVCache_LayerWise(layer_idx=self.layer_idx, hh_size=config.hh_size,  # 200
            recent_size=config.
            recent_size,  # 200
            k_seq_dim=2,  # k的长度的维度
            v_seq_dim=2, hh_ratio=config.hh_ratio,  # 0.1
            recent_ratio=config.recent_ratio,  # 0.1
            hh_layer=config.hh_layer)


    def chunk_state(self,hidden_states):
        token_size = hidden_states.size(-2)
        indices = list(range(0, token_size, self.block_size))
        blocks = []
        if self.mask is None:
            for start in indices:
                end = start + self.block_size
                block = hidden_states[:, :, start:end, :]
                # if block.size(-2) < block_size:
                #     padding_size = block_size - block.size(-2)
                #     block = torch.nn.functional.pad(block, (0, 0, 0, padding_size))
                blocks.append(block)
        else:
            start = 0
            for i in range(token_size):
                if torch.all(self.mask[i] == 1):
                    block = hidden_states[:, :, start:i+1, :] 
                    blocks.append(block)
                    start = i + 1 
        return blocks
        
    def chunk_merge(self, chunk_list, indices):
        selected_blocks = []
        for i in range(len(chunk_list)):
            if i in indices:
                selected_blocks.append(chunk_list[i])
            else:
                device,dtype = chunk_list[i].device,chunk_list[i].dtype
                B, H, l, D = chunk_list[i].shape
                merged = torch.zeros((B, H, l, D), device=device, dtype=dtype)
                selected_blocks.append(merged)
        # selected_blocks = [chunk_list[i] for i in indices]
        merged_tensor = torch.cat(selected_blocks, dim=-2)
        return merged_tensor
        
    def eviction_block(self,query_states,key_states,value_states, keep_ratio = 0.2, recent_ratio = 0.2):
        block_query = self.chunk_state(query_states)
        block_key = self.chunk_state(key_states)
        block_values = self.chunk_state(value_states)
        block_weight = []
        r = int(len(block_query) * recent_ratio)
        for i in range(len(block_query) - r):
            key = block_key[i].repeat_interleave(self.num_key_value_groups, dim=1)  # [1, 32, 31676, 128]
            query = block_query[i]
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            attention_mask = torch.full([1, query.shape[-2], key.shape[-2]], torch.finfo(query.dtype).min, device=query.device, dtype=query.dtype, )
            for j in range(query.shape[-2]):
                attention_mask[..., j, 0: j + key.shape[-2] - query.shape[-2] + 1] = 0
            causal_mask = attention_mask[:, -query.shape[-2]:, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # [1, 32, 32, 31676]
            block_weight.append((attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean().item())
        k = int(len(block_weight) * keep_ratio)
        block_weight = torch.tensor(block_weight)
        # print(torch.allclose(block_weight, block_weight[0].expand_as(block_weight)))
        _, indices = torch.topk(block_weight,k, largest=True)
        indices = torch.sort(indices).values.tolist()
        for i in range(len(block_query)-r,len(block_query)):
            indices.append(i)
        return self.chunk_merge(block_query,indices),self.chunk_merge(block_key,indices), self.chunk_merge(block_values,indices)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_values: Optional[Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs: Unpack[FlashAttentionKwargs], ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs, )
        # print(f"attn_weights={attn_weights.shape},hidden_states={hidden_states.shape},query_states={query_states.shape}")
        # print(attention_entropy(attn_output))
        # import ipdb; ipdb.set_trace()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if hidden_states.shape[-2]>=100:
            query_states,key_states,value_states = self.eviction_block(query_states,key_states,value_states)
        if past_key_values is not None and hidden_states.shape[-2]>=100:
            past_key_values.layers[self.layer_idx].keys = key_states
            past_key_values.layers[self.layer_idx].values = value_states
        return attn_output, attn_weights

    def _clean_cache(self):
        self.kv_cache._clean_scores()




class StreamingLLMQwen2_5OmniAttention_drop(H2OQwen2_5OmniAttention_drop):
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = StreamingLLMKVCache_LayerWise(layer_idx=self.layer_idx, hh_size=config.hh_size, recent_size=config.recent_size, k_seq_dim=2, v_seq_dim=2, hh_ratio=config.hh_ratio, recent_ratio=config.recent_ratio)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, output_attentions: bool = False, use_cache: bool = False, cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(f"The input hidden states seems to be silently casted in float32, this might be related to"
                                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                                f" {target_dtype}.")

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None and self.layer_idx >= self.config.max_window_layers):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, sliding_window=sliding_window, is_causal=self.is_causal, use_top_left_mask=self._flash_attn_uses_top_left_mask, )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        if q_len != 1:
            self.kv_cache(past_key_value, attn_weights)
        return attn_output, attn_weights, past_key_value


class StreamingLLMKVCache_LayerWise:
    def __init__(self, layer_idx: Optional[int] = None, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, hh_ratio=None, recent_ratio=None, pattern="b",):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"StreamingLLMKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None
        self.seq_len = None
        self.pattern = pattern  # self.heatmap = Heatmap(layer_num=self.layer_num)

    def __call__(self, past_key_values, attn_weights=None):
        if self.hh_ratio is not None:
            self.cache_size = int(past_key_values[self.layer_idx][0].shape[2] * (self.hh_ratio + self.recent_ratio))
        self.hh_size = self.cache_size // 2  # 70
        self.recent_size = self.cache_size - self.hh_size

        if past_key_values is None:
            return None

        # seq_len = past_key_values[0].size(self.k_seq_dim)
        seq_len = past_key_values[self.layer_idx][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[self.layer_idx][0].shape  # torch.Size([1, 8, 18, 128])
        # import ipdb; ipdb.set_trace()
        keep_sink = torch.arange(0, self.hh_size, device=past_key_values[self.layer_idx][0].device).repeat(num_heads, 1)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values[self.layer_idx][0].device).repeat(num_heads, 1)
        keep_idx = torch.cat([keep_sink, keep_recent], dim=-1)

        mask = torch.zeros(num_heads, seq_len, dtype=torch.bool).to(past_key_values[self.layer_idx][0].device)
        mask = mask.scatter(-1, keep_idx, 1)
        # import ipdb; ipdb.set_trace()
        k_hh_recent = past_key_values[self.layer_idx][0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[self.layer_idx][1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        if k_hh_recent.size(-2) != self.cache_size:
            raise ValueError(f"Cache should be of size {self.cache_size}, but is"
                             f" {k_hh_recent.size(-2)}")

        past_key_values.key_cache[self.layer_idx] = k_hh_recent
        past_key_values.value_cache[self.layer_idx] = v_hh_recent
        return past_key_values

    def _clean_scores(self):
        self.hh_score = None


class QuestKVCache_LayerWise:
    def __init__(self, layer_idx: Optional[int] = None, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, hh_ratio=None, recent_ratio=None, hh_layer=27, ):
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                                "when creating this class.")
        if self.layer_idx == 0:
            print(f"H2O KVCache-LayerWise: {hh_size}, {recent_size}, {hh_ratio}, {recent_ratio} {hh_layer}")  # 200, 200
        self.hh_size = hh_size  # 200
        self.recent_size = recent_size  # 200
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size  # 400
        self.k_seq_dim = k_seq_dim  # 2
        self.v_seq_dim = v_seq_dim  # 2
        self.hh_ratio = hh_ratio  # 0.1
        self.recent_ratio = recent_ratio  # 0.1
        self.hh_score = None
        self.seq_len = None
        self.hh_layer = hh_layer

    def __call__(self, past_key_values, query_states):
        return past_key_values

    def __call__1(self, past_key_values, query_states):
        return past_key_values

    def _update_hh_score(self, attn_score_cache):
        ############## stop here and find the bug ##############
        num_new_tokens = attn_score_cache.shape[2]  # 1436

        # attn_score_cache = attn_score_cache[:, :, 2186:, :]
        if self.hh_score is None:  # --
            self.hh_score = attn_score_cache.sum(0).sum(1)  # torch.Size([4, 1436])
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)  # torch.Size([4, 1437])
            attn_score_cache[..., :-num_new_tokens] = attn_score_cache[..., :-num_new_tokens] + self.hh_score  # 18
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None


class QuestQwen2_5OmniAttention_drop(H2OQwen2_5OmniAttention_drop):
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = QuestKVCache_LayerWise(layer_idx=self.layer_idx, hh_size=config.hh_size, recent_size=config.recent_size, k_seq_dim=2, v_seq_dim=2, hh_ratio=config.hh_ratio, recent_ratio=config.recent_ratio)

    def quest(self, past_key_value):
        key_cache = past_key_value.key_cache[self.layer_idx]
        value_cache = past_key_value.value_cache[self.layer_idx]

        return past_key_value

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, output_attentions: bool = False, use_cache: bool = False, cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  # torch.Size([1, 509, 3584])
        key_states = self.k_proj(hidden_states)  # torch.Size([1, 509, 512])
        value_states = self.v_proj(hidden_states)  # torch.Size([1, 509, 512])

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # torch.Size([1, 28, 509, 128])
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # torch.Size([1, 4, 509, 128])
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # torch.Size([1, 4, 509, 128])

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        is_kvcache = True
        if q_len != 1 and is_kvcache:
            self.kv_cache(past_key_value, query_states)
        # import ipdb; ipdb.set_trace()
        key_states = past_key_value.key_cache[0]
        value_states = past_key_value.value_cache[0]

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states,  # torch.Size([1, 509, 28, 128])
            key_states,  # torch.Size([1, 509, 28, 128])
            value_states, attn_mask=causal_mask,  # torch.Size([1, 1, 509, 509])
            dropout_p=self.attention_dropout if self.training else 0.0, is_causal=is_causal, )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _clean_cache(self):
        self.kv_cache._clean_scores()


class NeedleLLMKVCache_LayerWise:
    def __init__(self, layer_idx: Optional[int] = None, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, hh_ratio=None, recent_ratio=None, ):
        self.layer_idx = layer_idx
        if layer_idx == 0:
            print(f"NeedleLLMKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        if recent_size and hh_size:
            self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache=None):
        return past_key_values

    def _clean_scores(self):
        self.hh_score = None


class ChunkSimilarityKVCache_LayerWise:
    """
    基于块内相似度的KV Cache驱逐策略。
    
    核心思想：
    1. 将KV cache中的token按固定大小分块
    2. 对每个块，取块的最后一个token作为"块代表"
    3. 计算块内所有token的均值表示
    4. 计算最后一个token与块均值之间的余弦相似度
    5. 相似度越高 → 块内内容越一致/与当前查询越相关 → 优先保留
    6. 最近的若干块始终保留（recent window）
    """

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        block_size: int = 32,
        keep_ratio: float = 0.3,
        recent_ratio: float = 0.2,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.keep_ratio = keep_ratio
        self.recent_ratio = recent_ratio
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

        if layer_idx == 0:
            print(
                f"ChunkSimilarityKVCache-LayerWise: block_size={block_size}, "
                f"keep_ratio={keep_ratio}, recent_ratio={recent_ratio}"
            )

    def compute_chunk_similarity(self, key_states):
        """
        计算每个块的相似度分数。
        
        Args:
            key_states: [bsz, num_heads, seq_len, head_dim]
            
        Returns:
            chunk_scores: [num_chunks] 每个块的相似度分数（越高越应保留）
            chunk_boundaries: List[Tuple[int, int]] 每个块的 (start, end) 索引
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # 按固定大小分块
        chunk_boundaries = []
        start = 0
        while start < seq_len:
            end = min(start + self.block_size, seq_len)
            chunk_boundaries.append((start, end))
            start = end
        
        num_chunks = len(chunk_boundaries)
        chunk_scores = torch.zeros(num_chunks, device=key_states.device, dtype=torch.float32)
        
        for i, (s, e) in enumerate(chunk_boundaries):
            chunk = key_states[:, :, s:e, :]  # [bsz, num_heads, chunk_len, head_dim]
            chunk_len = e - s
            
            if chunk_len <= 1:
                # 块太小，给一个默认的中间分数
                chunk_scores[i] = 0.0
                continue
            
            # 块的最后一个token: [bsz, num_heads, head_dim]
            last_token = chunk[:, :, -1, :]
            
            # 块内所有token的均值: [bsz, num_heads, head_dim]
            chunk_mean = chunk.mean(dim=2)
            
            # 计算余弦相似度: [bsz, num_heads]
            cos_sim = F.cosine_similarity(last_token, chunk_mean, dim=-1)
            
            # 对所有batch和head取平均，得到该块的综合相似度分数
            chunk_scores[i] = cos_sim.mean()
        
        return chunk_scores, chunk_boundaries

    def __call__(self, past_key_values, key_states):
        """
        执行基于块相似度的KV cache驱逐。
        
        Args:
            past_key_values: DynamicCache对象
            key_states: [bsz, num_heads, seq_len, head_dim] 当前的key states
        """
        seq_len = past_key_values[self.layer_idx][0].size(self.k_seq_dim)
        
        # 计算需要保留的总token数
        total_keep = int(seq_len * (self.keep_ratio + self.recent_ratio))
        
        if seq_len <= total_keep:
            return past_key_values
        
        bsz, num_heads, _, head_dim = past_key_values[self.layer_idx][0].shape
        current_keys = past_key_values[self.layer_idx][0]  # [bsz, num_heads, seq_len, head_dim]
        
        # 计算recent区域大小
        recent_size = int(seq_len * self.recent_ratio)
        # 非recent区域
        non_recent_len = seq_len - recent_size
        
        if non_recent_len <= 0:
            return past_key_values
        
        # 只对非recent区域进行分块和打分
        non_recent_keys = current_keys[:, :, :non_recent_len, :]
        
        # 计算块相似度
        chunk_scores, chunk_boundaries = self.compute_chunk_similarity(non_recent_keys)
        
        num_chunks = len(chunk_boundaries)
        if num_chunks == 0:
            return past_key_values
        
        # 计算需要从非recent区域保留的token数
        keep_from_non_recent = int(seq_len * self.keep_ratio)
        # 需要保留的块数量（向上取整以保证足够token）
        num_keep_chunks = max(1, keep_from_non_recent // self.block_size)
        num_keep_chunks = min(num_keep_chunks, num_chunks)
        
        # 根据相似度分数选择top-k个块（分数越高越相关，优先保留）
        _, topk_chunk_indices = torch.topk(chunk_scores, num_keep_chunks, largest=True)
        topk_chunk_indices = topk_chunk_indices.sort().values  # 保持顺序
        
        # 构建保留的token mask
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=current_keys.device)
        
        # 保留选中的块中的所有token
        for idx in topk_chunk_indices:
            s, e = chunk_boundaries[idx.item()]
            keep_mask[s:e] = True
        
        # 保留recent区域的所有token
        keep_mask[non_recent_len:] = True
        
        # 应用mask到每个head（所有head共享相同的保留决策）
        k_kept = current_keys[:, :, keep_mask, :]
        v_kept = past_key_values[self.layer_idx][1][:, :, keep_mask, :]
        
        past_key_values.layers[self.layer_idx].keys = k_kept
        past_key_values.layers[self.layer_idx].values = v_kept
        
        return past_key_values

    def _clean_scores(self):
        self.hh_score = None


class ChunkSimilarityAttention_drop(nn.Module):
    """
    基于块内相似度的KV Cache驱逐Attention模块。
    
    驱逐策略：
    1. 将输入token的key states按固定大小分块
    2. 对每个块，计算最后一个token与块内所有token均值的余弦相似度
    3. 相似度越高的块与当前查询越相关，优先保留
    4. 最近的token窗口始终保留
    
    该策略的直觉：
    - 块的最后一个token自然"总结"了前面的内容（由于causal attention）
    - 如果最后一个token与块均值高度相似，说明块内内容具有高内聚性
    - 内聚性高的块更可能包含重要的连贯信息，应优先保留
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, block_size: int = 32):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.block_size = block_size

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        # 块相似度KV Cache
        self.kv_cache = ChunkSimilarityKVCache_LayerWise(
            layer_idx=self.layer_idx,
            block_size=block_size,
            keep_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # 在prefill阶段（q_len > 1）执行块相似度驱逐
        if q_len > 1 and past_key_values is not None:
            self.kv_cache(past_key_values, key_states)

        return attn_output, attn_weights

    def _clean_cache(self):
        self.kv_cache._clean_scores()


class NeedleLLMQwen2_5OmniAttention_drop(H2OQwen2_5OmniAttention_drop):
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.kv_cache = NeedleLLMKVCache_LayerWise(layer_idx=self.layer_idx, hh_size=config.hh_size, recent_size=config.recent_size, k_seq_dim=2, v_seq_dim=2, hh_ratio=config.hh_ratio, recent_ratio=config.recent_ratio)  # self.kv_cache = StreamingLLMKVCache_LayerWise(  #     layer_idx=self.layer_idx,  #     hh_size=config.hh_size,  #     recent_size=config.recent_size,  #     k_seq_dim=2,  #     v_seq_dim=2,  #     hh_ratio=config.hh_ratio,  #     recent_ratio=config.recent_ratio  # )

    def quest(self, past_key_value):
        key_cache = past_key_value.key_cache[self.layer_idx]
        value_cache = past_key_value.value_cache[self.layer_idx]

        return past_key_value

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, output_attentions: bool = False, use_cache: bool = False, cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(f"The input hidden states seems to be silently casted in float32, this might be related to"
                                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                                f" {target_dtype}.")

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None and self.layer_idx >= self.config.max_window_layers):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, sliding_window=sliding_window, is_causal=self.is_causal, use_top_left_mask=self._flash_attn_uses_top_left_mask, )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        is_kvcache = False
        if q_len != 1 and is_kvcache:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            query_states = query_states[:, :, -32:, :]
            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if self.layer_idx == self.config.num_hidden_layers - 1 and self.num_heads == 28 and self.hidden_size == 3584:
                attn_weights = torch.matmul(query_states / math.sqrt(self.head_dim), key_states.transpose(2, 3))
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            causal_mask = attention_mask[:, :, -query_states.shape[-2]:, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            self.kv_cache(past_key_value, attn_weights)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _clean_cache(self):
        self.kv_cache._clean_scores()