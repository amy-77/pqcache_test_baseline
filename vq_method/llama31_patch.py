import os
import math
import torch
import types
from torch import nn
import matplotlib.pyplot as plt
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaModel,
    repeat_kv
)
from .baseline_compressor import *
from .flash_attn_with_score import flash_attn_with_score
from .retrieval_based.pq_search import *
from .retrieval_based.sparq import *
from flash_attn import flash_attn_func
import seaborn as sns
from loguru import logger
import numpy as np



# # 主要组件层次结构
# VQLlama31ForCausalLM (主类)
#     ↓
# PPLlamaModelPatch (模型补丁)
#     ↓
# LlamaDecoderLayerPatch (层补丁)
#     ↓
# LlamaAttentionPatch (注意力补丁)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x = (q * cos)
    # y = (rotate_half(q) * sin)
    q_embed = x.add_(rotate_half(q).mul_(sin))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def layer2device(idx, layer_cnt):
    gpu_in_use = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    step = math.ceil(layer_cnt / gpu_in_use)
    return torch.device(f"cuda:{idx // step}")


def get_device(layer: nn.Module):
    for param in layer.parameters():
        return param.device


def LlamaAttentionPatch(attn: LlamaAttention, config, idx):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 1. 基础检查和投影
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Do not support bsz > 1 yet."
        # 2. QKV投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # 3. 重塑为多头格式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # 4. 检查是否是第一次前向传播
        # first_time = True  → Prefill阶段（处理完整输入序列）
        # first_time = False → Decoding阶段（生成下一个token）
        first_time = (position_ids.nelement() != 1)
        # print(f"[DEBUG] Layer {self.layer_idx}: first_time = {first_time}, position_ids.nelement() = {position_ids.nelement()}")
        # 5. 应用RoPE位置编码
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
       
        # 6. 检查是否是第一次前向传播
        if first_time:
            self.seq_cnt += 1
            self.fwd_cnt = 0
        
        
        if self.compressor == "original":
            # 使用标准Flash Attention，无压缩
            print("使用原始Flash Attention，无压缩")
            # print(f"first_time: {first_time}, past_key_value is None: {past_key_value is None}")
            if past_key_value is not None:
                    # 只打印第一层的repeat_kv后尺寸
                # if self.layer_idx == 0:
                #     print(f"Original Layer{self.layer_idx}: {'Prefill' if first_time else 'Decode'} - key_states before update: {key_states.shape}")
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)
                
            # 只打印第一层的尺寸
            # if self.layer_idx == 0:
            #     print(f"Original Layer{self.layer_idx}: {'Prefill' if first_time else 'Decode'} - key_states after update: {key_states.shape}")
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
                
            if self.fwd_cnt <= 0 and self.idx <= 0:
                print(f"Using naive flash-attn, NO COMPRESSION IS CONVEYED NOW, {os.getpid()}")
            
            # TODO: 这里的Decoding能否换成varlen_func来加速？
            attn_output = flash_attn_func(
                    query_states.transpose(1,2),
                    key_states.transpose(1,2),
                    value_states.transpose(1,2),
                    causal = True
                ).transpose(1,2)
            
            # ==================== 添加Neff计算 ====================
            # 只在prefill阶段且启用Neff分析时计算，prefill阶段注意力稀疏预判
            # if first_time and hasattr(self, 'enable_neff_analysis') and self.enable_neff_analysis:
                # print(f"[DEBUG] Layer {self.layer_idx}: 开始计算Neff统计...")
                # 暂时注释掉Neff计算以避免内存问题
                # self.compute_neff_statistics(query_states, key_states, value_states)
                # print(f"[DEBUG] Layer {self.layer_idx}: Neff统计已跳过（避免内存问题）")
            # =====================================================
            
            
        elif self.compressor in ["pq_search", "sparq_f"]: 
            
            if first_time:  # Prefill阶段
                # 1. 将4041个token送入PQ压缩缓存
                # if self.layer_idx == 0:
                #     print(f"Layer{self.layer_idx}: query_states.shape={query_states.shape}") #query_states.shape=torch.Size([1, 32, 2853, 128])
                #     print(f"Layer{self.layer_idx}: key_states.shape={key_states.shape}") # key_states.shape=torch.Size([1, 8, 2853, 128])
                #     print(f"Layer{self.layer_idx}: value_states.shape={value_states.shape}") # value_states.shape=torch.Size([1, 8, 2853, 128])
                #     print("进入prefill_attn + pq_search")
                # 内部处理:
                # 1. 构建PQ索引（聚类、码本生成）
                # 2. 压缩存储所有历史token
                # 3. 使用Flash Attention计算注意力
                # 4. 输出：attn_output（注意力计算结果）      
                attn_output, _ = self.kvcache_quantizer.prefill_attn(query_states, (key_states, value_states))
                # attn_output.shape = [1, 32, 4041, 128]  ✅ 正常！
                # 2. 只把最后一个token存入官方缓存
                _,_ = past_key_value.update(key_states[...,:1].clone(), value_states[...,:1].clone(), self.layer_idx, None)
                
                # if self.layer_idx == 0:
                #     # 我觉得挺奇怪的，这个地方的dim=1，len=2853
                #     print("key_states[...,:1].shape = ", key_states[...,:1].shape) # torch.Size([1, 8, 2853, 1])
                #     print("value_states[...,:1].shape = ", value_states[...,:1].shape) # torch.Size([1, 8, 2853, 1])
                #     print("past_key_value.key_cache[self.layer_idx].shape = ", past_key_value.key_cache[self.layer_idx].shape) # torch.Size([1, 8, 2853, 1])
                #     print("past_key_value.value_cache[self.layer_idx].shape = ", past_key_value.value_cache[self.layer_idx].shape) # torch.Size([1, 8, 2853, 1])    
            
            else:  # Decoding阶段
                
                # GQA, Query头数 > Key/Value头数
                key_states = repeat_kv(key_states, self.num_key_value_groups) 
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                #从 torch.Size([1, 8, 1, 128]) 变成 torch.Size([1, 32, 1, 128])             
                # 2. decoding_attn内部会:
                #    - 从PQ压缩缓存检索重要的历史token
                #    - 与当前新token组合计算注意力
                # if self.layer_idx == 0:
                    # print("进入decoding_attn + pq_search + euc_metric")
                    # print("query_states.shape = ", query_states.shape) # torch.Size([1, 32, 1, 128])
                    # print("key_states.shape = ", key_states.shape) # torch.Size([1, 32, 1, 128])
                    # print("value_states.shape = ", value_states.shape) # torch.Size([1, 32, 1, 128]) 
                    
                attn_output = self.kvcache_quantizer.decoding_attn(self.num_key_value_groups, query_states, key_states, value_states).to(query_states.dtype)
                attn_output = attn_output.to(query_states.dtype)
        
                
        else:  # 使用自定义的Flash Attention，有压缩
            
            assert past_key_value is not None
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)
            kv = (key_states, value_states)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if self.use_flash_attn and first_time:  # Prefill阶段
                # 自定义Flash Attention调用：不仅计算attention，还计算每个token的重要性分数，  attn_output是attention输出，score是重要性分数矩阵
                attn_output, score = flash_attn_with_score(query_states, key_states, value_states, \
                                                            phase="prefill", gumbel_adjustment=False, \
                                                            score_func=self.score_func)
                score = score.reshape([bsz, self.num_key_value_heads, self.num_key_value_groups, q_len])
                
                if self.score_func == "sum":
                    score = score.sum(dim=2)
                elif self.score_func == "max":
                    score = score.max(dim=2).values
                else:
                    raise Exception(f"Given score func {self.score_func} do not support yet.")
                # 根据注意力分数进行压缩，得到压缩后的key和value
                compressed_k, compressed_v, _ = self.kvcache_quantizer.apply(kv,  attention_score=score, 
                                                                            query_states=query_states)
            else:
                # 标准注意力计算
                attention_mask = None 
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = self.kvcache_quantizer.restore(attn_weights, self.num_key_value_groups).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
        # 第5部分：输出处理
        # 1. 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        # 2. 输出注意力权重
        if not output_attentions:
            attn_weights = None
        # 3. 更新计数器
        self.fwd_cnt += 1
        # 4. 返回输出
        return attn_output, attn_weights, past_key_value
    


    # ==================== 添加Neff计算函数 ====================
    def compute_neff_statistics(self, query_states, key_states, value_states):
        """
        计算Neff统计量，优化内存使用避免OOM
        """
        if not self.enable_neff_analysis:
            return
            
        bsz, num_heads, seq_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[2]
        
        if seq_len != kv_seq_len:
            # print(f"[DEBUG] 跳过decode阶段的Neff计算: seq_len={seq_len}, kv_seq_len={kv_seq_len}")
            return
        
        # 内存优化：分批计算并直接累积统计量，避免创建完整注意力矩阵
        batch_size = 64  # 增大batch_size提高效率
        
        # 初始化累积统计量
        total_entropy = 0.0
        total_tokens = 0
        total_neff = 0.0
        
        for i in range(0, seq_len, batch_size):
            end_i = min(i + batch_size, seq_len)
            query_batch = query_states[:, :, i:end_i, :]  # [bsz, num_heads, batch_size, head_dim]
            
            # 计算这个batch的注意力分数
            attention_scores_batch = torch.matmul(query_batch, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            # 应用causal mask (只对当前batch的query位置)
            causal_mask = torch.triu(torch.ones(end_i - i, kv_seq_len, device=attention_scores_batch.device), 
                                   diagonal=1 if i == 0 else 1 - i)
            attention_scores_batch = attention_scores_batch.masked_fill(causal_mask.bool(), float('-inf'))
            
            # 计算softmax权重
            attention_weights_batch = torch.softmax(attention_scores_batch, dim=-1)  # [bsz, num_heads, batch_size, kv_seq_len]
            
            # 直接在这个batch中计算Neff统计
            batch_tokens = end_i - i
            
            # 计算entropy: -sum(p * log(p))
            log_attention = torch.log(attention_weights_batch + 1e-8)
            entropy_batch = -torch.sum(attention_weights_batch * log_attention, dim=-1)  # [bsz, num_heads, batch_size]
            
            # 计算Neff: exp(entropy)
            neff_batch = torch.exp(entropy_batch)  # [bsz, num_heads, batch_size]
            
            # 累积统计
            total_entropy += entropy_batch.sum().item()
            total_neff += neff_batch.sum().item()
            total_tokens += batch_tokens * bsz * num_heads
            
            # 清理临时变量以释放内存
            del attention_scores_batch, attention_weights_batch, entropy_batch, neff_batch
            torch.cuda.empty_cache()
        
        # 计算平均值
        avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0
        avg_neff = total_neff / total_tokens if total_tokens > 0 else 0.0
        
        # 详细的调试信息（只在前几层打印）
        # if self.layer_idx <= 1:
        #     print(f"\n=== Layer {self.layer_idx} Neff计算详解 ===")
        #     print(f"输入形状: query_states={query_states.shape}, key_states={key_states.shape}")
        #     print(f"序列长度: seq_len={seq_len}, kv_seq_len={kv_seq_len}")
        #     print(f"平均熵: {avg_entropy:.4f}")
        #     print(f"平均Neff: {avg_neff:.4f}")
        
        # 保存到全局统计中（如果需要的话）
        if not hasattr(self, 'neff_statistics'):
            self.neff_statistics = []
        
        self.neff_statistics.append({
            'layer_idx': self.layer_idx,
            'seq_len': seq_len,
            'avg_entropy': avg_entropy,
            'avg_neff': avg_neff,
            'total_tokens': total_tokens
        })
        
        # print(f"[DEBUG] Layer {self.layer_idx}: 完成Neff计算, avg_neff={avg_neff:.4f}, tokens={total_tokens}")
    # =========================================================

    attn.forward = types.MethodType(forward, attn)
    attn.compute_neff_statistics = types.MethodType(compute_neff_statistics, attn)
    attn.use_flash_attn = True
    attn.fwd_cnt = 0
    attn.idx = idx
    attn.score_func = config.score_func
    attn.compressor = config.compressor
    attn.seq_cnt = -1
    
    # ==================== 添加Neff分析开关 ====================
    attn.enable_neff_analysis = getattr(config, 'enable_neff_analysis', False)
    # print(f"[DEBUG] Layer {attn.layer_idx}: enable_neff_analysis = {attn.enable_neff_analysis}")
    # =========================================================
    
    if config.compressor == "h2o":
        attn.kvcache_quantizer = KVCacheH2OOfficial(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            config.sink_size,
        )

    elif config.compressor == "no_drop_lb":
        assert attn.score_func == "sum", "full KV limited based compressor only accept sum function"
        attn.kvcache_quantizer = fullKVLimitBasedCompressor(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            config.gqa,
            config.sink_size,
        )

    elif config.compressor == "pq_search":
        # print(f"----------------进入 LlamaAttentionPatch类的 __init__()函数-----------------")
        attn.kvcache_quantizer = PqBasedSearchCompressor(
            config.compress_ratio,
            config.recent_ratio,
            config.n_subvec_per_head,
            config.n_subbits,
            config.gqa,
            config.sink_size,
            layer_idx = attn.idx,
            cur_device=layer2device(attn.idx, config.num_hidden_layers),
            max_iter = config.max_iter,
            kv_head = config.num_key_value_heads,
            dim = config.hidden_size // config.num_attention_heads,
            num_layer_cnt = config.num_hidden_layers
        )

    elif config.compressor == "sparq_f":
        if os.environ.get("MODE","off") == "profile":
            raise NotImplementedError("profile mode for Sparq is not done yet.")
        else:
            if attn.idx <= 2:
                print(f"Using Sparq Compressor, gpu version.")
            attn.kvcache_quantizer = SparQCompressor(
                config.compress_ratio,
                config.recent_ratio,
                config.sink_size,
                config.gqa,
                r = config.topr,
                idx = attn.idx,
                model_config = config,
                layer_idx = attn.idx,
                cur_device=layer2device(attn.idx, config.num_hidden_layers),
                kv_head = config.num_key_value_heads,
                dim = config.hidden_size // config.num_key_value_heads
            )
    elif config.compressor == "original":
        pass
    else:
        raise Exception("Invalid compression strategy name")



    
def LlamaDecoderLayerPatch(layer: LlamaDecoderLayer, config, layer_idx):
    """
    优化的transformer层的forward方法
    """
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                **kwargs,):

        residual = hidden_states.clone()
        batch, seq_len, embed_dim = hidden_states.shape
        # hidden_states = self.input_layernorm(hidden_states)
        for start_idx in range(0, seq_len, 32000):
            end_idx = min(seq_len, start_idx + 32000)
            hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(
                hidden_states[:, start_idx:end_idx, :]
            )

        # Self Attention
        """
        调用当前层的self_attn的forward方法，
        self_attn是一个对象，self_attn()自动调用self_attn.forward()
        而此时的self_attn都已经在初始化的时候被LlamaAttentionPatch补丁替换了，
        
        self_attn()属于LlamaAttentionPatch对象，self_attn.forward()自动调用LlamaAttentionPatch.forward()
        """
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # hidden_states = residual + hidden_states
        hidden_states.add_(residual)
        del(residual)
        
        # hidden_states = residual + hidden_states
        n_chunks = math.ceil(seq_len / 32000)
        avg_chunk_size = math.ceil(seq_len // n_chunks)
        for start_idx in range(0, seq_len, avg_chunk_size):
            end_idx = min(seq_len, start_idx + avg_chunk_size)
            part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
            part_hidden_states = self.post_attention_layernorm(part_hidden_states)
            part_hidden_states = self.mlp(part_hidden_states)
            hidden_states[:, start_idx:end_idx, :] += part_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs

    """
    初始化: 替换原层的forward方法，并安装VQ补丁
    """
    layer.forward = types.MethodType(forward, layer)
    layer.device = layer2device(layer_idx, config.num_hidden_layers)
    """
    此时layer.self_attn是LlamaAttention对象，layer.self_attn()自动调用LlamaAttentionPatch.forward()
    """
    LlamaAttentionPatch(layer.self_attn, config, layer_idx)
    
    return layer.half()



def PPLlamaModelPatch(model:LlamaModel, config):
    """
    接收一个旧的LlamaModel对象，给它打补丁，返回修改后的同一个对象
    函数整体功能：
    这个函数是一个模型补丁函数，用于将标准的Llama模型转换为支持管线并行(PP)和向量量化(VQ)压缩的优化版本。
    主要实现以下功能：
    1. 重写模型的forward方法，支持跨GPU的管线并行计算
    2. 将模型的不同层分配到不同的GPU上，实现内存分散
    3. 添加对VQ压缩的支持，优化KV cache的内存使用
    4. 转换为半精度(FP16)以节省内存
    
    输入参数说明：
    model: LlamaModel - 标准的Llama模型实例，包含完整的模型结构和参数
    config - 模型配置对象，包含模型的各种参数设置，如层数、隐藏维度、管线大小等
    
    输出：
    返回修改后的LlamaModel实例，具有新的forward方法和优化的GPU分配
    """
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,          # 输入的token ID序列，形状为(batch_size, seq_len)
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于忽略padding token
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码ID，指定每个token的位置
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,  # 过去的KV cache，用于加速生成
        inputs_embeds: Optional[torch.FloatTensor] = None,  # inputs_embeds 是input_ids查完 embedding table 后的向量
        use_cache: Optional[bool] = None,            # 是否使用KV cache
        output_attentions: Optional[bool] = None,    # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出所有层的隐藏状态
        return_dict: Optional[bool] = None,          # 是否返回字典格式的输出
        cache_position: Optional[torch.LongTensor] = None,  # VQ压缩中的cache位置信息
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        重写的forward方法，支持管线并行和VQ压缩
        """
        
        # ==================== 第一部分：参数初始化和验证 ====================
        # 设置默认参数值，如果未提供则使用config中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 验证输入参数：inputs_embeds必须为None（这个实现不支持预计算嵌入）
        assert inputs_embeds is None
        # 确保input_ids和inputs_embeds不能同时提供或同时为空
        # input_ids 是词的编号，inputs_embeds 是input_ids查完 embedding table 后的向量
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # ==================== 第二部分：输入处理和缓存设置 ====================
        # 将input_ids转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # self.embed_tokens是一个embedding layer，输入input_ids，输出inputs_embeds
        return_legacy_cache = False

        # 处理legacy格式的past_key_values，转换为新的Cache格式
        if use_cache and not isinstance(past_key_values, Cache):  
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # 验证必需的参数：cache_position和position_ids必须提供
        if cache_position is None or position_ids is None:
            raise Exception("We don't expect this case to happen")
        
        # ==================== 第三部分：初始化变量 ====================
        causal_mask = None                           # 因果掩码，这里设为None（在attention层中处理）
        hidden_states = inputs_embeds                # 初始隐藏状态就是输入嵌入
        position_embeddings = None                   # 位置嵌入，这里设为None

        # 初始化输出收集器  ()表示元组
        all_hidden_states = () if output_hidden_states else None  # 用于收集所有层的隐藏状态
        all_self_attns = () if output_attentions else None        # 用于收集所有层的注意力权重
        next_decoder_cache = None                    # 下一轮的decoder cache

        # ==================== 第四部分：管线并行处理每一层 ====================
        # idx 是层数，decoder_layer 是当前层
        for idx, decoder_layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，收集当前层的隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 获取当前层对应的past_key_value
            # past_key_values 是所有层的KV cache，len(past_key_values) == len(self.layers) 表示有所有层的KV cache
            past_key_value = past_key_values[idx] if len(past_key_values) == len(self.layers) else None
            
            if past_key_value is not None:
                # 验证past_key_value在正确的设备上
                assert past_key_values[idx][0].device == get_device(decoder_layer)
            
            # ==================== 设备间数据传输（管线并行的核心） ====================
            # 将 hidden_states 移动到当前层所在的GPU
            if hidden_states.device != decoder_layer.device:
                hidden_states = hidden_states.to(decoder_layer.device)

            # 将 position_ids 移动到当前层所在的GPU
            if position_ids.device != decoder_layer.device:
                position_ids = position_ids.to(decoder_layer.device)

            # 将 attention_mask 移动到当前层所在的GPU
            if attention_mask is not None and attention_mask.device != decoder_layer.device:
                attention_mask = attention_mask.to(decoder_layer.device)

            # ==================== 调用当前层的forward方法 ====================
            """
            调用当前层的forward方法，
            layer是一个对象，decoder_layer()自动调用decoder_layer.forward(),
            而此时所有的self.layers都已经在初始化的时候被LlamaDecoderLayerPatch补丁替换了，
            decoder_layer()属于LlamaDecoderLayerPatch对象，decoder_layer.forward()自动调用LlamaDecoderLayerPatch中定义的forward
            """
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # 更新hidden_states为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果使用cache，收集cache信息
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            # 如果需要输出注意力权重，收集当前层的注意力权重
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # ==================== 第五部分：最终处理和输出 ====================
        # 将最终的hidden_states移动到norm层所在的GPU
        if hidden_states.device != get_device(self.norm):
            hidden_states = hidden_states.to(get_device(self.norm))
        # 应用最终的层归一化
        hidden_states = self.norm(hidden_states)

        # 如果需要输出隐藏状态，添加最终的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # 设置最终的cache
        next_cache = next_decoder_cache if use_cache else None

        # 如果需要返回legacy格式的cache，进行转换
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        # 根据return_dict参数决定返回格式
        if not return_dict:
            # 返回元组格式
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # 返回字典格式
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,         # 最后一层的隐藏状态
            past_key_values=next_cache,              # 更新后的KV cache
            hidden_states=all_hidden_states,         # 所有层的隐藏状态
            attentions=all_self_attns,               # 所有层的注意力权重
        )


    # ==================== 第六部分：模型配置和GPU分配 ====================
    """
    模型初始化（只执行一次）
    执行一次，在VQLlama31ForCausalLM.init()中会调用PPLlamaModelPatch函数，此时执行函数体line725-742
    """
    
    
    # 设置模型的词汇表大小
    model.vocab_size = config.vocab_size
    # print("qyl----------PPLlamaModelPatch.forward()----------:\n", model.forward)
    model.forward = types.MethodType(forward, model)
    """
    将新的forward方法绑定到模型实例
    右边的forward函数就是PPLlamaModelPatch定义的新函数
    从此以后，调用model.forward() 或 model() 都会执行PPLlamaModelPatch中定义的forward
    """
    # 将embed_tokens固定在cuda:0上（CUDA_VISIBLE_DEVICES会映射到正确的物理GPU）
    model.embed_tokens = model.embed_tokens.to(torch.device("cuda:0"))
    # ==================== 第七部分：管线并行层分配 ====================
    # 对每个decoder layer调用LlamaDecoderLayerPatch做进一步的优化
    for i in range(config.num_hidden_layers):
        # 将每一层移动到对应的GPU上，并应用LlamaDecoderLayerPatch优化
        """
        model.layers[0]=LlamaDecoderLayerPatch对象，  初始化所有层，用补丁LlamaDecoderLayerPatch优化
        model.layers[1]=LlamaDecoderLayerPatch对象， （总共32层）        
        """
        model.layers[i] = LlamaDecoderLayerPatch(model.layers[i].to(layer2device(i, config.num_hidden_layers)), config, i)
    # 将最终的归一化层放在最后一个GPU上
    model.norm = model.norm.to(torch.device(f"cuda:{config.pp_size - 1}"))

    # ==================== 第八部分：最终配置 ====================
    # 禁用梯度检查点（因为管线并行不需要）
    model.gradient_checkpointing = False
    # 调用模型的post_init方法进行最终初始化
    model.post_init()
    # 返回转换为半精度的模型
    return model.half()
    




"""
VQLlama31ForCausalLM (最顶层的主类)
     ↓
PPLlamaModelPatch (整个模型的补丁)
     ↓  
LlamaDecoderLayerPatch (每一层的补丁)
     ↓
LlamaAttentionPatch (注意力机制的补丁)



VQLlama31ForCausalLM.forward()  ← 外层，处理语言模型逻辑
    ↓ 第874行调用
    self.model()  ← 调用PPLlamaModelPatch.forward()
        ↓ 处理Pipeline并行、VQ优化
        LlamaDecoderLayerPatch.forward()  ← 每层的优化
            ↓ 处理内存优化
            LlamaAttentionPatch.forward()  ← Attention层的VQ优化和Neff分析
"""


# 只有一个对象，只有一个 self.model 属性

class VQLlama31ForCausalLM(LlamaForCausalLM):
    """
    重写父类的init方法，实现VQ压缩
    重写父类的forward方法，调用VQ补丁后的模型
    重写父类的prepare_inputs_for_generation方法，为生成任务准备输入
    """
    def __init__(self, config):
        # 此时 self = 当前正在创建的 VQLlama31ForCausalLM 对象
        a = time.perf_counter()
        #  步骤1：调用父类的__init__方法，在当前对象上执行
        super().__init__(config)
        # 执行完后，当前对象已经有了： self.model = LlamaModel(config) 【标准模型】
        # self.lm_head = nn.Linear(...)  # 输出投影头
        
        b = time.perf_counter()
        # print(f"父类初始化后，self.model类型: {type(self.model)}")  # LlamaModel

        # 步骤2：修改当前对象的model属性
        # 初始化模型：只在对象实例化该类时调用init函数， self.model 从标准LlamaModel变成了打补丁的PPLlamaModelPatch
        self.model = PPLlamaModelPatch(self.model, config) 
        #    ↑                             ↑
        #  当前对象的model属性        当前对象的model属性（标准版）
        #  执行后：self.model = 打补丁的LlamaModel 【优化模型】
        # print(f"经过PPLlamaModelPatch补丁初始化后，self.model是什么: {self.model}")
        # print(f"self.model类型: {type(self.model)}")

        # lm_head 把输出投影头挪到最后一块卡并转 FP16 (cuda:pp_size-1)
        self.lm_head = self.lm_head.to(torch.device(f"cuda:{config.pp_size - 1}")).half()
        # embed_tokens 把词向量表固定在 cuda:0 并转 FP16（CUDA_VISIBLE_DEVICES映射到正确GPU）
        self.model.embed_tokens = self.model.embed_tokens.to(torch.device("cuda:0")).half()
        self.layer_num = config.num_hidden_layers
        self.kv_head_cnt = config.num_key_value_heads
        self._device = torch.device("cuda:0")
        self.fwd_cnt = 0
        self.gen_seq_cnt = 0
        self.prefill_len = 0

        c = time.perf_counter()
        # print(f"Init model from llama patch, Time elapsed:{c - b}, {b - a}")
        self.gradient_checkpointing = False
        self.post_init()
    
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None, 
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None, 
        position_ids=None, 
        use_cache=True,
        **kwargs,
    ):  
        # 判断是否是解码阶段（cache_position只有一个元素）
        is_decoding = (cache_position.nelement() == 1)
        
        if past_key_values is not None:
            assert inputs_embeds is None
            if is_decoding:
                assert input_ids.shape[1] != cache_position.shape[0]  
                input_ids = input_ids[:, cache_position]
            else:
                assert input_ids.shape[1] == cache_position.shape[0]

        
        if attention_mask is not None and position_ids is None:
            
            assert len(attention_mask.shape) == 2 and attention_mask.shape[0] == 1, attention_mask.shape
            assert attention_mask.nelement() == (cache_position[-1].item()+1), f"{attention_mask.nelement()},{cache_position}"
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs



    """
    重写父类的forward方法， 调用VQ补丁后的模型
    定义 VQLlama31ForCausalLM.forward() 
    """
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        """
        !!!!!! 请注意: 这里的self.model已经是PPLlamaModelPatch补丁改造后的版本，所以调用PPLlamaModelPatch中定义的forward()
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:,-1:,:])
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        loss = loss.to(torch.device(f"cuda:0")) if loss is not None else None
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_state.to(torch.device(f"cuda:0"))
        logits = logits.to(torch.device(f"cuda:0"))
        
        if outputs.attentions is not None:
            outputs.attentions = outputs.attentions.to(torch.device(f"cuda:0"))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )