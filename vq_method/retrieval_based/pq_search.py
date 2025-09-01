from multiprocessing.managers import SharedMemoryManager
import os
import torch
import torch.multiprocessing as mp
from kmeans_gpu import KMeans as KMeans_gpu  # try kmeans on GPU
from typing import Optional, List, Tuple
import numpy as np
import math
import time
from .sparq_official.methods.ann_attention import MistralAttentionWithANN, Settings
from flash_attn import flash_attn_func

from .retrieval_based_compressor import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeans_sklr
from .multi_core_compressor_v2 import MultiCoreCompressor_v2
import sys
import os.path as osp
from .cache_manager import init_gpu_cache_manager
from loguru import logger
from .global_timer import global_timer

CHECK_RECALL = eval(os.environ.get("CHECK_RECALL", "0"))
SYNC_TEST_TIME = eval(os.environ.get("SYNC_TEST_TIME","0"))

pq_compute_time = 0

# All those configs are based on mistral model architecture.
# TODO: Only init those two object for master process.

# MultiCoreCompressor_v2 是在 initialize_objects 中实例化的
def initialize_objects(config, model):
    global global_compressor
    global cache_managers
    global total_layer_num, pp_size, layer_per_rank # pp_size 表示GPU数量，layer_per_rank 表示每个GPU管理的层数
    
    global H2DStream
    H2DStream = torch.cuda.Stream()
    
    MAX_CPU_IN_USE=64
    MAX_WORKER_CNT=64

    cache_managers = []
    cpu_key_bufs = []
    offload_events = []

    total_layer_num = config.num_hidden_layers
    
    # 安全获取 pp_size，如果 CUDA_VISIBLE_DEVICES 不存在则使用 config.pp_size
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        pp_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        pp_size = getattr(config, 'pp_size', 1)
        logger.info(f"CUDA_VISIBLE_DEVICES not set, using config.pp_size={pp_size}")
    
    layer_per_rank = total_layer_num // pp_size
    
    
    for rank in range(pp_size): # 遍历每个GPU
        # 为每个GPU创建独立的缓存管理器：init_gpu_cache_manager
        cache_manager = init_gpu_cache_manager(
                                        layer_cnt = config.num_hidden_layers // pp_size, # 每个GPU管理的层数
                                        n_kv_head = config.num_key_value_heads, # 每个head的子向量数量
                                        total_max_len = config.max_seq_len, # 最大序列长度
                                        dim = config.hidden_size // config.num_attention_heads, # 每个子向量的维度
                                        device = torch.device(f"cuda:{rank}"), # 当前GPU的设备
                                        dtype = torch.float16, # 数据类型
                                        compress_ratio = config.compress_ratio, # 压缩比 = 0.5    
                                        local_ratio = config.recent_ratio, 
                                        sink_size = config.sink_size, 
                                        global_cache_size = config.global_cache_size, # 全局缓存大小 = 4096
                                        cache_block_size = config.cache_block_size, # 缓存块大小 = 128
                                        cache_topk = config.cache_topk,
                                    )
        cache_managers.append(cache_manager)
        cpu_key_bufs += cache_manager.cpu_key_buffers # 将每个GPU的key_buffers添加到cpu_key_bufs列表中
        offload_events += cache_manager.offload_events  # GPU→CPU同步事件

    # Assume that we utilize 64 cpu cores.
    subvec = int(os.environ.get("SUBVEC", "2"))
    process_cnt = min(config.num_key_value_heads * subvec, MAX_WORKER_CNT)
    
    # print(f"----------------进入 MultiCoreCompressor_v2类的 __init__()函数-----------------")
    # print(f"process_cnt = {process_cnt}")   
    #4. 多进程压缩器创建 
    global_compressor = MultiCoreCompressor_v2(cpu_key_bufs, # 每个GPU的key_buffers列表
                                                offload_events, # GPU→CPU同步事件
                                                process_cnt = process_cnt, # 进程数量
                                                core_per_process = MAX_CPU_IN_USE // process_cnt, # 每个进程使用的CPU核心数
                                                max_km_groups=config.num_key_value_heads * subvec, # 每个head的子向量数量
                                                max_seq_len=config.max_seq_len, # 最大序列长度
                                                dim=(config.hidden_size // config.num_attention_heads) // subvec, # 每个子向量的维度
                                                max_cent_cnt= 2 ** int(os.environ.get("SUBBITS", "6")), # 聚类中心数量
                                                max_task_cnt=32, # 最大任务数量
                                                metric=os.environ.get("METRIC","euc"), 
                                                layer_cnt = config.num_hidden_layers, # 总层数
                                                model_name=model) # 模型名称

    logger.info("Multi-core compressor init done.")


def wait():
    global global_compressor
    global_compressor.wait_for_km_result()


def del_objects():
    global global_compressor
    global cache_managers
    del global_compressor
    for m in cache_managers:
        del m
    

class PqBasedSearchCompressor(RetrievalBasedCompressor):
    all_pq_compressors = []
    
    def __init__(self, compress_ratio, recent_ratio, n_subvec_per_head, n_subbits, gqa, sink_size = 32, **kwargs):
        self.compress_ratio = compress_ratio 
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.topk_ratio = 1 - self.recent_ratio
        if n_subvec_per_head not in [1,2,4,8,16]:
            raise Exception("PQ subvec must in 1 2 4 8 16")
        self.n_subvec_per_head = n_subvec_per_head
        self.n_subbits = n_subbits
        self.recent_size = 0
        self.prefill_length = 0
        self.topk_size = 0
        self.layer_idx = kwargs["layer_idx"]
        self.rank = self.layer_idx // layer_per_rank
        self.code_book = None
        self.centroids = None
        self.future = None
        self.km_done = False
        self.ip2l2_phi = None
        self.GQA = gqa

        self.all_layer_cnt = kwargs["num_layer_cnt"]

        self.selected_idx_arr = []
        self.seq_cnt = 0

        n_kv_heads = kwargs["kv_head"]
        dim = kwargs["dim"]
        device = kwargs["cur_device"]
        self.max_iter = kwargs["max_iter"]
        
        if SYNC_TEST_TIME:
            self.prefetch_event = torch.cuda.Event(enable_timing=True)
            self.prefetch_event_start = torch.cuda.Event(enable_timing=True)
            self.prefetch_event_end = torch.cuda.Event(enable_timing=True)

            self.pq_start_event = torch.cuda.Event(enable_timing=True)
            self.pq_end_event = torch.cuda.Event(enable_timing=True)
            global_timer.append_compute_event(self.pq_start_event, self.pq_end_event)
            if self.layer_idx == 0:
                self.layer_0_start = torch.cuda.Event(enable_timing=True)
                self.layer_0_end = torch.cuda.Event(enable_timing=True)
        
        self.prefetch_event = torch.cuda.Event()
        
        self.gpu_key_for_recall_check = None
    
        # if self.layer_idx <= 1:
            # print(f"GQA is {self.GQA}")
        super().__init__(**kwargs)
        
        # Used for prefetch
        PqBasedSearchCompressor.all_pq_compressors.append(self)
    

    def build_index_cpu_multi_core_sklr(self, xb, cent_cnt) -> torch.Tensor:
        bsz, kv_heads, n_subvec_per_head, n_xb, subvec_d = xb.shape
        if n_xb > cent_cnt:
            self.valid_n_xb = n_xb
            xb = xb.reshape([bsz * kv_heads * n_subvec_per_head, n_xb, subvec_d]) 
            # if self.layer_idx == 0:
                # print("xb.shape = ", xb.shape) # 
                # print("--------------------------进入global_compressor.compress---------------------------")
                # print("max_iter = ", self.max_iter) # 10
                # print("cent_cnt = ", cent_cnt) # 64=2^6
            self.centroids, self.code_book, self.shm_set_idx, ip2l2_phi = global_compressor.compress(xb, cent_cnt=cent_cnt, 
                                                                                          max_iter=self.max_iter, 
                                                                                          layer_idx=self.layer_idx)
            # if self.layer_idx == 0:
            #     print("--------------------------------从compress输出后--------------------------------")
            #     print("self.centroids.shape = ", self.centroids.shape) #  torch.Size([16, 64, 64]) # [子向量空间数, 聚类中心数, 中心向量维度]
            #     print("self.code_book.shape = ", self.code_book.shape) #  torch.Size([50000, 16]) 
            #     print("self.shm_set_idx = ", self.shm_set_idx) #  0
            #     print("ip2l2_phi = ", ip2l2_phi) # none
                
            # code_book is a big buffer that reserve places for generated token in the future.
            self.code_book = self.code_book.reshape([bsz, -1, kv_heads, n_subvec_per_head])
            self.centroids = self.centroids.reshape([bsz, kv_heads, n_subvec_per_head, cent_cnt, -1])
            # if self.layer_idx == 0:
            #     print("reshaped self.centroids.shape = ", self.centroids.shape) #  torch.Size([1, 8, 2, 64, 64])
            #     print("reshaped self.code_book.shape = ", self.code_book.shape) #  torch.Size([1, 50000, 8, 2])
                
            return ip2l2_phi
        
        return None


    def _ip2l2_preprocess(self, xb: torch.Tensor, phi):
        assert xb.device != torch.device("cpu")
        assert phi.shape == (xb.shape[0], 1, 1)
        norms = (xb ** 2).sum(dim=2, keepdim=True) # n_groups, n_xb, 1
        extracol = torch.sqrt(phi - norms)
        return torch.concat((xb, extracol), dim=2)


    # 异步预取下一层需要的PQ编码数据。
    def prefetch_codebook(self):
        with torch.cuda.stream(H2DStream): # 使用专用cuda流
            if SYNC_TEST_TIME and global_timer.can_record():
                self.prefetch_event_start.record()
            # 异步数据传输：将CPU上的centroids和code_book张量移动到GPU上，并转换为float16类型和int64类型
            self.gpu_centroids = self.centroids.to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = self.code_book.to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64)
            self.prefetch_event.record() # 传输完成后记录一个cuda事件

            if SYNC_TEST_TIME and global_timer.can_record(): 
                self.prefetch_event_end.record() # 记录预取事件结束时间
                global_timer.append_transfer_time_tuples(self.prefetch_event_start, self.prefetch_event_end)



    def predict_index_cpu(self, vec: np.ndarray):
        assert vec.shape[-2] == 1
        bsz, n_kv_heads, n_subvec_per_head, q_len, subvec_d = vec.shape
        if global_compressor.metric == "ip":
            vec = self._ip2l2_preprocess(vec.reshape([bsz * n_kv_heads * n_subvec_per_head, q_len, subvec_d]), self.ip2l2_phi)
            subvec_d = vec.shape[-1]
        assert subvec_d == self.centroids.shape[-1]
        cent_cnt = self.centroids.shape[-2]
        vec = vec.reshape((-1, subvec_d))[:,None,:] # n_subspace, 1, subvec_d
        centroids = self.centroids.reshape((-1, cent_cnt, subvec_d)) # n_subspace, cent_cnt, subvec_d
        distances = torch.tensor(np.sum((centroids - vec) ** 2, axis=-1)) # n_subspace, cent_cnt
        return distances.min(dim=-1).indices.reshape([bsz, n_kv_heads, n_subvec_per_head, 1]).numpy()
    

    # 将token的key部分转换为PQ编码
    def predict_index_gpu(self, vec: torch.Tensor):
        assert vec.shape[-2] == 1
        bsz, n_kv_heads, n_subvec_per_head, q_len, subvec_d = vec.shape
        if global_compressor.metric == "ip":
            vec = self._ip2l2_preprocess(vec.reshape([bsz * n_kv_heads * n_subvec_per_head, q_len, subvec_d]), self.ip2l2_phi)
            subvec_d = vec.shape[-1]
        assert subvec_d == self.centroids.shape[-1]
        cent_cnt = self.centroids.shape[-2]
        vec = vec.reshape((-1, subvec_d))[:,None,:] # n_subspace, 1, subvec_d
        centroids = self.gpu_centroids.reshape((-1, cent_cnt, subvec_d)) # n_subspace, cent_cnt, subvec_d
        distances = torch.tensor(torch.sum((centroids - vec) ** 2, axis=-1)) # n_subspace, cent_cnt
        return distances.min(dim=-1).indices.reshape([bsz, 1, n_kv_heads, n_subvec_per_head])





    def prefill_attn(
        self,
        query,
        past_key_value: torch.Tensor,
        use_gpu = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        
        self.gpu_centroids = None
        self.centroids = None
        self.code_book = None
        self.gpu_code_book = None
        self.km_done = False
        self.ip2l2_phi = None
        self.past_token_cnt = 0
        self.seq_cnt += 1
        if self.layer_idx == 0:
            global_compressor.refresh_pool()

        key_states, value_states = past_key_value
        # if self.layer_idx == 0:
            # print("key_states.shape = ", key_states.shape) # torch.Size([1, 8, 2853, 128])
        bsz, kv_heads, kv_seq_len, dim = key_states.shape
        # batch_size = 1, kv_heads = 8, kv_seq_len = 2853, dim = 128

        assert bsz == 1, "Do not support bsz > 1 in adaptive compression mode yet."
        self.recent_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * self.recent_ratio)
        # if self.layer_idx == 0:
        #     print("kv_seq_len-self.sink_size = ", kv_seq_len-self.sink_size) # 2853-32=2821
        #     print("self.compress_ratio = ", self.compress_ratio) # 0.05
        #     print("(kv_seq_len-self.sink_size) * self.compress_ratio = ", (kv_seq_len-self.sink_size) * self.compress_ratio) # 2821*0.05=141.05
        #     print("self.recent_ratio = ", self.recent_ratio) # 0.3
        #     print("self.recent_size = ", self.recent_size) # 42=141.05*0.3
            
        self.prefill_length = kv_seq_len
        self.topk_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * (1 - self.recent_ratio))
        # if self.layer_idx == 0:
        #     print("self.topk_size = ", self.topk_size) # 141.05*(1-0.3)=98.735
        
        # There is no need to compress sink token
        xb = key_states[:,:, self.sink_size:, :]
        # if self.layer_idx == 0:
        #     print("xb.shape = ", xb.shape) # 
        n_xb = kv_seq_len - self.sink_size
        # if self.layer_idx == 0:
        #     print("n_xb = ", n_xb) # 

        subvec_d = dim // self.n_subvec_per_head #  self.n_subvec_per_head表示每个head的子向量数量
        centroid_cnt = 2 ** self.n_subbits # self.n_subbits表示每个子向量包含的比特数,centroid_cnt=2^bits表示子向量的数量
        if self.layer_idx == 0:
            print("subvec_d = ", subvec_d) # 128/2=64
            print("centroid_cnt = ", centroid_cnt) # 64=2^6
        
        xb = xb.reshape(bsz, kv_heads, n_xb, self.n_subvec_per_head, subvec_d).transpose(2,3)
        # if self.layer_idx == 0:
        #     print("xb.shape = ", xb.shape) # 
        #     print("--------------------------------进入cache_managers[self.rank].init--------------------------------")
        
        # self.rank 表示当前layer idx下分配的GPU设备编号，用于多GPU并行处理：self.rank = self.layer_idx // layer_per_rank，一个gpu可能管理多个层，把对应的层分配到对应的gpu
        cache_managers[self.rank].init(key_states, value_states, self.layer_idx,self.topk_size)
        # Do compression, in async manner. self.ip2l2 will be set to None if clustering metric is euc.
        # if self.layer_idx == 0:
        #     print("xb.shape = ", xb.shape) #   
        #     print("centroid_cnt = ", centroid_cnt) # 64=2^6
        #     print("--------------------------------进入build_index_cpu--------------------------------")
        
        self.ip2l2_phi = self.build_index_cpu_multi_core_sklr(xb, centroid_cnt)
        
        # if self.layer_idx == 0:
        #     print("self.ip2l2_phi = ", self.ip2l2_phi) # 
        
        # if self.layer_idx == 0:
        #     print("--------------------------------进入flash_attn_func--------------------------------")
        #     print("query.shape = ", query.shape) # 
        #     print("key_states.shape = ", key_states.shape) # 
        #     print("value_states.shape = ", value_states.shape) # 
            
        attn_output = flash_attn_func(query.transpose(1,2), key_states.transpose(1,2), value_states.transpose(1,2), causal = True).transpose(1,2)
        # if self.layer_idx == 0:
        #     print("attn_output.shape = ", attn_output.shape) # 
            
        self.kv_cache_cnt = np.zeros([bsz*kv_heads], dtype=np.int64)
        self.past_token_cnt = key_states.shape[-2]
        # self.gpu_key_for_recall_check = key_states
        return attn_output, self.kv_cache_cnt



    def decoding_attn_GQA_euc(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v   
    ):  
        # 如果PQ压缩未完成，则使用原始的softmax计算attention
        if self.code_book is None: # skip this situation
            attn_output = torch.matmul(torch.softmax(query @ repeat_k.transpose(2,3) / math.sqrt(query.shape[-1]), dim=-1), repeat_v)
            return attn_output

        if SYNC_TEST_TIME and global_timer.can_record():
            self.pq_start_event.record()

        bsz, n_heads, n_kv_seqlen, dim = repeat_k.shape
        _, kv_head, n_subvec_per_head, cent_cnt, subvec_d = self.centroids.shape
        assert query.shape[2] == 1, "Do not support multi query pq_search yet."

        # past_token_cnt: 已处理的token总数， recent_size: 保留的recent token数， sink_size: 开头的token数
        # n_topk_candidate: 可用于top-k检索的token数
        recent_index = self.past_token_cnt - self.recent_size  
        n_topk_candidate = recent_index - self.sink_size
        
        # 这行代码的作用是将重复扩展的key和value张量还原为原始的未重复状态。
        # if self.layer_idx == 0: 
        #     print("=" * 60)
        #     print("张量形状调试信息:")
        #     print(f"query.shape = {query.shape}")
        #     print(f"repeat_k.shape = {repeat_k.shape}")
        #     print(f"repeat_v.shape = {repeat_v.shape}")
        #     print(f"num_key_value_groups = {num_key_value_groups}")
            
        k, v = unrepeat(repeat_k, num_key_value_groups, 1), unrepeat(repeat_v, num_key_value_groups, 1)
        
        # if self.layer_idx == 0: 
        #     print(f"k.shape (after unrepeat) = {k.shape}")
        #     print(f"v.shape (after unrepeat) = {v.shape}")
        #     print(f"形状变化: repeat_k {repeat_k.shape} -> k {k.shape}")
        #     print(f"形状变化: repeat_v {repeat_v.shape} -> v {v.shape}")
        #     print("=" * 60) 
        
        # 步骤1:等待PQ聚类完成，获取质心和码本， # 预取下一层的PQ数据（流水线优化）

        if not self.km_done: # K-means完成标志
            global_compressor.wait_for_km_result(self.shm_set_idx) # 阻塞等待指定索引的K-means聚类完成
            self.km_done = True
        
        if self.layer_idx == 0:
            if SYNC_TEST_TIME and global_timer.can_record():
                self.layer_0_start.record()
                
            # 将CPU上的centroids和code_book张量移动到GPU上，并转换为float16类型和int64类型
            self.gpu_centroids = torch.Tensor(self.centroids).to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = torch.Tensor(self.code_book).to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64) # 对编码本进行维度重排：permute([0,2,3,1])

            if SYNC_TEST_TIME and global_timer.can_record():
                self.layer_0_end.record()
                global_timer.append_transfer_time_tuples(self.layer_0_start, self.layer_0_end)
       
        else: # 如果当前层不是第0层，则等待prefetch_event完成
            self.prefetch_event.wait()
        
        
        if self.layer_idx < (self.all_layer_cnt - 1):
            PqBasedSearchCompressor.all_pq_compressors[self.layer_idx+1].prefetch_codebook()

        query_trans = query.reshape([bsz, n_heads, 1, n_subvec_per_head, subvec_d]).transpose(2,3) # query: [bsz, n_heads, n_subvec_per_head, q_len, subvec_d]
        
        # 扩展和重排质心数据，扩展后形状：[bsz, n_heads, n_subvec_per_head, cent_cnt, subvec_d]，原先是kv_heads
        # if self.layer_idx == 0:
        #     print(f"code_book.size = {self.code_book.shape}")
        #     print(f"code_book[0,0,0,0] = {self.code_book[0,0,0,0]}")
        #     # 打印centroids的size和第1个子空间的第一个类中心
        #     print(f"centroids.size = {self.centroids.shape}") #torch.Size([1, 8, 2, 64, 64])
        #     print(f"centroids[0,0,:,:] = {self.centroids[0,0,:,:]}")
            
            
        repeat_centroids = repeat(self.gpu_centroids, size=num_key_value_groups, dim_idx=1).transpose(3,4)
        repeat_code_book = repeat(self.gpu_code_book, size=num_key_value_groups, dim_idx=1)
      
        # Sink token don't have their pq indices, and tokens within local window can be ignored.
        # 只获取top-k位置的码本
        repeat_code_book = repeat_code_book[...,:n_topk_candidate] 
        #query的每个子向量与对应质心集合中所有质心 的内积，是一个完整的相似度查找表
        
        # 打印第1个head的前10个token的码本
        # if self.layer_idx == 0:
        #     print("=" * 60)
        #     print("第1个head的前10个token的PQ码本:")
        #     print("repeat_code_book[0, 0, :, :10] =")
        #     print(repeat_code_book[0, 0, :, :10])
        #     print("每行表示一个子向量位置，每列表示一个token")
        #     print("数值范围: 0-63 (对应64个聚类中心)")
        #     print("=" * 60)
        
        qk_table = torch.matmul(query_trans, repeat_centroids) # [bsz, n_heads, n_subvec_per_head, q_len, cent_cnt]      
        # 打印head 0下的qk_table
        # if self.layer_idx == 0:
        #     print("=" * 60)
        #     print("head 0 下的 qk_table:")
        #     # 只打印前2个batch、前2个head、前2个子向量、queryd的第1个token到所有centroid的 qk内积距离
        #     print("qk_table[0, 0, :, 0, :] =")
        #     print(qk_table[0, 0, :, 0, :])
        #     print("qk_table.shape =", qk_table.shape)
        #     print("=" * 60)
       

        
        
        # 步骤2: PQ近似相似度计算 (第406-447行) 获取qk_table: [bsz, n_heads, n_subvec_per_head, q_len, cent_cnt]
        dummy_weight = torch.gather(qk_table[:,:,:,0,:], -1, repeat_code_book[:,:,:,:]).sum(dim=-2) # [1, 32, 2780]
        
        # # 打印第1个head的前10个token的距离
        # if self.layer_idx == 0:
        #     print("=" * 60)
        #     print("第1个head的前10个token的距离（dummy_weight）:")
        #     # 假设dummy_weight形状为 [bsz, n_heads, q_len]，打印第一个batch，第1个head，前10个token的距离
        #     print("dummy_weight[0, 0, :10] =")
        #     print(dummy_weight[0, 0, :10])
        #     print("=" * 60)
        
        
        # if self.layer_idx == 0:
        #     print(f"dummy_weight.size = {dummy_weight.shape}")
        #     print("repeat_code_book.size = ", repeat_code_book.shape) #torch.Size([1, 32, 2, 2780])
            
            
        dummy_softmax_scale = math.sqrt(dim)
        dummy_score = torch.softmax(dummy_weight / dummy_softmax_scale, dim=-1)
        # 处理GQA的头数聚合
        # if self.layer_idx == 0:
        #     print(f"dummy_score reshape 前的 shape: {dummy_score.shape}")
        dummy_score = dummy_score.reshape([bsz, kv_head, num_key_value_groups, 1, n_topk_candidate])
        # if self.layer_idx == 0:
        #     print(f"dummy_score reshape 后的 shape: {dummy_score.shape}")
        dummy_score = torch.sum(dummy_score, dim=2) # reduce
        # if self.layer_idx == 0:
        #     print(f"dummy_score 求和后的 shape: {dummy_score.shape}")
            
        topk_indices = dummy_score.topk(self.topk_size, dim=-1, largest = True, sorted=False).indices # [bsz, kv_head, q_len, topk]
        # if self.layer_idx == 0:
        #     print(f"topk_indices.shape = {topk_indices.shape}")

        if CHECK_RECALL:
            k_, v_ = cache_managers[self.rank].fetch_all_key_value(self.layer_idx, self.past_token_cnt)
            recall, recall_mean, recall_var = calc_recall(query, k_.transpose(1,2), topk_indices, num_key_value_groups, self.topk_size)
            if self.layer_idx == 0:
                logger.info(f"{recall},{recall_mean},{recall_var}")
            # print(f"layer {self.layer_idx} recall:{recall}, recall_mean:{recall_mean}, recall_var:{recall_var}")
        
        #fetch_and_concat_kv_w_cache 函数会构建完整的KV缓存
        #final_k_gpu = [Sink tokens] + [TopK tokens] + [Recent tokens] + [空位给Current token]
        final_k_gpu, final_v_gpu = cache_managers[self.rank].fetch_and_concat_kv_w_cache(topk_indices.squeeze(2).squeeze(0), self.layer_idx)
        # if self.layer_idx == 0:
        #     print("=" * 60)
        #     print("final_k_gpu.shape = ", final_k_gpu.shape) # 
        #     print("final_v_gpu.shape = ", final_v_gpu.shape) # 
        #     print("=" * 60)
        
        assert final_k_gpu.shape[-2] == self.sink_size + self.recent_size + self.topk_size + 1, f"{final_k_gpu.shape[-2]},{self.sink_size + self.recent_size + self.topk_size + 1}"
        
        # 将当前token的KV数据添加到最终的attention缓存中
        final_k_gpu[:,:,-1:,:].copy_(k, non_blocking=True)
        final_v_gpu[:,:,-1:,:].copy_(v, non_blocking=True)
        
        # if self.layer_idx == 0: 
        #     print("执行flash_attn_func前的final_k_gpu和final_v_gpu的shape:")
        #     print("=" * 60)
        #     print("final_k_gpu.shape = ", final_k_gpu.shape) # 
        #     print("final_v_gpu.shape = ", final_v_gpu.shape) # 
        #     print("=" * 60)
        
        attn_output = flash_attn_func(
            query.transpose(1,2),
            repeat(final_k_gpu, num_key_value_groups, 1).transpose(1,2),
            repeat(final_v_gpu, num_key_value_groups, 1).transpose(1,2),
            causal=True
        ).transpose(1,2)
        
        # 添加: 新token自动替换最久远的token， 从recent window中被替换出来的最久远token的key部分。
        to_evict_key = cache_managers[self.rank].add_new_token(k, v, self.layer_idx)  
        
        # if self.layer_idx == 0:
        #     print("=" * 60)
        #     print("to_evict_key.shape = ", to_evict_key.shape) # 
        #     print("=" * 60)
        # 被local window丢弃的token进入历史token池， 需要预测其PQ编码才能参与下一步decode的top-k检索      
        #  valid_n_xb： 已有PQ编码的token数量
        # n_topk_candidate： 可用于top-k检索的token数
        if n_topk_candidate == self.valid_n_xb:
            # if self.layer_idx <= 0:
            #     print("Predicting generated token")
            # 计算当前token的索引
            to_pass_index = self.sink_size + self.valid_n_xb
            # 确保当前token的索引是正确的
            assert (to_pass_index + self.recent_size) == self.past_token_cnt, f"{to_pass_index}, {self.recent_size}, {n_kv_seqlen}, {self.past_token_cnt}"
            # 获取被local window丢弃的token的key部分 
            to_predict_k = to_evict_key
            # 预测当前token的PQ编码
            indices = self.predict_index_gpu(to_predict_k.reshape([bsz, kv_head, 1, n_subvec_per_head, subvec_d]).transpose(2,3))
            # 更新code_book， 将预测的PQ编码添加到code_book中
            self.code_book[:, n_topk_candidate:n_topk_candidate+1,:,:].copy_(indices, non_blocking=True) # NOTE: Let's neglect its overhead for now.
            # 更新valid_n_xb， 表示当前token已离开local window， 可以参与下一步decode的top-k检索
            self.valid_n_xb += 1
            # print("=" * 60)
            # print("layer is ", self.layer_idx, "self.valid_n_xb = ", self.valid_n_xb) # 
            # print("=" * 60)
        # 更新past_token_cnt， 表示当前token已处理
        if SYNC_TEST_TIME and global_timer.can_record():
            self.pq_end_event.record()

        self.past_token_cnt += 1
        return attn_output






    def decoding_attn_GQA_ip(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v   
    ):
        if self.code_book is None: # skip this situation
            attn_output = torch.matmul(torch.softmax(query @ repeat_k.transpose(2,3) / math.sqrt(query.shape[-1]), dim=-1), repeat_v)
            return attn_output
        if np.random.randint(0,10000) % 5000 == 0:
            logger.info("Using ip2l2 metric to decoding!")
    
        bsz, n_heads, n_kv_seqlen, dim = repeat_k.shape
        _, kv_head, n_subvec_per_head, cent_cnt, subvec_d = self.centroids.shape
        assert query.shape[2] == 1, "Do not support multi query pq_search yet."

        recent_index = self.past_token_cnt - self.recent_size
        n_topk_candidate = recent_index - self.sink_size
        
        k, v = unrepeat(repeat_k, num_key_value_groups, 1), unrepeat(repeat_v, num_key_value_groups, 1)

        if not self.km_done:
            global_compressor.wait_for_km_result(self.shm_set_idx)
            self.km_done = True
        
        if self.layer_idx == 0:
            self.gpu_centroids = torch.Tensor(self.centroids).to(self.device, non_blocking=True).to(torch.float16)
            self.gpu_code_book = torch.Tensor(self.code_book).to(self.device, non_blocking=True).permute([0,2,3,1]).to(torch.int64)
        else:
            self.prefetch_event.wait()
        
        # NOTE: prefetch.
        if self.layer_idx < (self.all_layer_cnt - 1):
            PqBasedSearchCompressor.all_pq_compressors[self.layer_idx+1].prefetch_codebook()

        query_trans = query.reshape([bsz, n_heads, 1, n_subvec_per_head, dim // n_subvec_per_head]) \
                                    .transpose(2,3) # query: [bsz, n_heads, n_subvec_per_head, q_len, subvec_d]
        aug_query_trans = self.augment_xq(query_trans.reshape([-1, dim // n_subvec_per_head])).reshape([bsz, n_heads,n_subvec_per_head,  1, subvec_d])
        
        repeat_centroids = repeat(self.gpu_centroids, size=num_key_value_groups, dim_idx=1)
        repeat_code_book = repeat(self.gpu_code_book, size=num_key_value_groups, dim_idx=1)

        # Sink token don't have their pq indices, and tokens within local window can be igonored.
        repeat_code_book = repeat_code_book[...,:n_topk_candidate] 

        # NOTE: Main method
        qk_table = torch.sum((aug_query_trans - repeat_centroids) ** 2, dim = -1, keepdim=True) # [bsz, n_heads, n_subvec_per_head, cent_cnt, 1]
        dummy_distance = query.new_zeros([bsz, n_heads, 1, n_topk_candidate])
        
        # TODO: optimize here
        for i in range(0, n_subvec_per_head):
            distance_piece = torch.gather(qk_table[:,:,i,:,0], -1, repeat_code_book[:,:,i,:])
            dummy_distance[:,:,0,:] += distance_piece
        
        dummy_score = torch.sum(dummy_distance.reshape([bsz, kv_head, num_key_value_groups, 1, n_topk_candidate]), dim = 2) # reduce
        topk_indices = dummy_score.topk(self.topk_size, dim=-1, largest = False).indices # [bsz, kv_head, q_len, n_xb]
        # END NOTE

        ratio, x, y = calc_recall(query, self.gpu_key_for_recall_check[...,self.sink_size:recent_index,:], topk_indices, num_key_value_groups, topk_indices.shape[-1])
        if np.random.randint(0,10000) % 5000 == 0:
            print(f"Recall in cur attn:{ratio}, mean:{x}, var:{y}, PQ_search_GQA, Ignore local and sink? yes!")

        final_k_gpu, final_v_gpu = cache_managers[self.rank].fetch_and_concat_kv_w_cache(topk_indices.squeeze(2).squeeze(0), self.layer_idx)

        assert final_k_gpu.shape[-2] == self.sink_size + self.recent_size + self.topk_size + 1, f"{final_k_gpu.shape[-2]},{self.sink_size + self.recent_size + self.topk_size + 1}"
        final_k_gpu[:,:,-1:,:].copy_(k, non_blocking=True)
        final_v_gpu[:,:,-1:,:].copy_(v, non_blocking=True)

        attn_output = flash_attn_func(
            query.transpose(1,2),
            repeat(final_k_gpu, num_key_value_groups, 1).transpose(1,2),
            repeat(final_v_gpu, num_key_value_groups, 1).transpose(1,2),
            causal=True
        ).transpose(1,2)

        to_evict_key = cache_managers[self.rank].add_new_token(k, v, self.layer_idx)
        if self.gpu_key_for_recall_check is not None:
            self.gpu_key_for_recall_check = torch.concat([self.gpu_key_for_recall_check, k], dim = -2)
        # If one token gonna pass local window in next decoding step while do not have its pq indices, 
        # we need to predict its pq indices.
        if n_topk_candidate == self.code_book.shape[-1]:
            if self.layer_idx <= 0:
                print("Predicting generated token")
            to_pass_index = self.sink_size + self.code_book.shape[-1]
            assert (to_pass_index + self.recent_size) == self.past_token_cnt, f"{to_pass_index}, {self.recent_size}, {n_kv_seqlen}"
            to_predict_k = to_evict_key
            indices = self.predict_index_gpu(to_predict_k.reshape([bsz, kv_head, 1, n_subvec_per_head, subvec_d]).transpose(2,3))
            self.code_book[0, n_topk_candidate:n_topk_candidate+1,:,:].copy_(indices, non_blocking=True) # NOTE: Let's neglect its overhead for now.
            
        self.past_token_cnt += 1

        return attn_output


    def augment_xq(self, xq): 
        extracol = torch.zeros(len(xq), dtype=xq.dtype, device = xq.device)
        return torch.hstack((xq, extracol.reshape(-1, 1)))



    def decoding_attn(
        self,
        num_key_value_groups: int,
        query, # bsz, n_heads, q_len, dim
        repeat_k, repeat_v
    ):
        if self.GQA:
            if global_compressor.metric == "euc":
                # print("decoding_attn: euc metric to decoding!")
                target_func = self.decoding_attn_GQA_euc
                
            elif global_compressor.metric == "ip":
                # print("decoding_attn: ip metric to decoding!")
                target_func = self.decoding_attn_GQA_ip
        else:
            raise Exception("wo GQA not supported currently")

        return target_func(num_key_value_groups, query, repeat_k, repeat_v)