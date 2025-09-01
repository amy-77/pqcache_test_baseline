import concurrent.futures as future
import os
import torch
import numpy as np
import time
import math
from typing import List

import torch.multiprocessing as mp
from torch.multiprocessing import Condition, Event, Value, Queue
import json
from loguru import logger
from tqdm import tqdm

def _ip2l2_preprocess(xb: torch.Tensor):
    norms = (xb ** 2).sum(dim=-1, keepdim=True) # n_groups, n_xb, 1
    phi = norms.max(dim = -2, keepdim=True).values
    extracol = torch.sqrt(phi - norms)
    return torch.concat((xb, extracol), dim=-1), phi


def pin_shm(bufs: List[torch.Tensor]):
    # Copy from https://gist.github.com/colesbury/aae4361feb596353fcd219339d6c44ab
    cudart = torch.cuda.cudart()
    for buf in bufs:
        err = cudart.cudaHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), 0)
        assert buf.is_shared()
        assert buf.is_pinned()
        assert err == 0, err

def unpin_shm(bufs: List[torch.Tensor]):
    # Copy from https://gist.github.com/colesbury/aae4361feb596353fcd219339d6c44ab
    lib = torch.cuda.cudart()
    for buf in bufs:
        err = lib.cudaHostUnregister(buf.data_ptr())
        assert err == 0, err

class SharedMemSet:
    def __init__(self, cpu_key_buf, max_km_groups, max_cent_cnt, max_seq_len, dim, **kwargs) -> None:
        self.xb_shared_tensor = cpu_key_buf

        self.cb_shared_tensor = torch.zeros(    
                                        [max_km_groups, max_cent_cnt, dim],
                                        dtype = torch.float16, 
                                        ).share_memory_()
        pin_shm([self.cb_shared_tensor])
        self.id_shared_tensor = torch.zeros(    
                                        [max_seq_len, max_km_groups],
                                        dtype = torch.int64,     
                                        ).share_memory_()
        pin_shm([self.id_shared_tensor])
        
        self.in_use = False # Not shared by worker process

    def free(self):
        self.km_task_event.clear()
        self.offload_done_cpu_event.clear()
        self.begin_offload_cpu_event.clear()
        self.in_use = False
    
    def init_sync_tools(self):
        self.offload_done_cpu_event = Event() # New event
        self.begin_offload_cpu_event = Event()
        
        self.km_task_event = Event()
        self.km_task_event.set()
        self.done_flag = Value("i", 0)
        

# No synchorization operation is needed.
class MultiProcSharedTensorPool:
    def __init__(self, cpu_key_bufs,**kwargs) -> None:
        self.shared_mem_block_sets = [
            SharedMemSet(cpu_key_bufs[i], **kwargs) 
            for i in range(kwargs["max_size"])
        ]
    
    def init_sync_tools(self):
        for shm_block_set in self.shared_mem_block_sets:
            shm_block_set.init_sync_tools()

    def get(self):
        while True:
            for i, shm_set in enumerate(self.shared_mem_block_sets):
                if not shm_set.in_use:
                    shm_set.in_use = True       
                    return i


def compute_kmeans_worker(
        idx, n_proc, n_core, core_offset,
        shmpool: MultiProcSharedTensorPool,
        task_queue: Queue,
        layer_cnt: int,
        metric: str,
        task_cnt: int,
        seed: int,
        time_value: Value,
        profiling_sync: Value
):  
    cpu_num = mp.cpu_count()
    cur_pid = os.getpid()
    if n_core >= 1:
        cpu_use = int(n_core)
        if idx == -1:
            os.sched_setaffinity(cur_pid, list(range(cpu_num))[-2:-1])
        else:
            os.sched_setaffinity(cur_pid, list(range(cpu_num))[core_offset + cpu_use * (idx+1) : core_offset + cpu_use * (idx+2)])
    else:
        assert int(n_core * task_cnt) == 1, f"{n_core},{task_cnt}"
        cpu_use = 1
        core_idx = math.floor(n_core*idx)
        os.sched_setaffinity(cur_pid, list(range(cpu_num))[core_idx:core_idx + 1])

    # NOTE: It is necessary to not patch sklearn before setting CPU affinity.
    # There are some environments in which doing patch earlier will fail CPU affinity setting.
    # TEMPORARILY DISABLED: Intel optimization causes compatibility issues with numpy 1.26.4
    # try:
    #     import sklearnex
    #     sklearnex.patch_sklearn(verbose = False) # for Intel Xeon CPU to acclerate
    # except:
    #     pass
    from sklearn.cluster import KMeans

    init_cent_idx = None
    last_n_xb, last_cent_cnt = 0, 0

    counter = 0

    while True:
        shm_set_idx, shape, cent_cnt, max_iter, is_profiling = task_queue.get()
        np.random.seed(seed)
        if shape is None and cent_cnt == 0:
            # Worker exit gracefully
            exit()
        n_groups, n_xb, dim = shape
        
        # DEBUG: Worker接收到任务
        # print(f"----------------进入 compute_kmeans_worker-----------------")
        # print(f"[DEBUG] Worker {idx} received task:")
        # print(f"  shm_set_idx: {shm_set_idx}")
        # print(f"  shape: {shape}")
        # print(f"  cent_cnt: {cent_cnt}")
        # print(f"  max_iter: {max_iter}")
        # print(f"  is_profiling: {is_profiling}")

        if last_n_xb != shape[1] or last_cent_cnt != cent_cnt:
            init_cent_idx = np.random.choice(np.arange(n_xb), size = cent_cnt, replace = False)
            last_n_xb = n_xb
            last_cent_cnt = cent_cnt
        
        a = time.perf_counter()
        for shm_set_idx in range(layer_cnt):
            shm_set = shmpool.shared_mem_block_sets[shm_set_idx]
            if not is_profiling:
                # 等待shm_set_idx层的GPU-CPU 传输完成
                shm_set.offload_done_cpu_event.wait() # Wait for the completion of KV offloading.
            cpu_key_buf = shm_set.xb_shared_tensor # [1, total_max_len, n_kv_head, dim]
            _, total_max_len, n_kv_head, hidden_size = cpu_key_buf.shape
            
            # We may not need to handle multiple kmeans task by one worker process
            # to get rid of CPU oversubscription
            for task_idx in range(task_cnt):
                # 所有层都需要做数据重塑
                xb_reshaped1 = shm_set.xb_shared_tensor.reshape([1, total_max_len, n_kv_head, hidden_size // dim, dim])
                xb_reshaped2 = xb_reshaped1.reshape([1, total_max_len, n_groups, dim])
                sub_vector_group_idx = idx*task_cnt + task_idx
                xb_array = xb_reshaped2[0, :n_xb, sub_vector_group_idx]
                
                # Shape processing for kmeans clustering
                
                if metric == "ip":
                    xb_array, _ = _ip2l2_preprocess(xb_array)
                    xb_array = xb_array.numpy()
                    assert xb_array.shape[-1] == (dim + 1)
                else:
                    xb_array = xb_array.numpy()
                
                cb_shm_tensor = shm_set.cb_shared_tensor
                indices_shm_tensor = shm_set.id_shared_tensor
                
                km = KMeans(
                        n_clusters = cent_cnt,
                        n_init=1,
                        init=xb_array[init_cent_idx], 
                        tol = 0.0001,
                        # copy_x=True,
                        verbose=False,
                        max_iter=max_iter,
                        random_state=0,
                        algorithm="lloyd"
                )   
                # 只在第一层打印K-means详细信息
                # if shm_set_idx == 0:
                #     print(f"\n[KMEANS] 开始对子向量组 {sub_vector_group_idx} 进行K-means聚类...")
                #     print(f"[KMEANS] 输入数据: {xb_array.shape} ({xb_array.shape[0]} 个 {xb_array.shape[1]} 维向量)")
                #     print(f"[KMEANS] 聚类参数: n_clusters={cent_cnt}, max_iter={max_iter}")
                
                result = km.fit(xb_array)
                
                # 只在第一层打印聚类结果
                # if shm_set_idx == 0:
                    # print(f"\n[RESULT] K-means聚类完成!")
                    # print(f"[RESULT] 实际迭代次数: {result.n_iter_}")
                    # print(f"[RESULT] 聚类中心数量: {result.cluster_centers_.shape[0]}")
                    # print(f"[RESULT] 每个聚类中心维度: {result.cluster_centers_.shape[1]}")
                    # print(f"[RESULT] 样本标签数量: {len(result.labels_)}")
                    
                    # 只为第一个worker的第一个子向量组打印详细结果
                    # if idx == 0 and task_idx == 0:
                        # print(f"\n[DETAIL] 子向量组 {sub_vector_group_idx} 的详细聚类结果:")
                        # # print(f"[DETAIL] 聚类中心 (前5个):")
                        # for i in range(min(5, result.cluster_centers_.shape[0])):
                        #     center = result.cluster_centers_[i]
                            # print(f"  中心 {i}: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}, ..., {center[-1]:.4f}]")
                        
                        # print(f"\n[DETAIL] 样本标签 (前100个):")
                        # labels_to_show = result.labels_[:100]
                        # print(f"  标签: {labels_to_show}")
                        
                        # # 统计每个聚类的样本数量
                        # unique_labels, counts = np.unique(result.labels_, return_counts=True)
                        # print(f"\n[DETAIL] 聚类分布统计:")
                        # for label, count in zip(unique_labels[:10], counts[:10]):  # 只显示前10个
                        #     print(f"  聚类 {label}: {count} 个样本")
                        # if len(unique_labels) > 10:
                        #     print(f"  ... (还有 {len(unique_labels)-10} 个聚类)")
                    
                    # print(f"[RESULT] 聚类结果已保存到共享内存")
                
                # 保存结果到共享内存        
                assert cb_shm_tensor[idx*task_cnt + task_idx].shape == torch.from_numpy(result.cluster_centers_).shape, \
                        f"{cb_shm_tensor[idx*task_cnt + task_idx].shape},{torch.from_numpy(result.cluster_centers_).shape}"
                cb_shm_tensor[idx*task_cnt + task_idx].copy_(torch.from_numpy(result.cluster_centers_))
                indices_shm_tensor[:n_xb, idx*task_cnt + task_idx].copy_(torch.from_numpy(result.labels_))

            shm_set.done_flag.acquire()
            shm_set.done_flag.value += 1
            if shm_set.done_flag.value == n_proc:
                shm_set.km_task_event.set()
                shm_set.done_flag.value = 0
            shm_set.done_flag.release()
            counter += 1

        b = time.perf_counter()
        
        if is_profiling:
            time_value.acquire()
            time_value.value = max(time_value.value, (b-a)/(layer_cnt*task_cnt))
            time_value.release()

            profiling_sync.acquire()
            profiling_sync.value += 1
            profiling_sync.release()





def cuda_event_synchronizer(
    offload_events: List[torch.cuda.Event],
    shmpool: MultiProcSharedTensorPool,
    task_queue: Queue,
    layer_cnt: int,
):  
    while True:
        finish = task_queue.get()
        # print(f"----------------进入 cuda_event_synchronizer-----------------")
        # print(f"[DEBUG] cuda_event_synchronizer received message: {finish}")
        if finish == 0:
        #     print("[DEBUG] cuda_event_synchronizer exiting...")
            exit()
        
        for shm_set_idx in range(layer_cnt):
            shm_set = shmpool.shared_mem_block_sets[shm_set_idx]
            if finish == 1: # NOTE: Otherwise Profiling
                # print(f"[DEBUG] Waiting for GPU->CPU offload completion for layer {shm_set_idx}")
                shm_set.begin_offload_cpu_event.wait()
                offload_events[shm_set_idx].synchronize() # Wait for the completion of KV offloading.
                # print(f"[DEBUG] GPU->CPU offload completed for layer {shm_set_idx}")
            shm_set.offload_done_cpu_event.set()
            # print(f"[DEBUG] Signaled workers can start clustering for layer {shm_set_idx}")

# 4090
prefill_coef = {
    "llama": [4.75329e-011, 2.53902e-06, 4.22124e-04],
    "mistral": [4.86069848e-11, 2.52501805e-06, 5.10610107e-04]
}






class MultiCoreCompressor_v2:
    def __init__(
            self, 
            cpu_key_bufs,
            offload_events,
            process_cnt, 
            core_per_process, 
            max_km_groups,
            max_seq_len,
            dim,
            max_cent_cnt,
            max_task_cnt,
            metric = "euc",
            layer_cnt = 32,
            model_name = "mistral"
        ) -> None:
        self.metric = metric
        if self.metric == "ip":
            dim += 1
        
        self.km_dim = dim
        self.max_km_groups = max_km_groups
        self.max_seq_len = max_seq_len
        self.cent_cnt = max_cent_cnt

        self.proc_cnt = process_cnt
        self.n_core_per_proc = core_per_process
        mp.set_start_method("spawn", force=True)

        self.compress_cnt = 0

        self.shm_pool = MultiProcSharedTensorPool(
                                    cpu_key_bufs,
                                    max_km_groups = max_km_groups, 
                                    max_cent_cnt = max_cent_cnt,
                                    max_seq_len = max_seq_len,
                                    dim = dim, 
                                    max_size = layer_cnt
                                    )
        
        self.shm_pool.init_sync_tools()
        self.queues = [Queue(max_task_cnt) for _ in range(process_cnt)]
        self.master_queue = Queue(max_task_cnt)

        self.processes = []
        assert self.proc_cnt <= 64, f"{self.proc_cnt},{process_cnt}"
        self.cuda_sync_proc = mp.Process(
            target = cuda_event_synchronizer,            
            args=(offload_events, self.shm_pool, self.master_queue, layer_cnt)
        )
        self.cuda_sync_proc.start()

        self.time_value = Value("f", -1)
        self.profiling_sync = Value("i", 0)

        core_offset = eval(os.environ.get("CORE_OFFSET", "0"))

        for i in range(self.proc_cnt):
            p = mp.Process(
                    target = compute_kmeans_worker, 
                    args = (i, self.proc_cnt, self.n_core_per_proc, core_offset, 
                            self.shm_pool, self.queues[i], layer_cnt,
                            self.metric, self.max_km_groups//self.proc_cnt,
                            eval(os.environ.get("RANDOM_SEED","4321")), 
                            self.time_value, self.profiling_sync)
                )
            p.start()
            self.processes.append(p)

        self.kmeans_coef = None

        # Try to load existing clustering coef. 
        # If no coef conf have been archived, we will profile clustering time to generate a new item.
        if os.path.exists("./cluster_config.json"):
            with open("./cluster_config.json", "r") as f:
                cfg = json.load(f)
        else:
            cfg = dict()
        
        try:
            self.kmeans_coef = cfg[f"{dim}_{max_cent_cnt}_{core_per_process}"]
            print(f"KMeans coef is {self.kmeans_coef}. If you are running this system in a new hardware environment, please generate a new configuration file")
        except:
            self.regress_kmeans_time(self.cent_cnt)
            print(f"KMeans coef is {self.kmeans_coef}. New cluster coef has been archived.")
            cfg[f"{dim}_{max_cent_cnt}_{core_per_process}"] = {
                                                            "3_iter":self.iter_3_coef.tolist(), 
                                                            "per_iter":self.per_iter_coef.tolist()
                                                        }
        
            with open("./cluster_config.json", "w") as f:
                json.dump(cfg, f)
        
            self.kmeans_coef = {"3_iter":self.iter_3_coef, "per_iter":self.per_iter_coef}

        if "llama" in model_name or "Llama" in model_name:
            self.prefill_coef = prefill_coef["llama"]
        elif "mistral" in model_name or "Mistral" in model_name:
            self.prefill_coef = prefill_coef["mistral"]
        else:
            raise Exception("Unsupported model name.")
        print("We will predict prefill computation time assuming the model is", model_name)
       
        print(f"Compressor init done, config: \n \
                process_cnt = {process_cnt}, \n \
                core_per_process = {core_per_process}, \n \
                max_km_groups = {max_km_groups}, \n \
                max_seq_len = {max_seq_len}, \n \
                dim = {dim}, \n \
                max_cent_cnt = {max_cent_cnt}, \n \
                max_task_cnt = {max_task_cnt} \n \
                metric is {metric}")

    
    def __del__(self):
        for queue in self.queues:
            queue.put((0, None, 0, 0, None))
        self.master_queue.put(0)
    
    
    
    def regress_kmeans_time(self, cent_cnt):
        shared_ = self.shm_pool.shared_mem_block_sets[0].xb_shared_tensor 
        n_groups = shared_.shape[2] * shared_.shape[3] // self.km_dim
        print("Cannot find existing clustering config. We will do some profiling now. n_groups is", n_groups)
        result_dict = dict()
        seq_lens = []
        base_latency = []
        per_iter_latency = []

        seqlen_list = list(range(100, 2000, 200)) + list(range(2000, 20000, 2000))

        for seq_len in tqdm(seqlen_list):
            if cent_cnt > seq_len:
                continue
            seq_lens.append(seq_len)
            for max_iter in [3, 9]:
                self._compress_task(cent_cnt, torch.empty([n_groups, seq_len, self.km_dim]), cent_cnt, max_iter, True)
                while True:
                    time.sleep(2)
                    self.profiling_sync.acquire()
                    if self.profiling_sync.value == len(self.processes):
                        self.profiling_sync.value = 0
                        self.profiling_sync.release()
                        break
                    self.profiling_sync.release()

                self.time_value.acquire()
                latency = self.time_value.value
                assert latency > 0, latency
                self.time_value.value = 0
                self.time_value.release()
                
                result_dict.setdefault(seq_len, dict())
                result_dict[seq_len][max_iter] = latency
        
        for seq_len, time_ in result_dict.items():
            base_latency.append(time_[3]) 
            per_iter_latency.append((time_[9] - time_[3]) / 6)

        self.iter_3_coef = np.polyfit(seq_lens, base_latency, 1)
        self.per_iter_coef = np.polyfit(seq_lens, per_iter_latency, 1)
        
        
        
    def borrow_cpu_buffer(self, bufs: List[torch.Tensor], offload_events:List[torch.cuda.Event]):
        for i, shm_set in enumerate(self.shm_pool.shared_mem_block_sets):
            shm_set.xb_shared_tensor = bufs[i]
            shm_set.offload_event = offload_events[i]


    def compress(self, x: torch.Tensor, cent_cnt: int, max_iter: int, layer_idx: int):
        # print(f"----------------进入 MultiCoreCompressor_v2类的 compress()函数-----------------")
        assert len(x.shape) == 3
        assert x.dtype == torch.float16
        n_groups, n_xb, dim = x.shape
        # if layer_idx == 0:
        #     print("x.shape = ", x.shape) #  torch.Size([16, 64, 64])
 
        free_shm_set_idx = layer_idx
        if self.metric == "ip":
            _, ip2l2_phi = _ip2l2_preprocess(x)
            dim += 1
        else:
            ip2l2_phi = None
        
        # 分配聚类中心存储空间     cb_shared_tensor = 预分配的共享内存张量
        dst_codebook = self.shm_pool.shared_mem_block_sets[free_shm_set_idx].cb_shared_tensor[:n_groups, :cent_cnt, :]
        #dst_codebook.shape = [16, 64, 64] 含义：为16个子向量空间各分配64个聚类中心的存储位置
        dst_indices = self.shm_pool.shared_mem_block_sets[free_shm_set_idx].id_shared_tensor
        #shape = [50000, 16] 含义：为16个子向量空间各分配50000个聚类中心的存储位置
        
        if max_iter is None or max_iter == 0:
            gpu_compute_time = (self.prefill_coef[0]*n_xb**2 + self.prefill_coef[1]*n_xb + self.prefill_coef[2])
            # kmeans_base 前3轮迭代的基础时间
            kmeans_base = self.kmeans_coef["3_iter"][0] * n_xb + self.kmeans_coef["3_iter"][1]
            # kmeans_per_round 每轮额外迭代的时间
            kmeans_per_round = self.kmeans_coef["per_iter"][0] * n_xb + self.kmeans_coef["per_iter"][1]
            # 计算最大迭代轮数
            max_iter = int((gpu_compute_time - kmeans_base) / kmeans_per_round  +  3) 
            # 确保最大迭代轮数在3到300之间
            max_iter = max(max_iter, 3)
            max_iter = min(max_iter, 300)
            
            if layer_idx == 0:
                logger.info(f"multi core压缩器，max_iter:{max_iter}, base:{kmeans_base}, per_iter:{kmeans_per_round}, gpu_perlayer: {gpu_compute_time}") 
        else:
            if layer_idx == 0:
                logger.info(f"multi core压缩器，max_iter:{max_iter}") 
            pass

        if layer_idx == 0:
            self.compress_cnt += 1
            self._compress_task(cent_cnt, x, free_shm_set_idx, max_iter, False)

        assert not self.shm_pool.shared_mem_block_sets[free_shm_set_idx].begin_offload_cpu_event.is_set()
        self.shm_pool.shared_mem_block_sets[free_shm_set_idx].begin_offload_cpu_event.set()

        return dst_codebook, dst_indices, free_shm_set_idx, ip2l2_phi



    #将一个大的聚类任务拆分并分发给多个worker进程
    def _compress_task(
                self,
                cent_cnt,
                input_tensor,
                shm_set_idx,
                max_iter,
                is_profiling
            ):
        n_groups, n_xb, input_dim = input_tensor.shape
        # n_groups = 16    # 16个子向量空间
        # n_xb = 2821      # 2821个token
        # input_dim = 64   # 每个子向量64维
        
        # DEBUG: 打印调用信息
        # import traceback
        # print("=" * 60)
        # print(f"----------------进入 _compress_task-----------------")
        # print(f"[DEBUG] _compress_task called with:")
        # print(f"  cent_cnt: {cent_cnt}")
        # print(f"  shape: {input_tensor.shape}")
        # print(f"  max_iter: {max_iter}")
        # print(f"  is_profiling: {is_profiling}")
        # print(f"[DEBUG] Call stack:")
        # traceback.print_stack()
        # print("=" * 60)
        
        # 2. 通知主协调器 1：正常聚类任务   2：性能测试任务（用于拟合时间系数）
        message = 1 if not is_profiling else 2
        print(f"[DEBUG] Sending message {message} to master_queue")
        self.master_queue.put(message)
        # 3. 将任务信息（子向量空间数、token数、维度、聚类中心数、最大迭代轮数、是否为性能测试任务）发送到所有工作进程
        for queue in self.queues:
            queue.put((shm_set_idx, (n_groups, n_xb, input_dim), cent_cnt, max_iter, is_profiling))
    
    
    
    def wait_for_km_result(self, shm_set_idx=None):
        if shm_set_idx is None:
            shm_set_idx = len(self.shm_pool.shared_mem_block_sets) - 1
        done = self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.is_set()
        if not done:
            while not self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.wait(timeout=2):
                logger.info("Waiting for km task to complete.")
        self.shm_pool.shared_mem_block_sets[shm_set_idx].km_task_event.wait()




    def refresh_pool(self):
        for shm_set in self.shm_pool.shared_mem_block_sets:
            shm_set.in_use = False
            if not shm_set.km_task_event.is_set():
                while not shm_set.km_task_event.wait(timeout=2):
                    logger.info("Waiting for km task to complete.")
            shm_set.free()













def test():
    n_subvec_per_head = 2
    cent_cnt = 64
    compressor = MultiCoreCompressor_v2(16, 4)
    a = time.perf_counter()
    futures = []
    for i in range(16):
        key = torch.load(f"./kv_tensors/key_{i}_sample1.pt").to("cuda:0")
        bsz, n_kv_heads, n_xb, dim = key.shape
        key = key.reshape([bsz, n_kv_heads, n_xb, n_subvec_per_head, dim // n_subvec_per_head])
        key = key.transpose(2,3).flatten(1,2).flatten(0,1)
        futures.append(compressor.compress(key, cent_cnt, "euc")[2])
    
    for f in futures:
        f.result()
    
    print(f"{time.perf_counter() - a}")
    a = time.perf_counter()
    futures = []
    
    for i in range(16):
        key = torch.load(f"./kv_tensors/key_{i}_sample1.pt").to("cuda:0")
        bsz, n_kv_heads, n_xb, dim = key.shape
        key = key.reshape([bsz, n_kv_heads, n_xb, n_subvec_per_head, dim // n_subvec_per_head])
        key = key.transpose(2,3).flatten(1,2).flatten(0,1)
        futures.append(compressor.compress(key, cent_cnt, "euc")[2])
    for f in futures:
        f.result()
    print(f"{time.perf_counter() - a}")

    time.sleep(10)
    # del compressor