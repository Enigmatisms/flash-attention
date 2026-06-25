# FM-4 Overlap Bridge (STEP 1)

把 FM-3（FA3 / sm_90a）已验证的 NVSHMEM SM 级通信-overlap runtime，接到 FM-4（FA4 / cute-DSL / SM100）路径上的最小 native bridge。

本目录是迁移总方案（`/root/work/fm-4-overlap.md`）的第一步落地：不重写 NVSHMEM 成 cute-DSL，而是保留 FM-3 native runtime，用一个很薄的 `extern "C"` bridge 把 `OverlapCommunicator` 的 SRBuffer 裸指针包装成 cute 能消费的 tensor view。

## STEP 1 验证什么

只证明数据通路，不追求性能：

1. NVSHMEM init（rank0 生成 unique id，broadcast 后各 rank 传入）。
2. 把计算 stream handle（int）经 C-ABI 传进去。
3. 把 SRBuffer 裸指针取回来，零拷贝包成 `cute.Tensor`。

没有真实 attention kernel，没有 true overlap。`run_ag_and_wait` 里的等待是 bridge 端一个故意的强等 shim（`cudaDeviceSynchronize`），把 AG 退化成阻塞 all-gather，只为验证指针传递 + layout + cute 消费正确。后续 step 才把这个 shim 换成 kernel 内的 `write_ptr` readiness 判定。

## 目录结构

```
flash_mask/overlap/
├── __init__.py            软 import 守卫（import 永不失败）
├── overlap_runtime.py     ctypes 加载 + uid bootstrap + init + stream 传入 + view 构造
├── README.md              本文件
└── csrc/
    ├── CMakeLists.txt      照搬 distributed/CMakeLists.txt 的 NVSHMEM 链接配方 + 自建 nvshmem target + 符号隔离
    ├── fm4_overlap_bridge.cu   extern "C" 薄封装
    └── fm4_overlap.map     version script，只导出 fm4_overlap_*
```

bridge `.so` 把三个 TU 编进一个 shared lib：`fm4_overlap_bridge.cu`（新增）、`overlap_comm.cu`、`sep_sr_buffer.cu`（后两个从 `csrc/flashmask_v2/distributed/` 复用，不改），静态链接预编译的 `libnvshmem.a` 加 UID bootstrap host `.so`。

## 构建

bridge 是 opt-in 组件，不会被 `FLASHMASK_BUILD=all` 带上（因为它需要 NVSHMEM 和 H100/B200）。必须显式请求 `ovl`。

### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `NVSHMEM_HOME` | `/root/work/Paddle/build/third_party/install/nvshmem` | NVSHMEM 安装根目录（含 `include/` 与 `lib/`）；传给子 CMake 的 `NVSHMEM_INSTALL_DIR` |
| `FM4_OVERLAP_CUDA_ARCH` | `90a` | 目标 sm arch，B200 设 `100a` |
| `FM4_OVERLAP_CUTLASS_INC` | 自动探测 | cutlass include 目录（须含 `cutlass/bfloat16.h`）。不设时按 FA4 submodule、Paddle 自带 cutlass 顺序探测 |

### H100（复用预编译 sm_90 NVSHMEM）

```bash
cd flashmask
FLASHMASK_BUILD=fa4,ovl pip install -e . --no-build-isolation
# 期望：子 CMake configure + build 无 error，libfm4_overlap.so 拷进 flash_mask/overlap/
python -c "from flash_mask.overlap import overlap_runtime as r; r._load(); print('loaded OK')"
```

### B200（需要 sm_100 NVSHMEM）

Paddle 预编译的 `libnvshmem.a` 设备码只有 sm_90（cuobjdump 已确认），所以 B200 上必须先备好一份 sm_100 的 NVSHMEM（重编，或指向 Paddle 自带的 sm_100 third_party），再：

```bash
cd flashmask
NVSHMEM_HOME=<sm100 NVSHMEM 路径> \
FM4_OVERLAP_CUDA_ARCH=100a \
FLASHMASK_BUILD=fa4,ovl pip install -e . --no-build-isolation
```

`100a` 要求 CUDA >= 12.8。

## 运行（H100 / B200，需 nranks >= 2）

`nranks == 1` 时 overlap 无意义（且 init 的 S_local dispatch 之外的拓扑约束也要求多卡）。单机 2 卡示例（`paddle.distributed.launch` 或 mpirun 起 2 进程）：

```python
import paddle
import paddle.distributed as dist
import cutlass.cute as cute
from flash_mask.overlap import overlap_runtime as r

dist.init_parallel_env()
rank, nranks = dist.get_rank(), dist.get_world_size()

uid = r.bootstrap_unique_id(rank)                       # rank0 生成 + broadcast
k = paddle.randn([1, 8192, 8, 128], dtype='bfloat16')   # (B, S_local, H, D)
v = paddle.randn([1, 8192, 8, 128], dtype='bfloat16')
r.init_overlap(k, v, rank, nranks, uid, mask_head=1)    # NVSHMEM init（不解引用 k/v）

stream = r.current_stream_handle()                      # 传入计算 stream
# write_ptr 是一个 device int buffer 地址；STEP 1 强等路径下其值不被使用，
# 只为填 kernel 签名。可用任意合法 device int 指针（例如一个 1 元素 int tensor）。
write_ptr = paddle.zeros([1], dtype='int32').data_ptr()
# run_ag_and_wait 内部照搬 forward launch template 顺序：
#   wait_sr_buffer_empty -> compute_chunk_mask -> update_kv -> run_ag -> wait_ag_done。
# compute_chunk_mask 用一份合成的全 attention mask（lt_start = S_total, ut_end = 0），
# 让 copy_chunk_mask 判定每个 chunk 都不被跳过，从而退化成完整 all-gather。
s_total = r.run_ag_and_wait(k, v, write_ptr, stream)    # 全量 AG + 强等

k_view, v_view = r.sr_kv_cute_views()                   # 取出 SRBuffer view
cute.print_tensor(k_view)                               # cute 成功消费 NVSHMEM 内存
print('S_total =', s_total)                             # == S_local * nranks

r.destroy()
```

### 「正确」长什么样

1. `init_overlap` 不抛 uid 校验异常，NVSHMEM init 日志出现 rank / nranks。
2. `current_stream_handle()` 拿到非零 handle。
3. `sr_kv_cute_views()` 返回的 cute.Tensor，shape == `(B, S_local*nranks, H, D)`，地址 == `fm4_overlap_k_data()`。
4. `cute.print_tensor(k_view)` 打印出 AG 后的 K（本 rank 段是本地 K，其余段是远端拉取值），证明 CPP -> Python -> Cute 的 SRBuffer 反向透传成立。每个 chunk 都应有真实数据，不应出现整段未被拉取的空洞（说明 compute_chunk_mask 的全 attention mask 生效，AG 没有错误跳过任何 chunk）。

## 设计要点

- cute 不需要知道底层是 NVSHMEM 内存，只要拿到合法 device pointer 加 dtype/shape/stride，就能像普通 tensor 读写。数据 readiness 不由 DLPack / cute tensor 表达，而由 FM-3 已有的 `write_ptr`、event、semaphore 保证（STEP 1 用强等 shim 代替）。
- 指针过界一律 `uint64_t`，Python 只见 int，`reinterpret_cast` 回真实类型，位等价。
- `OverlapCommunicator` 不暴露给 Python，生命周期完全在 C++ 端的 static `unique_ptr` 单例里，到进程退出。
- init 不解引用 k/v：`OverlapCommunicator` 构造函数虽接收 k/v 指针但从不解引用（只存 shape），所以 bridge 的 `fm4_overlap_init` 直接丢掉 k/v 参数，内部传 nullptr。本地 KV 拷贝发生在后面的 `update_kv`。
- AG 前必须算 chunk mask：`copy_chunk_mask` 是 `block_work_ids` 里由 `cudaMallocAsync` 分到的一段子区域，而 `cudaMallocAsync` 不清零，AG 的 remote-get kernel 又会读它来判断某个 256/512 行的 KV chunk 能否整段跳过（`remote_get_kernel.cuh:383`）。所以 STEP 1 照搬 forward template 在 `run_ag` 前调一次 `compute_chunk_mask`，喂一份合成的全 attention mask（`lt_start = S_total`、`ut_end = 0`），让 kernel 里的 `is_masked = (lt_start <= ut_end)` 判定恒为假，每个 chunk 都被拷贝，退化成正确的完整 all-gather。这份 mask 只在 bridge 透传真实指针的前提下成立，mask 张量必须在 `run_ag` 消费完之前一直存活（kernel 跑在内部 comm_stream 上，是异步的）。

## 已知遗留风险（记录，不阻塞 STEP 1）

- 符号隔离（version-script + `-Bsymbolic` + `--exclude-libs,ALL`）只能挡 ELF interposition，挡不住 NVSHMEM plugin `.so` 静态状态共享这个更深的根因。STEP 1 是单实例、不要求与 DeepEP 共存，故此风险记录但不处理。
- arch `100a` 路径本地（A100）无法编译验证，靠静态审查保证；首次在 B200 上验证。
