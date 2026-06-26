import paddle
import paddle.distributed as dist
from flash_mask.overlap import overlap_runtime as r

dist.init_parallel_env()
rank, nranks = dist.get_rank(), dist.get_world_size()

B, S_LOCAL, H, D = 1, 8192, 8, 128

uid = r.bootstrap_unique_id(rank)
print(f"[py rank{rank}] STEP-1 uid ok", flush=True)

def local_k_of(src_rank):
    paddle.seed(1234 + src_rank)
    return paddle.randn([B, S_LOCAL, H, D], dtype='float32').astype('bfloat16')

k = local_k_of(rank)
v = k.clone()
r.init_overlap(k, v, rank, nranks, uid, mask_head=1)
print(f"[py rank{rank}] STEP-2 init ok", flush=True)

stream = r.current_stream_handle()
# write_ptr allocation
write_ptr_t = paddle.zeros([1], dtype='int32')
write_ptr = write_ptr_t.data_ptr()
s_total = r.run_ag_and_wait(k, v, write_ptr, stream)

# force sync in run_ag_and_wait

assert s_total == S_LOCAL * nranks, (s_total, S_LOCAL * nranks)
print(f"[py rank{rank}] STEP-3 ag ok, s_total={s_total}", flush=True)

# write_ptr INT_MAX checking
INT_MAX = 2147483647
wptr_val = r.peek_wptr(write_ptr)
wptr_ok = (wptr_val == INT_MAX)
print(f"[py rank{rank}] STEP-3b write_ptr = {wptr_val} expect={INT_MAX} "
      f"{'OK' if wptr_ok else 'MISMATCH'}", flush=True)

# compare with paddle.distributed.all_gather
gathered = []
dist.all_gather(gathered, k)
ref_ag = paddle.concat(gathered, axis=1)        # rank identity order
print(f"[py rank{rank}] STEP-4 paddle all_gather ok, shape={list(ref_ag.shape)}", flush=True)

# fwd shoule reverse the rank order
sr_k = paddle.empty([B, s_total, H, D], dtype='bfloat16')
r.copy_k_into(sr_k)                             # d2d copy
sr_v = paddle.empty([B, s_total, H, D], dtype='bfloat16')
r.copy_v_into(sr_v)                             # d2d copy
print(f"[py rank{rank}] STEP-5 copy SRBuffer K/V -> paddle tensor ok", flush=True)

# acc testing: should be exact match (bitwise)
seg = S_LOCAL
ref_segs = []
for s in range(nranks):
    src_rank = (rank - (nranks - 1 - s)) % nranks
    ref_segs.append(ref_ag[:, src_rank * seg:(src_rank + 1) * seg, :, :])
ref_reordered = paddle.concat(ref_segs, axis=1)

bitwise_equal = bool(paddle.equal(sr_k, ref_reordered).all().item())

# fix seed regenerate for testing
gt_segs = [local_k_of((rank - (nranks - 1 - s)) % nranks) for s in range(nranks)]
gt = paddle.concat(gt_segs, axis=1)
bitwise_equal_gt = bool(paddle.equal(sr_k, gt).all().item())

# v == k.clone(), reuse the same reference
v_equal = bool(paddle.equal(sr_v, ref_reordered).all().item())
v_equal_gt = bool(paddle.equal(sr_v, gt).all().item())

print(f"[py rank{rank}] seg order src_rank = "
      f"{[(rank - (nranks - 1 - s)) % nranks for s in range(nranks)]}", flush=True)
print(f"[py rank{rank}] STEP-6 K bitwise vs paddle.all_gather(reordered) = "
      f"{'PASS' if bitwise_equal else 'FAIL'}", flush=True)
print(f"[py rank{rank}] STEP-6 K bitwise vs recomputed ground-truth      = "
      f"{'PASS' if bitwise_equal_gt else 'FAIL'}", flush=True)
print(f"[py rank{rank}] STEP-6 V bitwise vs paddle.all_gather(reordered) = "
      f"{'PASS' if v_equal else 'FAIL'}", flush=True)
print(f"[py rank{rank}] STEP-6 V bitwise vs recomputed ground-truth      = "
      f"{'PASS' if v_equal_gt else 'FAIL'}", flush=True)

final_ok = (bitwise_equal and bitwise_equal_gt and
            v_equal and v_equal_gt and wptr_ok)
print(f"[py rank{rank}] STEP-6 RESULT {'PASS' if final_ok else 'FAIL'} "
      f"(K={bitwise_equal and bitwise_equal_gt}, V={v_equal and v_equal_gt}, "
      f"write_ptr={wptr_ok})", flush=True)

if not final_ok:
    # check for any failures
    for s in range(nranks):
        a = sr_k[:, s * seg:(s + 1) * seg, :, :]
        b = ref_reordered[:, s * seg:(s + 1) * seg, :, :]
        eq = bool(paddle.equal(a, b).all().item())
        print(f"[py rank{rank}]   seg {s}: equal={eq} "
              f"sr_head={a.flatten()[:3].astype('float32').tolist()} "
              f"ref_head={b.flatten()[:3].astype('float32').tolist()}", flush=True)

r.destroy()
print(f"[py rank{rank}] STEP-7 destroy ok", flush=True)
