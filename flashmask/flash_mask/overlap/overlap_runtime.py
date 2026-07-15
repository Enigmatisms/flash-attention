"""FM-4 Overlap Python runtime: ctypes front-end over ``libfm4_overlap.so`` (the
``extern "C"`` wrapper around the FM-3 ``flashmask::comm`` NVSHMEM singleton).
Importing never loads the .so; ``_load()`` raises only on first use if missing.
"""

import ctypes
import os
from typing import NamedTuple

_UID_NBYTES = 128  # sizeof(nvshmemx_uniqueid_t)
_LIB = None


def _find_so(here):
    for f in os.listdir(here):
        if f.startswith("libfm4_overlap") and f.endswith(".so"):
            return os.path.join(here, f)
    return None


def _load():
    """Load the bridge .so and bind argtypes/restype. Cached after first call."""
    global _LIB
    if _LIB is not None:
        return _LIB

    here = os.path.dirname(os.path.abspath(__file__))
    so_path = _find_so(here)
    if so_path is None:
        raise RuntimeError(
            "FM4 overlap extension (libfm4_overlap.so) not found next to "
            f"{__file__}. Built only when the 'ovl' component is selected and "
            "NVSHMEM is available."
        )

    # RTLD_GLOBAL so the bridge's NVSHMEM symbols are visible to the dlopen'd
    # bootstrap/transport plugins at runtime.
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    lib.fm4_overlap_get_unique_id.argtypes = [ctypes.c_char_p]
    lib.fm4_overlap_get_unique_id.restype = ctypes.c_int

    # (b, s, h, d, rank, nranks) ints, uid bytes, mask_head int.
    lib.fm4_overlap_init.argtypes = [ctypes.c_int] * 6 + [ctypes.c_char_p, ctypes.c_int]
    lib.fm4_overlap_init.restype = ctypes.c_int

    for name in ("fm4_overlap_k_data", "fm4_overlap_v_data"):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_uint64
    for name in ("fm4_overlap_s_local", "fm4_overlap_nranks"):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_int

    lib.fm4_overlap_update_kv.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
    lib.fm4_overlap_update_kv.restype = None

    lib.fm4_overlap_run_ag.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.fm4_overlap_run_ag.restype = None

    # (lt_start_ptr, ut_end_ptr, compute_stream) uint64, fwd int.
    lib.fm4_overlap_compute_chunk_mask.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int]
    lib.fm4_overlap_compute_chunk_mask.restype = None

    for name in ("fm4_overlap_wait_sr_buffer_empty", "fm4_overlap_wait_reset_stream_coordinator"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_uint64]
        fn.restype = None
    lib.fm4_overlap_reset_ag_counter.argtypes = [ctypes.c_uint64]
    lib.fm4_overlap_reset_ag_counter.restype = None

    lib.fm4_overlap_wait_wptr_init.argtypes = []
    lib.fm4_overlap_wait_wptr_init.restype = None

    lib.fm4_overlap_sync_comm_stream.argtypes = []
    lib.fm4_overlap_sync_comm_stream.restype = None

    _LIB = lib
    return lib


# B/H/D are Python-side (from the K/V shape); stashed to rebuild the gathered view.
_KV_SHAPE = None   # (B, H, D); S_total = s_local() * nranks()

# NVSHMEM unique id is a process-level constant: bootstrap once (one broadcast),
# then reuse on every reconfigure.
_UID = None


def is_available():
    """True if the bridge .so can be loaded (built + NVSHMEM present)."""
    try:
        _load()
        return True
    except Exception:
        return False


def bootstrap_unique_id(rank, group=None):
    """group-local rank 0 generates the NVSHMEM unique id and broadcasts it; every
    rank returns the same 128-byte id. Cached process-wide (the id never changes)."""
    global _UID
    if _UID is not None:
        return _UID
    import numpy as np
    import paddle
    import paddle.distributed as dist

    lib = _load()
    buf = (ctypes.c_uint8 * _UID_NBYTES)()
    if rank == 0:
        lib.fm4_overlap_get_unique_id(ctypes.cast(buf, ctypes.c_char_p))

    src_global = group.ranks[0] if group is not None else 0
    arr = np.frombuffer(bytes(buf), dtype=np.uint8).copy()
    t = paddle.to_tensor(arr, dtype="uint8")
    dist.broadcast(t, src=src_global, group=group)
    _UID = bytes(t.numpy().tobytes())
    return _UID


def init_overlap(k, v, rank, nranks, uid_bytes, mask_head=1):
    """Create or reconfigure the C++ singleton from k/v shape + topology (only
    k.shape is read; the local-KV copy into the SRBuffer happens in update_kv)."""
    global _KV_SHAPE
    lib = _load()
    b, s_local, h, d = (int(x) for x in k.shape)
    _KV_SHAPE = (b, h, d)
    rc = lib.fm4_overlap_init(b, s_local, h, d, int(rank), int(nranks), uid_bytes, int(mask_head))
    if rc != 1:
        raise RuntimeError("fm4_overlap_init failed")


def ensure_initialized(k, v, group, mask_head=1):
    """Bootstrap the unique id once, then forward shape/topology to the C++ singleton
    every step. init_singleton_instance reconfigures (and reallocs the SRBuffer) only
    when they actually change, so unconditional forwarding is cheap and correct."""
    uid = bootstrap_unique_id(group.rank, group=group)
    init_overlap(k, v, group.rank, group.world_size, uid, mask_head=mask_head)


def _data_ptr(t):
    return int(t.data_ptr())


def _s_total():
    """Total gathered seqlen after the all-gather: S_local * nranks."""
    lib = _load()
    return lib.fm4_overlap_s_local() * lib.fm4_overlap_nranks()


def update_kv(k, v, fwd=True):
    """Copy local K/V into the SRBuffer (cudaMemcpyAsync on comm_stream)."""
    _load().fm4_overlap_update_kv(_data_ptr(k), _data_ptr(v), 1 if fwd else 0)


def run_ag(write_ptr_addr, fwd=True):
    """Launch the AG remote-get kernel; write_ptr_addr is the caller-owned int buffer
    the kernel signals completion into."""
    lib = _load()
    s_total = ctypes.c_int(0)
    lib.fm4_overlap_run_ag(int(write_ptr_addr), ctypes.byref(s_total), 1 if fwd else 0)
    return s_total.value


def wait_sr_buffer_empty(compute_stream):
    """Stream-order compute_stream behind the prior step draining the SRBuffer, so the
    next update_kv waits for the prior AG's consumers without blocking overlap."""
    _load().fm4_overlap_wait_sr_buffer_empty(int(compute_stream))


def wait_reset_stream_coordinator(compute_stream):
    """Hold compute_stream until the comm kernel has occupied its SMs (deadlock guard if
    compute grabbed every SM first)."""
    _load().fm4_overlap_wait_reset_stream_coordinator(int(compute_stream))


def reset_ag_counter(compute_stream):
    """Reset the persistent AG scheduler counter before each gather."""
    _load().fm4_overlap_reset_ag_counter(int(compute_stream))


def wait_wptr_init():
    """comm_stream waits for the counter reset (wptr_init event) to land before the
    remote-get kernel reads block_cnt_semaphore. Pairs with reset_ag_counter."""
    _load().fm4_overlap_wait_wptr_init()


def sync_comm_stream():
    """Host-block until the internal comm_stream has fully drained before the kernel
    consumes the gathered SRBuffer."""
    _load().fm4_overlap_sync_comm_stream()


def _sparse_chunk_mask_cols(startend_row_indices):
    """Slice (lt_start, ut_end) columns into contiguous (B, H_mask, S_total) int32."""
    num_vecs = startend_row_indices.shape[-1]
    assert num_vecs in (2, 4), f"overlap mask must be 2/4-vec (non-causal), got {num_vecs}"
    lt_start = startend_row_indices[..., 0].contiguous()
    ut_end = startend_row_indices[..., num_vecs - 1].contiguous()
    return lt_start, ut_end


def compute_chunk_mask_sparse(startend_row_indices, compute_stream, fwd=True):
    """Drive copy_chunk_mask so the AG kernel skips fully-masked KV chunks. The check
    runs async on compute_stream reading the two tensors, so they MUST outlive run_ag's
    launch -- returned for keepalive."""
    lt_start, ut_end = _sparse_chunk_mask_cols(startend_row_indices)
    _load().fm4_overlap_compute_chunk_mask(
        _data_ptr(lt_start), _data_ptr(ut_end), int(compute_stream), 1 if fwd else 0
    )
    return lt_start, ut_end


def current_stream_handle():
    """Current compute-stream handle as a plain int (the cuda_stream pointer)."""
    import paddle

    return int(paddle.device.current_stream().stream_base.cuda_stream)


class SrKvView(NamedTuple):
    """Cross-jit-safe handle to the gathered SRBuffer K/V."""
    k_addr: int
    v_addr: int
    shape: tuple  # (B, S_total, H, D)


class ForwardAgArgs(NamedTuple):
    view: SrKvView
    write_ptr: object
    kv_chunk_size: int


class BackwardAgArgs(NamedTuple):
    view: SrKvView
    write_ptr: object
    kv_chunk_size: int


def sr_kv_view_args():
    """SrKvView for the gathered SRBuffer K/V (post all-gather). SRBuffer is bf16."""
    if _KV_SHAPE is None:
        raise RuntimeError("init_overlap must be called before sr_kv_view_args")
    lib = _load()
    b, h, d = _KV_SHAPE
    return SrKvView(
        k_addr=lib.fm4_overlap_k_data(),
        v_addr=lib.fm4_overlap_v_data(),
        shape=(b, _s_total(), h, d),
    )


def _start_ag(k, v, startend_row_indices, compute_stream, *, fwd, write_ptr=0):
    wait_sr_buffer_empty(compute_stream)
    mask_keepalive = compute_chunk_mask_sparse(
        startend_row_indices, compute_stream, fwd=fwd
    )
    update_kv(k, v, fwd=fwd)
    reset_ag_counter(compute_stream)
    wait_wptr_init()
    run_ag(write_ptr, fwd=fwd)
    wait_reset_stream_coordinator(compute_stream)
    return sr_kv_view_args(), mask_keepalive


def start_forward_ag(k, v, startend_row_indices, compute_stream):
    """Launch forward sparse AG asynchronously for cumulative-frontier gating."""
    import paddle

    write_ptr = paddle.zeros([1], dtype=paddle.int32)
    view, _ = _start_ag(
        k,
        v,
        startend_row_indices,
        compute_stream,
        fwd=True,
        write_ptr=int(write_ptr.data_ptr()),
    )
    return ForwardAgArgs(view, write_ptr, int(k.shape[1]))


def start_backward_ag(k, v, startend_row_indices, compute_stream):
    """Launch backward sparse AG asynchronously. The non-splitted comm kernel's bwd
    traversal fills the remote KV region left-to-right, advancing the same wptr row
    frontier (atomicMax) as forward, so the compute gate waits on it exactly like
    forward -- only the edge differs (bwd gates the tile's right edge)."""
    import paddle

    write_ptr = paddle.zeros([1], dtype=paddle.int32)
    view, _ = _start_ag(
        k,
        v,
        startend_row_indices,
        compute_stream,
        fwd=False,
        write_ptr=int(write_ptr.data_ptr()),
    )
    return BackwardAgArgs(view, write_ptr, int(k.shape[1]))
