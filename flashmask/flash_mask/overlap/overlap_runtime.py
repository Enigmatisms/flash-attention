"""FM-4 Overlap Python runtime (STEP 1).

Thin ctypes front-end over the ``libfm4_overlap.so`` bridge (which itself is a
``extern "C"`` wrapper around the FM-3 ``flashmask::comm`` NVSHMEM singleton).

STEP 1 only proves the data path:
  1. NVSHMEM init via a broadcast unique-id,
  2. passing a compute stream handle (int) across the C-ABI,
  3. pulling the SRBuffer raw device pointer back out and wrapping it as a
     ``cute.Tensor`` (zero-copy) via ``make_gmem_tensor_from_addr``.

No real attention kernel, no true overlap. ``run_ag_and_wait`` deliberately
serializes (the bridge's ``wait_ag_done`` is a blunt device sync), which is fine
because STEP 1 is correctness-only.

The ``.so`` is built by ``flashmask/setup.py`` only when the 'ovl' component is
requested AND NVSHMEM is available. If it is missing, ``_load()`` raises a clear
error but importing this module never fails, so FA4 keeps importing on machines
that did not build overlap (e.g. the A100 dev box).
"""

import ctypes
import os

# nvshmemx_uniqueid_t is 128 bytes in practice. The bridge writes exactly
# sizeof(nvshmemx_uniqueid_t) bytes; we size the Python buffer to match.
_UID_NBYTES = 128

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
            f"{__file__}. It is built only when the 'ovl' component is selected "
            "and NVSHMEM is available (H100/B200). On the A100 dev box it is not "
            "built by design."
        )

    # RTLD_GLOBAL so the bridge's NVSHMEM symbols are visible to the
    # dlopen'd bootstrap/transport plugins at runtime.
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    lib.fm4_overlap_get_unique_id.argtypes = [ctypes.c_char_p]
    lib.fm4_overlap_get_unique_id.restype = ctypes.c_int

    # (b, s, h, d, rank, nranks) ints, then uid bytes, then mask_head int.
    lib.fm4_overlap_init.argtypes = [ctypes.c_int] * 6 + [
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.fm4_overlap_init.restype = ctypes.c_int

    lib.fm4_overlap_is_initialized.argtypes = []
    lib.fm4_overlap_is_initialized.restype = ctypes.c_int

    lib.fm4_overlap_destroy.argtypes = []
    lib.fm4_overlap_destroy.restype = None

    lib.fm4_overlap_k_data.argtypes = []
    lib.fm4_overlap_k_data.restype = ctypes.c_uint64
    lib.fm4_overlap_v_data.argtypes = []
    lib.fm4_overlap_v_data.restype = ctypes.c_uint64

    lib.fm4_overlap_s_local.argtypes = []
    lib.fm4_overlap_s_local.restype = ctypes.c_int
    lib.fm4_overlap_nranks.argtypes = []
    lib.fm4_overlap_nranks.restype = ctypes.c_int

    lib.fm4_overlap_update_kv.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fm4_overlap_update_kv.restype = None

    lib.fm4_overlap_run_ag.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.fm4_overlap_run_ag.restype = None

    # (lt_start_ptr, ut_end_ptr, compute_stream) uint64, then fwd int.
    lib.fm4_overlap_compute_chunk_mask.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fm4_overlap_compute_chunk_mask.restype = None

    for name in (
        "fm4_overlap_wait_sr_buffer_empty",
        "fm4_overlap_wait_reset_stream_coordinator",
        "fm4_overlap_wait_ag_done",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_uint64]
        fn.restype = None

    lib.fm4_overlap_wait_wptr_init.argtypes = []
    lib.fm4_overlap_wait_wptr_init.restype = None

    _LIB = lib
    return lib


# The bridge only exposes S_local and nranks. B/H/D and H_mask are known at init
# time on the Python side (from the K/V tensor shape + mask_head arg), so we stash
# them here to rebuild the SRBuffer view shape and the chunk-mask tensors later.
# Single process-local config mirrors the single C++ singleton.
_KV_SHAPE = None  # (B, H, D); S_total = s_local() * nranks()
_MASK_HEAD = None  # H_mask, used to shape the (B, H_mask, S_total) chunk-mask buffers


def is_available():
    """True if the bridge .so can be loaded (built + NVSHMEM present)."""
    try:
        _load()
        return True
    except Exception:
        return False


def bootstrap_unique_id(rank, group=None):
    """rank 0 generates the NVSHMEM unique id; broadcast it to all ranks.

    Returns the 128-byte id (identical on every rank) to feed into init_overlap.
    """
    import paddle
    import paddle.distributed as dist

    lib = _load()
    buf = (ctypes.c_uint8 * _UID_NBYTES)()
    if rank == 0:
        lib.fm4_overlap_get_unique_id(ctypes.cast(buf, ctypes.c_char_p))

    # uint8 tensor of the raw id bytes; broadcast from rank 0.
    t = paddle.to_tensor(bytearray(buf), dtype="uint8")
    dist.broadcast(t, src=0, group=group)
    return bytes(t.numpy().tobytes())


def init_overlap(k, v, rank, nranks, uid_bytes, mask_head=1):
    """Create/reuse the C++ singleton from shapes + topology (no data deref).

    k/v are Paddle tensors of shape (B, S_local, H, D), bf16. Only their shape
    is used by init; the local-KV copy into the SRBuffer happens in update_kv.
    """
    global _KV_SHAPE, _MASK_HEAD
    lib = _load()
    b, s_local, h, d = (int(x) for x in k.shape)
    _KV_SHAPE = (b, h, d)
    _MASK_HEAD = int(mask_head)
    rc = lib.fm4_overlap_init(
        b, s_local, h, d, int(rank), int(nranks), uid_bytes, int(mask_head)
    )
    if rc != 1:
        raise RuntimeError("fm4_overlap_init failed")


def _data_ptr(t):
    # Paddle tensor device pointer as a Python int -> c_uint64 (bit identity).
    return int(t.data_ptr())


def update_kv(k, v, fwd=True):
    """Copy local K/V into the SRBuffer (cudaMemcpyAsync on comm_stream)."""
    lib = _load()
    lib.fm4_overlap_update_kv(_data_ptr(k), _data_ptr(v), 1 if fwd else 0)


def run_ag(write_ptr, fwd=True):
    """Launch the AG remote-get kernel; returns S_total (= S_local * nranks)."""
    lib = _load()
    s_total = ctypes.c_int(0)
    lib.fm4_overlap_run_ag(
        int(write_ptr), ctypes.byref(s_total), 1 if fwd else 0
    )
    return s_total.value


def _full_attention_mask_tensors(s_total):
    """Build the two int32 mask buffers that force a FULL all-gather.

    The AG remote-get kernel decides whether to skip a KV chunk by reading
    copy_chunk_mask, which compute_chunk_mask fills from lt_start / ut_end with
    the predicate ``is_masked = (lt_start <= ut_end)`` (remote_get_kernel.cuh:207),
    AND-reduced across H_mask heads. A chunk is copied only when it is NOT masked,
    so to copy everything (full attention, skip nothing) we need lt_start > ut_end
    on every row: fill lt_start = s_total and ut_end = 0, giving the predicate
    ``s_total <= 0`` which is false for all rows.

    The buffers are laid out (B, H_mask, S_total) row-major int32, matching the
    kernel's batch_offset = blockIdx.y * num_head * head_stride with
    head_stride = S_total. Returns (lt_start, ut_end) Paddle tensors; the caller
    MUST keep them alive until the AG kernel has consumed them, because the kernel
    runs asynchronously on the internal comm_stream.
    """
    import paddle

    if _KV_SHAPE is None or _MASK_HEAD is None:
        raise RuntimeError("init_overlap must be called before compute_chunk_mask")
    b, _h, _d = _KV_SHAPE
    shape = [b, _MASK_HEAD, s_total]
    lt_start = paddle.full(shape, s_total, dtype="int32")
    ut_end = paddle.zeros(shape, dtype="int32")
    return lt_start, ut_end


def compute_chunk_mask(compute_stream, fwd=True):
    """Fill copy_chunk_mask for a full-attention all-gather (skip nothing).

    Mirrors flash_fwd_launch_template.h:124: this MUST be called before run_ag,
    otherwise the AG kernel reads an uninitialized copy_chunk_mask (a cudaMallocAsync
    sub-region of block_work_ids, which is not zeroed) and skips/copies arbitrary
    chunks. Returns the (lt_start, ut_end) tensors so the caller can keep them alive
    across the subsequent async AG launch.
    """
    lib = _load()
    s_total = lib.fm4_overlap_s_local() * lib.fm4_overlap_nranks()
    lt_start, ut_end = _full_attention_mask_tensors(s_total)
    lib.fm4_overlap_compute_chunk_mask(
        _data_ptr(lt_start), _data_ptr(ut_end), int(compute_stream), 1 if fwd else 0
    )
    return lt_start, ut_end


def current_stream_handle():
    """Compute-stream handle as a plain int, matching interface.py:334."""
    import paddle

    return int(paddle.device.current_stream().stream_base.cuda_stream)


def run_ag_and_wait(k, v, write_ptr, compute_stream):
    """STEP-1 helper: full single-shot all-gather, then force the compute stream
    to wait for it (deliberately non-overlap; wait_ag_done is a device sync).

    Mirrors the forward launch template ordering (flash_fwd_launch_template.h:123-145):
        wait_sr_buffer_empty -> compute_chunk_mask -> update_kv -> run_ag -> wait_ag_done.

    k/v are the local Paddle K/V tensors (B, S_local, H, D) that get copied into
    the SRBuffer by update_kv. Returns S_total (= S_local * nranks).
    """
    lib = _load()
    lib.fm4_overlap_wait_sr_buffer_empty(int(compute_stream))
    # Keep the mask tensors referenced until run_ag has consumed copy_chunk_mask;
    # the AG kernel runs async on comm_stream, so dropping them earlier could free
    # the device memory the BlockSparsityCheck kernel is still reading from.
    _mask_keepalive = compute_chunk_mask(compute_stream, fwd=True)
    update_kv(k, v, fwd=True)
    s_total = run_ag(write_ptr, fwd=True)
    lib.fm4_overlap_wait_ag_done(int(compute_stream))
    del _mask_keepalive
    return s_total


def sr_kv_cute_views():
    """Wrap the SRBuffer K and V raw pointers as zero-copy cute.Tensors.

    Returns (k_view, v_view), each a (B, S_total, H, D) BSHD cute.Tensor whose
    data pointer == fm4_overlap_k_data()/v_data(). Readiness is NOT expressed by
    these views; it is enforced separately (the wait_ag_done shim in STEP 1).
    """
    import cutlass

    from flash_mask.cute.utils import make_gmem_tensor_from_addr

    if _KV_SHAPE is None:
        raise RuntimeError("init_overlap must be called before sr_kv_cute_views")

    lib = _load()
    b, h, d = _KV_SHAPE
    s_total = lib.fm4_overlap_s_local() * lib.fm4_overlap_nranks()
    shape = (b, s_total, h, d)
    stride = (s_total * h * d, h * d, d, 1)

    k_addr = lib.fm4_overlap_k_data()
    v_addr = lib.fm4_overlap_v_data()
    k_view = make_gmem_tensor_from_addr(k_addr, shape, stride, cutlass.BFloat16, align=16)
    v_view = make_gmem_tensor_from_addr(v_addr, shape, stride, cutlass.BFloat16, align=16)
    return k_view, v_view


def destroy():
    """Tear down the C++ singleton (device sync + nvshmem barrier + reset)."""
    global _KV_SHAPE, _MASK_HEAD
    lib = _load()
    lib.fm4_overlap_destroy()
    _KV_SHAPE = None
    _MASK_HEAD = None

