// FM-4 Overlap bridge: a thin extern "C" wrapper around the FM-3
// `flashmask::comm` NVSHMEM overlap singleton, so Python (via ctypes) can
// drive the proven FM-3 communication runtime and pull the SRBuffer raw
// pointer back out to be wrapped as a cute tensor.
//
// Design:
//   - Every function is extern "C", prefixed fm4_overlap_, so a single
//     version-script glob (fm4_overlap_*) can export exactly this surface.
//   - All pointers / handles cross the ABI boundary as uint64_t. Python only
//     ever passes plain ints; we reinterpret_cast back to the real type here.
//   - The OverlapCommunicator class is never exposed. Its lifetime lives
//     entirely on the C++ side in the FM-3 static unique_ptr singleton.
//
// Step-1 scope: prove init + stream-in + SRBuffer-pointer-out + cute wrap.
// No real SM100 attention kernel, no true overlap. wait_ag_done is a
// deliberate forced-wait shim that degrades AG to a blocking all-gather,
// just enough to prove the data in the SRBuffer is correct and cute can
// read it. Later steps replace the shim with in-kernel write_ptr waits.

#include "overlap_comm.cuh"
#include "cutlass/bfloat16.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

using bf16 = cutlass::bfloat16_t;

namespace {

// Convert an opaque uint64 handle coming from Python into a cudaStream_t.
// A value of 0 maps to the default stream (cudaStream_t)0, which is fine.
inline cudaStream_t as_stream(uint64_t handle) {
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(handle));
}

}  // namespace

extern "C" {

// rank 0 generates the NVSHMEM unique id (sizeof(nvshmemx_uniqueid_t) bytes,
// 128 in practice) into `out`. Caller must broadcast it to every rank and pass
// the same bytes into fm4_overlap_init. Returns 1 on success.
//
// We call nvshmemx_get_uniqueid directly rather than reusing flash_api.cu's
// ::get_nvshmem_unique_id() free function: that symbol lives in a TU we do not
// link into this bridge .so, and it is gated behind the same
// NVSHMEM_DISTRIBUTED_OVERLAP macro anyway. Calling the NVSHMEM API here keeps
// the bridge self-contained.
int fm4_overlap_get_unique_id(uint8_t* out) {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::memcpy(out, &unique_id, sizeof(nvshmemx_uniqueid_t));
    return 1;
}

// Create or reuse the singleton. Internally goes through
// flashmask::comm::init_singleton_instance; first call constructs the
// OverlapCommunicator (which NVSHMEM-inits, creates comm_stream/events, and
// nvshmem_malloc's the SRBuffer).
//
// NOTE: init only needs shape (b/s/h/d) and topology (rank/nranks/uid). It does
// NOT need the real K/V data pointers. The OverlapCommunicator constructor
// (overlap_comm.cu:154-243) takes k/v pointers but never dereferences them
// (see the author's note at overlap_comm.cu:167), so we pass nullptr here.
// The actual local-K/V copy into the SRBuffer happens later via
// fm4_overlap_update_kv, mirroring flash_fwd_launch_template.h:125.
//
// Returns 1 on success.
int fm4_overlap_init(int b_kv, int s_kv, int h_kv, int d_kv,
                     int rank, int nranks,
                     const uint8_t* unique_id, int mask_head) {
    flashmask::comm::init_singleton_instance(
        /*k_data=*/static_cast<const bf16*>(nullptr),
        /*v_data=*/static_cast<const bf16*>(nullptr),
        b_kv, s_kv, h_kv, d_kv,
        rank, nranks,
        unique_id, mask_head);
    return 1;
}

// 1 if the singleton has been created, 0 otherwise.
int fm4_overlap_is_initialized() {
    return flashmask::comm::is_singleton_null() ? 0 : 1;
}

// Destroy the singleton (cudaDeviceSynchronize + nvshmem_barrier_all + reset).
// NVSHMEM is, by FM-3 design, not finalized (MANUAL_CLEANUP=false).
void fm4_overlap_destroy() {
    flashmask::comm::destroy_singleton();
}

// SRBuffer K base pointer as a raw uint64. This is the address that gets
// wrapped into a cute tensor on the Python side (the whole point of step 1).
uint64_t fm4_overlap_k_data() {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().k_data());
}

uint64_t fm4_overlap_v_data() {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().v_data());
}

// Local seqlen chunk length (S_local). After AG, S_total = s_local * nranks.
int fm4_overlap_s_local() {
    return flashmask::comm::singleton().s_local();
}

// Number of PEs participating (== internal _total_n_pes).
int fm4_overlap_nranks() {
    return flashmask::comm::singleton().nranks();
}

// Copy local K/V into the SRBuffer and (inside update_kv_buffer) kick off the
// transfer-prep on the internal comm_stream. fwd should be 1 for step 1.
void fm4_overlap_update_kv(uint64_t k_ptr, uint64_t v_ptr, int fwd) {
    flashmask::comm::singleton().update_kv_buffer(
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(k_ptr)),
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(v_ptr)),
        fwd != 0);
}

// Compute the per-chunk sparsity mask (copy_chunk_mask) on the internal
// comm_stream. This MUST run before fm4_overlap_run_ag: the AG remote-get
// kernel reads copy_chunk_mask[mask_index] (remote_get_kernel.cuh:383) to decide
// whether a 256/512-row KV chunk is fully masked and can be skipped. That buffer
// is a sub-region of block_work_ids allocated by cudaMallocAsync, which does NOT
// zero memory, so without this call the AG kernel would read garbage and skip /
// copy arbitrary chunks, corrupting the all-gather.
//
// lt_start_ptr / ut_end_ptr are device int32 buffers of shape (B, H_mask, S_total)
// in row-major order (S_total = s_local * nranks). compute_chunk_mask requires
// both non-null (overlap_comm.cu:453) and tolerates (with a warning) lt_end /
// ut_start being null, which is exactly what we pass. For a full-attention smoke
// test, Python fills lt_start = S_total and ut_end = 0 so that the kernel's
// "is_masked = (lt_start <= ut_end)" predicate is false everywhere and every
// chunk is copied. stream is the stream the BlockSparsityCheck kernel is launched
// on; pass the compute stream so the mask write is ordered before the AG kernel,
// which the comm_stream picks up after wait_sr_buffer_empty's event handshake.
void fm4_overlap_compute_chunk_mask(uint64_t lt_start_ptr, uint64_t ut_end_ptr,
                                    uint64_t stream, int fwd) {
    flashmask::comm::singleton().compute_chunk_mask(
        reinterpret_cast<const int*>(static_cast<uintptr_t>(lt_start_ptr)),
        /*lt_end_ptr=*/nullptr,
        /*ut_start_ptr=*/nullptr,
        reinterpret_cast<const int*>(static_cast<uintptr_t>(ut_end_ptr)),
        as_stream(stream), fwd != 0);
}

// Launch the All-Gather remote-get kernel on the internal comm_stream.
// write_ptr_dev points at a small device int buffer (its value is unused in the
// step-1 forced-wait path; it only fills the kernel signature). The communicator
// rewrites S to S_local * nranks; we return that via s_total_out.
//
// IMPORTANT: fm4_overlap_compute_chunk_mask must have been called first (see its
// comment) so copy_chunk_mask holds a valid mask, otherwise the remote-get kernel
// reads uninitialized memory.
void fm4_overlap_run_ag(uint64_t write_ptr_dev, int* s_total_out, int fwd) {
    int S = 0;  // run_overlap_ag_kernel sets S = S_local * _total_n_pes
    flashmask::comm::singleton().run_overlap_ag_kernel(
        reinterpret_cast<int*>(static_cast<uintptr_t>(write_ptr_dev)),
        S, fwd != 0);
    if (s_total_out) {
        *s_total_out = S;
    }
}

// comp_stream notifies comm_stream that the local SRBuffer chunk may be reused.
void fm4_overlap_wait_sr_buffer_empty(uint64_t compute_stream) {
    flashmask::comm::singleton().wait_sr_buffer_empty(as_stream(compute_stream));
}

// comm_stream waits on the wptr_init event (write_ptr usable). In step 1, with
// no prepare_flashmask recording wptr_init, this is effectively a no-op wait;
// exposed now so the C-ABI stays stable when step 3 wires in true overlap.
void fm4_overlap_wait_wptr_init() {
    flashmask::comm::singleton().wait_wptr_init();
}

// compute stream waits until the comm kernel is actually scheduled onto SMs.
void fm4_overlap_wait_reset_stream_coordinator(uint64_t stream) {
    flashmask::comm::singleton().wait_reset_stream_coordinator(as_stream(stream));
}

// Step-1 forced-wait shim. Make the compute stream wait until the AG transfer
// done on comm_stream is visible. We reuse the communicator's existing
// sr_usable event: record it on the comm_stream, then have the compute stream
// wait on it. This deliberately serializes (no overlap) and is only here to
// prove pointer hand-off + layout + cute consumption are correct. Step 3
// replaces this with in-kernel write_ptr readiness checks.
void fm4_overlap_wait_ag_done(uint64_t compute_stream) {
    // sr_usable is a public cudaEvent_t on the communicator. The comm_stream
    // is private, but wait_reset_stream_coordinator already guarantees the comm
    // kernel is scheduled; to make the data visible we synchronize the comm work
    // onto the compute stream via the event recorded on the comm side.
    //
    // The communicator does not expose comm_stream directly, so the simplest
    // faithful step-1 barrier is a full device synchronize here. This is the
    // bluntest possible "AG is finished" guarantee and is acceptable because
    // step 1 is explicitly non-overlap correctness-only.
    (void)compute_stream;
    cudaDeviceSynchronize();
}

}  // extern "C"
