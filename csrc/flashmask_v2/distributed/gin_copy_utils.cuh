#pragma once

#include <cuda_runtime.h>

#include "nccl_gin_backend.cuh"

namespace flashmask::shmem {

__device__ __forceinline__ void copy_two_buffers_block(
    void* dst1, void* dst2, const void* src1, const void* src2, int bytes) {
    auto* dst1_vec = static_cast<int4*>(dst1);
    auto* dst2_vec = static_cast<int4*>(dst2);
    const auto* src1_vec = static_cast<const int4*>(src1);
    const auto* src2_vec = static_cast<const int4*>(src2);
    const int count = bytes / sizeof(int4);
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst1_vec[i] = src1_vec[i];
        dst2_vec[i] = src2_vec[i];
    }
}

// Building a GIN work request needs far more live state than the bulk copy
// above. The comm kernels run at 512 threads with 4 CTAs per SM, i.e. a hard
// 32-register budget, so inlining this into them makes ptxas spill the copy
// loop -- a cost every chunk pays, including the intra-node ones that never
// reach here. Keeping it out of line confines that pressure to its own frame.
// `static` is required: with external linkage the ABI register count is fixed
// before ptxas sees the caller, and separable compilation then rejects a
// 32-register entry calling a 126-register function.
static __device__ __noinline__ void network_get_two_buffers(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    const auto& src_region = gin::find_region(src1);
    const auto& dst_region = gin::find_region(dst1);
    const int peer = gin::team_peer(world_peer);
    auto transport = gin::make_gin();
    transport.get(
        gin::network_team(), peer,
        src_region.window, gin::region_offset(src_region, src1),
        dst_region.window, gin::region_offset(dst_region, dst1),
        bytes, ncclCoopThread(), ncclGin_None(),
        ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
    transport.get(
        gin::network_team(), peer,
        src_region.window, gin::region_offset(src_region, src2),
        dst_region.window, gin::region_offset(dst_region, dst2),
        bytes, ncclCoopThread(), ncclGin_None(),
        ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
    // Settle only the peer we just read from. flush() instead walks all team
    // ranks of this context, issuing a CST + CQ wait per peer from this single
    // thread, and would also block on unrelated in-flight reads posted by CTAs
    // that share this context.
    ncclGinRequest_t request;
    transport.flushAsync(gin::network_team(), peer, &request, ncclCoopThread());
    transport.wait(request, ncclCoopThread());
}

__device__ __forceinline__ void two_buffers_getmem_block(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    __syncthreads();
    if (auto* peer_src1 = gin::get_lsa_ptr(src1, world_peer); peer_src1 != nullptr) {
        auto* peer_src2 = gin::get_lsa_ptr(src2, world_peer);
        copy_two_buffers_block(dst1, dst2, peer_src1, peer_src2, bytes);
        __syncthreads();
        return;
    }

    if (threadIdx.x == 0) {
        network_get_two_buffers(dst1, dst2, src1, src2, bytes, world_peer);
    }
    __syncthreads();
}

// Out of line for the same reason as network_get_two_buffers.
static __device__ __noinline__ void network_put_two_buffers(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    const auto& src_region = gin::find_region(src1);
    const auto& dst_region = gin::find_region(dst1);
    // Route every CTA's put through the shared ordered context (one QP per
    // peer) so that the winner CTA's later remote_strong_add on the same
    // context can settle ALL these puts to this peer -- see try_commit_rank.
    auto transport = gin::make_gin_ordered();
    transport.put(
        gin::network_team(), gin::team_peer(world_peer),
        dst_region.window, gin::region_offset(dst_region, dst1),
        src_region.window, gin::region_offset(src_region, src1),
        bytes, ncclGin_None(), ncclGin_None(), ncclCoopThread(),
        ncclGin_None(), cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
    transport.put(
        gin::network_team(), gin::team_peer(world_peer),
        dst_region.window, gin::region_offset(dst_region, dst2),
        src_region.window, gin::region_offset(src_region, src2),
        bytes, ncclGin_None(), ncclGin_None(), ncclCoopThread(),
        ncclGin_None(), cuda::thread_scope_thread, cuda::thread_scope_device,
        ncclGinOptFlagsDefault);
}

__device__ __forceinline__ void two_buffers_putmem_block(
    void* dst1, void* dst2, const void* src1, const void* src2,
    int bytes, int world_peer) {
    __syncthreads();
    if (auto* peer_dst1 = gin::get_lsa_ptr(dst1, world_peer); peer_dst1 != nullptr) {
        auto* peer_dst2 = gin::get_lsa_ptr(dst2, world_peer);
        copy_two_buffers_block(peer_dst1, peer_dst2, src1, src2, bytes);
        __threadfence_system();
        __syncthreads();
        return;
    }

    if (threadIdx.x == 0) {
        network_put_two_buffers(dst1, dst2, src1, src2, bytes, world_peer);
    }
    __syncthreads();
}

}  // namespace flashmask::shmem
