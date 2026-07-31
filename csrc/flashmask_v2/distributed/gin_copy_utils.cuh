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
        const auto& src_region = gin::find_region(src1);
        const auto& dst_region = gin::find_region(dst1);
        auto transport = gin::make_gin();
        transport.get(
            gin::network_team(), gin::team_peer(world_peer),
            src_region.window, gin::region_offset(src_region, src1),
            dst_region.window, gin::region_offset(dst_region, dst1),
            bytes, ncclCoopThread(), ncclGin_None(),
            ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
        transport.get(
            gin::network_team(), gin::team_peer(world_peer),
            src_region.window, gin::region_offset(src_region, src2),
            dst_region.window, gin::region_offset(dst_region, dst2),
            bytes, ncclCoopThread(), ncclGin_None(),
            ncclGinOptFlagsDefault, ncclGin_SegmentDevice());
        transport.flush(ncclCoopThread());
    }
    __syncthreads();
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
    __syncthreads();
}

}  // namespace flashmask::shmem
