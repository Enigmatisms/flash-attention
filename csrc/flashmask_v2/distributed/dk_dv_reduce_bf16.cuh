#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cutlass/bfloat16.h>
#include <cutlass/array.h>

namespace flashmask {

using bf16 = cutlass::bfloat16_t;
using bf16x4 = cutlass::Array<bf16, 4>;

// bf16x4 and float4 conversion, so that we can use fp32 accumulation
__device__ __forceinline__ float4 to_float4(bf16x4 in) {
    return {
        static_cast<float>(in[0]),
        static_cast<float>(in[1]),
        static_cast<float>(in[2]),
        static_cast<float>(in[3])
    };
}

__device__ __forceinline__ bf16x4 to_bf16x4(float4 in) {
    bf16x4 out;
    out[0] = static_cast<bf16>(in.x);
    out[1] = static_cast<bf16>(in.y);
    out[2] = static_cast<bf16>(in.z);
    out[3] = static_cast<bf16>(in.w);
    return out;
}

/**
 * Note that the input buffer and output buffer has shape mismatch:
 * @param dx_send_recv the shape is (B, S_local * num_chunks, H, D)
 * @param dx_accum the shape is (B, S_local, H, D)
 *
 * So we need to calculate different batch stride for input and output
 *
 * Cross-segment accumulation stays in the fp32 scratch (dx_f32); only the last
 * segment rounds to bf16, into dx_out. This matches the single rounding of the
 * non-overlap reduce. For a single segment (is_first && is_last) the scratch is
 * never touched and needs no allocation.
*/
template <int num_chunks = 4, bool is_first = true, bool is_last = true>
__global__ __launch_bounds__(128, 8)
void ReducedKdVKernel(
    const bf16* __restrict__ dk_recv,
    const bf16* __restrict__ dv_recv,
    float* __restrict__ dk_f32,
    float* __restrict__ dv_f32,
    bf16* __restrict__ dk_out,
    bf16* __restrict__ dv_out,
    const int num_tasks_per_batch       // S_chunk * H * D / 512
) {
    static constexpr int elem_per_block = 512;
    const int b = blockIdx.y;           // batch

    const int64_t elem_per_chunk = int64_t(num_tasks_per_batch) * elem_per_block;    // chunk stride
    const int64_t b_offset_accum = b * elem_per_chunk;
    const int64_t b_offset_sr = b_offset_accum * num_chunks;

    // task offset is small_chunk offset + thread offset
    auto reduce_op = [&](
        const bf16* const __restrict__ src_recv,
        float* const __restrict__ dst_f32,
        bf16* const __restrict__ dst_out, int64_t task_offset
    ) {
        const int64_t accum_offset = b_offset_accum + task_offset;
        float4 acc = make_float4(0, 0, 0, 0);
        if constexpr (!is_first) {
            acc = *reinterpret_cast<const float4*>(dst_f32 + accum_offset);
        }
        // use higher precision to do the reduce
        const int64_t base_offset = b_offset_sr + task_offset;
        #pragma unroll
        for (int c = 0; c < num_chunks; ++c) {
            float4 temp_v = to_float4(
                *reinterpret_cast<const bf16x4*>(src_recv + c * elem_per_chunk + base_offset)
            );
            acc.x += temp_v.x;
            acc.y += temp_v.y;
            acc.z += temp_v.z;
            acc.w += temp_v.w;
        }

        if constexpr (is_last) {
            *reinterpret_cast<bf16x4*>(dst_out + accum_offset) = to_bf16x4(acc);
        } else {
            *reinterpret_cast<float4*>(dst_f32 + accum_offset) = acc;
        }
    };

    for (int task_idx = blockIdx.x; task_idx < num_tasks_per_batch; task_idx += gridDim.x) {
        const int64_t task_offset = int64_t(task_idx) * elem_per_block + 4 * threadIdx.x;

        reduce_op(dk_recv, dk_f32, dk_out, task_offset);
        if (dv_recv != nullptr) reduce_op(dv_recv, dv_f32, dv_out, task_offset);
    }
}

#define ReduceKernelLaunch(_num_chunk, _is_first, _is_last)                              \
    ReducedKdVKernel<_num_chunk, _is_first, _is_last><<<grid, 128, 0, stream>>>(         \
        dk_recv, dv_recv, dk_f32, dv_f32, dk_out, dv_out, num_tasks_per_chunk)

#define ChunkDipatchKernelLaunch(num_chunk, is_first, is_last)                           \
    switch (num_chunk) {                                                                \
        case 4: { ReduceKernelLaunch(4, is_first, is_last); break; }                    \
        case 2: { ReduceKernelLaunch(2, is_first, is_last); break; }                    \
        case 8: { ReduceKernelLaunch(8, is_first, is_last); break; }                    \
        case 1: { ReduceKernelLaunch(1, is_first, is_last); break; }                    \
    default:                                                                            \
        throw std::invalid_argument(                                                    \
            "[FlashMask Overlap] num_chunks must be one of {1, 2, 4, 8}, got: "         \
            + std::to_string(num_chunk));                                               \
    }

/**
 * This function calls the dK, dV reduce kernel.
 * @param dv_recv nullptr skips the dV half entirely (kv_shared merges dV into dK,
 *  leaving dv_f32 / dv_out untouched).
 * @param is_first The first segment overwrites the fp32 scratch instead of
 *  accumulating into it, so the scratch never needs to be zeroed.
 * @param is_last The last segment rounds the fp32 sum into the bf16 output
 *  (dk_out, dv_out); the others keep it in the scratch (dk_f32, dv_f32).
 *  With a single segment both are true and the scratch is unused.
*/
void launch_dk_dv_reduce(
    const bf16* dk_recv,
    const bf16* dv_recv,
    float* dk_f32, float* dv_f32,
    bf16* dk_out, bf16* dv_out,
    int B, int S_chunk, int H, int D,
    int num_chunks, bool is_first, bool is_last, cudaStream_t stream
) {
    // 128 threads, each reduces 4 bf16
    static constexpr int elem_per_block = 512;
    size_t elem_per_chunk = static_cast<size_t>(S_chunk) * H * D;
    // a typical value: 8192 * 8 * 128 / 512 = 16384
    int num_tasks_per_chunk = elem_per_chunk / elem_per_block;

    // typically, B = 1, so we have 2048 CTAs --> 16 CTAs per SM = 128 SMs
    // the reduce speed shouldn't be a bottleneck, so it's OK to allocate more SMs
    dim3 grid(std::max(2048 / B, 128), B);

    if (is_first) {
        if (is_last) { ChunkDipatchKernelLaunch(num_chunks, true, true); }
        else { ChunkDipatchKernelLaunch(num_chunks, true, false); }
    } else {
        if (is_last) { ChunkDipatchKernelLaunch(num_chunks, false, true); }
        else { ChunkDipatchKernelLaunch(num_chunks, false, false); }
    }
}

#undef ChunkDipatchKernelLaunch
#undef ReduceKernelLaunch

}   // namespace flashmask