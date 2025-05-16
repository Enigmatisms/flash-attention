#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace flash {

template <int kBlockM, int kBlockN, int kStages>
class FlashMask {

  bool partially_masked;
  using index_t = uint32_t;
  int32_t const n_block_min;
  int32_t n_block;
  int32_t const m_block;

  index_t const row_offset_flash_mask;
  index_t const row_offset_flash_mask_nblock;

  int32_t* const s_lt_start = nullptr;
  int32_t* const s_lt_end = nullptr;
  int32_t* const s_ut_start = nullptr;
  int32_t* const s_ut_end = nullptr;

  int32_t* const g_lt_start;
  int32_t* const g_lt_end;
  int32_t* const g_lt_start_max;
  int32_t* const g_lt_start_min;
  int32_t* const g_lt_end_max;
  int32_t* const g_lt_end_min;

  int32_t* const g_ut_start;
  int32_t* const g_ut_end;
  int32_t* const g_ut_start_max;
  int32_t* const g_ut_start_min;
  int32_t* const g_ut_end_max;
  int32_t* const g_ut_end_min;

  struct Params {
    int32_t* const smem;
    int const m_block;
    int const bidh;
    int const bidb;
    int32_t const n_block_max;
    int32_t const n_block_min;

    int32_t const seqlen_k;

    // FlashMask
    int const h_flashmask;
    int const h_h_flashmask_ratio;
    
    int * __restrict__ const lt_start_ptr = nullptr;
    int * __restrict__ const lt_end_ptr = nullptr;
    
    int * __restrict__ const ut_start_ptr = nullptr;
    int * __restrict__ const ut_end_ptr = nullptr;
    
    int * __restrict__ const flashmask_maxmin_ptr = nullptr;
    
    int * __restrict__ const lt_start_nblockmax = nullptr;
    int * __restrict__ const lt_start_nblockmin = nullptr;
    
    int * __restrict__ const lt_end_nblockmax = nullptr;
    int * __restrict__ const lt_end_nblockmin = nullptr;
    
    int * __restrict__ const ut_start_nblockmax = nullptr;
    int * __restrict__ const ut_start_nblockmin = nullptr;
    
    int * __restrict__ const ut_end_nblockmax = nullptr;
    int * __restrict__ const ut_end_nblockmin = nullptr;
  };

public:
  CUTLASS_DEVICE
  FlashMask(Params params) :
    n_block_min(params.n_block_min),
    n_block(params.n_block_max - 1),
    m_block(params.m_block),

    row_offset_flash_mask((params.bidb * params.h_flashmask + params.bidh / params.h_h_flashmask_ratio) * params.seqlen_k),
    row_offset_flash_mask_nblock((params.bidb * params.h_flashmask + params.bidh / params.h_h_flashmask_ratio) * cute::ceil_div(params.seqlen_k, kBlockN)),

    s_lt_start(params.smem),
    s_lt_end(params.smem + kBlockN * kStages),
    s_ut_start(params.smem + 2 * kBlockN * kStages),
    s_ut_end(params.smem + 3 * kBlockN * kStages),

    g_lt_start(params.lt_start_ptr + row_offset_flash_mask),
    g_lt_end(params.lt_end_ptr + row_offset_flash_mask),
    g_lt_start_max(params.lt_start_nblockmax + row_offset_flash_mask_nblock),
    g_lt_start_min(params.lt_start_nblockmin + row_offset_flash_mask_nblock),
    g_lt_end_max(params.lt_end_nblockmax + row_offset_flash_mask_nblock),
    g_lt_end_min(params.lt_end_nblockmin + row_offset_flash_mask_nblock),

    g_ut_start(params.ut_start_ptr + row_offset_flash_mask),
    g_ut_end(params.ut_end_ptr + row_offset_flash_mask),
    g_ut_start_max(params.ut_start_nblockmax + row_offset_flash_mask_nblock),
    g_ut_start_min(params.ut_start_nblockmin + row_offset_flash_mask_nblock),
    g_ut_end_max(params.ut_end_nblockmax + row_offset_flash_mask_nblock),
    g_ut_end_min(params.ut_end_nblockmin + row_offset_flash_mask_nblock) {}

  CUTLASS_DEVICE
  int32_t get_n_block(int32_t n_block_min) {
    for(; n_block >= n_block_min; n_block--) {
        // TODO(umiswing): optimize gmem read? although a broadcast should be performed
        int32_t lt_start_max = g_lt_start_max[n_block];
        int32_t lt_start_min = g_lt_start_min[n_block];

        int32_t lt_end_max = g_lt_end_max[n_block];
        int32_t lt_end_min = g_lt_end_min[n_block];

        int32_t ut_start_max = g_ut_start_max[n_block];
        int32_t ut_start_min = g_ut_start_min[n_block];

        int32_t ut_end_max = g_ut_end_max[n_block];
        int32_t ut_end_min = g_ut_end_min[n_block];

        if(m_block * kBlockM >= lt_start_max && (m_block + 1) * kBlockM <= lt_end_min)
            continue;
        if(m_block * kBlockM >= ut_start_max && (m_block + 1) * kBlockM <= ut_end_min)
            continue;
        if(m_block * kBlockM < lt_end_max && (m_block + 1) > lt_start_min)
            partially_masked = true;
        else if(m_block * kBlockM < ut_end_max && (m_block + 1) > ut_start_min)
            partially_masked = true;
        else
            partially_masked = false;
        return n_block;
    }
    return n_block_min - 1;
  }

  CUTLASS_DEVICE
  void load(int const thread_idx, int const threads_num, int index) {
    if(n_block < n_block_min) return;
    if(!partially_masked) return;
    #pragma unroll
    for(int idx = thread_idx; idx < kBlockN; idx += threads_num) {
      s_lt_start[idx + index * kBlockN] = g_lt_start[n_block * kBlockN + idx];
      s_lt_end[idx + index * kBlockN] = g_lt_end[n_block * kBlockN + idx];
      s_ut_start[idx + index * kBlockN] = g_ut_start[n_block * kBlockN + idx];
      s_ut_end[idx + index * kBlockN] = g_ut_end[n_block * kBlockN + idx];
    }
    cutlass::arch::NamedBarrier::sync(threads_num, static_cast<uint32_t>(FwdNamedBarriers::FlashMask));            
  }

  template <typename TiledMma, typename Engine, typename Layout>
  CUTLASS_DEVICE
  void apply(Tensor<Engine, Layout> &tSrS, int const thread_idx) {
      if(n_block < n_block_min) return;
      if(!partially_masked) return;

      // static_assert(!PackGQA);
      // static_assert(!SwapAB);

      auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

      Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = thread_mma.partition_C(cS);
      Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
      Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));

      static constexpr int Row = 0, Col = 1;

      #pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col_idx = get<Col>(tScS_rowcol(m, n)); // col_idx within a block
          if(row_idx >= s_lt_start[col_idx] && row_idx < s_lt_end[col_idx])
              tSrS_rowcol(m, n) = -INFINITY;
          if(row_idx >= s_ut_start[col_idx] && row_idx < s_ut_end[col_idx])
              tSrS_rowcol(m, n) = -INFINITY;
        }
      }
    }

};

namespace flashmask {
  template<int kBlockN>
  __global__
  void scanMaxMinKernel(
      const int *input, int b, int n, int *maxo, int *mino) {
    int bid = threadIdx.y + blockIdx.y * blockDim.y;
    if (bid >= b) {
      return;
    }
    int i_offset = bid * n;
    const int o_n = (n + kBlockN - 1) / kBlockN;
    input = input + i_offset;
    // constexpr int nums = kBlockN / 32;  // ensure N % 32 == 0
    constexpr int nums = (kBlockN + 31) / 32;
    int warpId = blockIdx.x;      // ensure blockDim.x == 32
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int maxv, minv;
    int idx = warpId * kBlockN + tid;
    if (warpId * kBlockN + kBlockN > n) {
      maxv = 0;
      minv = INT_MAX;
  #pragma unroll
      for (int i = 0; i < nums; i++) {
        if (idx < n && lane_id + i * 32 < kBlockN) {
          maxv = max(maxv, input[idx]);
          minv = min(minv, input[idx]);
        }
        idx += 32;
      }
    } else {
      maxv = 0;
      minv = INT_MAX;
  #pragma unroll
      for (int i = 0; i < nums; i++) {
        if(lane_id + i * 32 < kBlockN) {
          maxv = max(maxv, input[idx]);
          minv = min(minv, input[idx]);
          idx += 32;
        }
      }
    }
    __syncwarp();
    maxv = __reduce_max_sync(0xffffffff, maxv);
    minv = __reduce_min_sync(0xffffffff, minv);
    if (tid == 0) {
      maxo[bid * o_n + warpId] = maxv;
      mino[bid * o_n + warpId] = minv;
    }
  }

  template <int kBlockN>
  void scanMaxMinGpu(
      const int *input, int b, int n, int *maxo, int *mino, cudaStream_t stream) {
    // static_assert(kBlockN % 32 == 0, "kBlockN must be a multiple of 32");
    dim3 block(32, 4);
    dim3 grid((n + kBlockN - 1) / kBlockN, (b + 3) / 4);
    scanMaxMinKernel<kBlockN><<<grid, block, 0, stream>>>(input, b, n, maxo, mino);
  }

  template <int kBlockN>
  void prepare_block_maxmin(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.lt_start_ptr == nullptr &&
        params.ut_end_ptr == nullptr) {
      return;
    }
    int *nblock_smask = params.flashmask_maxmin_ptr;
    const int nblock_seqlen = (params.seqlen_k + kBlockN - 1) / kBlockN;
    const int nblock_masklen = params.b * params.h_flashmask * nblock_seqlen;
    params.lt_start_nblockmax = nblock_smask;
    params.lt_start_nblockmin = nblock_smask + nblock_masklen;
    params.ut_end_nblockmax = nblock_smask + 2 * nblock_masklen;
    params.ut_end_nblockmin = nblock_smask + 3 * nblock_masklen;
    params.lt_end_nblockmax = nblock_smask + 4 * nblock_masklen;
    params.lt_end_nblockmin = nblock_smask + 5 * nblock_masklen;
    params.ut_start_nblockmax = nblock_smask + 6 * nblock_masklen;
    params.ut_start_nblockmin = nblock_smask + 7 * nblock_masklen;
    if (params.lt_start_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          static_cast<const int *>(params.lt_start_ptr),
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.lt_start_nblockmax,
          params.lt_start_nblockmin,
          stream);
    } else {
      params.lt_start_nblockmax = nullptr;
      params.lt_start_nblockmin = nullptr;
    }
    if (params.ut_end_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(static_cast<const int *>(params.ut_end_ptr),
                    params.b * params.h_flashmask,
                    params.seqlen_k,
                    params.ut_end_nblockmax,
                    params.ut_end_nblockmin,
                    stream);
    } else {
      params.ut_end_nblockmax = nullptr;
      params.ut_end_nblockmin = nullptr;
    }
    if (params.lt_end_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          static_cast<const int *>(params.lt_end_ptr),
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.lt_end_nblockmax,
          params.lt_end_nblockmin,
          stream);
    } else {
      params.lt_end_nblockmax = nullptr;
      params.lt_end_nblockmin = nullptr;
    }
    if (params.ut_start_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          static_cast<const int *>(params.ut_start_ptr),
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.ut_start_nblockmax,
          params.ut_start_nblockmin,
          stream);
    } else {
      params.ut_start_nblockmax = nullptr;
      params.ut_start_nblockmin = nullptr;
    }
  }
} // namespace flashmask
} // namespace flash
