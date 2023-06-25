/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <fmha.h>
#include <fmha/utils.h>
#include <fmha/smem_tile.h>
#include <fmha/gmem_tile.h>
#include <fmha/mask.h>
#include <fmha/softmax.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO store Blockmask to shared memory to avoid access to global memory.
template <int BlockmaskGrainRow = 256, int BlockmaskGrainCol = 16, int ThreadBlockTileRow = 256, int ThreadBlockTileCol = 16>
struct Blockmask {

    template<typename Params>
    __device__ Blockmask(const Params &params, int loop_step_idx) :
        // assert BlockmaskGrainRow is divisible by compute_blocksize_row
	// assert BlockmaskGrainCol * ELEMENT_BYTES is divisible by BYTES_PER_LDG
        blockmask_ptr(params.blockmask + loop_step_idx * (ThreadBlockTileRow / BlockmaskGrainRow)  * params.seqlen_q / BlockmaskGrainCol) {
    }

    __device__ int mask_val(int block_row_idx) const {
        return blockmask_ptr[block_row_idx];
    }

    // get the mask_val for the element in (row, col) of a block
    __device__ int mask_val(int row, int col) const {
        return mask_val(col/BlockmaskGrainCol + row/BlockmaskGrainRow*params.seqlen_q/BlockmaskGrainCol);
    }

    __device__ bool predicate(int row, int col) const {
        return mask_val(row, col) != -1;
    }

    __device__ int threadblock_tile_row_idx() const {
        return mask_val(0) / (ThreadBlockTileCol / BlockmaskGrainCol) / 4;
    }

    __device__ bool threadblock_tile_predicate() const {
        for(int i=0;i<ThreadBlockTileRow / BlockmaskGrainRow;i++) {
            for(int j=0;j<ThreadBlockTileCol / BlockmaskGrainCol;i++) {
                if(mask_val(i,j)==-1) return false;
	    }
	}
	return true;
    }

    const int *blockmask_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
