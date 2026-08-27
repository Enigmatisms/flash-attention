/**
 * Separated S-R buffer for a2a based all-gather 
*/
#pragma once

#include "nccl_gin_backend.cuh"
#include <stdexcept>
#include <cstring>
#include <vector>

namespace flashmask {

// RAII object of separate send/recv buffer

#define CLAMP_IDX(idx) (idx % _capacity)
template <typename KVType>
class SepSRBuffer {
using SemaphoreType = int64_t;

private:
    gin::Context& _context;
    KVType* _dk_data;
    KVType* _dv_data;
    SemaphoreType* _semaphores;
    bool _allocated;
    int _team;

    // offset to the recv buffer (2 * chunks_per_seg * k_numel)
    size_t _buf_offset;
    int _semaphore_size;
    size_t _single_k_numel;    // allocated capacity per-K (B * S_local * H * D)
    int _chunks_per_seg;       // chunks per segment (layout-defining)

    const int _capacity;
    std::vector<cudaEvent_t> _empty_states; 

    // this object cannot be moved or copied
    SepSRBuffer(const SepSRBuffer&) = delete;
    SepSRBuffer(SepSRBuffer&&) = delete;
    SepSRBuffer& operator=(const SepSRBuffer&) = delete;
    SepSRBuffer& operator=(SepSRBuffer&&) = delete;
public:
    explicit SepSRBuffer(
        gin::Context& context,
        size_t single_k_numel,
        int semaphore_size,
        int chunks_per_seg,
        int buffer_capacity = 1,
        int team = 0
    );

    void team_bar() const {
        gin::barrier(_context);
    }

    void team_bar_on_stream(cudaStream_t stream) const {
        gin::barrier_on_stream(_context, stream);
    }

    void release();
    void release_for_realloc() { release(); }

    ~SepSRBuffer() noexcept;

    // [K_send, V_send] --> buf_offset size, therefore 2 * buf_offset is the double buffer offset
    inline KVType* k_send(int seg_idx) const { return _dk_data + CLAMP_IDX(seg_idx) * 2 * _buf_offset; }
    inline KVType* v_send(int seg_idx) const { return _dv_data + CLAMP_IDX(seg_idx) * 2 * _buf_offset; }
    inline KVType* k_recv(int seg_idx) const { return _dk_data + (CLAMP_IDX(seg_idx) * 2 + 1) * _buf_offset; }
    inline KVType* v_recv(int seg_idx) const { return _dv_data + (CLAMP_IDX(seg_idx) * 2 + 1) * _buf_offset; }
    inline SemaphoreType* semaphores(int seg_idx) const { return _semaphores + CLAMP_IDX(seg_idx) * _semaphore_size; }

    // Zero every recv slot; the sparse RS puts leave holes the reduce would otherwise read.
    void initialize_buffer(int self_rank);

    void wait_buffer(int seg_idx, cudaStream_t stream) {
        cudaStreamWaitEvent(stream, _empty_states[CLAMP_IDX(seg_idx)]);
    }

    void release_buffer(int seg_idx, cudaStream_t stream) {
        cudaEventRecord(_empty_states[CLAMP_IDX(seg_idx)], stream);
    }

    // clear recv buffer (so that reduce won't op on dirty data)
    void zero_recv_buf(int seg_idx, cudaStream_t comm_stream);

    inline bool is_valid() const noexcept {
        return _allocated && _dk_data && _dv_data && _semaphores && _team != -1;
    }

    size_t capacity() const noexcept {
        return _single_k_numel;
    }

    int get_chunks_per_seg() const noexcept {
        return _chunks_per_seg;
    }

    int team() const noexcept {
        return _team;
    }
};

#undef CLAMP_IDX

}   // namespace flashmask