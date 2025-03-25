#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM192
template<>
void run_mla_bwd_<90, cutlass::bfloat16_t, 192, false>(Flash_bwd_params &params, cudaStream_t stream) {
  printf("\nwsm debug run_mla_bwd_()\n");
  run_mha_bwd_hdim192_hdimv128<90, cutlass::bfloat16_t, false>(params, stream);
}
template<>
void run_mla_bwd_<90, cutlass::bfloat16_t, 192, true>(Flash_bwd_params &params, cudaStream_t stream) {
  printf("\nwsm debug run_mla_bwd_()\n");
  run_mha_bwd_hdim192_hdimv128<90, cutlass::bfloat16_t, true>(params, stream);
}
#endif
