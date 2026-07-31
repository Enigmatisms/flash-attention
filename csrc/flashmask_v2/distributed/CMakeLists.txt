cmake_minimum_required(VERSION 3.9)
find_package(CUDAToolkit REQUIRED)

set(NCCL_INSTALL_DIR "" CACHE PATH "NCCL install dir (include/ + lib/)")
if(NOT NCCL_INSTALL_DIR)
    message(FATAL_ERROR "NCCL_INSTALL_DIR is required for FlashMask distributed")
endif()
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h nccl_device.h
    PATHS "${NCCL_INSTALL_DIR}/include"
    NO_DEFAULT_PATH
)
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS "${NCCL_INSTALL_DIR}/lib" "${NCCL_INSTALL_DIR}/lib64"
    NO_DEFAULT_PATH
)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL headers and libnccl.so are required under ${NCCL_INSTALL_DIR}")
endif()

set(DISTRIBUTED_SOURCES
    overlap_comm.cu
    sep_sr_buffer.cu
    nccl_gin_backend.cu
)

add_library(flashmaskv2_distributed STATIC
    ${DISTRIBUTED_SOURCES}
)

target_compile_definitions(flashmaskv2_distributed PRIVATE
    FLASHMASK_GIN_DEVICE_API=1
)

target_compile_options(flashmaskv2_distributed PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler="-fPIC"
    -Xcompiler="-O3"
    -std=c++17
    --use_fast_math
    -lineinfo
    -DNDEBUG
    --expt-relaxed-constexpr
    >)

target_include_directories(flashmaskv2_distributed PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include          # include cutlass
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(flashmaskv2_distributed PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARY}
)

set_target_properties(flashmaskv2_distributed PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    # the following lines tries to prevent PTX JIT
    # getting rid of the need for loading libnvidia-ptxjitcompiler.so
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "90a-real"
)
