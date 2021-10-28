#pragma once

using namespace quda;

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__) && CUDA_VERSION >= 9000
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#ifdef __CUDACC_RTC__
// WAR for CUDA < 11 which prevents the use of cuda_fp16.h in cub with nvrtc
struct __half { };
#endif

#if CUDA_VERSION >= 11000
#include <cub/block/block_reduce.cuh>
#else
#include <cub_legacy/block/block_reduce.cuh>
#endif
