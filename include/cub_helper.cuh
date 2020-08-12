#pragma once

using namespace quda;

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */

#if defined(__HIP__)
#include <hipcub/hipcub.hpp>
namespace cub=hipcub;
#include <hipcub/rocprim/block/block_reduce.hpp>
#endif

#if defined(__NVCC__)
#include <cub/block/block_reduce.cuh>
#endif

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__) && CUDA_VERSION >= 9000
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#ifdef __CUDACC_RTC__
// WAR for CUDA < 11 which prevents the use of cuda_fp16.h in cub with nvrtc
struct __half { };
#endif

