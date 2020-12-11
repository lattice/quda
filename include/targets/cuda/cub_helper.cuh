#pragma once

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#if defined(__clang__) && defined(__CUDA__)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

using namespace quda;

#include <cub/block/block_reduce.cuh>
namespace QudaCub = cub;

// Stuff shared between CUDA and HIP
#include <cub_helper_shared.h>

