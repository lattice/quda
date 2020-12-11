#pragma once

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */


using namespace quda;

#include <hipcub/block/block_reduce.cuh>
namespace QudaCub = hipcub;
#include <cub_helper_shared.h>

