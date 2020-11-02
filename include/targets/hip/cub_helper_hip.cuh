#pragma once

using namespace quda;

/**
   @file cub_helper.cuh

   @section Description
   Include this file as opposed to cub headers directly to ensure
   correct compilation with clang and nvrtc
 */

// ensures we use shfl_sync and not shfl when compiling with clang
#include <hipcub/hipcub.hpp>
namespace QudaCub = hipcub;
