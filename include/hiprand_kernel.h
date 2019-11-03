// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPRAND_KERNEL_H_
#define HIPRAND_KERNEL_H_

#ifndef QUALIFIERS
#define QUALIFIERS __forceinline__ __device__ __host__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>
#include <hiprand.h>

/** \addtogroup hipranddevice
 *
 *  @{
 */

 /**
 * \def HIPRAND_PHILOX4x32_DEFAULT_SEED
 * \brief Default seed for PHILOX4x32 PRNG.
 */
#define HIPRAND_PHILOX4x32_DEFAULT_SEED 0ULL
 /**
 * \def HIPRAND_XORWOW_DEFAULT_SEED
 * \brief Default seed for XORWOW PRNG.
 */
#define HIPRAND_XORWOW_DEFAULT_SEED 0ULL
 /**
 * \def HIPRAND_MRG32K3A_DEFAULT_SEED
 * \brief Default seed for MRG32K3A PRNG.
 */
#define HIPRAND_MRG32K3A_DEFAULT_SEED 12345ULL
/** @} */ // end of group hipranddevice

#ifdef __HIP_PLATFORM_HCC__
#include "hiprand_kernel_hcc.h"
#else
#include "hiprand_kernel_nvcc.h"
#endif

#endif // HIPRAND_KERNEL_H_
