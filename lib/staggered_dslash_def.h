// dslash_def.h - Dslash kernel definitions

// There are currently 288 different variants of the Dslash kernel,
// each one characterized by a set of 6 options, where each option can
// take one of several values (3*3*4*2*2*2 = 288).  This file is
// structured so that the C preprocessor loops through all 288
// variants (in a manner resembling a counter), sets the appropriate
// macros, and defines the corresponding functions.
//
// As an example of the function naming conventions, consider
//
// dslashSHS12DaggerXpayKernel(float4* out, int oddBit, float a).
//
// This is a Dslash^dagger kernel where the gauge field is read in single
// precision (S), the spinor field is read in half precision (H), the clover
// term is read in single precision (S), each gauge matrix is reconstructed
// from 12 real numbers, and the result is multiplied by "a" and summed
// with an input vector (Xpay).  More generally, each function name is given
// by the concatenation of the following 6 fields, with "dslash" at the
// beginning and "Kernel" at the end:
//
// DD_PREC_F = D, S, H
// DD_CPREC_F = D, S, H, [blank]; the latter corresponds to plain Wilson
// DD_RECON_F = 12, 8
// DD_DAG_F = Dagger, [blank]
// DD_XPAY_F = Xpay, [blank]

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_DAG 0
#define DD_XPAY 0
#define DD_RECON 0
#define DD_PREC 0
#endif

// set options for current iteration

#define DD_FNAME staggeredDslash

#if (DD_DAG==0) // no dagger
#define DD_DAG_F
#else           // dagger
#define DD_DAG_F Dagger
#endif

#if (DD_XPAY==0) // no xpay 
#define DD_XPAY_F 
#define DD_PARAM5 const DslashParam param
#else            // xpay
#if (DD_PREC == 0)
#define DD_PARAM5 const DslashParam param, const double2 *x, const float *xNorm, const double a
#elif (DD_PREC == 1) 
#define DD_PARAM5 const DslashParam param, const float2 *x, const float *xNorm, const float a
#else
#define DD_PARAM5 const DslashParam param, const short2 *x, const float *xNorm, const float a
#endif
#define DD_XPAY_F Axpy
#define DSLASH_AXPY
#endif

#if (DD_RECON==0) // reconstruct from 8 reals
#define DD_RECON_F 8
#if (DD_PREC==0)
#define DD_PARAM2 const double2 *fatGauge0, const double2 *fatGauge1, const double2* longGauge0, const double2* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_DOUBLE
#elif (DD_PREC==1)
#define DD_PARAM2 const float2 *fatGauge0, const float2 *fatGauge1, const float4* longGauge0, const float4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_SINGLE
#else
#define DD_PARAM2 const short2 *fatGauge0, const short2* fatGauge1, const short4* longGauge0, const short4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_HALF
#endif

#elif (DD_RECON ==1)// reconstruct from 12 reals

#define DD_RECON_F 12
#if (DD_PREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_DOUBLE
#define DD_PARAM2 const double2 *fatGauge0, const double2 *fatGauge1,  const double2* longGauge0, const double2* longGauge1
#elif (DD_PREC==1)
#define DD_PARAM2 const float2 *fatGauge0, const float2 *fatGauge1, const float4* longGauge0, const float4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_SINGLE
#else
#define DD_PARAM2 const short2 *fatGauge0, const short2 *fatGauge1, const short4* longGauge0, const short4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_HALF
#endif

#else //18 reconstruct
#define DD_RECON_F 18
#define RECONSTRUCT_GAUGE_MATRIX(dir, gauge, idx, sign)
#if (DD_PREC==0)
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_18_DOUBLE
#define DD_PARAM2 const double2 *fatGauge0, const double2 *fatGauge1,  const double2* longGauge0, const double2* longGauge1
#elif (DD_PREC==1)
#define DD_PARAM2 const float2 *fatGauge0, const float2 *fatGauge1, const float2* longGauge0, const float2* longGauge1
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_18_SINGLE
#else 
#define DD_PARAM2 const short2 *fatGauge0, const short2 *fatGauge1, const short2* longGauge0, const short2* longGauge1
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_18_HALF
#endif

#endif

#if (DD_PREC==0) // double-precision fields

// gauge field
#define DD_PREC_F D
#ifndef DIRECT_ACCESS_FAT_LINK
#define FATLINK0TEX fatGauge0TexDouble
#define FATLINK1TEX fatGauge1TexDouble
#else
#define FATLINK0TEX fatGauge0
#define FATLINK1TEX fatGauge1
#endif

#ifndef DIRECT_ACCESS_LONG_LINK //longlink access
#define LONGLINK0TEX longGauge0TexDouble
#define LONGLINK1TEX longGauge1TexDouble
#else
#define LONGLINK0TEX longGauge0
#define LONGLINK1TEX longGauge1
#endif

#define GAUGE_DOUBLE

// spinor fields
#define DD_PARAM1 double2* g_out, float *null1
#define DD_PARAM4 const double2* in, const float *null4
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#ifndef DIRECT_ACCESS_SPINOR
#define SPINORTEX spinorTexDouble
#else
#define SPINORTEX in
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_DOUBLE2
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_DOUBLE
#define SPINOR_DOUBLE
#if (DD_XPAY==1)
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ST_ACCUM_DOUBLE
#endif


#elif (DD_PREC==1) // single-precision fields

// gauge fields
#define DD_PREC_F S

#ifndef DIRECT_ACCESS_FAT_LINK
#define FATLINK0TEX fatGauge0TexSingle
#define FATLINK1TEX fatGauge1TexSingle
#else
#define FATLINK0TEX fatGauge0
#define FATLINK1TEX fatGauge1
#endif

#ifndef DIRECT_ACCESS_LONG_LINK //longlink access
#if (DD_RECON ==2)
#define LONGLINK0TEX longGauge0TexSingle_norecon
#define LONGLINK1TEX longGauge1TexSingle_norecon
#else
#define LONGLINK0TEX longGauge0TexSingle
#define LONGLINK1TEX longGauge1TexSingle
#endif
#else
#define LONGLINK0TEX longGauge0
#define LONGLINK1TEX longGauge1
#endif

// spinor fields
#define DD_PARAM1 float2* g_out, float *null1
#define DD_PARAM4 const float2* in, const float *null4
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_SINGLE
#ifndef DIRECT_ACCESS_SPINOR
#define SPINORTEX spinorTexSingle2
#else
#define SPINORTEX in
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_FLOAT2
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#if (DD_XPAY==1)
#define ACCUMTEX accumTexSingle2
#define READ_ACCUM READ_ST_ACCUM_SINGLE
#endif


#else             // half-precision fields

// gauge fields
#define DD_PREC_F H
#define FATLINK0TEX fatGauge0TexHalf
#define FATLINK1TEX fatGauge1TexHalf
#if (DD_RECON ==2)
#define LONGLINK0TEX longGauge0TexHalf_norecon
#define LONGLINK1TEX longGauge1TexHalf_norecon
#else
#define LONGLINK0TEX longGauge0TexHalf
#define LONGLINK1TEX longGauge1TexHalf
#endif

#define READ_SPINOR READ_ST_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_HALF
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_HALF
#define SPINORTEX spinorTexHalf2
#define DD_PARAM1 short2* g_out, float *outNorm
#define DD_PARAM4 const short2* in, const float *inNorm
#define WRITE_SPINOR WRITE_ST_SPINOR_SHORT2
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_HALF
#if (DD_XPAY==1)
#define ACCUMTEX accumTexHalf2
#define READ_ACCUM READ_ST_ACCUM_HALF
#endif

#endif

// only build double precision if supported
#if !(__CUDA_ARCH__ < 130 && DD_PREC == 0) 

#define DD_CONCAT(n,r,d,x) n ## r ## d ## x ## Kernel
#define DD_FUNC(n,r,d,x) DD_CONCAT(n,r,d,x)

// define the kernel
__global__ void	DD_FUNC(DD_FNAME, DD_RECON_F, DD_DAG_F, DD_XPAY_F)
  (DD_PARAM1, DD_PARAM2,  DD_PARAM4, DD_PARAM5) {
#ifdef GPU_STAGGERED_DIRAC
  #include "staggered_dslash_core.h"
#endif
}


#endif




// clean up

#undef DD_PREC_F
#undef DD_RECON_F
#undef DD_DAG_F
#undef DD_XPAY_F
#undef DD_PARAM1
#undef DD_PARAM2
#undef DD_PARAM4
#undef DD_PARAM5
#undef DD_FNAME
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_XPAY
#undef DSLASH_AXPY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef FATLINK0TEX
#undef FATLINK1TEX
#undef LONGLINK0TEX
#undef LONGLINK1TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_SPINOR
#undef READ_AND_SUM_SPINOR
#undef ACCUMTEX
#undef READ_ACCUM
#undef CLOVERTEX
#undef READ_CLOVER
#undef DSLASH_CLOVER
#undef GAUGE_DOUBLE
#undef SPINOR_DOUBLE
#undef CLOVER_DOUBLE
#undef READ_FAT_MATRIX
#undef READ_LONG_MATRIX
#undef READ_1ST_NBR_SPINOR
#undef READ_3RD_NBR_SPINOR

// prepare next set of options, or clean up after final iteration

#if (DD_DAG==0)
#undef DD_DAG
#define DD_DAG 1
#else
#undef DD_DAG
#define DD_DAG 0

#if (DD_XPAY==0)
#undef DD_XPAY
#define DD_XPAY 1
#else
#undef DD_XPAY
#define DD_XPAY 0

#if (DD_RECON==0)
#undef DD_RECON
#define DD_RECON 1
#elif (DD_RECON ==1)
#undef DD_RECON
#define DD_RECON 2
#else
#undef DD_RECON
#define DD_RECON 0

#if (DD_PREC==0)
#undef DD_PREC
#define DD_PREC 1
#elif (DD_PREC==1)
#undef DD_PREC
#define DD_PREC 2
#else
#undef DD_PREC
#define DD_PREC 0

#undef DD_LOOP
#undef DD_DAG
#undef DD_XPAY
#undef DD_RECON
#undef DD_PREC

#endif // DD_PREC
#endif // DD_RECON
#endif // DD_XPAY
#endif // DD_DAG

#ifdef DD_LOOP
#include "staggered_dslash_def.h"
#endif
