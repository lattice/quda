// staggered_dslash_def.h - staggered Dslash kernel definitions
//
// See comments in wilson_dslash_def.h

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_AXPY 0
#define DD_RECON 0
#define DD_PREC 0
#endif

// set options for current iteration

#define DD_FNAME staggeredDslash

#if (DD_AXPY==0) // no axpy
#define DD_AXPY_F 
#else            // axpy
#define DD_AXPY_F Axpy
#define DSLASH_AXPY
#endif

#if (DD_PREC == 0)
#define DD_PARAM_AXPY const double2 *x, const float *xNorm, const double a, const DslashParam param
#elif (DD_PREC == 1) 
#define DD_PARAM_AXPY const float2 *x, const float *xNorm, const float a, const DslashParam param
#else
#define DD_PARAM_AXPY const short2 *x, const float *xNorm, const float a, const DslashParam param
#endif


#if (DD_RECON==0) // reconstruct from 8 reals
#define DD_RECON_F 8

#if (DD_PREC==0) // DOUBLE PRECISION
#define DD_PARAM_GAUGE const double2 *fatGauge0, const double2 *fatGauge1, const double2* longGauge0, const double2* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_DOUBLE

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_DOUBLE2(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_DOUBLE2_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION
#define DD_PARAM_GAUGE const float2 *fatGauge0, const float2 *fatGauge1, const float4* longGauge0, const float4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_FLOAT4(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_FLOAT4_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION
#define DD_PARAM_GAUGE const short2 *fatGauge0, const short2* fatGauge1, const short4* longGauge0, const short4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE

/*#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
#else*/
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
/*#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_SHORT4(LONG, gauge, dir, idx, long_ga_stride)
#else*/
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_8_SHORT4_TEX(LONG, gauge, dir, idx, long_ga_stride)
//#endif // DIRECT_ACCESS_LONG_LINK

#endif // DD_PREC

#elif (DD_RECON ==1)// reconstruct from 12 reals

#define DD_RECON_F 12

#if (DD_PREC==0) // DOUBLE PRECISION
#define DD_PARAM_GAUGE const double2 *fatGauge0, const double2 *fatGauge1,  const double2* longGauge0, const double2* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_DOUBLE2(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_DOUBLE2_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION
#define DD_PARAM_GAUGE const float2 *fatGauge0, const float2 *fatGauge1, const float4* longGauge0, const float4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_FLOAT4(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_FLOAT4_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCESS_LONG_LINK

#else // HALF PRECISION
#define DD_PARAM_GAUGE const short2 *fatGauge0, const short2 *fatGauge1, const short4* longGauge0, const short4* longGauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE

/*#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
#else*/
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
/*#endif // DIRECT_ACCCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_SHORT4(LONG, gauge, dir, idx, long_ga_stride)
#else*/
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_12_SHORT4_TEX(LONG, gauge, dir, idx, long_ga_stride)
									//#endif // DIRECT_ACCCESS_LONG_LINK

#endif // DD_PREC

#else //18 reconstruct
#define DD_RECON_F 18
#define RECONSTRUCT_GAUGE_MATRIX(dir, gauge, idx, sign)

#if (DD_PREC==0) // DOUBLE PRECISION
#define DD_PARAM_GAUGE const double2 *fatGauge0, const double2 *fatGauge1,  const double2* longGauge0, const double2* longGauge1

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_DOUBLE2_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCCESS_LONG_LINK

#elif (DD_PREC==1) // SINGLE PRECISION

#define DD_PARAM_GAUGE const float2 *fatGauge0, const float2 *fatGauge1, const float4* longGauge0, const float4* longGauge1

#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2(FAT, gauge, dir, idx, fat_ga_stride)
#else
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2_TEX(FAT, gauge, dir, idx, fat_ga_stride)
#endif // DIRECT_ACCCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2(LONG, gauge, dir, idx, long_ga_stride)
#else
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_FLOAT2_TEX(LONG, gauge, dir, idx, long_ga_stride)
#endif // DIRECT_ACCCESS_LONG_LINK

#else  // HALF PRECISION

#define DD_PARAM_GAUGE const short2 *fatGauge0, const short2 *fatGauge1, const short4* longGauge0, const short4* longGauge1

/*#ifdef DIRECT_ACCESS_FAT_LINK
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
#else*/
#define READ_FAT_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2_TEX(FAT, gauge, dir, idx, fat_ga_stride); RESCALE2(FAT, fat_ga_max);
									 /*#endif // DIRECT_ACCESS_FAT_LINK
#ifdef DIRECT_ACCESS_LONG_LINK
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2(LONG, gauge, dir, idx, long_ga_stride)
#else*/
#define READ_LONG_MATRIX(gauge, dir, idx) READ_GAUGE_MATRIX_18_SHORT2_TEX(LONG, gauge, dir, idx, long_ga_stride)
									 //#endif // DIRECT_ACCCESS_LONG_LINK

#endif // DD_PREC

#endif // DD_RECON

#if (DD_PREC==0) // double-precision fields

// gauge field
#define DD_PREC_F D
#if (defined DIRECT_ACCESS_FAT_LINK) || (defined FERMI_NO_DBLE_TEX)
#define FATLINK0TEX fatGauge0
#define FATLINK1TEX fatGauge1
#else
#define FATLINK0TEX fatGauge0TexDouble
#define FATLINK1TEX fatGauge1TexDouble
#endif

#if (defined DIRECT_ACCESS_LONG_LINK) || (defined FERMI_NO_DBLE_TEX)
#define LONGLINK0TEX longGauge0
#define LONGLINK1TEX longGauge1
#else
#define LONGLINK0TEX longGauge0TexDouble
#define LONGLINK1TEX longGauge1TexDouble
#endif

#define GAUGE_DOUBLE

// spinor fields
#define DD_PARAM_OUT double2* out, float *null1
#define DD_PARAM_IN const double2* in, const float *null4
#if (defined DIRECT_ACCESS_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define SPINORTEX in
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_DOUBLE
#else
#define SPINORTEX spinorTexDouble
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE_TEX
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_DOUBLE_TEX
#endif
#if (defined DIRECT_ACCESS_INTER) || (defined FERMI_NO_DBLE_TEX)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#define INTERTEX out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_DOUBLE_TEX
#define INTERTEX interTexDouble
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
#if (DD_AXPY==1)
#if (defined DIRECT_ACCESS_ACCUM) || (defined FERMI_NO_DBLE_TEX)
#define ACCUMTEX x
#define READ_ACCUM READ_ST_ACCUM_DOUBLE
#else
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ST_ACCUM_DOUBLE_TEX
#endif
#endif // DD_AXPY


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
#define DD_PARAM_OUT float2* out, float *null1
#define DD_PARAM_IN const float2* in, const float *null4
#ifndef DIRECT_ACCESS_SPINOR
#define SPINORTEX spinorTexSingle2
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE_TEX
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_SINGLE_TEX
#else
#define SPINORTEX in
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_SINGLE
#endif
#if (defined DIRECT_ACCESS_INTER)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR
#define INTERTEX out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_SINGLE_TEX
#define INTERTEX interTexSingle2
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_FLOAT2
#if (DD_AXPY==1)
#if (defined DIRECT_ACCESS_ACCUM)
#define ACCUMTEX x
#define READ_ACCUM READ_ST_ACCUM_SINGLE
#else
#define ACCUMTEX accumTexSingle2
#define READ_ACCUM READ_ST_ACCUM_SINGLE_TEX
#endif
#endif // DD_AXPY


#else             // half-precision fields

// all reads done through texture cache regardless

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

#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_HALF_TEX
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_HALF_TEX
#define SPINORTEX spinorTexHalf2
#define DD_PARAM_OUT short2* out, float *outNorm
#define DD_PARAM_IN const short2* in, const float *inNorm
#if (defined DIRECT_ACCESS_INTER)
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_HALF
#define INTERTEX out
#else
#define READ_AND_SUM_SPINOR READ_AND_SUM_ST_SPINOR_HALF_TEX
#define INTERTEX interTexHalf2
#endif
#define WRITE_SPINOR WRITE_ST_SPINOR_SHORT2
#if (DD_AXPY==1)
#define ACCUMTEX accumTexHalf2
#define READ_ACCUM READ_ST_ACCUM_HALF_TEX
#endif // DD_AXPY

#endif

// only build double precision if supported
#if !(__COMPUTE_CAPABILITY__ < 130 && DD_PREC == 0) 

#define DD_CONCAT(n,r,x) n ## r ## x ## Kernel
#define DD_FUNC(n,r,x) DD_CONCAT(n,r,x)

// define the kernel

template <KernelType kernel_type>
__global__ void	DD_FUNC(DD_FNAME, DD_RECON_F, DD_AXPY_F)
  (DD_PARAM_OUT, DD_PARAM_GAUGE, DD_PARAM_IN, DD_PARAM_AXPY) {
#ifdef GPU_STAGGERED_DIRAC
  #include "staggered_dslash_core.h"
#endif
}

#endif // !(__COMPUTE_CAPABILITY__ < 130 && DD_PREC == 0)


// clean up

#undef DD_PREC_F
#undef DD_RECON_F
#undef DD_AXPY_F
#undef DD_PARAM_OUT
#undef DD_PARAM_GAUGE
#undef DD_PARAM_IN
#undef DD_PARAM_AXPY
#undef DD_FNAME
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_AXPY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef FATLINK0TEX
#undef FATLINK1TEX
#undef LONGLINK0TEX
#undef LONGLINK1TEX
#undef SPINORTEX
#undef WRITE_SPINOR
#undef READ_AND_SUM_SPINOR
#undef INTERTEX
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

#if (DD_AXPY==0)
#undef DD_AXPY
#define DD_AXPY 1
#else
#undef DD_AXPY
#define DD_AXPY 0

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
#undef DD_AXPY
#undef DD_RECON
#undef DD_PREC

#endif // DD_PREC
#endif // DD_RECON
#endif // DD_AXPY

#ifdef DD_LOOP
#include "staggered_dslash_def.h"
#endif
