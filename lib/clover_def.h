// clover_def.h - clover kernel definitions
//
// See comments in wilson_dslash_def.h

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_XPAY 0
#define DD_PREC 0
#endif

// set options for current iteration

#if (DD_XPAY==0) // no xpay 
#define DD_XPAY_F 
#define DD_PARAM4 DslashParam param
#else            // xpay
#define DD_XPAY_F Xpay
#if (DD_PREC == 0)
#define DD_PARAM4 DslashParam param, double a
#else
#define DD_PARAM4 DslashParam param, float a
#endif
#define DSLASH_XPAY
#endif

#if (DD_PREC==0) // double-precision spinor field
#define DD_PREC_F D
#define DD_PARAM1 double2* out, float *null1
#define DD_PARAM3 const double2* in, const float *null3
#if (defined DIRECT_ACCESS_WILSON_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define SPINORTEX spinorTexDouble
#endif
#if (DD_XPAY==1)  // never used
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ACCUM_DOUBLE
#endif
#define SPINOR_DOUBLE
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
#elif (DD_PREC==1) // single-precision spinor field
#define DD_PREC_F S
#define DD_PARAM1 float4* out, float *null1
#define DD_PARAM3 const float4* in, const float *null3
#ifdef DIRECT_ACCESS_WILSON_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define SPINORTEX spinorTexSingle
#endif
#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexSingle
#define READ_ACCUM READ_ACCUM_SINGLE
#endif
#else            // half-precision spinor field
#define DD_PREC_F H
#ifdef DIRECT_ACCESS_WILSON_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define SPINORTEX spinorTexHalf
#endif
#define DD_PARAM1 short4* out, float *outNorm
#define DD_PARAM3 const short4* in, const float *inNorm
#define WRITE_SPINOR WRITE_SPINOR_SHORT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexHalf
#define READ_ACCUM READ_ACCUM_HALF
#endif
#endif

#if (DD_PREC==0) // double-precision clover term
#define DD_PREC_F D
#define DD_PARAM2 const double2* clover, const float *null
#if (defined DIRECT_ACCESS_CLOVER) || (defined FERMI_NO_DBLE_TEX)
#define CLOVERTEX clover
#define READ_CLOVER READ_CLOVER_DOUBLE
#else
#define CLOVERTEX cloverTexDouble
#define READ_CLOVER READ_CLOVER_DOUBLE_TEX
#endif
#define CLOVER_DOUBLE
#elif (DD_PREC==1) // single-precision clover term
#define DD_PREC_F S
#define DD_PARAM2 const float4* clover, const float *null
#ifdef DIRECT_ACCESS_CLOVER
#define CLOVERTEX clover
#define READ_CLOVER READ_CLOVER_SINGLE
#else
#define CLOVERTEX cloverTexSingle
#define READ_CLOVER READ_CLOVER_SINGLE_TEX
#endif
#else               // half-precision clover term
#define DD_PREC_F H
#define DD_PARAM2 const short4* clover, const float *cloverNorm
#ifdef DIRECT_ACCESS_CLOVER
#define CLOVERTEX clover
#define READ_CLOVER READ_CLOVER_HALF
#else
#define CLOVERTEX cloverTexHalf
#define READ_CLOVER READ_CLOVER_HALF_TEX
#endif
#endif

//#define DD_CONCAT(s,c,x) clover ## s ## c ## x ## Kernel
#define DD_CONCAT(x) clover ## x ## Kernel
#define DD_FUNC(x) DD_CONCAT(x)

// define the kernel
#if !(__COMPUTE_CAPABILITY__ < 130 && DD_PREC == 0)

__global__ void DD_FUNC(DD_XPAY_F)(DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4) {

#ifdef GPU_CLOVER_DIRAC
#include "clover_core.h"
#endif

}

#endif

// clean up

#undef DD_PREC_F
#undef DD_XPAY_F
#undef DD_PARAM1
#undef DD_PARAM2
#undef DD_PARAM3
#undef DD_PARAM4
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_XPAY
#undef READ_SPINOR
#undef SPINORTEX
#undef WRITE_SPINOR
#undef ACCUMTEX
#undef READ_ACCUM
#undef CLOVERTEX
#undef READ_CLOVER
#undef GAUGE_DOUBLE
#undef SPINOR_DOUBLE
#undef CLOVER_DOUBLE

// prepare next set of options, or clean up after final iteration

//#if (DD_XPAY==0)   // xpay variant is not needed
//#undef DD_XPAY
//#define DD_XPAY 1
//#else
//#undef DD_XPAY
//#define DD_XPAY 0

#if (DD_PREC==0)
#undef DD_PREC
#define DD_PREC 1
#elif (DD_PREC==1)
#undef DD_PREC
#define DD_PREC 2
#else

#undef DD_LOOP
#undef DD_XPAY
#undef DD_PREC

#endif // DD_PREC
//#endif // DD_XPAY

#ifdef DD_LOOP
#include "clover_def.h"
#endif
