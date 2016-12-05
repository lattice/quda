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
#else            // xpay
#define DD_XPAY_F Xpay
#define DSLASH_XPAY
#endif

#if (DD_PREC==0) // double-precision spinor field
#define DD_PREC_F D
#define FLOATN double2
#if (defined DIRECT_ACCESS_WILSON_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define SPINORTEX param.in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexDouble
#endif // USE_TEXTURE_OBJECTS
#endif
#if (DD_XPAY==1)  // never used
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ACCUM_DOUBLE
#endif
#define SPINOR_DOUBLE
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
#elif (DD_PREC==1) // single-precision spinor field
#define DD_PREC_F S
#define FLOATN float4
#ifdef DIRECT_ACCESS_WILSON_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define SPINORTEX param.in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexSingle
#endif // USE_TEXTURE_OBJECTS
#endif
#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexSingle
#define READ_ACCUM READ_ACCUM_SINGLE
#endif
#else            // half-precision spinor field
#define DD_PREC_F H
#define FLOATN short4
#ifdef DIRECT_ACCESS_WILSON_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define SPINORTEX param.in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#ifdef USE_TEXTURE_OBJECTS
#define SPINORTEX param.inTex
#else
#define SPINORTEX spinorTexHalf
#endif // USE_TEXTURE_OBJECTS
#endif
#define WRITE_SPINOR WRITE_SPINOR_SHORT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexHalf
#define READ_ACCUM READ_ACCUM_HALF
#endif
#endif

#if (DD_PREC==0) // double-precision clover term
#define DD_PREC_F D
#if (defined DIRECT_ACCESS_CLOVER) || (defined FERMI_NO_DBLE_TEX)
#define CLOVERTEX param.clover
#define READ_CLOVER READ_CLOVER_DOUBLE
#else
#ifdef USE_TEXTURE_OBJECTS
#define CLOVERTEX (param.cloverTex)
#else
#define CLOVERTEX cloverTexDouble
#endif
#define READ_CLOVER READ_CLOVER_DOUBLE_TEX
#endif

#define CLOVER_DOUBLE
#elif (DD_PREC==1) // single-precision clover term
#define DD_PREC_F S
#ifdef DIRECT_ACCESS_CLOVER
#define CLOVERTEX param.clover
#define READ_CLOVER READ_CLOVER_SINGLE
#else
#ifdef USE_TEXTURE_OBJECTS
#define CLOVERTEX (param.cloverTex)
#else
#define CLOVERTEX cloverTexSingle
#endif
#define READ_CLOVER READ_CLOVER_SINGLE_TEX
#endif
#else               // half-precision clover term
#define DD_PREC_F H
#ifdef DIRECT_ACCESS_CLOVER
#define CLOVERTEX param.clover
#define CLOVERTEXNORM param.cloverNorm
#define READ_CLOVER READ_CLOVER_HALF
#else
#ifdef USE_TEXTURE_OBJECTS
#define CLOVERTEX (param.cloverTex)
#define CLOVERTEXNORM (param.cloverNormTex)
#else
#define CLOVERTEX cloverTexHalf
#define CLOVERTEXNORM cloverTexNorm
#endif
#define READ_CLOVER READ_CLOVER_HALF_TEX
#endif
#endif

//#define DD_CONCAT(s,c,x) clover ## s ## c ## x ## Kernel
#define DD_CONCAT(p,x) clover ## p ## x ## Kernel
#define DD_FUNC(p,x) DD_CONCAT(p,x)

// define the kernel

__global__ void DD_FUNC(DD_PREC_F,DD_XPAY_F)(const DslashParam param) {

#ifdef GPU_CLOVER_DIRAC
#include "clover_core.h"
#endif

}

// clean up

#undef FLOATN
#undef DD_PREC_F
#undef DD_XPAY_F
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
