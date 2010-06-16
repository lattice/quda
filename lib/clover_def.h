// clover_def.h - clover kernel definitions

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_XPAY 0
#define DD_SPREC 0
#define DD_CPREC 0
#endif

// set options for current iteration

#if (DD_XPAY==0) // no xpay 
#define DD_XPAY_F 
#define DD_PARAM2 int oddBit
#else            // xpay
#define DD_XPAY_F Xpay
#if (DD_SPREC == 0)
#define DD_PARAM2 int oddBit, double a
#else
#define DD_PARAM2 int oddBit, float a
#endif
#define DSLASH_XPAY
#endif

#if (DD_SPREC==0) // double-precision spinor field
#define DD_SPREC_F D
#define DD_PARAM1 double2* g_out
#define READ_SPINOR READ_SPINOR_DOUBLE
#define SPINORTEX spinorTexDouble
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
#if (DD_XPAY==1)
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ACCUM_DOUBLE
#endif
#elif (DD_SPREC==1) // single-precision spinor field
#define DD_SPREC_F S
#define DD_PARAM1 float4* g_out
#define READ_SPINOR READ_SPINOR_SINGLE
#define SPINORTEX spinorTexSingle
#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexSingle
#define READ_ACCUM READ_ACCUM_SINGLE
#endif
#else            // half-precision spinor field
#define DD_SPREC_F H
#define READ_SPINOR READ_SPINOR_HALF
#define SPINORTEX spinorTexHalf
#define DD_PARAM1 short4* g_out, float *c
#define WRITE_SPINOR WRITE_SPINOR_SHORT4
#if (DD_XPAY==1)
#define ACCUMTEX accumTexHalf
#define READ_ACCUM READ_ACCUM_HALF
#endif
#endif

#if (DD_CPREC==0) // double-precision clover term
#define DD_CPREC_F D
#define CLOVERTEX cloverTexDouble
#define READ_CLOVER READ_CLOVER_DOUBLE
#define CLOVER_DOUBLE
#elif (DD_CPREC==1) // single-precision clover term
#define DD_CPREC_F S
#define CLOVERTEX cloverTexSingle
#define READ_CLOVER READ_CLOVER_SINGLE
#else               // half-precision clover term
#define DD_CPREC_F H
#define CLOVERTEX cloverTexHalf
#define READ_CLOVER READ_CLOVER_HALF
#endif

#define DD_CONCAT(s,c,x) clover ## s ## c ## x ## Kernel
#define DD_FUNC(s,c,x) DD_CONCAT(s,c,x)

// define the kernel

#if !(__CUDA_ARCH__ < 130 && (DD_SPREC == 0 || DD_CPREC == 0))

__global__ void
DD_FUNC(DD_SPREC_F, DD_CPREC_F, DD_XPAY_F)(DD_PARAM1, DD_PARAM2) {
#include "clover_core.h"
}

#endif

// clean up

#undef DD_SPREC_F
#undef DD_CPREC_F
#undef DD_XPAY_F
#undef DD_PARAM1
#undef DD_PARAM2
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

#if (DD_SPREC==0)
#undef DD_SPREC
#define DD_SPREC 1
#elif (DD_SPREC==1)
#undef DD_SPREC
#define DD_SPREC 2
#else
#undef DD_SPREC
#define DD_SPREC 0

#if (DD_CPREC==0)
#undef DD_CPREC
#define DD_CPREC 1
#elif (DD_CPREC==1)
#undef DD_CPREC
#define DD_CPREC 2
#else

#undef DD_LOOP
#undef DD_XPAY
#undef DD_SPREC
#undef DD_CPREC

#endif // DD_CPREC
#endif // DD_SPREC
//#endif // DD_XPAY

#ifdef DD_LOOP
#include "clover_def.h"
#endif
