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
// dslashSHS12DaggerXpayKernel(float4* g_out, int oddBit, float a).
//
// This is a Dslash^dagger kernel where the gauge field is read in single
// precision (S), the spinor field is read in half precision (H), the clover
// term is read in single precision (S), each gauge matrix is reconstructed
// from 12 real numbers, and the result is multiplied by "a" and summed
// with an input vector (Xpay).  More generally, each function name is given
// by the concatenation of the following 6 fields, with "dslash" at the
// beginning and "Kernel" at the end:
//
// DD_GPREC_F = D, S, H
// DD_SPREC_F = D, S, H
// DD_CPREC_F = D, S, H, [blank]; the latter corresponds to plain Wilson
// DD_RECON_F = 12, 8
// DD_DAG_F = Dagger, [blank]
// DD_XPAY_F = Xpay, [blank]

// initialize on first iteration

#ifndef DD3D_LOOP
#define DD3D_LOOP

#define DD3D_XPAY 0
#define DD3D_RECON 0
#define DD3D_GPREC 0
#define DD3D_SPREC 0
#define DD3D_CPREC 0 //
#endif

// set options for current iteration

#if (DD3D_DAG==0) // no dagger
#define DD3D_DAG_F
#else           // dagger
#define DD3D_DAG_F Dagger
#endif

#if (DD3D_XPAY==0) // no xpay 
#define DD3D_XPAY_F 
#define DD3D_PARAM2 int oddBit
#else            // xpay
#define DD3D_XPAY_F Xpay
#if (DD3D_SPREC == 0)
#define DD3D_PARAM2 int oddBit, double a
#else
#define DD3D_PARAM2 int oddBit, float a
#endif
#define DSLASH_XPAY
#endif

#if (DD3D_RECON==0) // reconstruct from 12 reals
#define DD3D_RECON_F 12
#if (DD3D_GPREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_DOUBLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_DOUBLE
#elif (DD3D_GPREC==1)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
#endif
#else             // reconstruct from 8 reals
#define DD3D_RECON_F 8
#if (DD3D_GPREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_DOUBLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_DOUBLE
#elif (DD3D_GPREC==1)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_SINGLE
#else
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#endif
#endif

#if (DD3D_GPREC==0) // double-precision gauge field
#define DD3D_GPREC_F D
#define GAUGE0TEX gauge0TexDouble
#define GAUGE1TEX gauge1TexDouble
#define GAUGE_DOUBLE
#elif (DD3D_GPREC==1) // single-precision gauge field
#define DD3D_GPREC_F S
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#else             // half-precision gauge field
#define DD3D_GPREC_F H
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#endif

#if (DD3D_SPREC==0) // double-precision spinor field
#define DD3D_SPREC_F D
#define DD3D_PARAM1 double2* g_out
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX spinorTexDouble
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
#if (DD3D_XPAY==1)
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ACCUM_DOUBLE
#endif
#elif (DD3D_SPREC==1) // single-precision spinor field
#define DD3D_SPREC_F S
#define DD3D_PARAM1 float4* g_out
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX spinorTexSingle
#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
#if (DD3D_XPAY==1)
#define ACCUMTEX accumTexSingle
#define READ_ACCUM READ_ACCUM_SINGLE
#endif
#else            // half-precision spinor field
#define DD3D_SPREC_F H
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX spinorTexHalf
#define DD3D_PARAM1 short4* g_out, float *c
#define WRITE_SPINOR WRITE_SPINOR_SHORT4
#if (DD3D_XPAY==1)
#define ACCUMTEX accumTexHalf
#define READ_ACCUM READ_ACCUM_HALF
#endif
#endif

#if (DD3D_CPREC==0) // double-precision clover term
#define DD3D_CPREC_F D
#define CLOVERTEX cloverTexDouble
#define READ_CLOVER READ_CLOVER_DOUBLE
#define DSLASH_CLOVER
#define CLOVER_DOUBLE
#elif (DD3D_CPREC==1) // single-precision clover term
#define DD3D_CPREC_F S
#define CLOVERTEX cloverTexSingle
#define READ_CLOVER READ_CLOVER_SINGLE
#define DSLASH_CLOVER
#elif (DD3D_CPREC==2) // half-precision clover term
#define DD3D_CPREC_F H
#define CLOVERTEX cloverTexHalf
#define READ_CLOVER READ_CLOVER_HALF
#define DSLASH_CLOVER
#else             // no clover term
#define DD3D_CPREC_F
#endif

#if !(__CUDA_ARCH__ != 130 && (DD3D_SPREC == 0 || DD3D_GPREC == 0 || DD3D_CPREC == 0))

#define DD3D_CONCAT(g,s,c,r,d,x) dslash3D ## g ## s ## c ## r ## d ## x ## Kernel
#define DD3D_FUNC(g,s,c,r,d,x) DD3D_CONCAT(g,s,c,r,d,x)

// define the kernel

__global__ void
DD3D_FUNC(DD3D_GPREC_F, DD3D_SPREC_F, DD3D_CPREC_F, DD3D_RECON_F, DD3D_DAG_F, DD3D_XPAY_F)(DD3D_PARAM1, DD3D_PARAM2) {
#if DD3D_DAG
#include "dslash_3d_dagger_core.h"
#else
#include "dslash_3d_core.h"
#endif
}

#endif

// clean up

#undef DD3D_GPREC_F
#undef DD3D_SPREC_F
#undef DD3D_CPREC_F
#undef DD3D_RECON_F
#undef DD3D_DAG_F
#undef DD3D_XPAY_F
#undef DD3D_PARAM1
#undef DD3D_PARAM2
#undef DD3D_CONCAT
#undef DD3D_FUNC

#undef DSLASH_XPAY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef GAUGE0TEX
#undef GAUGE1TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_SPINOR
#undef ACCUMTEX
#undef READ_ACCUM
#undef CLOVERTEX
#undef READ_CLOVER
#undef DSLASH_CLOVER
#undef GAUGE_DOUBLE
#undef SPINOR_DOUBLE
#undef CLOVER_DOUBLE

// prepare next set of options, or clean up after final iteration

//#if (DD3D_DAG==0)
//#undef DD3D_DAG
//#define DD3D_DAG 1
//#else
//#undef DD3D_DAG
//#define DD3D_DAG 0

#if (DD3D_XPAY==0)
#undef DD3D_XPAY
#define DD3D_XPAY 1
#else
#undef DD3D_XPAY
#define DD3D_XPAY 0

#if (DD3D_RECON==0)
#undef DD3D_RECON
#define DD3D_RECON 1
#else
#undef DD3D_RECON
#define DD3D_RECON 0

#if (DD3D_GPREC==0)
#undef DD3D_GPREC
#define DD3D_GPREC 1
#elif (DD3D_GPREC==1)
#undef DD3D_GPREC
#define DD3D_GPREC 2
#else
#undef DD3D_GPREC
#define DD3D_GPREC 0

#if (DD3D_SPREC==0)
#undef DD3D_SPREC
#define DD3D_SPREC 1
#elif (DD3D_SPREC==1)
#undef DD3D_SPREC
#define DD3D_SPREC 2
#else

#undef DD3D_SPREC
#define DD3D_SPREC 0

#if (DD3D_CPREC==0)
#undef DD3D_CPREC
#define DD3D_CPREC 1
#elif (DD3D_CPREC==1)
#undef DD3D_CPREC
#define DD3D_CPREC 2
#elif (DD3D_CPREC==2)
#undef DD3D_CPREC
#define DD3D_CPREC 3

#else

#undef DD3D_LOOP
#undef DD3D_DAG
#undef DD3D_XPAY
#undef DD3D_RECON
#undef DD3D_GPREC
#undef DD3D_SPREC
#undef DD3D_CPREC

#endif // DD3D_CPREC
#endif // DD3D_SPREC
#endif // DD3D_GPREC
#endif // DD3D_RECON
#endif // DD3D_XPAY
//#endif // DD3D_DAG

#ifdef DD3D_LOOP
#include "dslash_3d_def.h"
#endif
