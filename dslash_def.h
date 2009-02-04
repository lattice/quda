// All dslash definitions 

__global__ void
dslashSS12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSS12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSS12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSS12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashSS8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSS8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSS8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSS8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

//////////////////////////

__global__ void
dslashHS12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHS12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHS12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHS12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashHS8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHS8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHS8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHS8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexSingle
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

// All dslash definitions 

__global__ void
dslashSH12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSH12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSH12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSH12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashSH8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSH8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSH8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSH8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

//////////////////////////

__global__ void
dslashHH12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHH12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHH12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHH12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashHH8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHH8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHH8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHH8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#define SPINORTEX spinorTexHalf
#include "dslash_dagger_core.h"
#undef SPINORTEX
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

