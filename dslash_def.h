// All dslash definitions 

__global__ void
dslashSingle12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSingle12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSingle12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSingle12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashSingle8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSingle8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashSingle8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashSingle8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8
#define GAUGE0TEX gauge0TexSingle
#define GAUGE1TEX gauge1TexSingle
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

//////////////////////////

__global__ void
dslashHalf12Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHalf12DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHalf12XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHalf12DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY

__global__ void
dslashHalf8Kernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHalf8DaggerKernel(float4* g_out, int oddBit) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}


#define DSLASH_XPAY

__global__ void
dslashHalf8XpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

__global__ void
dslashHalf8DaggerXpayKernel(float4* g_out, int oddBit, float a) {
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#define GAUGE0TEX gauge0TexHalf
#define GAUGE1TEX gauge1TexHalf
#include "dslash_dagger_core.h"
#undef GAUGE1TEX
#undef GAUGE0TEX
#undef READ_GAUGE_MATRIX
}

#undef DSLASH_XPAY



