#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <util_quda.h>

#define REDUCE_THREADS 128
#define REDUCE_MAX_BLOCKS 64

#define REDUCE_DOUBLE 64
#define REDUCE_KAHAN 32

#if (__CUDA_ARCH__ == 130)
#define REDUCE_TYPE REDUCE_DOUBLE
#define QudaSumFloat double
#define QudaSumComplex cuDoubleComplex
#define QudaSumFloat3 double3
#else
#define REDUCE_TYPE REDUCE_KAHAN
#define QudaSumFloat float
#define QudaSumComplex cuComplex
#define QudaSumFloat3 float3
#endif

#define RECONSTRUCT_HALF_SPINOR(a, texHalf, texNorm, length)		\
  float4 a##0 = tex1Dfetch(texHalf, i + 0*length);			\
  float4 a##1 = tex1Dfetch(texHalf, i + 1*length);			\
  float4 a##2 = tex1Dfetch(texHalf, i + 2*length);			\
  float4 a##3 = tex1Dfetch(texHalf, i + 3*length);			\
  float4 a##4 = tex1Dfetch(texHalf, i + 4*length);			\
  float4 a##5 = tex1Dfetch(texHalf, i + 5*length);			\
  {float b = tex1Dfetch(texNorm, i);					\
  (a##0).x *= b; (a##0).y *= b; (a##0).z *= b; (a##0).w *= b;		\
  (a##1).x *= b; (a##1).y *= b; (a##1).z *= b; (a##1).w *= b;		\
  (a##2).x *= b; (a##2).y *= b; (a##2).z *= b; (a##2).w *= b;		\
  (a##3).x *= b; (a##3).y *= b; (a##3).z *= b; (a##3).w *= b;		\
  (a##4).x *= b; (a##4).y *= b; (a##4).z *= b; (a##4).w *= b;		\
  (a##5).x *= b; (a##5).y *= b; (a##5).z *= b; (a##5).w *= b;}

#define CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, n, a, length)		\
  {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));			\
  float c1 = fmaxf(fabsf((a##0).z), fabsf((a##0).w));			\
  float c2 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));			\
  float c3 = fmaxf(fabsf((a##1).z), fabsf((a##1).w));			\
  float c4 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));			\
  float c5 = fmaxf(fabsf((a##2).z), fabsf((a##2).w));			\
  float c6 = fmaxf(fabsf((a##3).x), fabsf((a##3).y));			\
  float c7 = fmaxf(fabsf((a##3).z), fabsf((a##3).w));			\
  float c8 = fmaxf(fabsf((a##4).x), fabsf((a##4).y));			\
  float c9 = fmaxf(fabsf((a##4).z), fabsf((a##4).w));			\
  float c10 = fmaxf(fabsf((a##5).x), fabsf((a##5).y));			\
  float c11 = fmaxf(fabsf((a##5).z), fabsf((a##5).w));			\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); c2 = fmaxf(c4, c5);		\
  c3 = fmaxf(c6, c7); c4 = fmaxf(c8, c9); c5 = fmaxf(c10, c11);		\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); c2 = fmaxf(c4, c5);		\
  c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);				\
  n[i] = c0;								\
  float C = __fdividef(MAX_SHORT, c0);					\
  h[i+0*length] = make_short4((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y), \
			      (short)(C*(float)(a##0).z), (short)(C*(float)(a##0).w)); \
  h[i+1*length] = make_short4((short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y), \
			      (short)(C*(float)(a##1).z), (short)(C*(float)(a##1).w)); \
  h[i+2*length] = make_short4((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y), \
			      (short)(C*(float)(a##2).z), (short)(C*(float)(a##2).w)); \
  h[i+3*length] = make_short4((short)(C*(float)(a##3).x), (short)(C*(float)(a##3).y), \
			      (short)(C*(float)(a##3).z), (short)(C*(float)(a##3).w)); \
  h[i+4*length] = make_short4((short)(C*(float)(a##4).x), (short)(C*(float)(a##4).y), \
			      (short)(C*(float)(a##4).z), (short)(C*(float)(a##4).w)); \
  h[i+5*length] = make_short4((short)(C*(float)(a##5).x), (short)(C*(float)(a##5).y),	\
			      (short)(C*(float)(a##5).z), (short)(C*(float)(a##5).w));}

#define CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, n, a, length)		\
  {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));			\
  float c1 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));		     	\
  float c2 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));			\
  float c3 = fmaxf(fabsf((a##3).x), fabsf((a##3).y));			\
  float c4 = fmaxf(fabsf((a##4).x), fabsf((a##4).y));			\
  float c5 = fmaxf(fabsf((a##5).x), fabsf((a##5).y));			\
  float c6 = fmaxf(fabsf((a##6).x), fabsf((a##6).y));			\
  float c7 = fmaxf(fabsf((a##7).x), fabsf((a##7).y));			\
  float c8 = fmaxf(fabsf((a##8).x), fabsf((a##8).y));			\
  float c9 = fmaxf(fabsf((a##9).x), fabsf((a##9).y));			\
  float c10 = fmaxf(fabsf((a##10).x), fabsf((a##10).y));		\
  float c11 = fmaxf(fabsf((a##11).x), fabsf((a##11).y));		\
  c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3);  c2 = fmaxf(c4, c5); c3 = fmaxf(c6, c7); \
  c4 = fmaxf(c8, c9); c5 = fmaxf(c10, c11); c0 = fmaxf(c0, c1); c1 = fmaxf(c2, c3); \
  c2 = fmaxf(c4, c5); c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);		\
  n[i] = c0;								\
  float C = __fdividef(MAX_SHORT, c0);					\
  h[i+0*length] = make_short4((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y), \
			      (short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y)); \
  h[i+1*length] = make_short4((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y), \
			      (short)(C*(float)(a##3).x), (short)(C*(float)(a##3).y)); \
  h[i+2*length] = make_short4((short)(C*(float)(a##4).x), (short)(C*(float)(a##4).y), \
			      (short)(C*(float)(a##5).x), (short)(C*(float)(a##5).y)); \
  h[i+3*length] = make_short4((short)(C*(float)(a##6).x), (short)(C*(float)(a##6).y), \
			      (short)(C*(float)(a##7).x), (short)(C*(float)(a##7).y)); \
  h[i+4*length] = make_short4((short)(C*(float)(a##8).x), (short)(C*(float)(a##8).y), \
			      (short)(C*(float)(a##9).x), (short)(C*(float)(a##9).y)); \
  h[i+5*length] = make_short4((short)(C*(float)(a##10).x), (short)(C*(float)(a##10).y),	\
			      (short)(C*(float)(a##11).x), (short)(C*(float)(a##11).y));}

#define SUM_FLOAT4(sum, a)			\
  float sum = a.x + a.y + a.z + a.w;

#define REAL_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w

#define IMAG_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z

#define AX_FLOAT4(a, X)				\
  X.x *= a; X.y *= a; X.z *= a; X.w *= a;

#define XPY_FLOAT4(X, Y)		     \
  Y.x += X.x; Y.y += X.y; Y.z += X.z; Y.w += X.w;

#define XMY_FLOAT4(X, Y)		     \
  Y.x = X.x - Y.x; Y.y = X.y - X.y; Y.z = X.z - Y.z; Y.w = X.w - Y.w;

#define MXPY_FLOAT4(X, Y)		     \
  Y.x -= X.x; Y.y -= X.y; Y.z -= X.z; Y.w -= X.w;

#define AXPY_FLOAT4(a, X, Y)		     \
  Y.x += a*X.x;	Y.y += a*X.y;		     \
  Y.z += a*X.z;	Y.w += a*X.w;

#define AXPBY_FLOAT4(a, X, b, Y)		\
  Y.x = a*X.x + b*Y.x; Y.y = a*X.y + b*Y.y;	\
  Y.z = a*X.z + b*Y.z; Y.w = a*X.w + b*Y.w;

#define XPAY_FLOAT4(X, a, Y)			     \
  Y.x = X.x + a*Y.x; Y.y = X.y + a*Y.y;		     \
  Y.z = X.z + a*Y.z; Y.w = X.w + a*Y.w;

#define CAXPY_FLOAT4(a, X, Y) \
  Y.x += a.x*X.x - a.y*X.y;   \
  Y.y += a.y*X.x + a.x*X.y;   \
  Y.z += a.x*X.z - a.y*X.w;   \
  Y.w += a.y*X.z + a.x*X.w;

#define CMAXPY_FLOAT4(a, X, Y)			\
  Y.x -= (a.x*X.x - a.y*X.y);			\
  Y.y -= (a.y*X.x + a.x*X.y);			\
  Y.z -= (a.x*X.z - a.y*X.w);			\
  Y.w -= (a.y*X.z + a.x*X.w);

#define CAXPBY_FLOAT4(a, X, b, Y)		 \
  Y.x = a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Y.y = a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Y.z = a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Y.w = a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

#define CXPAYPBZ_FLOAT4(X, a, Y, b, Z)		\
  {float2 z;					       \
  z.x = X.x + a.x*Y.x - a.y*Y.y + b.x*Z.x - b.y*Z.y;   \
  z.y = X.y + a.y*Y.x + a.x*Y.y + b.y*Z.x + b.x*Z.y;   \
  Z.x = z.x; Z.y = z.y;				       \
  z.x = X.z + a.x*Y.z - a.y*Y.w + b.x*Z.z - b.y*Z.w;   \
  z.y = X.w + a.y*Y.z + a.x*Y.w + b.y*Z.z + b.x*Z.w;   \
  Z.z = z.x; Z.w = z.y;}			       \  

#define CAXPBYPZ_FLOAT4(a, X, b, Y, Z)		  \
  Z.x += a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Z.y += a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Z.z += a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Z.w += a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf1;
texture<float, 1, cudaReadModeElementType> texNorm1;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf2;
texture<float, 1, cudaReadModeElementType> texNorm2;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf3;
texture<float, 1, cudaReadModeElementType> texNorm3;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf4;
texture<float, 1, cudaReadModeElementType> texNorm4;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> texHalf5;
texture<float, 1, cudaReadModeElementType> texNorm5;

#if (__CUDA_ARCH__ == 130)
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
    int4 v = tex1Dfetch(t,i);
    return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#endif

inline void checkSpinor(ParitySpinor &a, ParitySpinor &b) {
  if (a.precision != b.precision) {
    printf("checkSpinor error, precisions do not match: %d %d\n", a.precision, b.precision);
    exit(-1);
  }

  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// For kernels with precision conversion built in
inline void checkSpinorLength(ParitySpinor &a, ParitySpinor &b) {
  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// cuda's floating point format, IEEE-754, represents the floating point
// zero as 4 zero bytes
void zeroCuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(double));
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(float));
  } else {
    cudaMemset(a.spinor, 0, a.length*sizeof(short));
    cudaMemset(a.spinorNorm, 0, a.length*sizeof(float)/spinorSiteSize);
  }
}

__global__ void convertDSKernel(double2 *dst, float4 *src, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    for (int k=0; k<6; k++) {
      dst[2*k*length+i].x = src[k*length+i].x;
      dst[2*k*length+i].y = src[k*length+i].y;
      dst[(2*k+1)*length+i].x = src[k*length+i].z;
      dst[(2*k+1)*length+i].y = src[k*length+i].w;
    }
    i += gridSize;
  }   
}

__global__ void convertSDKernel(float4 *dst, double2 *src, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    for (int k=0; k<6; k++) {
      dst[k*length+i].x = src[2*k*length+i].x;
      dst[k*length+i].y = src[2*k*length+i].y;
      dst[k*length+i].z = src[(2*k+1)*length+i].x;
      dst[k*length+i].w = src[(2*k+1)*length+i].y;
    }
    i += gridSize;
  }   
}

__global__ void convertHSKernel(short4 *h, float *norm, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < length) {
    float4 F0 = tex1Dfetch(spinorTexSingle, i + 0*length);
    float4 F1 = tex1Dfetch(spinorTexSingle, i + 1*length);
    float4 F2 = tex1Dfetch(spinorTexSingle, i + 2*length);
    float4 F3 = tex1Dfetch(spinorTexSingle, i + 3*length);
    float4 F4 = tex1Dfetch(spinorTexSingle, i + 4*length);
    float4 F5 = tex1Dfetch(spinorTexSingle, i + 5*length);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, norm, F, length);
    i += gridSize;
  }

}

__global__ void convertSHKernel(float4 *res, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  while (i<length) {
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
    res[0*length+i] = I0;
    res[1*length+i] = I1;
    res[2*length+i] = I2;
    res[3*length+i] = I3;
    res[4*length+i] = I4;
    res[5*length+i] = I5;
    i += gridSize;
  }
}

__global__ void convertHDKernel(short4 *h, float *norm, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < length) {
    double2 F0 = fetch_double2(spinorTexDouble, i+0*length);
    double2 F1 = fetch_double2(spinorTexDouble, i+1*length);
    double2 F2 = fetch_double2(spinorTexDouble, i+2*length);
    double2 F3 = fetch_double2(spinorTexDouble, i+3*length);
    double2 F4 = fetch_double2(spinorTexDouble, i+4*length);
    double2 F5 = fetch_double2(spinorTexDouble, i+5*length);
    double2 F6 = fetch_double2(spinorTexDouble, i+6*length);
    double2 F7 = fetch_double2(spinorTexDouble, i+7*length);
    double2 F8 = fetch_double2(spinorTexDouble, i+8*length);
    double2 F9 = fetch_double2(spinorTexDouble, i+9*length);
    double2 F10 = fetch_double2(spinorTexDouble, i+10*length);
    double2 F11 = fetch_double2(spinorTexDouble, i+11*length);
    CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, norm, F, length);
    i += gridSize;
  }
}

__global__ void convertDHKernel(double2 *res, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < length) {
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
    res[0*length+i] = make_double2(I0.x, I0.y);
    res[1*length+i] = make_double2(I0.z, I0.w);
    res[2*length+i] = make_double2(I1.x, I1.y);
    res[3*length+i] = make_double2(I1.z, I1.w);
    res[4*length+i] = make_double2(I2.x, I2.y);
    res[5*length+i] = make_double2(I2.z, I2.w);
    res[6*length+i] = make_double2(I3.x, I3.y);
    res[7*length+i] = make_double2(I3.z, I3.w);
    res[8*length+i] = make_double2(I4.x, I4.y);
    res[9*length+i] = make_double2(I4.z, I4.w);
    res[10*length+i] = make_double2(I5.x, I5.y);
    res[11*length+i] = make_double2(I5.z, I5.w);
    i += gridSize;
  }

}

void copyCuda(ParitySpinor dst, ParitySpinor src) {
  checkSpinorLength(dst, src);

  int convertLength = dst.length / spinorSiteSize;
  int blocks = min(REDUCE_MAX_BLOCKS, max(convertLength/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
    convertDSKernel<<<dimGrid, dimBlock>>>((double2*)dst.spinor, (float4*)src.spinor, convertLength);
  } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
    convertSDKernel<<<dimGrid, dimBlock>>>((float4*)dst.spinor, (double2*)src.spinor, convertLength);
  } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = dst.length*sizeof(short);
    cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, src.spinorNorm, spinor_bytes/12);
    convertSHKernel<<<dimGrid, dimBlock>>>((float4*)dst.spinor, convertLength);
  } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
    int spinor_bytes = dst.length*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, src.spinor, spinor_bytes); 
    convertHSKernel<<<dimGrid, dimBlock>>>((short4*)dst.spinor, (float*)dst.spinorNorm, convertLength);
  } else if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = dst.length*sizeof(short);
    cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, src.spinorNorm, spinor_bytes/12);
    convertDHKernel<<<dimGrid, dimBlock>>>((double2*)dst.spinor, convertLength);
  } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
    int spinor_bytes = dst.length*sizeof(double);
    cudaBindTexture(0, spinorTexDouble, src.spinor, spinor_bytes); 
    convertHDKernel<<<dimGrid, dimBlock>>>((short4*)dst.spinor, (float*)dst.spinorNorm, convertLength);
  } else if (dst.precision == QUDA_DOUBLE_PRECISION) {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(double), cudaMemcpyDeviceToDevice);
  } else if (dst.precision == QUDA_SINGLE_PRECISION) {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(float), cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(short), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst.spinorNorm, src.spinorNorm, dst.length*sizeof(float)/spinorSiteSize, cudaMemcpyDeviceToDevice);
  }
}


template <typename Float>
__global__ void axpbyKernel(Float a, Float *x, Float b, Float *y, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    y[i] = a*x[i] + b*y[i];
    i += gridSize;
  } 
}

__global__ void axpbyHKernel(float a, float b, short4 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    AXPBY_FLOAT4(a, x0, b, y0);
    AXPBY_FLOAT4(a, x1, b, y1);
    AXPBY_FLOAT4(a, x2, b, y2);
    AXPBY_FLOAT4(a, x3, b, y3);
    AXPBY_FLOAT4(a, x4, b, y4);
    AXPBY_FLOAT4(a, x5, b, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpbyCuda(double a, ParitySpinor x, double b, ParitySpinor y) {
  checkSpinor(x, y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpbyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, b, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpbyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float)b, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    axpbyHKernel<<<dimGrid, dimBlock>>>((float)a, (float)b, (short4*)y.spinor, 
					(float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void xpyKernel(Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += x[i];
    i += gridSize;
  } 
}

__global__ void xpyHKernel(short4 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    XPY_FLOAT4(x0, y0);
    XPY_FLOAT4(x1, y1);
    XPY_FLOAT4(x2, y2);
    XPY_FLOAT4(x3, y3);
    XPY_FLOAT4(x4, y4);
    XPY_FLOAT4(x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = x[i] + y[i]
void xpyCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    xpyKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    xpyKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    xpyHKernel<<<dimGrid, dimBlock>>>((short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void axpyKernel(Float a, Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += a*x[i];
    i += gridSize;
  } 
}

__global__ void axpyHKernel(float a, short4 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    AXPY_FLOAT4(a, x0, y0);
    AXPY_FLOAT4(a, x1, y1);
    AXPY_FLOAT4(a, x2, y2);
    AXPY_FLOAT4(a, x3, y3);
    AXPY_FLOAT4(a, x4, y4);
    AXPY_FLOAT4(a, x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = a*x[i] + y[i]
void axpyCuda(double a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    axpyHKernel<<<dimGrid, dimBlock>>>((float)a, (short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void xpayKernel(Float *x, Float a, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] = x[i] + a*y[i];
    i += gridSize;
  } 
}

__global__ void xpayHKernel(float a, short4 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    XPAY_FLOAT4(x0, a, y0);
    XPAY_FLOAT4(x1, a, y1);
    XPAY_FLOAT4(x2, a, y2);
    XPAY_FLOAT4(x3, a, y3);
    XPAY_FLOAT4(x4, a, y4);
    XPAY_FLOAT4(x5, a, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = x[i] + a*y[i]
void xpayCuda(ParitySpinor x, double a, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    xpayKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, a, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    xpayKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float)a, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    xpayHKernel<<<dimGrid, dimBlock>>>((float)a, (short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void mxpyKernel(Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] -= x[i];
    i += gridSize;
  } 
}

__global__ void mxpyHKernel(short4 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    MXPY_FLOAT4(x0, y0);
    MXPY_FLOAT4(x1, y1);
    MXPY_FLOAT4(x2, y2);
    MXPY_FLOAT4(x3, y3);
    MXPY_FLOAT4(x4, y4);
    MXPY_FLOAT4(x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}


// performs the operation y[i] -= x[i] (minus x plus y)
void mxpyQuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    mxpyKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    mxpyKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    mxpyHKernel<<<dimGrid, dimBlock>>>((short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void axKernel(Float a, Float *x, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    x[i] *= a;
    i += gridSize;
  } 
}

__global__ void axHKernel(float a, short4 *xH, float *xN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    AX_FLOAT4(a, x0); AX_FLOAT4(a, x1); AX_FLOAT4(a, x2);
    AX_FLOAT4(a, x3); AX_FLOAT4(a, x4); AX_FLOAT4(a, x5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, length);
    i += gridSize;
  } 
  
}

// performs the operation x[i] = a*x[i]
void axCuda(double a, ParitySpinor x) {
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    axHKernel<<<dimGrid, dimBlock>>>((float)a, (short4*)x.spinor, (float*)x.spinorNorm, x.length/spinorSiteSize);
  }
}

template <typename Float2>
__global__ void caxpyKernel(Float2 a, Float2 *x, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z;
    Z.x = x[i].x;
    Z.y = x[i].y;
    y[i].x += a.x*Z.x - a.y*Z.y;
    y[i].y += a.y*Z.x + a.x*Z.y;
    i += gridSize;
  } 
  
}

__global__ void caxpyHKernel(float2 a, short4 *yH, float *yN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    CAXPY_FLOAT4(a, x0, y0);
    CAXPY_FLOAT4(a, x1, y1);
    CAXPY_FLOAT4(a, x2, y2);
    CAXPY_FLOAT4(a, x3, y3);
    CAXPY_FLOAT4(a, x4, y4);
    CAXPY_FLOAT4(a, x5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] += a*x[i]
void caxpyCuda(double2 a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpyKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    caxpyKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    caxpyHKernel<<<dimGrid, dimBlock>>>(af2, (short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float2>
__global__ void caxpbyKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 Z1, Z2;
    Z1.x = x[i].x;
    Z2.y = x[i].y;
    Z1.x = y[i].x;
    Z2.y = y[i].y;
    y[i].x = a.x*Z1.x + b.x*Z2.x - a.y*Z1.y - b.y*Z2.y;
    y[i].y = a.y*Z1.x + b.y*Z2.x + a.x*Z1.y + b.x*Z2.y;
    i += gridSize;
  } 
}

__global__ void caxpbyHKernel(float2 a, float2 b, short4 *yH, float *yN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    CAXPBY_FLOAT4(a, x0, b, y0);
    CAXPBY_FLOAT4(a, x1, b, y1);
    CAXPBY_FLOAT4(a, x2, b, y2);
    CAXPBY_FLOAT4(a, x3, b, y3);
    CAXPBY_FLOAT4(a, x4, b, y4);
    CAXPBY_FLOAT4(a, x5, b, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  }   
}


// performs the operation y[i] = c*x[i] + b*y[i]
void caxbyCuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbyKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbyKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbyHKernel<<<dimGrid, dimBlock>>>(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

template <typename Float2>
__global__ void cxpaypbzKernel(Float2 *x, Float2 a, Float2 *y, Float2 b, Float2 *z, int len) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 T1, T2, T3;
    T1.x = x[i].x;
    T1.y = x[i].y;
    T2.x = y[i].x;
    T2.y = y[i].y;
    T3.x = z[i].x;
    T3.y = z[i].y;
    
    T1.x += a.x*T2.x - a.y*T2.y;
    T1.y += a.y*T2.x + a.x*T2.y;
    T1.x += b.x*T3.x - b.y*T3.y;
    T1.y += b.y*T3.x + b.x*T3.y;
    
    z[i].x = T1.x;
    z[i].y = T1.y;
    i += gridSize;
  } 
  
}

__global__ void cxpaypbzHKernel(float2 a, float2 b, short4 *zH, float *zN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, length);
    CXPAYPBZ_FLOAT4(x0, a, y0, b, z0);
    CXPAYPBZ_FLOAT4(x1, a, y1, b, z1);
    CXPAYPBZ_FLOAT4(x2, a, y2, b, z2);
    CXPAYPBZ_FLOAT4(x3, a, y3, b, z3);
    CXPAYPBZ_FLOAT4(x4, a, y4, b, z4);
    CXPAYPBZ_FLOAT4(x5, a, y5, b, z5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, length);
    i += gridSize;
  }   
}


// performs the operation z[i] = x[i] + a*y[i] + b*z[i]
void cxpaypbzCuda(ParitySpinor x, double2 a, ParitySpinor y, double2 b, ParitySpinor z) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    cxpaypbzKernel<<<dimGrid, dimBlock>>>((double2*)x.spinor, a, (double2*)y.spinor, b, (double2*)z.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    cxpaypbzKernel<<<dimGrid, dimBlock>>>((float2*)x.spinor, af2, (float2*)y.spinor, bf2, (float2*)z.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    cxpaypbzHKernel<<<dimGrid, dimBlock>>>(af2, bf2, (short4*)z.spinor, (float*)z.spinorNorm, z.length/spinorSiteSize);
  }
}

template <typename Float>
__global__ void axpyZpbxKernel(Float a, Float *x, Float *y, Float *z, Float b, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float x_i = x[i];
    y[i] += a*x_i;
    x[i] = z[i] + b*x_i;
    i += gridSize;
  }
}

__global__ void axpyZpbxHKernel(float a, float b, short4 *xH, float *xN, short4 *yH, float *yN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, length);
    AXPY_FLOAT4(a, x0, y0);
    XPAY_FLOAT4(z0, b, x0);
    AXPY_FLOAT4(a, x1, y1);
    XPAY_FLOAT4(z1, b, x1);
    AXPY_FLOAT4(a, x2, y2);
    XPAY_FLOAT4(z2, b, x2);
    AXPY_FLOAT4(a, x3, y3);
    XPAY_FLOAT4(z3, b, x3);
    AXPY_FLOAT4(a, x4, y4);
    XPAY_FLOAT4(z4, b, x4);
    AXPY_FLOAT4(a, x5, y5);
    XPAY_FLOAT4(z5, b, x5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, length);
    i += gridSize;
  }   
}


// performs the operations: {y[i] = a x[i] + y[i]; x[i] = z[i] + b x[i]}
void axpyZpbxCuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axpyZpbxKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, (double*)z.spinor, b, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axpyZpbxKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, (float*)z.spinor, (float)b, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    axpyZpbxHKernel<<<dimGrid, dimBlock>>>((float)a, (float)b, (short4*)x.spinor, (float*)x.spinorNorm,
					   (short4*)y.spinor, (float*)y.spinorNorm, z.length/spinorSiteSize);
  }
}

template <typename Float2>
__global__ void caxpbypzYmbwKernel(Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, int len) {

  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float2 X, Y, Z, W;
    X.x = x[i].x;
    X.y = x[i].y;
    Y.x = y[i].x;
    Y.y = y[i].y;
    W.x = w[i].x;
    W.y = w[i].y;
    
    Z.x = a.x*X.x - a.y*X.y;
    Z.y = a.y*X.x + a.x*X.y;
    Z.x += b.x*Y.x - b.y*Y.y;
    Z.y += b.y*Y.x + b.x*Y.y;
    Y.x -= b.x*W.x - b.y*W.y;
    Y.y -= b.y*W.x + b.x*W.y;	
    
    z[i].x += Z.x;
    z[i].y += Z.y;
    y[i].x = Y.x;
    y[i].y = Y.y;
    i += gridSize;
  } 
}

__global__ void caxpbypzYmbwHKernel(float2 a, float2 b, short4 *yH, float *yN, short4 *zH, float *zN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, length);
    RECONSTRUCT_HALF_SPINOR(w, texHalf4, texNorm4, length);
    CAXPBYPZ_FLOAT4(a, x0, b, y0, z0);
    CAXPBYPZ_FLOAT4(a, x1, b, y1, z1);
    CAXPBYPZ_FLOAT4(a, x2, b, y2, z2);
    CAXPBYPZ_FLOAT4(a, x3, b, y3, z3);
    CAXPBYPZ_FLOAT4(a, x4, b, y4, z4);
    CAXPBYPZ_FLOAT4(a, x5, b, y5, z5);
    CMAXPY_FLOAT4(b, w0, y0);
    CMAXPY_FLOAT4(b, w1, y1);
    CMAXPY_FLOAT4(b, w2, y2);
    CMAXPY_FLOAT4(b, w3, y3);
    CMAXPY_FLOAT4(b, w4, y4);
    CMAXPY_FLOAT4(b, w5, y5);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, length);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  }   
}

// performs the operation z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
void caxpbypzYmbwCuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y,
                      ParitySpinor z, ParitySpinor w) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  checkSpinor(x,w);
  int length = x.length/2;
  int blocks = min(REDUCE_MAX_BLOCKS, max(length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    caxpbypzYmbwKernel<<<dimGrid, dimBlock>>>(a, (double2*)x.spinor, b, (double2*)y.spinor, 
					  (double2*)z.spinor, (double2*)w.spinor, length); 
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbypzYmbwKernel<<<dimGrid, dimBlock>>>(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, 
					  (float2*)z.spinor, (float2*)w.spinor, length); 
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf4, w.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm4, w.spinorNorm, spinor_bytes/12); 
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    caxpbypzYmbwHKernel<<<dimGrid, dimBlock>>>(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm,
					       (short4*)z.spinor, (float*)z.spinorNorm, z.length/spinorSiteSize);
  }
}


// Computes c = a + b in "double single" precision.
__device__ void dsadd(QudaSumFloat &c0, QudaSumFloat &c1, const QudaSumFloat a0, 
		      const QudaSumFloat a1, const float b0, const float b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0 + b0;
  QudaSumFloat e = t1 - a0;
  QudaSumFloat t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;
  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (complex version)
__device__ void zcadd(QudaSumComplex &c0, QudaSumComplex &c1, const QudaSumComplex a0, 
		      const QudaSumComplex a1, const QudaSumComplex b0, const QudaSumComplex b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (float3 version)
__device__ void dsadd3(QudaSumFloat3 &c0, QudaSumFloat3 &c1, const QudaSumFloat3 a0, 
		       const QudaSumFloat3 a1, const QudaSumFloat3 b0, const QudaSumFloat3 b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.z + b0.z;
  e = t1 - a0.z;
  t2 = ((b0.z - e) + (a0.z - (t1 - e))) + a1.z + b1.z;
  // The result is t1 + t2, after normalization.
  c0.z = e = t1 + t2;
  c1.z = t2 - (e - t1);
}

//
// double sumCuda(float *a, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) sumF##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) a[i]
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) sumH##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, n);			\
  SUM_FLOAT4(s0, I0);							\
  SUM_FLOAT4(s1, I1);							\
  SUM_FLOAT4(s2, I2);							\
  SUM_FLOAT4(s3, I3);							\
  SUM_FLOAT4(s4, I4);							\
  SUM_FLOAT4(s5, I5);							\
  s0 += s1; s2 += s3; s4 += s5; s0 += s2; s0 += s4;
#define REDUCE_OPERATION(i) (I0.x)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double sumCuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return sumFCuda((double*)a.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return sumFCuda((float*)a.spinor, a.length);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    return sumHCuda((char*)0, a.length/spinorSiteSize);
  }
}

//
// double normCuda(float *a, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) normF##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*a[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// double normHCuda(char *, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) normH##suffix
#define REDUCE_TYPES Float *a // dummy type
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, n);			\
  REAL_DOT_FLOAT4(norm0, I0, I0);					\
  REAL_DOT_FLOAT4(norm1, I1, I1);					\
  REAL_DOT_FLOAT4(norm2, I2, I2);					\
  REAL_DOT_FLOAT4(norm3, I3, I3);					\
  REAL_DOT_FLOAT4(norm4, I4, I4);					\
  REAL_DOT_FLOAT4(norm5, I5, I5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double normCuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return normFCuda((double*)a.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return normFCuda((float*)a.spinor, a.length);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    return normHCuda((char*)0, a.length/spinorSiteSize);
  }
}



//
// double reDotProductFCuda(float *a, float *b, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductF##suffix
#define REDUCE_TYPES Float *a, Float *b
#define REDUCE_PARAMS a, b
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*b[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// double reDotProductHCuda(float *a, float *b, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductH##suffix
#define REDUCE_TYPES Float *a, Float *b
#define REDUCE_PARAMS a, b
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(aH, texHalf1, texNorm1, n);		\
  RECONSTRUCT_HALF_SPINOR(bH, texHalf2, texNorm2, n);		\
  REAL_DOT_FLOAT4(rdot0, aH0, bH0);					\
  REAL_DOT_FLOAT4(rdot1, aH1, bH1);					\
  REAL_DOT_FLOAT4(rdot2, aH2, bH2);					\
  REAL_DOT_FLOAT4(rdot3, aH3, bH3);					\
  REAL_DOT_FLOAT4(rdot4, aH4, bH4);					\
  REAL_DOT_FLOAT4(rdot5, aH5, bH5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_OPERATION(i) (rdot0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double reDotProductCuda(ParitySpinor a, ParitySpinor b) {
  checkSpinor(a, b);
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return reDotProductFCuda((double*)a.spinor, (double*)b.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return reDotProductFCuda((float*)a.spinor, (float*)b.spinor, a.length);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, b.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, b.spinorNorm, spinor_bytes/12);    
    return reDotProductHCuda((char*)0, (char*)0, a.length/spinorSiteSize);
  }
}

//
// double axpyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = a*x[i] + y[i]
// Second returns the norm of y
//

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormF##suffix
#define REDUCE_TYPES Float a, Float *x, Float *y
#define REDUCE_PARAMS a, x, y
#define REDUCE_AUXILIARY(i) y[i] = a*x[i] + y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormH##suffix
#define REDUCE_TYPES Float a, short4 *yH, float *yN
#define REDUCE_PARAMS a, yH, yN
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  AXPY_FLOAT4(a, x0, y0);						\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  AXPY_FLOAT4(a, x1, y1);						\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  AXPY_FLOAT4(a, x2, y2);						\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  AXPY_FLOAT4(a, x3, y3);						\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  AXPY_FLOAT4(a, x4, y4);						\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  AXPY_FLOAT4(a, x5, y5);						\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2; norm0 += norm4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double axpyNormCuda(double a, ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return axpyNormFCuda(a, (double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return axpyNormFCuda((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return axpyNormHCuda((float)a, (short4*)y.spinor, (float*)y.spinorNorm, x.length/spinorSiteSize);
  }
}


//
// double xmyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = x[i] - y[i]
// Second returns the norm of y
//

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormF##suffix
#define REDUCE_TYPES Float *x, Float *y
#define REDUCE_PARAMS x, y
#define REDUCE_AUXILIARY(i) y[i] = x[i] - y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormH##suffix
#define REDUCE_TYPES Float *d1, Float *d2, short4 *yH, float *yN
#define REDUCE_PARAMS d1, d2, yH, yN
#define REDUCE_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  XMY_FLOAT4(x0, y0);							\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  XMY_FLOAT4(x1, y1);							\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  XMY_FLOAT4(x2, y2);							\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  XMY_FLOAT4(x3, y3);							\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  XMY_FLOAT4(x4, y4);							\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  XMY_FLOAT4(x5, y5);							\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2; norm0 += norm4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double xmyNormCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return xmyNormFCuda((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return xmyNormFCuda((float*)x.spinor, (float*)y.spinor, x.length);
  } else { 
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return xmyNormHCuda((char*)0, (char*)0, (short4*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}


//
// double2 cDotProductCuda(float2 *a, float2 *b, int n) {}
//
template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductF##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b, Float c
#define REDUCE_PARAMS a, b, c
#define REDUCE_REAL_AUXILIARY(i)
#define REDUCE_IMAG_AUXILIARY(i)
#define REDUCE_REAL_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_IMAG_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductH##suffix
#define REDUCE_TYPES Float2 *a, Float b
#define REDUCE_PARAMS a, b
#define REDUCE_REAL_AUXILIARY(i)					\
  RECONSTRUCT_HALF_SPINOR(aH, texHalf1, texNorm1, n);		\
  RECONSTRUCT_HALF_SPINOR(bH, texHalf2, texNorm2, n);		\
  REAL_DOT_FLOAT4(rdot0, aH0, bH0);					\
  REAL_DOT_FLOAT4(rdot1, aH1, bH1);					\
  REAL_DOT_FLOAT4(rdot2, aH2, bH2);					\
  REAL_DOT_FLOAT4(rdot3, aH3, bH3);					\
  REAL_DOT_FLOAT4(rdot4, aH4, bH4);					\
  REAL_DOT_FLOAT4(rdot5, aH5, bH5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_IMAG_AUXILIARY(i)					\
  IMAG_DOT_FLOAT4(idot0, aH0, bH0);					\
  IMAG_DOT_FLOAT4(idot1, aH1, bH1);					\
  IMAG_DOT_FLOAT4(idot2, aH2, bH2);					\
  IMAG_DOT_FLOAT4(idot3, aH3, bH3);					\
  IMAG_DOT_FLOAT4(idot4, aH4, bH4);					\
  IMAG_DOT_FLOAT4(idot5, aH5, bH5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;
#define REDUCE_REAL_OPERATION(i) (rdot0)
#define REDUCE_IMAG_OPERATION(i) (idot0)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 cDotProductCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    char c = NULL;
    return cDotProductFCuda((double2*)x.spinor, (double2*)y.spinor, c, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    char c = NULL;
    return cDotProductFCuda((float2*)x.spinor, (float2*)y.spinor, c, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductHCuda((char*)0, (char*)0, x.length/spinorSiteSize);
  }
}

//
// double2 xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int n) {}
//
// First performs the operation y = x + a*y
// Second returns complex dot product (z,y)
//

template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) xpaycDotzyF##suffix
#define REDUCE_TYPES Float2 *x, Float a, Float2 *y, Float2 *z
#define REDUCE_PARAMS x, a, y, z
#define REDUCE_REAL_AUXILIARY(i) y[i].x = x[i].x + a*y[i].x
#define REDUCE_IMAG_AUXILIARY(i) y[i].y = x[i].y + a*y[i].y
#define REDUCE_REAL_OPERATION(i) (z[i].x*y[i].x + z[i].y*y[i].y)
#define REDUCE_IMAG_OPERATION(i) (z[i].x*y[i].y - z[i].y*y[i].x)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

template <typename Float, typename Float2>
#define REDUCE_FUNC_NAME(suffix) xpaycDotzyH##suffix
#define REDUCE_TYPES Float a, short4 *yH, Float2 *yN
#define REDUCE_PARAMS a, yH, yN
#define REDUCE_REAL_AUXILIARY(i)					\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, n);			\
  XPAY_FLOAT4(x0, a, y0);						\
  XPAY_FLOAT4(x1, a, y1);						\
  XPAY_FLOAT4(x2, a, y2);						\
  XPAY_FLOAT4(x3, a, y3);						\
  XPAY_FLOAT4(x4, a, y4);						\
  XPAY_FLOAT4(x5, a, y5);						\
  REAL_DOT_FLOAT4(rdot0, z0, y0);					\
  REAL_DOT_FLOAT4(rdot1, z1, y1);					\
  REAL_DOT_FLOAT4(rdot2, z2, y2);					\
  REAL_DOT_FLOAT4(rdot3, z3, y3);					\
  REAL_DOT_FLOAT4(rdot4, z4, y4);					\
  REAL_DOT_FLOAT4(rdot5, z5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_IMAG_AUXILIARY(i)					\
  IMAG_DOT_FLOAT4(idot0, z0, y0);					\
  IMAG_DOT_FLOAT4(idot1, z1, y1);					\
  IMAG_DOT_FLOAT4(idot2, z2, y2);					\
  IMAG_DOT_FLOAT4(idot3, z3, y3);					\
  IMAG_DOT_FLOAT4(idot4, z4, y4);					\
  IMAG_DOT_FLOAT4(idot5, z5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4; \
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);
#define REDUCE_REAL_OPERATION(i) (rdot0)
#define REDUCE_IMAG_OPERATION(i) (idot0)
#include "reduce_complex_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_REAL_AUXILIARY
#undef REDUCE_IMAG_AUXILIARY
#undef REDUCE_REAL_OPERATION
#undef REDUCE_IMAG_OPERATION

double2 xpayDotzyCuda(ParitySpinor x, double a, ParitySpinor y, ParitySpinor z) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return xpaycDotzyFCuda((double2*)x.spinor, a, (double2*)y.spinor, (double2*)z.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return xpaycDotzyFCuda((float2*)x.spinor, (float)a, (float2*)y.spinor, (float2*)z.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    return xpaycDotzyHCuda((float)a, (short4*)y.spinor, (float*)y.spinorNorm,  x.length/spinorSiteSize);
  }
}


//
// double3 cDotProductNormACuda(float2 *a, float2 *b, int n) {}
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormAF##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b
#define REDUCE_PARAMS a, b
#define REDUCE_X_AUXILIARY(i)
#define REDUCE_Y_AUXILIARY(i)
#define REDUCE_Z_AUXILIARY(i)
#define REDUCE_X_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_Y_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#define REDUCE_Z_OPERATION(i) (a[i].x*a[i].x + a[i].y*a[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormAH##suffix
#define REDUCE_TYPES Float2 *a
#define REDUCE_PARAMS a
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  REAL_DOT_FLOAT4(norm0, x0, x0);					\
  REAL_DOT_FLOAT4(norm1, x1, x1);					\
  REAL_DOT_FLOAT4(norm2, x2, x2);					\
  REAL_DOT_FLOAT4(norm3, x3, x3);					\
  REAL_DOT_FLOAT4(norm4, x4, x4);					\
  REAL_DOT_FLOAT4(norm5, x5, x5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_Y_AUXILIARY(i)						\
  REAL_DOT_FLOAT4(rdot0, x0, y0);					\
  REAL_DOT_FLOAT4(rdot1, x1, y1);					\
  REAL_DOT_FLOAT4(rdot2, x2, y2);					\
  REAL_DOT_FLOAT4(rdot3, x3, y3);					\
  REAL_DOT_FLOAT4(rdot4, x4, y4);					\
  REAL_DOT_FLOAT4(rdot5, x5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_Z_AUXILIARY(i)						\
  IMAG_DOT_FLOAT4(idot0, x0, y0);					\
  IMAG_DOT_FLOAT4(idot1, x1, y1);					\
  IMAG_DOT_FLOAT4(idot2, x2, y2);					\
  IMAG_DOT_FLOAT4(idot3, x3, y3);					\
  IMAG_DOT_FLOAT4(idot4, x4, y4);					\
  IMAG_DOT_FLOAT4(idot5, x5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;  
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

double3 cDotProductNormACuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormAFCuda((double2*)x.spinor, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return cDotProductNormAFCuda((float2*)x.spinor, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductNormAHCuda((char*)0,  x.length/spinorSiteSize);
  }
}

//
// double3 cDotProductNormBCuda(float2 *a, float2 *b, int n) {}
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormBF##suffix
#define REDUCE_TYPES Float2 *a, Float2 *b
#define REDUCE_PARAMS a, b
#define REDUCE_X_AUXILIARY(i)
#define REDUCE_Y_AUXILIARY(i)
#define REDUCE_Z_AUXILIARY(i)
#define REDUCE_X_OPERATION(i) (a[i].x*b[i].x + a[i].y*b[i].y)
#define REDUCE_Y_OPERATION(i) (a[i].x*b[i].y - a[i].y*b[i].x)
#define REDUCE_Z_OPERATION(i) (b[i].x*b[i].x + b[i].y*b[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) cDotProductNormBH##suffix
#define REDUCE_TYPES Float2 *a
#define REDUCE_PARAMS a
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4;
#define REDUCE_Y_AUXILIARY(i)						\
  REAL_DOT_FLOAT4(rdot0, x0, y0);					\
  REAL_DOT_FLOAT4(rdot1, x1, y1);					\
  REAL_DOT_FLOAT4(rdot2, x2, y2);					\
  REAL_DOT_FLOAT4(rdot3, x3, y3);					\
  REAL_DOT_FLOAT4(rdot4, x4, y4);					\
  REAL_DOT_FLOAT4(rdot5, x5, y5);					\
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4;
#define REDUCE_Z_AUXILIARY(i)						\
  IMAG_DOT_FLOAT4(idot0, x0, y0);					\
  IMAG_DOT_FLOAT4(idot1, x1, y1);					\
  IMAG_DOT_FLOAT4(idot2, x2, y2);					\
  IMAG_DOT_FLOAT4(idot3, x3, y3);					\
  IMAG_DOT_FLOAT4(idot4, x4, y4);					\
  IMAG_DOT_FLOAT4(idot5, x5, y5);					\
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4;  
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

double3 cDotProductNormBCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return cDotProductNormBFCuda((double2*)x.spinor, (double2*)y.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    return cDotProductNormBFCuda((float2*)x.spinor, (float2*)y.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    return cDotProductNormBHCuda((char*)0,  x.length/spinorSiteSize);
  }
}


//
// double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormYF##suffix
#define REDUCE_TYPES Float2 a, Float2 *x, Float2 b, Float2 *y, Float2 *z, Float2 *w, Float2 *u
#define REDUCE_PARAMS a, x, b, y, z, w, u
#define REDUCE_X_AUXILIARY(i) \
  Float2 W, X, Y;	      \
  X.x = x[i].x;		      \
  X.y = x[i].y;		      \
  Y.x = y[i].x;		      \
  Y.y = y[i].y;		      \
  W.x = w[i].x;		      \
  W.y = w[i].y;
#define REDUCE_Y_AUXILIARY(i)	    \
  Float2 Z;			    \
  Z.x = a.x*X.x - a.y*X.y;	    \
  Z.y = a.y*X.x + a.x*X.y;	    \
  Z.x += b.x*Y.x - b.y*Y.y;	    \
  Z.y += b.y*Y.x + b.x*Y.y;	    \
  Y.x -= b.x*W.x - b.y*W.y;	    \
  Y.y -= b.y*W.x + b.x*W.y;	
#define REDUCE_Z_AUXILIARY(i)	      \
  z[i].x += Z.x;		      \
  z[i].y += Z.y;		      \
  y[i].x = Y.x;			      \
  y[i].y = Y.y;
#define REDUCE_X_OPERATION(i) (u[i].x*y[i].x + u[i].y*y[i].y)
#define REDUCE_Y_OPERATION(i) (u[i].x*y[i].y - u[i].y*y[i].x)
#define REDUCE_Z_OPERATION(i) (y[i].x*y[i].x + y[i].y*y[i].y)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

//
// double3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
template <typename Float2>
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormYH##suffix
#define REDUCE_TYPES Float2 a, Float2 b, short4 *yH, float *yN, short4 *zH, float *zN
#define REDUCE_PARAMS a, b, yH, yN, zH, zN
#define REDUCE_X_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
  RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);			\
  RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, n);			\
  CAXPBYPZ_FLOAT4(a, x0, b, y0, z0);					\
  CAXPBYPZ_FLOAT4(a, x1, b, y1, z1);					\
  CAXPBYPZ_FLOAT4(a, x2, b, y2, z2);					\
  CAXPBYPZ_FLOAT4(a, x3, b, y3, z3);					\
  CAXPBYPZ_FLOAT4(a, x4, b, y4, z4);					\
  CAXPBYPZ_FLOAT4(a, x5, b, y5, z5);					\
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(zH, zN, z, n);			\
  RECONSTRUCT_HALF_SPINOR(w, texHalf4, texNorm4, n);			\
  CMAXPY_FLOAT4(b, w0, y0);						\
  CMAXPY_FLOAT4(b, w1, y1);						\
  CMAXPY_FLOAT4(b, w2, y2);						\
  CMAXPY_FLOAT4(b, w3, y3);						\
  CMAXPY_FLOAT4(b, w4, y4);						\
  CMAXPY_FLOAT4(b, w5, y5);						\
  REAL_DOT_FLOAT4(norm0, y0, y0);					\
  REAL_DOT_FLOAT4(norm1, y1, y1);					\
  REAL_DOT_FLOAT4(norm2, y2, y2);					\
  REAL_DOT_FLOAT4(norm3, y3, y3);					\
  REAL_DOT_FLOAT4(norm4, y4, y4);					\
  REAL_DOT_FLOAT4(norm5, y5, y5);					\
  CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);			
#define REDUCE_Y_AUXILIARY(i)						\
  RECONSTRUCT_HALF_SPINOR(u, texHalf5, texNorm5, n);			\
  REAL_DOT_FLOAT4(rdot0, u0, y0);					\
  REAL_DOT_FLOAT4(rdot1, u1, y1);					\
  REAL_DOT_FLOAT4(rdot2, u2, y2);					\
  REAL_DOT_FLOAT4(rdot3, u3, y3);					\
  REAL_DOT_FLOAT4(rdot4, u4, y4);					\
  REAL_DOT_FLOAT4(rdot5, u5, y5);					\
  IMAG_DOT_FLOAT4(idot0, u0, y0);					\
  IMAG_DOT_FLOAT4(idot1, u1, y1);					\
  IMAG_DOT_FLOAT4(idot2, u2, y2);					\
  IMAG_DOT_FLOAT4(idot3, u3, y3);					\
  IMAG_DOT_FLOAT4(idot4, u4, y4);					\
  IMAG_DOT_FLOAT4(idot5, u5, y5);					
#define REDUCE_Z_AUXILIARY(i)						\
  norm0 += norm1; norm2 += norm3; norm4 += norm5; norm0 += norm2, norm0 += norm4; \
  rdot0 += rdot1; rdot2 += rdot3; rdot4 += rdot5; rdot0 += rdot2; rdot0 += rdot4; \
  idot0 += idot1; idot2 += idot3; idot4 += idot5; idot0 += idot2; idot0 += idot4; 
#define REDUCE_X_OPERATION(i) (rdot0)
#define REDUCE_Y_OPERATION(i) (idot0)
#define REDUCE_Z_OPERATION(i) (norm0)
#include "reduce_triple_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_X_AUXILIARY
#undef REDUCE_Y_AUXILIARY
#undef REDUCE_Z_AUXILIARY
#undef REDUCE_X_OPERATION
#undef REDUCE_Y_OPERATION
#undef REDUCE_Z_OPERATION

// This convoluted kernel does the following: z += a*x + b*y, y -= b*w, norm = (y,y), dot = (u, y)
double3 caxpbypzYmbwcDotProductWYNormYQuda(double2 a, ParitySpinor x, double2 b, ParitySpinor y,
					   ParitySpinor z, ParitySpinor w, ParitySpinor u) {
  checkSpinor(x,y);
  checkSpinor(x,z);
  checkSpinor(x,w);
  checkSpinor(x,u);
  int length = x.length/2;
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    return caxpbypzYmbwcDotProductWYNormYFCuda(a, (double2*)x.spinor, b, (double2*)y.spinor, (double2*)z.spinor, 
					       (double2*)w.spinor, (double2*)u.spinor, length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    return caxpbypzYmbwcDotProductWYNormYFCuda(af2, (float2*)x.spinor, bf2, (float2*)y.spinor, (float2*)z.spinor,
					       (float2*)w.spinor, (float2*)u.spinor, length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf4, w.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm4, w.spinorNorm, spinor_bytes/12);    
    cudaBindTexture(0, texHalf5, u.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm5, u.spinorNorm, spinor_bytes/12);    
    float2 af2 = make_float2((float)a.x, (float)a.y);
    float2 bf2 = make_float2((float)b.x, (float)b.y);
    return caxpbypzYmbwcDotProductWYNormYHCuda(af2, bf2, (short4*)y.spinor, (float*)y.spinorNorm, 
					       (short4*)z.spinor, (float*)z.spinorNorm, y.length/spinorSiteSize);
  }
}



double cpuDouble(float *data, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++) sum += data[i];
  return sum;
}

/*
void blasTest() {
  int n = 3*1<<24;
  float *h_data = (float *)malloc(n*sizeof(float));
  float *d_data;
  if (cudaMalloc((void **)&d_data,  n*sizeof(float))) {
    printf("Error allocating d_data\n");
    exit(0);
  }
  
  for (int i = 0; i < n; i++) {
    h_data[i] = rand()/(float)RAND_MAX - 0.5; // n-1.0-i;
  }
  
  cudaMemcpy(d_data, h_data, n*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaThreadSynchronize();
  stopwatchStart();
  int LOOPS = 20;
  for (int i = 0; i < LOOPS; i++) {
    sumCuda(d_data, n);
  }
  cudaThreadSynchronize();
  float secs = stopwatchReadSeconds();
  
  printf("%f GiB/s\n", 1.e-9*n*sizeof(float)*LOOPS / secs);
  printf("Device: %f MiB\n", (float)n*sizeof(float) / (1 << 20));
  printf("Shared: %f KiB\n", (float)REDUCE_THREADS*sizeof(float) / (1 << 10));
  
  float correctDouble = cpuDouble(h_data, n);
  printf("Total: %f\n", correctDouble);
  printf("CUDA db: %f\n", fabs(correctDouble-sumCuda(d_data, n)));
  
  cudaFree(d_data) ;
  free(h_data);
}
*/
/*
void axpbyTest() {
    int n = 3 * 1 << 20;
    float *h_x = (float *)malloc(n*sizeof(float));
    float *h_y = (float *)malloc(n*sizeof(float));
    float *h_res = (float *)malloc(n*sizeof(float));
    
    float *d_x, *d_y;
    if (cudaMalloc((void **)&d_x,  n*sizeof(float))) {
      printf("Error allocating d_x\n");
      exit(0);
    }
    if (cudaMalloc((void **)&d_y,  n*sizeof(float))) {
      printf("Error allocating d_y\n");
      exit(0);
    }
    
    for (int i = 0; i < n; i++) {
        h_x[i] = 1;
        h_y[i] = 2;
    }
    
    cudaMemcpy(d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice);
    
    axpbyCuda(4, d_x, 3, d_y, n/2);
    
    cudaMemcpy( h_res, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float expect = (i < n/2) ? 4*h_x[i] + 3*h_y[i] : h_y[i];
        if (h_res[i] != expect)
            printf("FAILED %d : %f != %f\n", i, h_res[i], h_y[i]);
    }
    
    cudaFree(d_y);
    cudaFree(d_x);
    free(h_x);
    free(h_y);
}
*/
