#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <util_quda.h>

#define REDUCE_THREADS 128
#define REDUCE_MAX_BLOCKS 64

void zeroCuda(float* dst, int len) {
    // cuda's floating point format, IEEE-754, represents the floating point
    // zero as 4 zero bytes
    cudaMemset(dst, 0, len*sizeof(float));
}

void copyCuda(float* dst, float *src, int len) {
    cudaMemcpy(dst, src, len*sizeof(float), cudaMemcpyDeviceToDevice);
}


__global__ void axpbyKernel(float a, float *x, float b, float *y, int len) {
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
        y[i] = a*x[i] + b*y[i];
        i += gridSize;
    } 
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpbyCuda(float a, float *x, float b, float *y, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    axpbyKernel<<<dimGrid, dimBlock>>>(a, x, b, y, len);
}

// performs the operation y[i] = a*x[i] + y[i]
void axpyCuda(float a, float *x, float *y, int len) {
    axpbyCuda(a, x, 1.0, y, len);
}

// performs the operation y[i] = x[i] + a*y[i]
void xpayCuda(float *x, float a, float *y, int len) {
    axpbyCuda(1.0, x, a, y, len);
}

// performs the operation y[i] -= x[i] (minus x plus y)
void mxpyCuda(float *x, float *y, int len) {
    axpbyCuda(-1.0, x, 1.0, y, len);
}

// performs the operation x[i] = a*x[i]
void axCuda(float a, float *x, int len) {
    axpbyCuda(0.0, x, a, x, len);
}

__global__ void caxpyKernel(float2 a, float2 *x, float2 *y, int len) {

    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	float Xx = x[i].x;
	float Xy = x[i].y;
        y[i].x += a.x*Xx - a.y*Xy;
	y[i].y += a.y*Xx + a.x*Xy;
        i += gridSize;
    } 

}

// performs the operation y[i] += a*x[i]
void caxpyCuda(float2 a, float2 *x, float2 *y, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    caxpyKernel<<<dimGrid, dimBlock>>>(a, x, y, len);
}

__global__ void caxpbyKernel(float2 a, float2 *x, float2 b, float2 *y, int len) {

    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	float Xx = x[i].x;
	float Xy = x[i].y;
	float Yx = y[i].x;
	float Yy = y[i].y;
        y[i].x = a.x*Xx + b.x*Yx - a.y*Xy - b.y*Yy;
	y[i].y = a.y*Xx + b.y*Yx + a.x*Xy + b.x*Yy;
        i += gridSize;
    } 

}

// performs the operation y[i] = c*x[i] + b*y[i]
void caxbyCuda(float2 a, float2 *x, float2 b, float2 *y, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    caxpbyKernel<<<dimGrid, dimBlock>>>(a, x, b, y, len);   
}

__global__ void cxpaypbzKernel(float2 *x, float2 a, float2 *y, 
                               float2 b, float2 *z, int len) {

    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	float Xx = x[i].x;
	float Xy = x[i].y;
	float Yx = y[i].x;
	float Yy = y[i].y;
	float Zx = z[i].x;
	float Zy = z[i].y;
	
	Xx += a.x*Yx - a.y*Yy;
 	Xy += a.y*Yx + a.x*Yy;
	Xx += b.x*Zx - b.y*Zy;
	Xy += b.y*Zx + b.x*Zy;

	z[i].x = Xx;
	z[i].y = Xy;
        i += gridSize;
    } 

}

// performs the operation z[i] = x[i] + a*y[i] + b*z[i]
void cxpaypbzCuda(float2 *x, float2 a, float2 *y, float2 b, float2 *z, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    cxpaypbzKernel<<<dimGrid, dimBlock>>>(x, a, y, b, z, len);   
}

__global__ void axpyZpbxKernel(float a, float *x, float *y, float *z, float b, int len) {
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
        float x_i = x[i];
        y[i] = a*x_i + y[i];
        x[i] = z[i] + b*x_i;
        i += gridSize;
    }
}

// performs the operations: {y[i] = a x[i] + y[i]; x[i] = z[i] + b x[i]}
void axpyZpbxCuda(float a, float *x, float *y, float *z, float b, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    axpyZpbxKernel<<<dimGrid, dimBlock>>>(a, x, y, z, b, len);
}

__global__ void caxpbypzYmbwKernel(float2 a, float2 *x, float2 b, float2 *y, 
	                           float2 *z, float2 *w, int len) {

    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	float Xx = x[i].x;
	float Xy = x[i].y;
	float Yx = y[i].x;
	float Yy = y[i].y;
	float Wx = w[i].x;
	float Wy = w[i].y;

        float Zx = a.x*Xx - a.y*Xy;
	float Zy = a.y*Xx + a.x*Xy;
	Zx += b.x*Yx - b.y*Yy;
	Zy += b.y*Yx + b.x*Yy;
	Yx -= b.x*Wx - b.y*Wy;
	Yy -= b.y*Wx + b.x*Wy;	

	z[i].x += Zx;
	z[i].y += Zy;
	y[i].x = Yx;
	y[i].y = Yy;
        i += gridSize;
    } 

}

// performs the operation z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
void caxpbypzYmbwCuda(float2 a, float2 *x, float2 b, float2 *y,
                      float2 *z, float2 *w, int len) {
    int blocks = min(REDUCE_MAX_BLOCKS, max(len/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    caxpbypzYmbwKernel<<<dimGrid, dimBlock>>>(a, x, b, y, z, w, len);   
}


// performs the operation y[i] = a*x[i] + y[i], and returns norm(y)
// float axpyNormCuda(float a, float *x, float *y, int len);


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
// float sumCuda(float *a, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) sum##suffix
#define REDUCE_TYPES float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) a[i]
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// float normCuda(float *a, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) norm##suffix
#define REDUCE_TYPES float *a
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
// float reDotProductCuda(float *a, float *b, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) reDotProduct##suffix
#define REDUCE_TYPES float *a, float *b
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
// float2 cDotProductCuda(float2 *a, float2 *b, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) cDotProduct##suffix
#define REDUCE_TYPES float2 *a, float2 *b
#define REDUCE_PARAMS a, b
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



//
// float3 cDotProductNormACuda(float2 *a, float2 *b, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) cDotProductNormA##suffix
#define REDUCE_TYPES float2 *a, float2 *b
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


//
// float3 cDotProductNormBCuda(float2 *a, float2 *b, int n) {}
//
#define REDUCE_FUNC_NAME(suffix) cDotProductNormB##suffix
#define REDUCE_TYPES float2 *a, float2 *b
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


//
// float3 caxpbypzYmbwcDotProductWYNormYCuda(float2 a, float2 *x, float2 b, float2 *y, 
// float2 *z, float2 *w, float2 *u, int len)
//
#define REDUCE_FUNC_NAME(suffix) caxpbypzYmbwcDotProductWYNormY##suffix
#define REDUCE_TYPES float2 a, float2 *x, float2 b, float2 *y, float2 *z, float2 *w, float2 *u
#define REDUCE_PARAMS a, x, b, y, z, w, u
#define REDUCE_X_AUXILIARY(i) \
	float Xx = x[i].x; \
	float Xy = x[i].y; \
	float Yx = y[i].x; \
	float Yy = y[i].y; \
	float Wx = w[i].x; \
	float Wy = w[i].y;
#define REDUCE_Y_AUXILIARY(i) \
        float Zx = a.x*Xx - a.y*Xy; \
	float Zy = a.y*Xx + a.x*Xy; \
	Zx += b.x*Yx - b.y*Yy; \
	Zy += b.y*Yx + b.x*Yy; \
	Yx -= b.x*Wx - b.y*Wy; \
	Yy -= b.y*Wx + b.x*Wy;	
#define REDUCE_Z_AUXILIARY(i) \
	z[i].x += Zx; \
	z[i].y += Zy; \
	y[i].x = Yx; \
	y[i].y = Yy;
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
// float axpyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = a*x[i] + y[i]
// Second returns the norm of y
//

#define REDUCE_FUNC_NAME(suffix) axpyNorm##suffix
#define REDUCE_TYPES float a, float *x, float *y
#define REDUCE_PARAMS a, x, y
#define REDUCE_AUXILIARY(i) y[i] = a*x[i] + y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION


//
// float2 xpaycDotzyCuda(float2 *x, float a, float2 *y, float2 *z, int n) {}
//
// First performs the operation y = x - a*y
// Second returns complex dot product (z,y)
//

#define REDUCE_FUNC_NAME(suffix) xpaycDotzy##suffix
#define REDUCE_TYPES float2 *x, float a, float2 *y, float2 *z
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



double cpuDouble(float *data, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++)
        sum += data[i];
    return sum;
}

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
