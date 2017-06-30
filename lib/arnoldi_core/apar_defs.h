/*
 * apar_defs.h
 *
 *  Created on: 8.8.2013
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2013 Teemu Rantalaiho
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *
 *
 */

#ifndef APAR_DEFS_H_
#define APAR_DEFS_H_

#ifdef MULTI_GPU
#define USE_CUDA 1
#define P_ERROR_CHECKS 0
#else
#define USE_CUDA 0
#endif

#ifndef MANGLE
#define MANGLE(X) X
#endif

/* Trivial implementation in CPU */
#if !(USE_CUDA)

#define PARALLEL_KERNEL_BEGIN(NAME, INPUTTYPE, INPUT, INDEXNAME, MULTICOMPINDEX)                                    \
static inline void MANGLE(NAME##_function)(INPUTTYPE INPUT, int range_start, int range_end, int multiCompDim) {     \
int INDEXNAME;                                                                                                      \
for (INDEXNAME = range_start; INDEXNAME < range_end; INDEXNAME++ ){                                                 \
    int MULTICOMPINDEX;                                                                                             \
    for (MULTICOMPINDEX = 0; MULTICOMPINDEX < multiCompDim; MULTICOMPINDEX++ ){                                     \

#define PARALLEL_KERNEL_END() }}}

#define KERNEL_CALL(NAME, INPUT, A, B, NMULTI)                                  \
    do {                                                                        \
         MANGLE(NAME##_function)(INPUT, (A), (B), NMULTI);                      \
    } while (0)

#define PARALLEL_KERNEL_BEGIN2D(NAME, INPUTTYPE, INPUT, IDX1, IDX2, MULTICOMPINDEX)                                 \
static inline void MANGLE(NAME##_2dfunction)(INPUTTYPE INPUT, int xstart, int xend, int ystart, int yend ,int multiCompDim) {     \
int IDX2;                                                                                                           \
for (IDX2 = ystart; IDX2 < yend; IDX2++ ){                                                                          \
int IDX1;                                                                                                           \
for (IDX1 = xstart; IDX1 < xend; IDX1++ ){                                                                          \
    int MULTICOMPINDEX;                                                                                             \
    for (MULTICOMPINDEX = 0; MULTICOMPINDEX < multiCompDim; MULTICOMPINDEX++ ){                                     \

#define PARALLEL_KERNEL_END2D() }}}}

#define KERNEL_CALL2D(NAME, INPUT, XA, XB, YA, YB, NMULTI)                                  \
    do {                                                                                    \
         MANGLE(NAME##_2dfunction)(INPUT, (XA), (XB), (YA), (YB),NMULTI);                   \
    } while (0)



#define PARALLEL_REDUCE_BEGIN(NAME, INPUTTYPE, INPUT, INDEXNAME, OUT_TYPE, RESULT_NAME, MULTIINDEX)     \
    static inline OUT_TYPE MANGLE(NAME##_SumXformFunction)                                      \
                    (INPUTTYPE INPUT, int INDEXNAME, int MULTIINDEX) {                          \
      OUT_TYPE RESULT_NAME;                                                                     \
      {


        /* Ok - so xform code comes here. */
#define PARALLEL_REDUCE_SUMFUN(NAME, TMP_RESULT, RESULT_NAME, OUT_TYPE)                         \
      }                                                                                         \
      return RESULT_NAME;                                                                       \
    }                                                                                           \
    static inline OUT_TYPE MANGLE(NAME##_SumFunction)(OUT_TYPE TMP_RESULT, OUT_TYPE RESULT_NAME)\
    {

        /* And reduction code comes here. */
#define PARALLEL_REDUCE_END(RESULT_NAME)                            \
      return RESULT_NAME;                                           \
    }



#define FOR_RANGE_REDUCE_KERNEL(NAME, INPUT, RESULT, A, B, NMULTI, RESONDEV, ACCUMULATE)                            \
  do {                                                                                                              \
    int index;                                                                                                      \
    int multi_idx;                                                                                                  \
    (void)RESONDEV;                                                                                                 \
    if ((A) < (B)){                                                                                                 \
      if (ACCUMULATE)                                                                                               \
          *RESULT = MANGLE(NAME##_SumFunction)(*RESULT, MANGLE(NAME##_SumXformFunction)(INPUT, (A), 0));            \
      else                                                                                                          \
          *RESULT = MANGLE(NAME##_SumXformFunction)(INPUT, (A), 0);                                                 \
      for (multi_idx = 1; multi_idx < NMULTI; multi_idx++ )                                                         \
          *RESULT = MANGLE(NAME##_SumFunction)(*RESULT, MANGLE(NAME##_SumXformFunction)(INPUT, (A), multi_idx));    \
    }                                                                                                               \
    for (index = (A) + 1; index < (B); index++ )                                                                    \
      for (multi_idx = 0; multi_idx < NMULTI; multi_idx++ )                                                         \
          *RESULT = MANGLE(NAME##_SumFunction)(*RESULT, MANGLE(NAME##_SumXformFunction)(INPUT, index, multi_idx));  \
  } while (0)


// Still parallelized matvec-mul
#define PARALLEL_MVECMUL_BEGIN(NAME, INPUTTYPE, INPUT, ROWIDX, COLIDX, SRC_TYPE, SRCNAME, OUT_TYPE, RESULT_NAME)  \
static inline OUT_TYPE MANGLE(NAME##_mulEntryFunc)(INPUTTYPE INPUT, int ROWIDX, int COLIDX, SRC_TYPE SRCNAME) \
{                                                                                                                           \
OUT_TYPE RESULT_NAME;

#define PARALLEL_MVECMUL_SUMFUN(NAME, RESULT_NAME, RES2_NAME, OUT_TYPE)                                         \
       return RESULT_NAME;                                                                                      \
}                                                                                                               \
static inline OUT_TYPE MANGLE(NAME##_mulSumFunc)(OUT_TYPE RESULT_NAME, OUT_TYPE RES2_NAME){

#define PARALLEL_MVECMUL_STOREFUN(NAME, RESULT_NAME, OUT_TYPE, DST_TYPE, DST_NAME, DST_IDX)                     \
       return RESULT_NAME;                                                                                      \
}                                                                                                               \
static inline void MANGLE(NAME##_storeFunc)(OUT_TYPE RESULT_NAME, DST_TYPE DST_NAME, int DST_IDX){

#define PARALLEL_MVECMUL_END()                                                                                  \
}

#define CALL_MVECMUL_KERNEL(NAME, INPUT, SRC, SIZEX, SIZEY, DST, OUT_TYPE)                                      \
  do {                                                                                                          \
      int x,y;                                                                                                  \
      if (SIZEX > 0){                                                                                           \
          for(y=0; y < SIZEY; y++){                                                                             \
              OUT_TYPE res = MANGLE(NAME##_mulEntryFunc)(INPUT,y,0,SRC);                                        \
              for(x=1; x < SIZEX; x++){                                                                         \
                  OUT_TYPE res2 = MANGLE(NAME##_mulEntryFunc)(INPUT,y,x,SRC);                                   \
                  res = MANGLE(NAME##_mulSumFunc)(res, res2);                                                   \
              }                                                                                                 \
              MANGLE(NAME##_storeFunc)(res, DST, y);                                                            \
          }                                                                                                     \
      }                                                                                                         \
  } while(0)



#else
// CUDA impl here

#ifndef UNROLL_NLOG2_CUDA_STEPS
#define UNROLL_NLOG2_CUDA_STEPS 3
#endif

#include "cuda_reduce.h"
#include "cuda_matmul.h"
#include "cuda_forall.h"

// This seems to be more or less best choice for us
// (Don't worry about seemingly low thread-count, we run multiple blocks on
// each sm giving ok occupation)
#define BLOCK_SIZE_LOG2     6
#define BLOCK_SIZE          (1 << BLOCK_SIZE_LOG2) // for log = 6 this gives 64


// NOTE: This can be used for multi-stream support
#ifndef CURRENT_STREAM
#define CURRENT_STREAM()    0
#endif

#ifndef CURRENT_STREAM_TMPBUF
#define CURRENT_STREAM_TMPBUF()    NULL
#endif
#ifndef CURRENT_STREAM_TMPBUFSIZE
#define CURRENT_STREAM_TMPBUFSIZE()    NULL
#endif



//s_createStream
static void* createCudaStream(void){
    cudaStream_t res;
    cudaStreamCreate(&res);
    return (void*)res;
}
static void destroyCudaStream(void* stream){
    cudaStream_t str = (cudaStream_t)stream;
    cudaStreamDestroy(str);
}
static void waitCudaStream(void* stream){
    cudaStream_t str = (cudaStream_t)stream;
    cudaStreamSynchronize(str);
}


#define PARALLEL_KERNEL_BEGIN(NAME, INPUTTYPE, INPUT, INDEXNAME, MULTIINDEX)                        \
    struct MANGLE(NAME##_xformFunctor) {                                                            \
        __device__ /* __host__ */                                                                   \
        void operator() (INPUTTYPE INPUT, int INDEXNAME, int MULTIINDEX = 0) const {


#define PARALLEL_KERNEL_END() } } ;

#ifdef TEST_CUDA_ERROR
#undef TEST_CUDA_ERROR
#endif

#if P_ERROR_CHECKS
#define TEST_CUDA_ERROR(STR)                                                            \
        do {                                                                            \
            cudaError_t error = cudaGetLastError();                                     \
            if (error != cudaSuccess)                                                   \
                   printf("%s: Cudaerror = %s\n", STR, cudaGetErrorString( error ));    \
        } while(0)
#else
#define TEST_CUDA_ERROR(STR) do {} while(0)
#endif

#define KERNEL_CALL(NAME, INPUT, A, B, NMULTI)                                          \
  do {                                                                                  \
      MANGLE(NAME##_xformFunctor) functionObject;                                       \
      callTransformKernel<UNROLL_NLOG2_CUDA_STEPS>                                      \
          (INPUT, functionObject, A, B, NMULTI, CURRENT_STREAM());                      \
      TEST_CUDA_ERROR("transformkernel:"#NAME " ");                                     \
      } while (0)

#ifndef __global__
#define __global__
#endif


static __global__
void detectCudaArchKernel(int* res)
{
    int result;
#if __CUDA_ARCH__ >= 350
    result = 350;
#elif __CUDA_ARCH__ >= 300
    result = 300;
#elif __CUDA_ARCH__ >= 210
    result = 210;
#elif __CUDA_ARCH__ >= 200
    result = 200;
#elif __CUDA_ARCH__ >= 130
    result = 130;
#elif __CUDA_ARCH__ >= 120
    result = 120;
#elif __CUDA_ARCH__ >= 110
    result = 110;
#else
    result = 100;
#endif
    if (threadIdx.x == 0)
        *res = result;
}


static inline
int DetectCudaArch(void)
{
    // The only way to know from host-code, which device architecture our kernels have been generated
    // against, is to run a kernel that actually checks it.. :)
    dim3 grid = 1;
    //dim3 block = 32;
    static int result = 0;
    if (result == 0)
    {
        void* tmpBuf;
        cudaMalloc(&tmpBuf, sizeof(int));
        detectCudaArchKernel<<<grid, grid>>>((int*)tmpBuf);
        cudaMemcpy(&result, tmpBuf, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(tmpBuf);
    }
    return result;
}


template <bool old, typename TRANSFORMFUNTYPE, typename INPUTTYPE>
__global__
void dXformKernel2d(INPUTTYPE input, TRANSFORMFUNTYPE xformFun, int x0, int x1, int y0, int y1, int nMulti){
    if (old){
        for (int multiIdx = 0; multiIdx < nMulti; multiIdx++){
            int idx = threadIdx.x + x0 + (blockIdx.x << BLOCK_SIZE_LOG2);
            int idy = blockIdx.y + y0;
            while (idx < x1 && idy < y1){
              xformFun(input, idx, idy, multiIdx);
              idx += (blockDim.x << BLOCK_SIZE_LOG2);
              }
        }
    } else {
        int idx = threadIdx.x + x0 + (blockIdx.x << BLOCK_SIZE_LOG2);
        int idy = blockIdx.y + y0;
        int multiIdx = blockIdx.z;
        while (idx < x1 && idy < y1){
          xformFun(input, idx, idy, multiIdx);
          idx += (blockDim.x << BLOCK_SIZE_LOG2);
          }

    }
}


template <typename TRANSFORMFUNTYPE, typename INPUTTYPE>
static inline void call2dXformKernel(INPUTTYPE input, TRANSFORMFUNTYPE xformFun, int x0, int x1, int y0, int y1, int nMulti){
    int sizey = y1 - y0;
    int sizex = x1 - x0;
    dim3 block = BLOCK_SIZE;
    dim3 grid = sizex >> BLOCK_SIZE_LOG2;
    if ((grid.x << BLOCK_SIZE_LOG2) < sizex)
        grid.x++;
    grid.y = sizey;
    int cuda_arch = DetectCudaArch();
    if(sizey > 0 && nMulti > 0 && x1 > x0){
        if (cuda_arch >= 200){
            grid.z = nMulti;
            dXformKernel2d<false><<<grid, block, 0, CURRENT_STREAM()>>>(input, xformFun, x0, x1, y0, y1, nMulti);
        }
        else {
            dXformKernel2d<true><<<grid, block, 0, CURRENT_STREAM()>>>(input, xformFun, x0, x1, y0, y1, nMulti);
        }
    }
#if P_ERROR_CHECKS
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
           printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif

}

#define PARALLEL_KERNEL_BEGIN2D(NAME, INPUTTYPE, INPUT, IDX1, IDX2, MULTIINDEX)                                 \
                                                                                \
struct MANGLE(NAME##_functor2d)                                                 \
{                                                                               \
  __device__   /*__host__ */                                                    \
  void operator()(INPUTTYPE INPUT, int IDX1, int IDX2, int MULTIINDEX) const    \
                                                                                \


#define PARALLEL_KERNEL_END2D() };

#define KERNEL_CALL2D(NAME, INPUT, XA, XB, YA, YB, NMULTI)                                          \
  do {                                                                                              \
      MANGLE(NAME##_functor2d) functionObject;                                                      \
      call2dXformKernel(INPUT, functionObject, (XA), (XB), (YA), (YB), NMULTI);                     \
      /*TEST_CUDA_ERROR("transformkernel:"#NAME " ");*/                                             \
      } while (0)






#define PARALLEL_REDUCE_BEGIN(NAME, INPUTTYPE, INPUT, INDEXNAME, OUT_TYPE, RESULT_NAME, MULTIINDEX) \
    struct MANGLE(NAME##_sumxformFunctor) {                                                         \
        __device__ /* __host__ */                                                                   \
        OUT_TYPE operator() (INPUTTYPE INPUT, int INDEXNAME, int MULTIINDEX = 0) const {            \
          OUT_TYPE RESULT_NAME;{

// You should put here code to add together TMP_RESULT and RESULT both of type OUT_TYPE
// Btw. NAME and RESULT have to be same as above! return the result in the end (of OUT_TYPE)
#define PARALLEL_REDUCE_SUMFUN(NAME, TMP_RESULT, RESULT_NAME, OUT_TYPE)                 \
          }                                                                             \
          return RESULT_NAME;                                                           \
        }                                                                               \
    };                                                                                  \
    struct MANGLE(NAME##_sumFunctor){                                                   \
          __device__ __host__                                                           \
          OUT_TYPE operator() (OUT_TYPE RESULT_NAME, OUT_TYPE TMP_RESULT) const{        \


#define PARALLEL_REDUCE_END(RESULT_NAME)                \
        return RESULT_NAME;                             \
        }                                               \
    };

#define FOR_RANGE_REDUCE_KERNEL(NAME, INPUT, RESULT, A, B, NMULTI, RESONDEV, ACCUMULATE)\
  do {                                                                                  \
    int index_0 = (A);                                                                  \
    int index_1 = (B);                                                                  \
    MANGLE(NAME##_sumFunctor) sumFun;                                                   \
    MANGLE(NAME##_sumxformFunctor) transformFun;                                        \
    callReduceKernel(INPUT, transformFun, sumFun, index_0, index_1, RESULT, NMULTI, CURRENT_STREAM(), RESONDEV, CURRENT_STREAM_TMPBUF(), CURRENT_STREAM_TMPBUFSIZE(), ACCUMULATE);   \
  } while (0)









// Still parallelized matvec-mul
#define PARALLEL_MVECMUL_BEGIN(NAME, INPUTTYPE, INPUT, ROWIDX, COLIDX, SRC_TYPE, SRCNAME, OUT_TYPE, RESULT_NAME)            \
        struct MANGLE(NAME##_mulEntryFunc) {                                                                                \
inline __device__                                /* x */     /* y */                                                        \
        OUT_TYPE operator()(INPUTTYPE INPUT, int COLIDX, int ROWIDX, SRC_TYPE SRCNAME) const{                               \
            OUT_TYPE RESULT_NAME;{


#define PARALLEL_MVECMUL_SUMFUN(NAME, RESULT_NAME, RES2_NAME, OUT_TYPE)               \
        }                                                                             \
        return RESULT_NAME;                                                           \
      }                                                                               \
  };                                                                                  \
  struct MANGLE(NAME##_mulSumFunc){                                                   \
        __device__ __host__                                                           \
        OUT_TYPE operator() (OUT_TYPE RESULT_NAME, OUT_TYPE RES2_NAME) const{

#define PARALLEL_MVECMUL_STOREFUN(NAME, RESULT_NAME, OUT_TYPE, DST_TYPE, DST_NAME, DST_IDX) \
        return RESULT_NAME;                                                                 \
      }                                                                                     \
  };                                                                                        \
  struct MANGLE(NAME##_mulStoreFunc){                                                       \
        __device__ __host__                                                                 \
        void operator() (DST_TYPE DST_NAME, int DST_IDX, OUT_TYPE RESULT_NAME) const{

#define PARALLEL_MVECMUL_END()                                                              \
}};

#define CALL_MVECMUL_KERNEL(NAME, INPUT, SRC, SIZEX, SIZEY, DST, OUT_TYPE)                  \
  do {                                                                                      \
      struct MANGLE(NAME##_mulSumFunc) sumFun;                                              \
      struct MANGLE(NAME##_mulEntryFunc) transformFun;                                      \
      struct MANGLE(NAME##_mulStoreFunc) storeFun;                                          \
      callFullMatMul<OUT_TYPE>(INPUT, transformFun, sumFun, storeFun, SIZEX, SIZEY, SRC, DST, false, CURRENT_STREAM(), true); \
  } while(0)




#endif // USE_CUDA


#endif /* APAR_DEFS_H_ */
