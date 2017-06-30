#ifndef CUDA_REDUCE_H_
#define CUDA_REDUCE_H_
/*
 * cuda_reduce.h
 *
 *  Created on: 27.3.2012
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2011-2013 Teemu Rantalaiho
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

//#include "cuda_histogram.h"




// Public APIs
template <int nDim,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline
cudaError_t
callReduceKernelNDim(
    INPUTTYPE input, TRANSFORMFUNTYPE xformObj, SUMFUNTYPE sumfunObj,
    INDEXT* starts, INDEXT* ends, OUTPUTTYPE* out,
    int multiReduce = 1, // Convenience API - multiple entries per one input-index ran in parallel (simple multidim)
    cudaStream_t stream = 0, bool outInDev = false,
    void** tmpbuf = NULL,
    size_t* tmpbufSize = NULL,
    bool accumulate = true);


template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline
cudaError_t callReduceKernel(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor,
    INDEXTYPE start, INDEXTYPE end, OUTPUTTYPE* result,
    int multiReduce = 1, // Convenience API - multiple entries per one input-index ran in parallel (simple multidim)
    cudaStream_t stream = 0, bool outInDev = false,
    void** tmpbuf = NULL,
    size_t* tmpbufSize = NULL,
    bool accumulate = true);

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename MINUSFUNTYPE, typename OUTPUTTYPE>
static inline
cudaError_t callKahanReduceKernel(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, MINUSFUNTYPE minusFunctor,
    INDEXTYPE start, INDEXTYPE end, OUTPUTTYPE* result, OUTPUTTYPE zero,
    int multiReduce = 1, // Convenience API - multiple entries per one input-index ran in parallel (simple multidim)
    cudaStream_t stream = 0, bool outInDev = false,
    void** tmpbuf = NULL,
    size_t* tmpbufSize = NULL,
    bool accumulate = true);


template <int nDim,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename MINUSFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline
cudaError_t
callKahanReduceKernelNDim(
    INPUTTYPE input, TRANSFORMFUNTYPE xformObj, SUMFUNTYPE sumfunObj, MINUSFUNTYPE minusFunctor,
    INDEXT* starts, INDEXT* ends, OUTPUTTYPE* out, OUTPUTTYPE zero,
    int multiReduce = 1,
    cudaStream_t stream = 0, bool outInDev = false,
    void** tmpbuf = NULL,
    size_t* tmpbufSize = NULL,
    bool accumulate = true);



// End public APIs



#define REDUCE_BLOCK_SIZE_LOG2  7
#define REDUCE_BLOCK_SIZE       (1 << REDUCE_BLOCK_SIZE_LOG2)
#define MAX_reduce_STEPS        2048
#define R_ERROR_CHECKS          0
#define R_PRINT_ALL_MALLOCS     0
#if R_PRINT_ALL_MALLOCS
#define PRINT_MALLOC(BUFFER)    printf("%p  cudaMalloc'd(%s) at line %d\n", (BUFFER), #BUFFER , __LINE__)
#define PRINT_FREE(BUFFER)      printf("%p  cudaFree'd(%s) at line %d\n", (BUFFER),  #BUFFER , __LINE__)
#else
#define PRINT_MALLOC(BUFFER)
#define PRINT_FREE(BUFFER)
#endif

#ifdef UNROLL_NLOG2_CUDA_STEPS
#define REDUCE_UNROLL_LOG2      UNROLL_NLOG2_CUDA_STEPS
#else
#define REDUCE_UNROLL_LOG2      1
#endif

#define REDUCE_UNROLL           (1 << REDUCE_UNROLL_LOG2)

#if R_ERROR_CHECKS || R_PRINT_ALL_MALLOCS
#include <stdio.h>
#endif



#define FINAL_SUM_BLOCK_SIZE    64


template <typename SUMFUNTYPE, typename OUTPUTTYPE>
__global__
void finalSumKernel(SUMFUNTYPE sumfunObj, OUTPUTTYPE* blockOut, int maxblocks, OUTPUTTYPE* devout, bool accumulate)
{
    int myIdx = threadIdx.x;
    OUTPUTTYPE res;
    OUTPUTTYPE dvout;

    if (devout || !accumulate){
        maxblocks--;
        if (devout)
            dvout = *devout;
    }
    if (maxblocks < 31 /*blockDim.x*/){
      if (threadIdx.x == 0){
        res = blockOut[0];
        if (devout && accumulate)
            res = sumfunObj(res, dvout);
        for (int i = 1; i <= maxblocks; i++)
          res = sumfunObj(res, blockOut[i]);
        if (devout)
            *devout = res;
        else
            blockOut[0] = res;
      }
      return;
    }
    if (myIdx <= maxblocks){
        res = blockOut[myIdx];
        myIdx += FINAL_SUM_BLOCK_SIZE;
    }
    while (myIdx <= maxblocks)
    {
        res = sumfunObj(res, blockOut[myIdx]);
        myIdx += FINAL_SUM_BLOCK_SIZE;
    }

    {
        __shared__ OUTPUTTYPE shRes[64];
        shRes[threadIdx.x] = res;
        __syncthreads();
        if (threadIdx.x < 32 && (threadIdx.x + 32) <= maxblocks) shRes[threadIdx.x] = sumfunObj(shRes[threadIdx.x + 32], shRes[threadIdx.x]);
        __threadfence_block();
        if (threadIdx.x < 16) shRes[threadIdx.x] = sumfunObj(shRes[threadIdx.x + 16], shRes[threadIdx.x]);
        __threadfence_block();
        if (threadIdx.x < 8) shRes[threadIdx.x] = sumfunObj(shRes[threadIdx.x + 8], shRes[threadIdx.x]);
        __threadfence_block();
        if (threadIdx.x < 4) shRes[threadIdx.x] = sumfunObj(shRes[threadIdx.x + 4], shRes[threadIdx.x]);
        __threadfence_block();
        if (threadIdx.x < 2) shRes[threadIdx.x] = sumfunObj(shRes[threadIdx.x + 2], shRes[threadIdx.x]);
        __threadfence_block();
        if (threadIdx.x == 0) res = sumfunObj(shRes[threadIdx.x + 1], shRes[threadIdx.x]);
        __threadfence_block();

    }
    if (threadIdx.x == 0){
        if (devout){
            if (accumulate)
                *devout = sumfunObj(dvout, res);
            else
                *devout = res;
        }
        else{
            blockOut[0] = res;
        }
    }
}




template <bool firstPass, bool lastSteps, /*int nMultires,*/
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histoKernel_reduceStep(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT myStart, INDEXT end,
    OUTPUTTYPE* myRes)
{
    if (lastSteps)
    {
        if (myStart < end)
        {
            if (firstPass)
              *myRes = xformObj(input, myStart, blockIdx.y);
            else
              *myRes = sumfunObj(xformObj(input, myStart, blockIdx.y), *myRes);
        }
    }
    else
    {
      if (firstPass)
        *myRes = xformObj(input, myStart, blockIdx.y);
      else
        *myRes = sumfunObj(xformObj(input, myStart, blockIdx.y), *myRes);
    }
}
template <bool lastSteps,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
__global__
void histoKernel_reduce(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE* blockOut, int maxblocks,
    int nSteps, bool first)
{
    // Take care with extern - In order to have two instances of this template the
    // type of the extern variables cannot change
    // (ie. cannot use "extern __shared__ OUTPUTTYPE bins[]")
    extern __shared__ int cudahistogram_allbinstmp[];
    OUTPUTTYPE*  allbins = (OUTPUTTYPE*)&(*cudahistogram_allbinstmp);
    OUTPUTTYPE  myres;
    OUTPUTTYPE* ourOut = &blockOut[blockIdx.x + blockIdx.y * gridDim.x];
    // Now our block handles a continuos lump from myStart to myStart + nThreads*nSteps
    // Let's change it to go in thread-linear order from myStart jumping over all the blocks in each step
    // Therefore we start from 'start' + blockId * nThreads + tid, and we walk with stride nThreads * nBlocks
    int stride = gridDim.x << REDUCE_BLOCK_SIZE_LOG2;
    INDEXT myStart = start + (INDEXT)((blockIdx.x) << REDUCE_BLOCK_SIZE_LOG2) + (INDEXT)threadIdx.x;
    // Run loops - unroll 8 steps manually
    if (nSteps > 0){
      histoKernel_reduceStep<true, lastSteps>(input, xformObj, sumfunObj, myStart, end, &myres);
      myStart += stride;
      nSteps--;
    }
    if (nSteps > 0){
      int doNSteps = (nSteps) >> REDUCE_UNROLL_LOG2;
      if (lastSteps){
          while (myStart + (doNSteps * stride << REDUCE_UNROLL_LOG2) >= end){
              doNSteps--;
          }
      }
      for (int step = 0; step < doNSteps; step++)
      {
          #pragma unroll
          for (int substep = 0; substep < REDUCE_UNROLL; substep++){
            histoKernel_reduceStep<false, false>(input, xformObj, sumfunObj, myStart, end, &myres);
            myStart += stride;
          }
      }
      int nStepsLeft = (nSteps) - (doNSteps << REDUCE_UNROLL_LOG2);
      for (int step = 0; step < nStepsLeft; step++)
      {
          histoKernel_reduceStep<false, lastSteps>(input, xformObj, sumfunObj, myStart, end, &myres);
          myStart += stride;
      }
    }

#if 1
    {
        OUTPUTTYPE result;
        if (!first && threadIdx.x == 0)
            result = ourOut[0];

        allbins[threadIdx.x] = myres;
        // In the end combine results:
        #if REDUCE_BLOCK_SIZE > 32
        __syncthreads();
        #endif

        if (lastSteps && start + (INDEXT)((blockIdx.x + 1) << REDUCE_BLOCK_SIZE_LOG2) >= end)
        {
          // Safepath for last steps
          if (threadIdx.x == 0){
            INDEXT limit = end - (start + (INDEXT)((blockIdx.x) << REDUCE_BLOCK_SIZE_LOG2));
            //if (limit == 2) myres = sumfunObj(myres, myres);
            for (int tid = 1; tid < limit; tid++)
              myres = sumfunObj(allbins[tid], myres);
          }
        }
        else
        {
          // Example REDUCE_BLOCK_SIZE == 256
#if REDUCE_BLOCK_SIZE >= 128
          int limit = REDUCE_BLOCK_SIZE >> 1;     // Limit = 128
          if (threadIdx.x < limit)                // For all i < 128 Add a[i] <- a[i] + a[i+128]
              allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + limit]);
          else
              return; // Note - exited warps can't hang execution
          limit >>= 1;                            // Limit = 64
          int looplimit = ((REDUCE_BLOCK_SIZE_LOG2 - 2)); // Looplimit = 6
  #pragma unroll
          for (int loop = 5; loop < looplimit; loop++){   // One iteration of loop
              __syncthreads();                            // 1: For all i add a[i] <-  a[i] + a[i + 64]
              if (threadIdx.x < limit)
                  allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + limit]);
              limit >>= 1;                                // Limit = 32
          }
#endif
          if (threadIdx.x >= 32) return;
          __syncthreads();
          // Unroll rest manually
#if REDUCE_BLOCK_SIZE > 32
          allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 32]);
          __threadfence_block();
#endif
          allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 16]);
          __threadfence_block();
          allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 8]);
          __threadfence_block();
          allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 4]);
          __threadfence_block();
          allbins[threadIdx.x] = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 2]);
          __threadfence_block();
          myres = sumfunObj(allbins[threadIdx.x], allbins[threadIdx.x + 1]);
        }
        if (threadIdx.x == 0){
          if (first)
            result = myres;
          else
            result = sumfunObj(myres, result);
          ourOut[0] = result;
        }
    }
#endif
}


template <typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
          typename OUTPUTTYPE, typename INDEXT>
static
void callReduceImpl(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE* out,
    cudaDeviceProp* props,
    cudaStream_t stream,
    bool outInDev,
    int multiReduce,
    void** tmpbuf,
    size_t* tmpbufSize, bool accumulate)
{
    INDEXT size = end - start;
    if (end <= start)
    {
        return;
    }

    int maxblocks = 64;
    if (props) maxblocks = props->multiProcessorCount * 4;
    //int maxblocks = 16;


    if (size > 2*1024*1024 && multiReduce < 2){
        maxblocks *= 2;
    }

    if ((maxblocks << REDUCE_BLOCK_SIZE_LOG2) >= size){
      maxblocks = size >> REDUCE_BLOCK_SIZE_LOG2;
      if ((maxblocks << REDUCE_BLOCK_SIZE_LOG2) < size)
        maxblocks++;
    }


    OUTPUTTYPE* tmpOut;
    // Check whether user has supplied a temporary buffer - these can be really useful, as
    // any allocation or free implies a cpu-gpu synchronization
    {
        if (tmpbuf && tmpbufSize){
            if (*tmpbuf && *tmpbufSize >= (maxblocks*multiReduce + 1) * sizeof(OUTPUTTYPE)){
                tmpOut = (OUTPUTTYPE*)(*tmpbuf);
            }
            else {
                if (*tmpbuf){
                    cudaFree(*tmpbuf);
                    PRINT_FREE(*tmpbuf);
                }
                *tmpbufSize = (maxblocks*multiReduce + 1) * sizeof(OUTPUTTYPE);
                cudaMalloc(tmpbuf, *tmpbufSize);
                PRINT_MALLOC(*tmpbuf);

                tmpOut = (OUTPUTTYPE*)(*tmpbuf);
            }
        }
        else {
            cudaMalloc((void**)&tmpOut, (maxblocks*multiReduce + 1) * sizeof(OUTPUTTYPE));
            PRINT_MALLOC(tmpOut);
        }
    }
    int sharedNeeded;
    {
        int typesize = sizeof(OUTPUTTYPE);
        sharedNeeded = (typesize) << (REDUCE_BLOCK_SIZE_LOG2);
        //printf("reduce-bin, generic, Shared needed = %d\n", sharedNeeded);
    }
    // Determine number of local variables
    // reduce_LOCALLIMIT is total local size available for one block:

    int nSteps = size / (maxblocks << REDUCE_BLOCK_SIZE_LOG2);
    if (nSteps * maxblocks * REDUCE_BLOCK_SIZE < size) nSteps++;
    if (nSteps > MAX_reduce_STEPS) nSteps = MAX_reduce_STEPS;

    int nFullSteps = size / (nSteps * maxblocks * REDUCE_BLOCK_SIZE);

    dim3 grid(maxblocks, multiReduce, 1);
    dim3 block = REDUCE_BLOCK_SIZE;
    bool first = true;
    for (int i = 0; i < nFullSteps; i++)
    {
        histoKernel_reduce<false><<<grid, block, sharedNeeded, stream>>>(
            input, xformObj, sumfunObj, start, end, tmpOut, maxblocks, nSteps, i == 0);
        first = false;
        start += nSteps * maxblocks * REDUCE_BLOCK_SIZE;
#if R_ERROR_CHECKS
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
           printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
    }
    size = end - start;
    if (size > 0)
    {
        // Do what steps we still can do without checks
        nSteps = size / (maxblocks << REDUCE_BLOCK_SIZE_LOG2);
        if (nSteps * (maxblocks << REDUCE_BLOCK_SIZE_LOG2) < size) nSteps++;
        if (nSteps > 0)
        {
            histoKernel_reduce<true><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, tmpOut,  maxblocks, nSteps, first);
            start += nSteps * maxblocks * REDUCE_BLOCK_SIZE;
            first = false;
        }
    }
#if R_ERROR_CHECKS
    {
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess)
         printf("Cudaerror = %s\n", cudaGetErrorString( error ));
    }
#endif
    // Finally put together the result:
#if 1
    if (!outInDev && accumulate){
        if (stream != 0)
            cudaMemcpyAsync(&tmpOut[maxblocks*multiReduce], out, sizeof(OUTPUTTYPE), cudaMemcpyHostToDevice, stream);
        else
            cudaMemcpy(&tmpOut[maxblocks*multiReduce], out, sizeof(OUTPUTTYPE), cudaMemcpyHostToDevice);
    }
    // Let's do so that one block handles one bin
    grid.x = 1;
    grid.y = 1;
    //grid.x = nOut >> REDUCE_BLOCK_SIZE_LOG2;
    //if ((grid.x << REDUCE_BLOCK_SIZE_LOG2) < nOut) grid.x++;
    block.x = FINAL_SUM_BLOCK_SIZE;
    finalSumKernel<<<grid, block, 0, stream>>>(sumfunObj, tmpOut, maxblocks*multiReduce, outInDev ? out : NULL, accumulate);
#if R_ERROR_CHECKS
    {
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess)
         printf("Cudaerror (finalsumkernel) = %s\n", cudaGetErrorString( error ));
    }
#endif
    // TODO: Use async copy for the results as well?
    if (!outInDev){
        if (stream != 0)
            cudaMemcpyAsync(out, tmpOut, sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost, stream);
        else
            cudaMemcpy(out, tmpOut, sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
    }
#else
    {
      int i;
      OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(maxblocks * sizeof(OUTPUTTYPE));
      //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
      cudaMemcpy(h_tmp, tmpOut, maxblocks*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
      {
        OUTPUTTYPE res = *out;
        for (i = 0; i < maxblocks; i++)
        {
          res = sumfunObj(res, h_tmp[i]);
        }
        *out = res;
      }
      free(h_tmp);
    }
#endif
    if (!(tmpbuf && tmpbufSize)){
        cudaFree(tmpOut);
        PRINT_FREE(tmpOut);
    }
}


template <typename SUMFUNTYPE, typename OUTPUTTYPE, typename MINUSFUNTYPE>
__global__
void finalKahanKernel(SUMFUNTYPE sumfunObj, OUTPUTTYPE* blockOut, int maxblocks, MINUSFUNTYPE minusFun, OUTPUTTYPE zero)
{
    OUTPUTTYPE res;
    if (threadIdx.x == 0){
      res = zero;
      OUTPUTTYPE c = zero;
      for (int i = 0; i <= maxblocks; i++){
        OUTPUTTYPE y = minusFun(blockOut[i], c);
        OUTPUTTYPE t = sumfunObj(res, y);
        c = minusFun(minusFun(t, res), y);
        res = t;
      }
      blockOut[0] = res;
    }
    return;
}



template <bool lastSteps, /*int nMultires,*/
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT, typename MINUSFUNTYPE>
static inline __device__
void kahanKernel_reduceStep(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT myStart, INDEXT end,
    OUTPUTTYPE* myRes,
    OUTPUTTYPE* c,
    MINUSFUNTYPE minusFun)
{
    if (lastSteps)
    {
        if (myStart < end)
        {
            OUTPUTTYPE y = minusFun(xformObj(input, myStart, blockIdx.y), *c);
            OUTPUTTYPE t = sumfunObj(y, *myRes);
            *c = minusFun( minusFun( t, *myRes ), y);
            *myRes = t;
        }
    }
    else
    {
      // NOTE: We need the if (1) here to get loop unrolling out of the compiler (at least nvcc 3.2.16)
      if (1){
        OUTPUTTYPE y = minusFun(xformObj(input, myStart, blockIdx.y), *c);
        OUTPUTTYPE t = sumfunObj(y, *myRes);
        *c = minusFun( minusFun( t, *myRes ), y);
        *myRes = t;
      }
    }
}

template <bool lastSteps,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT, typename MINUSFUNTYPE>
__global__
void kahanKernel_reduce(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE* blockOut, int maxblocks,
    int nSteps, bool first,
    MINUSFUNTYPE minusFunObj, OUTPUTTYPE zero)
{
    // Take care with extern - In order to have two instances of this template the
    // type of the extern variables cannot change
    // (ie. cannot use "extern __shared__ OUTPUTTYPE bins[]")
    extern __shared__ int cudahistogram_allbinstmp[];
    OUTPUTTYPE* allbins = (OUTPUTTYPE*)&(*cudahistogram_allbinstmp);
    OUTPUTTYPE  myres = zero;
    OUTPUTTYPE  c = zero;
    OUTPUTTYPE* ourOut = &blockOut[blockIdx.x + blockIdx.y * gridDim.x];
    // Now our block handles a continuos lump from myStart to myStart + nThreads*nSteps
    // Let's change it to go in thread-linear order from myStart jumping over all the blocks in each step
    // Therefore we start from 'start' + blockId * nThreads + tid, and we walk with stride nThreads * nBlocks
    int stride = gridDim.x << REDUCE_BLOCK_SIZE_LOG2;
    INDEXT myStart = start + (INDEXT)((blockIdx.x) << REDUCE_BLOCK_SIZE_LOG2) + (INDEXT)threadIdx.x;
    // Run loops - unroll 8 steps manually
    if (nSteps > 0){
      int doNSteps = (nSteps) >> REDUCE_UNROLL_LOG2;
      if (lastSteps){
          while (myStart + (doNSteps * stride << REDUCE_UNROLL_LOG2) >= end){
              doNSteps--;
          }
      }
      for (int step = 0; step < doNSteps; step++)
      {
          #pragma unroll
          for (int substep = 0; substep < REDUCE_UNROLL; substep++){
            kahanKernel_reduceStep<false>(input, xformObj, sumfunObj, myStart, end, &myres, &c, minusFunObj);
            myStart += stride;
          }
      }
      int nStepsLeft = (nSteps) - (doNSteps << REDUCE_UNROLL_LOG2);
      for (int step = 0; step < nStepsLeft; step++)
      {
          kahanKernel_reduceStep<lastSteps>(input, xformObj, sumfunObj, myStart, end, &myres, &c, minusFunObj);
          myStart += stride;
      }
    }

#if 1
    {
        OUTPUTTYPE result;
        if (!first && threadIdx.x == 0)
            result = ourOut[0];

        allbins[threadIdx.x] = myres;
        // In the end combine results:
        #if REDUCE_BLOCK_SIZE > 32
        __syncthreads();
        #endif

        // Safepath for last steps
        if (threadIdx.x == 0){
          INDEXT limit = end - (start + (INDEXT)((blockIdx.x) << REDUCE_BLOCK_SIZE_LOG2));
          if (limit > REDUCE_BLOCK_SIZE) limit = REDUCE_BLOCK_SIZE;
          //if (limit == 2) myres = sumfunObj(myres, myres);
          for (int tid = 1; tid < limit; tid++){
            OUTPUTTYPE y = minusFunObj(allbins[tid], c);
            OUTPUTTYPE t = sumfunObj(y, myres);
            c = minusFunObj( minusFunObj( t, myres ), y);
            myres = t;
            //myres = sumfunObj(allbins[tid], myres);
          }
        }
        if (threadIdx.x == 0){
          if (first){
            result = myres;
          } else {
            OUTPUTTYPE y = minusFunObj(result, c);
            result = sumfunObj(myres, y);
          }
          ourOut[0] = result;
        }
    }
#endif
}




template <typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
          typename OUTPUTTYPE, typename INDEXT, typename MINUSFUNTYPE>
static
void callKahanReduceImpl(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE* out,
    cudaDeviceProp* props,
    cudaStream_t stream,
    bool outInDev,
    MINUSFUNTYPE minusFunObj, OUTPUTTYPE zero,
    int multidim,
    void** tmpbuf,
    size_t* tmpbufSize,
    bool accumulate
    )
{
    INDEXT size = end - start;
    if (end <= start)
    {
        return;
    }

    int maxblocks = 32;
    if (props) maxblocks = props->multiProcessorCount * 4;
    //int maxblocks = 16;


    if (size > 2*1024*1024 && multidim < 2){
        maxblocks *= 2;
    }

    if ((maxblocks << REDUCE_BLOCK_SIZE_LOG2) >= size){
      maxblocks = size >> REDUCE_BLOCK_SIZE_LOG2;
      if ((maxblocks << REDUCE_BLOCK_SIZE_LOG2) < size)
        maxblocks++;
    }

    // TODO: Magic constants..
    // With low bin-counts and large problems it seems beneficial to use
    // more blocks...
/*    if (size > 2*4096*4096)
        maxblocks *= 2;*/
    //printf("maxblocks = %d\n", maxblocks);

    OUTPUTTYPE* tmpOut;
    cudaMalloc((void**)&tmpOut, (maxblocks*multidim + 1) * sizeof(OUTPUTTYPE));
    PRINT_MALLOC(tmpOut);
    int sharedNeeded;
    {
        int typesize = sizeof(OUTPUTTYPE);
        sharedNeeded = (typesize) << (REDUCE_BLOCK_SIZE_LOG2);
        //printf("reduce-bin, generic, Shared needed = %d\n", sharedNeeded);
    }
    // Determine number of local variables
    // reduce_LOCALLIMIT is total local size available for one block:

    int nSteps = size / (maxblocks << REDUCE_BLOCK_SIZE_LOG2);
    if (nSteps * maxblocks * REDUCE_BLOCK_SIZE < size) nSteps++;
    if (nSteps > MAX_reduce_STEPS) nSteps = MAX_reduce_STEPS;

    int nFullSteps = size / (nSteps * maxblocks * REDUCE_BLOCK_SIZE);

    dim3 grid(maxblocks, multidim, 1);
    dim3 block = REDUCE_BLOCK_SIZE;
    bool first = true;
    for (int i = 0; i < nFullSteps; i++)
    {
        kahanKernel_reduce<false><<<grid, block, sharedNeeded, stream>>>(
            input, xformObj, sumfunObj, start, end, tmpOut, maxblocks, nSteps, i == 0, minusFunObj, zero);
        first = false;
        start += nSteps * maxblocks * REDUCE_BLOCK_SIZE;
#if R_ERROR_CHECKS
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
           printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
    }
    size = end - start;
    if (size > 0)
    {
        // Do what steps we still can do without checks
        nSteps = size / (maxblocks << REDUCE_BLOCK_SIZE_LOG2);
        if (nSteps * (maxblocks << REDUCE_BLOCK_SIZE_LOG2) < size) nSteps++;
        if (nSteps > 0)
        {
          kahanKernel_reduce<true><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, tmpOut,  maxblocks, nSteps, first, minusFunObj, zero);
            start += nSteps * maxblocks * REDUCE_BLOCK_SIZE;
            first = false;
        }
    }
#if R_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
       printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
    // Finally put together the result:
#if 1
    enum cudaMemcpyKind fromOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    enum cudaMemcpyKind toOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    if (stream != 0)
        cudaMemcpyAsync(&tmpOut[maxblocks*multidim], out, sizeof(OUTPUTTYPE), fromOut, stream);
    else
        cudaMemcpy(&tmpOut[maxblocks*multidim], out, sizeof(OUTPUTTYPE), fromOut);
    // Let's do so that one block handles one bin
    grid.x = 1;
    grid.y = 1;
    //grid.x = nOut >> REDUCE_BLOCK_SIZE_LOG2;
    //if ((grid.x << REDUCE_BLOCK_SIZE_LOG2) < nOut) grid.x++;
    block.x = FINAL_SUM_BLOCK_SIZE;
    finalKahanKernel<<<grid, block, 0, stream>>>(sumfunObj, tmpOut, maxblocks*multidim, minusFunObj, zero);
    // TODO: Use async copy for the results as well?
    if (stream != 0)
        cudaMemcpyAsync(out, tmpOut, sizeof(OUTPUTTYPE), toOut, stream);
    else
        cudaMemcpy(out, tmpOut, sizeof(OUTPUTTYPE), toOut);
#else
    {
      int i;
      OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(maxblocks * sizeof(OUTPUTTYPE));
      //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
      cudaMemcpy(h_tmp, tmpOut, maxblocks*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
      {
        OUTPUTTYPE res = *out;
        for (i = 0; i < maxblocks; i++)
        {
          res = sumfunObj(res, h_tmp[i]);
        }
        *out = res;
      }
      free(h_tmp);
    }
#endif
    cudaFree(tmpOut);
    PRINT_FREE(tmpOut);
}














template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE>
cudaError_t callReduceKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformFunctor,
    SUMFUNTYPE sumFunctor,
    INDEXTYPE start, INDEXTYPE end,
    OUTPUTTYPE* result,
    int multiReduce, // Convenience API - multiple entries per one input-index ran in parallel (simple multidim)
    cudaStream_t stream, bool outInDev,
    void** tmpbuf,
    size_t* tmpbufSize,bool accumulate)
{
    callReduceImpl(input, xformFunctor, sumFunctor, start, end, result, NULL, stream, outInDev, multiReduce, tmpbuf, tmpbufSize, accumulate);
    return cudaSuccess;
}

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename MINUSFUNTYPE, typename OUTPUTTYPE>
cudaError_t callKahanReduceKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformFunctor,
    SUMFUNTYPE sumFunctor,
    MINUSFUNTYPE minusFunctor,
    INDEXTYPE start, INDEXTYPE end,
    OUTPUTTYPE* result,
    OUTPUTTYPE zero,
    int multiReduce, // Convenience API - multiple entries per one input-index ran in parallel (simple multidim)
    cudaStream_t stream, bool outInDev,
    void** tmpbuf,
    size_t* tmpbufSize,
    bool accumulate)
{
    callKahanReduceImpl(input, xformFunctor, sumFunctor, start, end, result, NULL, stream, outInDev, minusFunctor, zero, multiReduce, tmpbuf, tmpbufSize, accumulate);
    return cudaSuccess;
}







template <typename nDimIndexFun, int nDim, typename USERINPUTTYPE, typename INDEXT, typename OUTPUTTYPE>
class wrapReduceInput
{
public:
    nDimIndexFun userIndexFun;
    INDEXT starts[nDim];
    //int ends[nDim];
    INDEXT sizes[nDim];
    float invSizes[nDim];
    __host__ __device__
    inline
    OUTPUTTYPE operator() (USERINPUTTYPE input, INDEXT i, int multiIndex) const {
    	INDEXT coords[nDim];
    	INDEXT tmpi = i;
  #pragma unroll
        for (int d=0; d < nDim - 1; d++)
        {
            // Example of how this logic works - imagine a cube of (10,100,1000), and take index 123 456
            // newI = 123 456 / 10 = 12 345, offset = 123 456 - 123 450 = 6 (this is our first coordinate!),
            // newI = 12 345 / 100 = 123,    offset = 12 345 - 12 300 = 45 (this is our second coordinate!),
            // newI = 123 / 1000 = 0,        offset = 123 - 0 = 123 (this is our last coordinate!)
            // Result = [123, 45, 6]
            INDEXT newI = (INDEXT)((float)tmpi * invSizes[d]);
            INDEXT offset = tmpi - newI * sizes[d];
            coords[d] = starts[d] + offset;
            tmpi = newI;
        }
        coords[nDim - 1] = starts[nDim - 1] + tmpi;
        // Now just call wrapped functor with right coordinate values
        return userIndexFun(input, coords, multiIndex);
    }
};



template <int nDim,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callReduceKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT* starts, INDEXT* ends,
    OUTPUTTYPE* out,
    int multiReduce,
    cudaStream_t stream, bool outInDev,
    void** tmpbuf,
    size_t* tmpbufSize,
    bool accumulate)
{
    wrapReduceInput<TRANSFORMFUNTYPE, nDim, INPUTTYPE, INDEXT, OUTPUTTYPE> wrapInput;
    INDEXT start = 0;
    INDEXT size = 1;
    for (int d = 0; d < nDim; d++)
    {
        wrapInput.starts[d] = starts[d];
        wrapInput.sizes[d] = ends[d] - starts[d];
        wrapInput.invSizes[d] = (float)(1.0 / ((double)wrapInput.sizes[d]));
        // Example: starts = [3, 10, 23], sizes = [10, 100, 1000]
        // start = 3 * 1 = 3, size = 10
        // start = 3 + 10 * 10 = 103, size = 10*100 = 1000
        // start = 103 + 1000*23 = 23 103, size = 1000*1000 = 1 000 000
        start += starts[d] * size;
        size *= wrapInput.sizes[d];
        if (ends[d] <= starts[d]) return cudaSuccess;
    }
    wrapInput.userIndexFun = xformObj;
    INDEXT end = start + size;

    return callReduceKernel(input, wrapInput, sumfunObj, start, end, out, multiReduce, stream, outInDev, tmpbuf, tmpbufSize, accumulate);
}



template <int nDim,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename MINUSFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callKahanReduceKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    MINUSFUNTYPE minusFunctor,
    INDEXT* starts, INDEXT* ends,
    OUTPUTTYPE* out,
    OUTPUTTYPE zero,
    int multiReduce,
    cudaStream_t stream, bool outInDev,
    void** tmpbuf,
    size_t* tmpbufSize,
    bool accumulate)
{
    wrapReduceInput<TRANSFORMFUNTYPE, nDim, INPUTTYPE, INDEXT, OUTPUTTYPE> wrapInput;
    INDEXT start = 0;
    INDEXT size = 1;
    for (int d = 0; d < nDim; d++)
    {
        wrapInput.starts[d] = starts[d];
        wrapInput.sizes[d] = ends[d] - starts[d];
        // Example: starts = [3, 10, 23], sizes = [10, 100, 1000]
        // start = 3 * 1 = 3, size = 10
        // start = 3 + 10 * 10 = 103, size = 10*100 = 1000
        // start = 103 + 1000*23 = 23 103, size = 1000*1000 = 1 000 000
        start += starts[d] * size;
        size *= wrapInput.sizes[d];
        if (ends[d] <= starts[d]) return cudaSuccess;
    }
    wrapInput.userIndexFun = xformObj;
    INDEXT end = start + size;
    return callKahanReduceKernel(input, wrapInput, sumfunObj, minusFunctor, start, end, out, zero, multiReduce, stream, outInDev, tmpbuf, tmpbufSize, accumulate);
}






#undef REDUCE_BLOCK_SIZE_LOG2
#undef REDUCE_BLOCK_SIZE
#undef MAX_reduce_STEPS
#undef R_ERROR_CHECKS

#endif // CUDA_REDUCE_H_

