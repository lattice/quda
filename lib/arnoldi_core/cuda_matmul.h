/*
 * cuda_matmul.h
 *
 *  Created on: 21.5.2013
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
#ifndef CUDA_MATMUL_H_
#define CUDA_MATMUL_H_

#include <cuda_runtime_api.h>

/*
 * Signatures:
 *
 *   TRANSFORMFUNTYPE:
 *     struct <name>{
 *       OUTPUTTYPE operator()( INPUTTYPE input, INDEXTYPE x, INDEXTYPE y, SRCTYPE src, INDEXTYPE srcx);
 *     };
 *     // NOTE: when srcx = (transpose==true) ? y : x;
 *   SUMFUNTYPE:
 *     struct <name>{
 *       OUTPUTTYPE operator()( OUTPUTTYPE x, OUTPUTTYPE y);
 *     };
 *   STOREFUNTYPE:
 *     struct <name> {
 *       void operator()( DSTTYPE result, INDEXTYPE i, OUTPUTTYPE var){
 *          result[i] = var;
 *       }
 *     };
 *
 */
template <typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename SRCTYPE, typename DSTTYPE>
static inline
cudaError_t callFullMatMul(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    bool transpose = false, cudaStream_t stream = 0, bool outInDev = true);


template <typename INDEXTYPE, typename RADIXTYPE>
static inline
cudaError_t callFloatMatMul(
    const RADIXTYPE* mat, INDEXTYPE sizex, INDEXTYPE sizey, INDEXTYPE stride,
    const RADIXTYPE* src, RADIXTYPE* result, bool transpose = false,
    cudaStream_t stream = 0, bool outInDev = true);




#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __device__
#define __device__
#endif



#define MM_BLOCKSIZE_LOG2   8
#define MM_BLOCKSIZE        (1<<MM_BLOCKSIZE_LOG2)

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename OUTPUTTYPE, typename SRCTYPE, typename DSTTYPE>
__global__
void callFullMatmulKernel(
        INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
        INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
        int nstepsX, INDEXTYPE starty = 0, OUTPUTTYPE* tmpOut = NULL)
{
    int tid = threadIdx.x;
    INDEXTYPE y = (INDEXTYPE)(blockIdx.y + starty);
    INDEXTYPE x = (INDEXTYPE)(tid + blockIdx.x * blockDim.x);
    OUTPUTTYPE myRes;
    int stride = gridDim.x << MM_BLOCKSIZE_LOG2;
    if (x < sizex && y < sizey){
        myRes = xformFunctor(input, x, y, src, x);
        x += stride;
    }
//#ifndef UNROLL_NLOG2_CUDA_STEPS
#define NUNROLL_LOG2 2
//#else
//#define NUNROLL_LOG2 UNROLL_NLOG2_CUDA_STEPS
//#endif

#define NUNROLL (1 << NUNROLL_LOG2)
    int nFullSteps = (nstepsX - 1) >> NUNROLL_LOG2;
    for (int fstep = 0; fstep < nFullSteps; fstep++){
#pragma unroll
        for (int substep = 0; substep < NUNROLL; substep++){
            OUTPUTTYPE tmpres = xformFunctor(input, x, y, src, x);
            myRes = sumFunctor(myRes, tmpres);
            x += stride;
        }
    }
    while (x < sizex){
        OUTPUTTYPE tmpres = xformFunctor(input, x, y, src, x);
        myRes = sumFunctor(myRes, tmpres);
        x += stride;
    }
    {
        __shared__ OUTPUTTYPE tmparr[MM_BLOCKSIZE];
        tmparr[tid] = myRes;
        __syncthreads();
        if (sizex < MM_BLOCKSIZE){
            if (tid == 0){
                for (int i = 1; i < sizex; i++)
                    myRes = sumFunctor(myRes, tmparr[i]);
            }
        } else {
            if (tid < 32){
#pragma unroll
                for (int i = 1; i < (MM_BLOCKSIZE >> 5); i++)
                    myRes = sumFunctor(myRes, tmparr[tid + (i<<5)]);
                tmparr[tid] = myRes;
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+16]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+8]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+4]);
                __threadfence_block();
                tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+2]);
                __threadfence_block();
                myRes = sumFunctor(tmparr[tid], tmparr[tid+1]);
            }
        }
    }
    if (threadIdx.x == 0){
        if (tmpOut)
            tmpOut[blockIdx.x + blockIdx.y * gridDim.x] = myRes;
        else
            storeDstFun(result, y, myRes);
    }
}

#define FINALSUMT 64

template <typename INDEXTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename OUTPUTTYPE, typename DSTTYPE>
__global__
void FinalSumsKernel(
        SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
        INDEXTYPE sizex, OUTPUTTYPE* tmpptr, DSTTYPE result, INDEXTYPE startY)
{
    __shared__ OUTPUTTYPE tmparr[FINALSUMT];
    OUTPUTTYPE myRes;
    int tid = threadIdx.x;
    if (tid < sizex)
        tmparr[tid] = tmpptr[tid + blockIdx.x * sizex];
    __syncthreads();
    if (tid >= FINALSUMT/2)
      return;
    if (sizex == FINALSUMT){
#if FINALSUMT == 64
        __threadfence_block();
        tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+32]);
#endif
        __threadfence_block();
        tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+16]);
        __threadfence_block();
        tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+8]);
        __threadfence_block();
        tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+4]);
        __threadfence_block();
        tmparr[tid] = sumFunctor(tmparr[tid], tmparr[tid+2]);
        __threadfence_block();
        myRes = sumFunctor(tmparr[tid], tmparr[tid+1]);
    } else {
        __threadfence_block();
        myRes = tmparr[0];
        for (int i = 1; i < sizex; i++)
            myRes = sumFunctor(myRes, tmparr[i]);
    }
    if (tid == 0)
        storeDstFun(result, blockIdx.x + startY, myRes);

}



static inline int divLog2RoundUp(int size, int divlog2)
{
    int div = 1 << divlog2;
    int paddedSize = (size + div - 1) & (~(div - 1));
    int res = paddedSize >> divlog2;
    return res;
}

#include <stdio.h>

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename OUTPUTTYPE, typename SRCTYPE, typename DSTTYPE>
static
cudaError_t callFullMatMulImpl(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    cudaStream_t stream, bool outInDev)
{
  //int stepsX = divLog2RoundUp(sizex, MM_BLOCKSIZE_LOG2);
    int stepsX = sizex >> MM_BLOCKSIZE_LOG2;
    INDEXTYPE blocksX = 1;
    OUTPUTTYPE* tmpptr = NULL;
    dim3 block = MM_BLOCKSIZE;
    if (sizey < 32 && stepsX > 64 * NUNROLL){
        blocksX = FINALSUMT;
        if (blocksX > stepsX)
            blocksX = stepsX;
        stepsX = (sizex/blocksX) >> MM_BLOCKSIZE_LOG2;
    }
    dim3 grid(blocksX, sizey, 1);

    if (!outInDev){
        printf("Sorry - no support yet for CPU-output buffers...\n");
        return cudaSuccess;
    }
    if (sizex <= 0 || sizey <= 0)
        return cudaSuccess;
    if (blocksX > 1){
        size_t needed = sizeof(OUTPUTTYPE) * blocksX * (sizey < 32768 ? sizey : 32768);
        cudaMalloc(&tmpptr, needed);
    }

/*    printf("block = (%d, %d, %d), grid = (%d,%d)\n", block.x, block.y, block.z, grid.x, grid.y);*/
    if (sizey > 32768){
        grid.y = 32768;
        int startY = 0;
        while (startY < sizey){
            if (startY + 32768 > sizey)
                grid.y = sizey - startY;
            callFullMatmulKernel
                    <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
                    <<<grid, block,0,stream>>>
                    (input, xformFunctor, sumFunctor, storeDstFun,
                    sizex, sizey, src, result, stepsX, startY, tmpptr);
            if (blocksX > 1){
                dim3 fgrid = 32768;
                dim3 fblock = FINALSUMT;
                FinalSumsKernel<INDEXTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, DSTTYPE>
                    <<<fgrid, fblock,0,stream>>>(sumFunctor, storeDstFun, blocksX, tmpptr, result, startY);
            }
            startY += 32768;
        }
    }
    else
    {
        callFullMatmulKernel
                <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
                <<<grid, block,0,stream>>>
                (input, xformFunctor, sumFunctor, storeDstFun, sizex, sizey, src, result, stepsX, 0, tmpptr);
        if (blocksX > 1){
            dim3 fgrid = sizey;
            dim3 fblock =  FINALSUMT;
            FinalSumsKernel<INDEXTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, DSTTYPE>
                <<<fgrid, fblock,0,stream>>>(sumFunctor, storeDstFun, blocksX, tmpptr, result, 0);
        }

    }
    return cudaGetLastError();
}

template <typename INDEXTYPE, typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INPUTTYPE, typename SRCTYPE>
struct FullMatMulTransposeWrapper {
	TRANSFORMFUNTYPE userFunctor;
	inline __device__
    OUTPUTTYPE operator()( INPUTTYPE input, INDEXTYPE x, INDEXTYPE y, SRCTYPE src, INDEXTYPE srcx){
        return userFunctor(input, y, x, src, x);
    }
};


template <typename OUTPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE, typename SUMFUNTYPE, typename STOREFUNTYPE, typename SRCTYPE, typename DSTTYPE>
cudaError_t callFullMatMul(
    INPUTTYPE input, TRANSFORMFUNTYPE xformFunctor, SUMFUNTYPE sumFunctor, STOREFUNTYPE storeDstFun,
    INDEXTYPE sizex, INDEXTYPE sizey, SRCTYPE src, DSTTYPE result,
    bool transpose, cudaStream_t stream, bool outInDev)
{
	cudaError_t err;
	if (transpose){
		struct FullMatMulTransposeWrapper<INDEXTYPE, OUTPUTTYPE, TRANSFORMFUNTYPE,INPUTTYPE,SRCTYPE> wrapperFun;
		wrapperFun.userFunctor = xformFunctor;
		err = callFullMatMulImpl
		        <FullMatMulTransposeWrapper<INDEXTYPE, OUTPUTTYPE, TRANSFORMFUNTYPE,INPUTTYPE,SRCTYPE>,
		            INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
		        (input, wrapperFun, sumFunctor, storeDstFun, sizey, sizex, src, result, stream, outInDev);
	}
	else
	{

	    err = callFullMatMulImpl
	            <TRANSFORMFUNTYPE, INDEXTYPE, INPUTTYPE, SUMFUNTYPE, STOREFUNTYPE, OUTPUTTYPE, SRCTYPE, DSTTYPE>
		        (input, xformFunctor, sumFunctor, storeDstFun, sizex, sizey, src, result, stream, outInDev);
	}
	return err;
}


template <typename INDEXTYPE, typename RADIXTYPE>
struct floatMatMulFun {
    INDEXTYPE stride;
    __device__
    RADIXTYPE operator()( const RADIXTYPE* data, INDEXTYPE x, INDEXTYPE y, const RADIXTYPE* src, INDEXTYPE srcx){
        RADIXTYPE m_ij = data[y*stride + x];
        RADIXTYPE res = m_ij*src[srcx];
        return res;
    }
};

template <typename RADIXTYPE>
struct radixSumFun {
    __device__
    RADIXTYPE operator()( RADIXTYPE a, RADIXTYPE b){
        RADIXTYPE res = a + b;
        return res;
    }
};
template <typename INDEXTYPE, typename RADIXTYPE>
struct storefloatFun {
    __device__
    void operator()( RADIXTYPE* result, INDEXTYPE i, RADIXTYPE var){
        result[i] = var;
    }
};



template <typename INDEXTYPE, typename RADIXTYPE>
cudaError_t callFloatMatMul(
    const RADIXTYPE* mat, INDEXTYPE sizex, INDEXTYPE sizey, INDEXTYPE stride,
    const RADIXTYPE* src, RADIXTYPE* result, bool transpose,
    cudaStream_t stream, bool outInDev)
{
    struct floatMatMulFun<INDEXTYPE, RADIXTYPE> mulfun;
    struct radixSumFun<RADIXTYPE> sumfun;
    struct storefloatFun<INDEXTYPE, RADIXTYPE> storeDstfun;
    mulfun.stride = stride;
    cudaError_t err = callFullMatMul<RADIXTYPE, floatMatMulFun<INDEXTYPE, RADIXTYPE>, INDEXTYPE, const RADIXTYPE*, radixSumFun<RADIXTYPE>,
                                     storefloatFun<INDEXTYPE, RADIXTYPE>, const RADIXTYPE*, RADIXTYPE*>
        (mat, mulfun, sumfun, storeDstfun, sizex, sizey, src, result, transpose, stream, outInDev);
    if (err != cudaSuccess)
    	printf("Error in callFullMatMul! err = %s\n", cudaGetErrorString(err));
    return err;
}



#endif /* CUDA_MATMUL_H_ */
