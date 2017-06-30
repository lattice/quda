/*
 * cuda_forall.h
 *
 *  Created on: 27.3.2012
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2011-2012 Teemu Rantalaiho
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

#include <cuda_runtime_api.h>

#define FOR_ALL_BLOCK_SIZE_LOG2     7
#define FOR_ALL_BLOCK_SIZE          (1 << FOR_ALL_BLOCK_SIZE_LOG2)

// As simple as possible

template <int UnrollLog2, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
__global__
void cuda_transFormKernel(INPUTTYPE input, TRANSFORMFUNTYPE functor, INDEXTYPE start, INDEXTYPE end, int nsteps) {
    INDEXTYPE idx = (INDEXTYPE)(blockIdx.x * blockDim.x + threadIdx.x) + start;
    int nFullnsteps = nsteps >> UnrollLog2;
    int stride = (blockDim.x * gridDim.x);
    for (int step = 0; step < nFullnsteps; step++ ){
#pragma unroll
        for (int substep = 0; substep < (1<<UnrollLog2); substep++){
            functor(input, idx, blockIdx.y);
            idx += stride;
        }
    }
    for (int step = nFullnsteps << UnrollLog2; step < nsteps; step++){
        if (idx < end){
          functor(input, idx, blockIdx.y);
        }
        idx += stride;
    }
}


template <int UnrollNlog2Steps, typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
static inline
void callTransformKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE functionObject,
    INDEXTYPE start, INDEXTYPE end,
    int nMultiXform = 1,
    cudaStream_t stream = 0) {
  if (end <= start)
    return;
  INDEXTYPE size = end - start;
  const dim3 block = FOR_ALL_BLOCK_SIZE;
  int paddedSize = (size + (FOR_ALL_BLOCK_SIZE) - 1) & (~((FOR_ALL_BLOCK_SIZE) - 1));
  dim3 grid = paddedSize >> ( FOR_ALL_BLOCK_SIZE_LOG2 );
  grid.y = nMultiXform;
  int steps = 1;
  if (grid.x > (1 << 12)){
      grid.x = (1 << 12);
      steps = size >> (FOR_ALL_BLOCK_SIZE_LOG2 + 12);
      if (steps << (FOR_ALL_BLOCK_SIZE_LOG2 + 12) < size) steps++;
  }
  cuda_transFormKernel<UnrollNlog2Steps><<<grid, block, 0, stream>>>(input, functionObject, start, end, steps);
}

// TODO: Error reporting?


template <typename nDimIndexFun, int nDim, typename USERINPUTTYPE, typename INDEXT>
class wrapXFormInput
{
public:
    nDimIndexFun userIndexFun;
    INDEXT starts[nDim];
    //int ends[nDim];
    INDEXT sizes[nDim];
    __host__ __device__
    void operator() (USERINPUTTYPE input, INDEXT i, int multiIndex) const {
    	INDEXT coords[nDim];
    	INDEXT tmpi = i;
  #pragma unroll
        for (int d=0; d < nDim; d++)
        {
            // Example of how this logic works - imagine a cube of (10,100,1000), and take index 123 456
            // newI = 123 456 / 10 = 12 345, offset = 123 456 - 123 450 = 6 (this is our first coordinate!),
            // newI = 12 345 / 100 = 123,    offset = 12 345 - 12 300 = 45 (this is our second coordinate!),
            // newI = 123 / 1000 = 0,        offset = 123 - 0 = 123 (this is our last coordinate!)
            // Result = [123, 45, 6]
            INDEXT newI = tmpi / sizes[d];
            INDEXT offset = tmpi - newI * sizes[d];
            coords[d] = starts[d] + offset;
            tmpi = newI;
        }
        // Now just call wrapped functor with right coordinate values
        userIndexFun(input, coords, multiIndex);
    }
};



template <int nDim, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXT>
cudaError_t
callXformKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    INDEXT* starts, INDEXT* ends,
    int nMultiXform = 1,
    cudaStream_t stream = 0)
{
    wrapXFormInput<TRANSFORMFUNTYPE, nDim, INPUTTYPE, INDEXT> wrapInput;
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

    callTransformKernel(input, wrapInput, start, end, nMultiXform, stream);
    return cudaSuccess;
}



#undef FOR_ALL_BLOCK_SIZE
#undef FOR_ALL_BLOCK_SIZE_LOG2



