__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumType *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*reduce_threads + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
  
  QudaSumType sum;
  ZERO(sum);
  while (i < n) {
    REDUCE_AUXILIARY(i);
    sum += REDUCE_OPERATION(i);
    i += gridSize;
  }

  extern __shared__ QudaSumFloat sdata[];
  QudaSumFloat *s = sdata + tid;
  
  SET_SUM(s, 0, sum);
  __syncthreads();
  
  // do reduction in shared mem
  if (reduce_threads >= 1024) { if (tid < 512) { ADD_SUM(s, 512); } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { ADD_SUM(s, 256); } __syncthreads(); }
  if (reduce_threads >= 256) { if (tid < 128) { ADD_SUM(s, 128); } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { ADD_SUM(s, 64); } __syncthreads(); }
  
  if (tid < 32) {
    volatile QudaSumFloat *sv = s;
    if (reduce_threads >=  64) { ADD_SUM(sv, 32); }
    if (reduce_threads >=  32) { ADD_SUM(sv, 16); }
    if (reduce_threads >=  16) { ADD_SUM(sv, 8); }
    if (reduce_threads >=  8)  { ADD_SUM(sv, 4); }
    if (reduce_threads >=  4)  { ADD_SUM(sv, 2); }
    if (reduce_threads >=  2)  { ADD_SUM(sv, 1); }
  }
  
  // write result for this block to global mem 
  if (tid == 0) WRITE_REDUCTION(g_odata[blockIdx.x], s, 0);
}

template <typename doubleN, typename Float, typename Float2>
doubleN REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {
  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(QudaSumType) : 
    blasBlock.x * sizeof(QudaSumType);

  QudaSumType *reduce = (QudaSumType*)d_reduce;
  if (blasBlock.x == 32) {
    REDUCE_FUNC_NAME(Kernel)<32><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else if (blasBlock.x == 64) {
    REDUCE_FUNC_NAME(Kernel)<64><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else if (blasBlock.x == 128) {
    REDUCE_FUNC_NAME(Kernel)<128><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else if (blasBlock.x == 256) {
    REDUCE_FUNC_NAME(Kernel)<256><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else if (blasBlock.x == 512) {
    REDUCE_FUNC_NAME(Kernel)<512><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else if (blasBlock.x == 1024) {
    REDUCE_FUNC_NAME(Kernel)<1024><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, reduce, n);
  } else {
    errorQuda("Reduction not implemented for %d threads", blasBlock.x);
  }

  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduce, d_reduce, blasGrid.x*sizeof(QudaSumType), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();

  doubleN cpu_sum = 0;
  for (unsigned int i = 0; i < blasGrid.x; i++) cpu_sum += h_reduce[i];
  reduceDoubleArray((double*)&cpu_sum, NREDUCE);

  return cpu_sum;
}
