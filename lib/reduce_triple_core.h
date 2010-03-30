
#if (REDUCE_TYPE == REDUCE_KAHAN)

#define DSACC(c0, c1, a0, a1) dsadd((c0), (c1), (c0), (c1), (a0), (a1))
#define DSACC3(c0, c1, a0, a1) dsadd3((c0), (c1), (c0), (c1), (a0), (a1))

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat3 *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(reduce_threads) + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
    
  QudaSumFloat acc0 = 0;
  QudaSumFloat acc1 = 0;
  QudaSumFloat acc2 = 0;
  QudaSumFloat acc3 = 0;
  QudaSumFloat acc4 = 0;
  QudaSumFloat acc5 = 0;

  while (i < n) {
    REDUCE_X_AUXILIARY(i);
    REDUCE_Y_AUXILIARY(i);
    REDUCE_Z_AUXILIARY(i);
    DSACC(acc0, acc1, REDUCE_X_OPERATION(i), 0);
    DSACC(acc2, acc3, REDUCE_Y_OPERATION(i), 0);
    DSACC(acc4, acc5, REDUCE_Z_OPERATION(i), 0);
    i += gridSize;
  }

  extern __shared__ QudaSumFloat3 tdata[];
  QudaSumFloat3 *s = tdata + 2*tid;
  s[0].x = acc0;
  s[1].x = acc1;
  s[0].y = acc2;
  s[1].y = acc3;
  s[0].z = acc4;
  s[1].z = acc5;
    
  __syncthreads();
    
  if (reduce_threads >= 1024) { if (tid < 512) { DSACC3(s[0],s[1],s[1024+0],s[1024+1]); } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { DSACC3(s[0],s[1],s[512+0],s[512+1]); } __syncthreads(); }    
  if (reduce_threads >= 256) { if (tid < 128) { DSACC3(s[0],s[1],s[256+0],s[256+1]); } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { DSACC3(s[0],s[1],s[128+0],s[128+1]); } __syncthreads(); }    

#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      if (reduce_threads >=  64) { DSACC3(s[0],s[1],s[64+0],s[64+1]); EMUSYNC; }
      if (reduce_threads >=  32) { DSACC3(s[0],s[1],s[32+0],s[32+1]); EMUSYNC; }
      if (reduce_threads >=  16) { DSACC3(s[0],s[1],s[16+0],s[16+1]); EMUSYNC; }
      if (reduce_threads >=   8) { DSACC3(s[0],s[1], s[8+0], s[8+1]); EMUSYNC; }
      if (reduce_threads >=   4) { DSACC3(s[0],s[1], s[4+0], s[4+1]); EMUSYNC; } 
      if (reduce_threads >=   2) { DSACC3(s[0],s[1], s[2+0], s[2+1]); EMUSYNC; }
    }
    
  // write result for this block to global mem as single QudaSumFloat3
  if (tid == 0) {
    g_odata[blockIdx.x].x = tdata[0].x+tdata[1].x;
    g_odata[blockIdx.x].y = tdata[0].y+tdata[1].y;
    g_odata[blockIdx.x].z = tdata[0].z+tdata[1].z;
  }
}

#else

#define SUM_DOUBLE3(i, j)			\
  s[i].x += s[j].x;				\
  s[i].y += s[j].y;				\
  s[i].z += s[j].z;

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat3 *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*reduce_threads + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;

  extern __shared__ QudaSumFloat3 tdata[];
  QudaSumFloat3 *s = tdata + tid;
  s[0].x = 0;
  s[0].y = 0;
  s[0].z = 0;

  while (i < n) {
    REDUCE_X_AUXILIARY(i);
    REDUCE_Y_AUXILIARY(i);
    REDUCE_Z_AUXILIARY(i);
    s[0].x += REDUCE_X_OPERATION(i);
    s[0].y += REDUCE_Y_OPERATION(i);
    s[0].z += REDUCE_Z_OPERATION(i);
    i += gridSize;
  }
  
  __syncthreads();

  // do reduction in shared mem
  if (reduce_threads >= 1024) 
    { if (tid < 512) { SUM_DOUBLE3(0, 512); } __syncthreads(); }
  if (reduce_threads >= 512) 
    { if (tid < 256) { SUM_DOUBLE3(0, 256); } __syncthreads(); }
  if (reduce_threads >= 256) 
    { if (tid < 128) { SUM_DOUBLE3(0, 128); } __syncthreads(); }
  if (reduce_threads >= 128) 
    { if (tid <  64) { SUM_DOUBLE3(0, 64); } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      if (reduce_threads >=  64) { SUM_DOUBLE3(0,32); EMUSYNC; }
      if (reduce_threads >=  32) { SUM_DOUBLE3(0,16); EMUSYNC; }
      if (reduce_threads >=  16) { SUM_DOUBLE3(0,8); EMUSYNC; }
      if (reduce_threads >=   8) { SUM_DOUBLE3(0,4); EMUSYNC; }
      if (reduce_threads >=   4) { SUM_DOUBLE3(0,2); EMUSYNC; }
      if (reduce_threads >=   2) { SUM_DOUBLE3(0,1); EMUSYNC; }
    }
    
  // write result for this block to global mem 
  if (tid == 0) {
    g_odata[blockIdx.x].x = s[0].x;
    g_odata[blockIdx.x].y = s[0].y;
    g_odata[blockIdx.x].z = s[0].z;
  }
}

#undef SUM_DOUBLE3

#endif

template <typename Float2>
double3 REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {

  setBlock(kernel, n, precision);
  
  if (n % blasBlock.x != 0) {
    errorQuda("reduce_triple_core: length %d must be a multiple of %d", n, blasBlock.x);
  }  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_triple_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
#if (REDUCE_TYPE == REDUCE_KAHAN)
  int smemSize = blasBlock.x * 2 * sizeof(QudaSumFloat3);
#else
  int smemSize = blasBlock.x * sizeof(QudaSumFloat3);
#endif

  if (blasBlock.x == 64) {
    REDUCE_FUNC_NAME(Kernel)<64><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  } else if (blasBlock.x == 128) {
    REDUCE_FUNC_NAME(Kernel)<128><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  } else if (blasBlock.x == 256) {
    REDUCE_FUNC_NAME(Kernel)<256><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  } else if (blasBlock.x == 512) {
    REDUCE_FUNC_NAME(Kernel)<512><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  } else if (blasBlock.x == 1024) {
    REDUCE_FUNC_NAME(Kernel)<1024><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  } else {
    errorQuda("Reduction not implemented for %d threads", blasBlock.x);
  }

  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduceFloat3, d_reduceFloat3, blasGrid.x*sizeof(QudaSumFloat3), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();
  
  double3 gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  gpu_result.z = 0;
  for (int i = 0; i < blasGrid.x; i++) {
    gpu_result.x += h_reduceFloat3[i].x;
    gpu_result.y += h_reduceFloat3[i].y;
    gpu_result.z += h_reduceFloat3[i].z;
  }
  
  return gpu_result;
}


