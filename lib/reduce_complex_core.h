
#if (REDUCE_TYPE == REDUCE_KAHAN)


#define DSACC(c0, c1, a0, a1) dsadd((c0), (c1), (c0), (c1), (a0), (a1))
#define ZCACC(c0, c1, a0, a1) zcadd((c0), (c1), (c0), (c1), (a0), (a1))

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumComplex *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(reduce_threads) + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
  
  QudaSumFloat acc0 = 0;
  QudaSumFloat acc1 = 0;
  QudaSumFloat acc2 = 0;
  QudaSumFloat acc3 = 0;
  
  while (i < n) {
    REDUCE_REAL_AUXILIARY(i);
    REDUCE_IMAG_AUXILIARY(i);
    DSACC(acc0, acc1, REDUCE_REAL_OPERATION(i), 0);
    DSACC(acc2, acc3, REDUCE_IMAG_OPERATION(i), 0);
    i += gridSize;
  }
  
  extern __shared__ QudaSumComplex cdata[];
  QudaSumComplex *s = cdata + 2*tid;
  s[0].x = acc0;
  s[1].x = acc1;
  s[0].y = acc2;
  s[1].y = acc3;
  
  __syncthreads();
  
  if (reduce_threads >= 1024) { if (tid < 512) { ZCACC(s[0],s[1],s[1024+0],s[1024+1]); } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { ZCACC(s[0],s[1],s[512+0],s[512+1]); } __syncthreads(); }    
  if (reduce_threads >= 256) { if (tid < 128) { ZCACC(s[0],s[1],s[256+0],s[256+1]); } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { ZCACC(s[0],s[1],s[128+0],s[128+1]); } __syncthreads(); }    

#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      volatile QudaSumComplex *sv = s;
      if (reduce_threads >=  64) { ZCACC(sv[0],sv[1],sv[64+0],sv[64+1]); EMUSYNC; }
      if (reduce_threads >=  32) { ZCACC(sv[0],sv[1],sv[32+0],sv[32+1]); EMUSYNC; }
      if (reduce_threads >=  16) { ZCACC(sv[0],sv[1],sv[16+0],sv[16+1]); EMUSYNC; }
      if (reduce_threads >=   8) { ZCACC(sv[0],sv[1], sv[8+0], sv[8+1]); EMUSYNC; }
      if (reduce_threads >=   4) { ZCACC(sv[0],sv[1], sv[4+0], sv[4+1]); EMUSYNC; }
      if (reduce_threads >=   2) { ZCACC(sv[0],sv[1], sv[2+0], sv[2+1]); EMUSYNC; }
    }
  
  // write result for this block to global mem as single QudaSumComplex
  if (tid == 0) {
    g_odata[blockIdx.x].x = cdata[0].x+cdata[1].x;
    g_odata[blockIdx.x].y = cdata[0].y+cdata[1].y;
  }
}

#else

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumComplex *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*reduce_threads + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
  
  extern __shared__ QudaSumComplex cdata[];
  QudaSumComplex *s = cdata + tid;

  QudaSumComplex sum;
  sum.x = 0.0;
  sum.y = 0.0;
  
  while (i < n) {
    REDUCE_REAL_AUXILIARY(i);
    REDUCE_IMAG_AUXILIARY(i);
    sum.x += REDUCE_REAL_OPERATION(i);
    sum.y += REDUCE_IMAG_OPERATION(i);
    i += gridSize;
  }
  s[0] = sum;
  __syncthreads();
  
  // do reduction in shared mem
  if (reduce_threads >= 1024) { if (tid < 512) { s[0].x += s[512].x; s[0].y += s[512].y; } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { s[0].x += s[256].x; s[0].y += s[256].y; } __syncthreads(); }
  if (reduce_threads >= 256) { if (tid < 128) { s[0].x += s[128].x; s[0].y += s[128].y; } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { s[0].x += s[ 64].x; s[0].y += s[ 64].y; } __syncthreads(); }
  
#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      volatile QudaSumComplex *sv = s;
      if (reduce_threads >=  64) { sv[0].x += sv[32].x; sv[0].y += sv[32].y; EMUSYNC; }
      if (reduce_threads >=  32) { sv[0].x += sv[16].x; sv[0].y += sv[16].y; EMUSYNC; }
      if (reduce_threads >=  16) { sv[0].x += sv[ 8].x; sv[0].y += sv[ 8].y; EMUSYNC; }
      if (reduce_threads >=   8) { sv[0].x += sv[ 4].x; sv[0].y += sv[ 4].y; EMUSYNC; }
      if (reduce_threads >=   4) { sv[0].x += sv[ 2].x; sv[0].y += sv[ 2].y; EMUSYNC; }
      if (reduce_threads >=   2) { sv[0].x += sv[ 1].x; sv[0].y += sv[ 1].y; EMUSYNC; }
    }
  
  // write result for this block to global mem 
  if (tid == 0) {
    g_odata[blockIdx.x].x = s[0].x;
    g_odata[blockIdx.x].y = s[0].y;
  }
}

#endif

template <typename Float, typename Float2>
cuDoubleComplex REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {

  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_complex: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
#if (REDUCE_TYPE == REDUCE_KAHAN)
  int smemSize = blasBlock.x * 2 * sizeof(QudaSumComplex);
#else
  int smemSize = blasBlock.x * sizeof(QudaSumComplex);
#endif

  if (blasBlock.x == 64) {
    REDUCE_FUNC_NAME(Kernel)<64><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else if (blasBlock.x == 128) {
    REDUCE_FUNC_NAME(Kernel)<128><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else if (blasBlock.x == 256) {
    REDUCE_FUNC_NAME(Kernel)<256><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else if (blasBlock.x == 512) {
    REDUCE_FUNC_NAME(Kernel)<512><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else if (blasBlock.x == 1024) {
    REDUCE_FUNC_NAME(Kernel)<1024><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else {
    errorQuda("Reduction not implemented for %d threads", blasBlock.x);
  }

  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduceComplex, d_reduceComplex, blasGrid.x*sizeof(QudaSumComplex), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();
  
  cuDoubleComplex gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  for (int i = 0; i < blasGrid.x; i++) {
    gpu_result.x += h_reduceComplex[i].x;
    gpu_result.y += h_reduceComplex[i].y;
  }
  
  return gpu_result;
}


