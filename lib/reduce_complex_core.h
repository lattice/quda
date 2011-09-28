
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

  if (tid < 32) {
    volatile QudaSumComplex *sv = s;
    if (reduce_threads >=  64) { ZCACC(sv[0],sv[1],sv[64+0],sv[64+1]); }
    if (reduce_threads >=  32) { ZCACC(sv[0],sv[1],sv[32+0],sv[32+1]); }
    if (reduce_threads >=  16) { ZCACC(sv[0],sv[1],sv[16+0],sv[16+1]); }
    if (reduce_threads >=   8) { ZCACC(sv[0],sv[1], sv[8+0], sv[8+1]); }
    if (reduce_threads >=   4) { ZCACC(sv[0],sv[1], sv[4+0], sv[4+1]); }
    if (reduce_threads >=   2) { ZCACC(sv[0],sv[1], sv[2+0], sv[2+1]); }
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

  // all real then imaginary to avoid bank conflicts
  extern __shared__ QudaSumFloat cdata[];
  QudaSumFloat *sx = cdata + tid;
  QudaSumFloat *sy = cdata + tid + reduce_threads;

  sx[0] = sum.x;
  sy[0] = sum.y;
  __syncthreads();
  
  // do reduction in shared mem
  if (reduce_threads >= 1024) { if (tid < 512) { sx[0] += sx[512]; sy[0] += sy[512]; } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { sx[0] += sx[256]; sy[0] += sy[256]; } __syncthreads(); }
  if (reduce_threads >= 256) { if (tid < 128) { sx[0] += sx[128]; sy[0] += sy[128]; } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { sx[0] += sx[ 64]; sy[0] += sy[ 64]; } __syncthreads(); }
  
  if (tid < 32) {
    volatile QudaSumFloat *svx = sx;
    volatile QudaSumFloat *svy = sy;
    if (reduce_threads >=  64) { svx[0] += svx[32]; svy[0] += svy[32]; }
    if (reduce_threads >=  32) { svx[0] += svx[16]; svy[0] += svy[16]; }
    if (reduce_threads >=  16) { svx[0] += svx[ 8]; svy[0] += svy[ 8]; }
    if (reduce_threads >=   8) { svx[0] += svx[ 4]; svy[0] += svy[ 4]; }
    if (reduce_threads >=   4) { svx[0] += svx[ 2]; svy[0] += svy[ 2]; }
    if (reduce_threads >=   2) { svx[0] += svx[ 1]; svy[0] += svy[ 1]; ; }
  }
  
  // write result for this block to global mem 
  if (tid == 0) {
    g_odata[blockIdx.x].x = sx[0];
    g_odata[blockIdx.x].y = sy[0];
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
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 4 * sizeof(QudaSumComplex) : blasBlock.x * 2 * sizeof(QudaSumComplex);
#else
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(QudaSumComplex) : blasBlock.x * sizeof(QudaSumComplex);
#endif

  if (blasBlock.x == 32) {
    REDUCE_FUNC_NAME(Kernel)<32><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  } else if (blasBlock.x == 64) {
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
  for (unsigned int i = 0; i < blasGrid.x; i++) {
    gpu_result.x += h_reduceComplex[i].x;
    gpu_result.y += h_reduceComplex[i].y;
  }

  reduceDoubleArray(&(gpu_result.x), 2);

  return gpu_result;
}


