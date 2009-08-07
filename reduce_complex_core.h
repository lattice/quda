
#if (REDUCE_TYPE == REDUCE_KAHAN)


#define DSACC(c0, c1, a0, a1) dsadd((c0), (c1), (c0), (c1), (a0), (a1))
#define ZCACC(c0, c1, a0, a1) zcadd((c0), (c1), (c0), (c1), (a0), (a1))

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumComplex *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(REDUCE_THREADS) + threadIdx.x;
  unsigned int gridSize = REDUCE_THREADS*gridDim.x;
  
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
  
  if (REDUCE_THREADS >= 256) { if (tid < 128) { ZCACC(s[0],s[1],s[256+0],s[256+1]); } __syncthreads(); }
  if (REDUCE_THREADS >= 128) { if (tid <  64) { ZCACC(s[0],s[1],s[128+0],s[128+1]); } __syncthreads(); }    
  if (tid < 32) {
    if (REDUCE_THREADS >=  64) { ZCACC(s[0],s[1],s[64+0],s[64+1]); }
    if (REDUCE_THREADS >=  32) { ZCACC(s[0],s[1],s[32+0],s[32+1]); }
    if (REDUCE_THREADS >=  16) { ZCACC(s[0],s[1],s[16+0],s[16+1]); }
    if (REDUCE_THREADS >=   8) { ZCACC(s[0],s[1], s[8+0], s[8+1]); }
    if (REDUCE_THREADS >=   4) { ZCACC(s[0],s[1], s[4+0], s[4+1]); }
    if (REDUCE_THREADS >=   2) { ZCACC(s[0],s[1], s[2+0], s[2+1]); }
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
  unsigned int i = blockIdx.x*REDUCE_THREADS + threadIdx.x;
  unsigned int gridSize = REDUCE_THREADS*gridDim.x;
  
  extern __shared__ QudaSumComplex cdata[];
  QudaSumComplex *s = cdata + tid;
  s[0].x = 0;
  s[0].y = 0;
  
  while (i < n) {
    REDUCE_REAL_AUXILIARY(i);
    REDUCE_IMAG_AUXILIARY(i);
    s[0].x += REDUCE_REAL_OPERATION(i);
    s[0].y += REDUCE_IMAG_OPERATION(i);
    i += gridSize;
  }
  __syncthreads();
  
  // do reduction in shared mem
  if (REDUCE_THREADS >= 256) { if (tid < 128) { s[0].x += s[128].x; s[0].y += s[128].y; } __syncthreads(); }
  if (REDUCE_THREADS >= 128) { if (tid <  64) { s[0].x += s[ 64].x; s[0].y += s[ 64].y; } __syncthreads(); }
  
  if (tid < 32) {
    if (REDUCE_THREADS >=  64) { s[0].x += s[32].x; s[0].y += s[32].y; }
    if (REDUCE_THREADS >=  32) { s[0].x += s[16].x; s[0].y += s[16].y; }
    if (REDUCE_THREADS >=  16) { s[0].x += s[ 8].x; s[0].y += s[ 8].y; }
    if (REDUCE_THREADS >=   8) { s[0].x += s[ 4].x; s[0].y += s[ 4].y; }
    if (REDUCE_THREADS >=   4) { s[0].x += s[ 2].x; s[0].y += s[ 2].y; }
    if (REDUCE_THREADS >=   2) { s[0].x += s[ 1].x; s[0].y += s[ 1].y; }
  }
  
  // write result for this block to global mem 
  if (tid == 0) {
    g_odata[blockIdx.x].x = s[0].x;
    g_odata[blockIdx.x].y = s[0].y;
  }
}

#endif

template <typename Float, typename Float2>
cuDoubleComplex REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n) {
  if (n % REDUCE_THREADS != 0) {
    printf("ERROR reduceCuda(): length must be a multiple of %d\n", REDUCE_THREADS);
    exit(-1);
  }
  
  // allocate arrays on device and host to store one QudaSumComplex for each block
  int blocks = min(REDUCE_MAX_BLOCKS, n / REDUCE_THREADS);
  initReduceComplex(blocks);
  
  // partial reduction; each block generates one number
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = REDUCE_THREADS * sizeof(QudaSumComplex);
  REDUCE_FUNC_NAME(Kernel)<<< dimGrid, dimBlock, smemSize >>>(REDUCE_PARAMS, d_reduceComplex, n);
  
  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduceComplex, d_reduceComplex, blocks*sizeof(QudaSumComplex), cudaMemcpyDeviceToHost);
  
  cuDoubleComplex gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  for (int i = 0; i < blocks; i++) {
    gpu_result.x += h_reduceComplex[i].x;
    gpu_result.y += h_reduceComplex[i].y;
  }
  
  return gpu_result;
}


