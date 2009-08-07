
#if (REDUCE_TYPE == REDUCE_KAHAN)

#define DSACC(c0, c1, a0, a1) dsadd((c0), (c1), (c0), (c1), (a0), (a1))
#define DSACC3(c0, c1, a0, a1) dsadd3((c0), (c1), (c0), (c1), (a0), (a1))

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat3 *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(REDUCE_THREADS) + threadIdx.x;
  unsigned int gridSize = REDUCE_THREADS*gridDim.x;
    
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
    
  if (REDUCE_THREADS >= 256) { if (tid < 128) { DSACC3(s[0],s[1],s[256+0],s[256+1]); } __syncthreads(); }
  if (REDUCE_THREADS >= 128) { if (tid <  64) { DSACC3(s[0],s[1],s[128+0],s[128+1]); } __syncthreads(); }    
  if (tid < 32) {
    if (REDUCE_THREADS >=  64) { DSACC3(s[0],s[1],s[64+0],s[64+1]); }
    if (REDUCE_THREADS >=  32) { DSACC3(s[0],s[1],s[32+0],s[32+1]); }
    if (REDUCE_THREADS >=  16) { DSACC3(s[0],s[1],s[16+0],s[16+1]); }
    if (REDUCE_THREADS >=   8) { DSACC3(s[0],s[1], s[8+0], s[8+1]); }
    if (REDUCE_THREADS >=   4) { DSACC3(s[0],s[1], s[4+0], s[4+1]); }
    if (REDUCE_THREADS >=   2) { DSACC3(s[0],s[1], s[2+0], s[2+1]); }
  }
    
  // write result for this block to global mem as single QudaSumFloat3
  if (tid == 0) {
    g_odata[blockIdx.x].x = tdata[0].x+tdata[1].x;
    g_odata[blockIdx.x].y = tdata[0].y+tdata[1].y;
    g_odata[blockIdx.x].z = tdata[0].z+tdata[1].z;
  }
}

#else

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat3 *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*REDUCE_THREADS + threadIdx.x;
  unsigned int gridSize = REDUCE_THREADS*gridDim.x;

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
  if (REDUCE_THREADS >= 256) 
    { if (tid < 128) { s[0].x += s[128].x; s[0].y += s[128].y; s[0].z += s[128].z; } __syncthreads(); }
  if (REDUCE_THREADS >= 128) 
    { if (tid <  64) { s[0].x += s[ 64].x; s[0].y += s[ 64].y; s[0].z += s[ 64].z; } __syncthreads(); }
    
  if (tid < 32) {
    if (REDUCE_THREADS >=  64) { s[0].x += s[32].x; s[0].y += s[32].y; s[0].z += s[32].z; }
    if (REDUCE_THREADS >=  32) { s[0].x += s[16].x; s[0].y += s[16].y; s[0].z += s[16].z; }
    if (REDUCE_THREADS >=  16) { s[0].x += s[ 8].x; s[0].y += s[ 8].y; s[0].z += s[ 8].z; }
    if (REDUCE_THREADS >=   8) { s[0].x += s[ 4].x; s[0].y += s[ 4].y; s[0].z += s[ 4].z; }
    if (REDUCE_THREADS >=   4) { s[0].x += s[ 2].x; s[0].y += s[ 2].y; s[0].z += s[ 2].z; }
    if (REDUCE_THREADS >=   2) { s[0].x += s[ 1].x; s[0].y += s[ 1].y; s[0].z += s[ 1].z; }
  }
    
  // write result for this block to global mem 
  if (tid == 0) {
    g_odata[blockIdx.x].x = s[0].x;
    g_odata[blockIdx.x].y = s[0].y;
    g_odata[blockIdx.x].z = s[0].z;
  }
}

#endif

template <typename Float2>
double3 REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n) {
  if (n % REDUCE_THREADS != 0) {
    printf("ERROR reduceCuda(): length must be a multiple of %d\n", REDUCE_THREADS);
    exit(-1);
  }
  
  // allocate arrays on device and host to store one QudaSumFloat3 for each block
  int blocks = min(REDUCE_MAX_BLOCKS, n / REDUCE_THREADS);
  initReduceFloat3(blocks);
  
  // partial reduction; each block generates one number
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize = REDUCE_THREADS * sizeof(QudaSumFloat3);
  REDUCE_FUNC_NAME(Kernel)<<< dimGrid, dimBlock, smemSize >>>(REDUCE_PARAMS, d_reduceFloat3, n);
  
  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduceFloat3, d_reduceFloat3, blocks*sizeof(QudaSumFloat3), cudaMemcpyDeviceToHost);
  
  double3 gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  gpu_result.z = 0;
  for (int i = 0; i < blocks; i++) {
    gpu_result.x += h_reduceFloat3[i].x;
    gpu_result.y += h_reduceFloat3[i].y;
    gpu_result.z += h_reduceFloat3[i].z;
  }
  
  return gpu_result;
}


