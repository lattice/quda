
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
  if (tid < 32) {
    if (reduce_threads >=  64) { DSACC3(s[0],s[1],s[64+0],s[64+1]); }
    if (reduce_threads >=  32) { DSACC3(s[0],s[1],s[32+0],s[32+1]); }
    if (reduce_threads >=  16) { DSACC3(s[0],s[1],s[16+0],s[16+1]); }
    if (reduce_threads >=   8) { DSACC3(s[0],s[1], s[8+0], s[8+1]); }
    if (reduce_threads >=   4) { DSACC3(s[0],s[1], s[4+0], s[4+1]); }
    if (reduce_threads >=   2) { DSACC3(s[0],s[1], s[2+0], s[2+1]); }
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
    { if (tid < 512) { s[0].x += s[512].x; s[0].y += s[512].y; s[0].z += s[512].z; } __syncthreads(); }
  if (reduce_threads >= 512) 
    { if (tid < 256) { s[0].x += s[256].x; s[0].y += s[256].y; s[0].z += s[256].z; } __syncthreads(); }
  if (reduce_threads >= 256) 
    { if (tid < 128) { s[0].x += s[128].x; s[0].y += s[128].y; s[0].z += s[128].z; } __syncthreads(); }
  if (reduce_threads >= 128) 
    { if (tid <  64) { s[0].x += s[ 64].x; s[0].y += s[ 64].y; s[0].z += s[ 64].z; } __syncthreads(); }
    
  if (tid < 32) {
    if (reduce_threads >=  64) { s[0].x += s[32].x; s[0].y += s[32].y; s[0].z += s[32].z; }
    if (reduce_threads >=  32) { s[0].x += s[16].x; s[0].y += s[16].y; s[0].z += s[16].z; }
    if (reduce_threads >=  16) { s[0].x += s[ 8].x; s[0].y += s[ 8].y; s[0].z += s[ 8].z; }
    if (reduce_threads >=   8) { s[0].x += s[ 4].x; s[0].y += s[ 4].y; s[0].z += s[ 4].z; }
    if (reduce_threads >=   4) { s[0].x += s[ 2].x; s[0].y += s[ 2].y; s[0].z += s[ 2].z; }
    if (reduce_threads >=   2) { s[0].x += s[ 1].x; s[0].y += s[ 1].y; s[0].z += s[ 1].z; }
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
double3 REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {

  setBlock(kernel, n, precision);
  
  if (n % blasBlock.x != 0) {
    printf("ERROR reduce_triple_core: length %d must be a multiple of %d\n", n, blasBlock.x);
    exit(-1);
  }
  
  if (blasBlock.x > REDUCE_MAX_BLOCKS) {
    printf("ERROR reduce_triple_core: block size greater then maximum permitted\n");
    exit(-1);
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
    printf("Reduction not implemented for %d threads\n", blasBlock.x);
    exit(-1);
  }

  // copy result from device to host, and perform final reduction on CPU
  cudaError_t error = cudaMemcpy(h_reduceFloat3, d_reduceFloat3, blasGrid.x*sizeof(QudaSumFloat3), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
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


