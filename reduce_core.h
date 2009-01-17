
#ifdef REDUCE_DOUBLE_PRECISION


#define DSACC(c0, c1, a0, a1) dsadd((c0), (c1), (c0), (c1), (a0), (a1))

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, float *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(REDUCE_THREADS) + threadIdx.x;
    unsigned int gridSize = REDUCE_THREADS*gridDim.x;
    
    float acc0 = 0;
    float acc1 = 0;

    while (i < n) {
        REDUCE_AUXILIARY(i);
        DSACC(acc0, acc1, REDUCE_OPERATION(i), 0);
        i += gridSize;
    }

    extern __shared__ float sdata[];
    float *s = sdata + 2*tid;
    s[0] = acc0;
    s[1] = acc1;
    
    __syncthreads();
    
    if (REDUCE_THREADS >= 256) { if (tid < 128) { DSACC(s[0],s[1],s[256+0],s[256+1]); } __syncthreads(); }
    if (REDUCE_THREADS >= 128) { if (tid <  64) { DSACC(s[0],s[1],s[128+0],s[128+1]); } __syncthreads(); }    
    if (tid < 32) {
        if (REDUCE_THREADS >=  64) { DSACC(s[0],s[1],s[64+0],s[64+1]); }
        if (REDUCE_THREADS >=  32) { DSACC(s[0],s[1],s[32+0],s[32+1]); }
        if (REDUCE_THREADS >=  16) { DSACC(s[0],s[1],s[16+0],s[16+1]); }
        if (REDUCE_THREADS >=   8) { DSACC(s[0],s[1], s[8+0], s[8+1]); }
        if (REDUCE_THREADS >=   4) { DSACC(s[0],s[1], s[4+0], s[4+1]); }
        if (REDUCE_THREADS >=   2) { DSACC(s[0],s[1], s[2+0], s[2+1]); }
    }
    
    // write result for this block to global mem as single float
    if (tid == 0) g_odata[blockIdx.x] = sdata[0]+sdata[1];
}


float REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n) {
    if (n % (REDUCE_THREADS) != 0) {
        printf("ERROR reduceCuda(): length must be a multiple of %d\n", (REDUCE_THREADS));
        return 0.;
    }
    
    // allocate arrays on device and host to store one float for each block
    int blocks = min(REDUCE_MAX_BLOCKS, n / (REDUCE_THREADS));
    float h_odata[REDUCE_MAX_BLOCKS];
    float *d_odata;
    
    if (cudaMalloc((void**) &d_odata, blocks*sizeof(float))) {
      printf("Error allocating reduction matrix\n");
      exit(0);
    }   
    
    // partial reduction; each block generates one number
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = 2 * REDUCE_THREADS * sizeof(float);
    REDUCE_FUNC_NAME(Kernel)<<< dimGrid, dimBlock, smemSize >>>(REDUCE_PARAMS, d_odata, n);
    
    // copy result from device to host, and perform final reduction on CPU
    cudaMemcpy(h_odata, d_odata, blocks*sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_result = 0;
    for (int i = 0; i < blocks; i++) 
        gpu_result += h_odata[i];
    
    cudaFree(d_odata);    
    return gpu_result;
}


#else


__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, float *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*REDUCE_THREADS + threadIdx.x;
    unsigned int gridSize = REDUCE_THREADS*gridDim.x;

    extern __shared__ float sdata[];
    float *s = sdata + tid;
    s[0] = 0;

    while (i < n) {
        REDUCE_AUXILIARY(i);
        s[0] += REDUCE_OPERATION(i);
        i += gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if (REDUCE_THREADS >= 256) { if (tid < 128) { s[0] += s[128]; } __syncthreads(); }
    if (REDUCE_THREADS >= 128) { if (tid <  64) { s[0] += s[ 64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (REDUCE_THREADS >=  64) { s[0] += s[32]; }
        if (REDUCE_THREADS >=  32) { s[0] += s[16]; }
        if (REDUCE_THREADS >=  16) { s[0] += s[ 8]; }
        if (REDUCE_THREADS >=   8) { s[0] += s[ 4]; }
        if (REDUCE_THREADS >=   4) { s[0] += s[ 2]; }
        if (REDUCE_THREADS >=   2) { s[0] += s[ 1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = s[0];
}


float REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n) {
    if (n % REDUCE_THREADS != 0) {
        printf("ERROR reduceCuda(): length must be a multiple of %d\n", REDUCE_THREADS);
        return 0.;
    }
    
    // allocate arrays on device and host to store one float for each block
    int blocks = min(REDUCE_MAX_BLOCKS, n / REDUCE_THREADS);
    float h_odata[REDUCE_MAX_BLOCKS];
    float *d_odata;
    if (cudaMalloc((void**) &d_odata, blocks*sizeof(float))) {
      printf("Error allocating reduction matrix\n");
      exit(0);
    }   
       
    // partial reduction; each block generates one number
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = REDUCE_THREADS * sizeof(float);
    REDUCE_FUNC_NAME(Kernel)<<< dimGrid, dimBlock, smemSize >>>(REDUCE_PARAMS, d_odata, n);
    
    // copy result from device to host, and perform final reduction on CPU
    cudaMemcpy(h_odata, d_odata, blocks*sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_result = 0;
    for (int i = 0; i < blocks; i++) 
        gpu_result += h_odata[i];
    
    cudaFree(d_odata);    
    return gpu_result;
}

#endif // REDUCE_DOUBLE_PRECISION
