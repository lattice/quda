
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <random_quda.h>
#include <cuda.h>
#include <quda_internal.h>

#include <comm_quda.h>
#include <index_helper.cuh>


namespace quda {

#if defined(GPU_GAUGE_ALG) || defined(GPU_GAUGE_TOOlS)
  
#define BLOCKSDIVUP(a, b)  (((a)+(b)-1)/(b))


dim3 GetBlockDim(size_t threads, size_t size){
    int blockx = BLOCKSDIVUP(size, threads);
    dim3 blocks(blockx,1,1);
    return blocks;
}




#  define CUDA_SAFE_CALL_NO_SYNC( call) {                               \
        cudaError err = call;                                           \
        if( cudaSuccess != err) {                                       \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString( err) );     \
            exit(EXIT_FAILURE);                                         \
        } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);   

/**
    @brief CUDA kernel to initialize CURAND RNG states
    @param state CURAND RNG state array
    @param seed initial seed for RNG
    @param rng_size size of the CURAND RNG state array
    @param node_offset this parameter is used to skip ahead the index in the sequence, usefull for multigpu. 
*/
__global__ void 
kernel_random(cuRNGState *state, int seed, int rng_size, int node_offset ){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < rng_size){
        /* Each thread gets same seed, a different sequence number, no offset */
        curand_init(seed, id + node_offset, 0, &state[id]);
    }
}

struct rngArg{
    int comm_dim[4];
    int comm_coord[4];
    int X[4];
};


__global__ void 
kernel_random(cuRNGState *state, int seed, int rng_size, int node_offset, rngArg arg ){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < rng_size){
        /* Each thread gets same seed, a different sequence number, no offset */
    #ifndef MULTI_GPU
        curand_init(seed, id + node_offset, 0, &state[id]);
    #else

    int x[4];
    getCoords(x, id, arg.X, 0);
    for(int i=0; i<4;i++) x[i] += arg.comm_coord[i] * arg.X[i];
    int idd = ((((x[3] * arg.comm_dim[2] * arg.X[2] + x[2]) * arg.comm_dim[1] * arg.X[1]) + x[1] ) * arg.comm_dim[0] * arg.X[0] + x[0]) >> 1 ;
    curand_init(seed, idd, 0, &state[id]);
    #endif
    }
}

/**
    @brief Call CUDA kernel to initialize CURAND RNG states
    @param state CURAND RNG state array
    @param seed initial seed for RNG
    @param rng_size size of the CURAND RNG state array
    @param node_offset this parameter is used to skip ahead the index in the sequence, usefull for multigpu. 
*/
void launch_kernel_random(cuRNGState *state, int seed, int rng_size, int node_offset, int X[4]){  
    dim3 nthreads(128,1,1);
    dim3 nblocks = GetBlockDim(nthreads.x, rng_size);
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig( kernel_random,	cudaFuncCachePreferL1));
    #ifndef MULTI_GPU
    kernel_random<<<nblocks,nthreads>>>(state, seed, rng_size, node_offset);
    #else
    rngArg arg;
    for(int i=0; i < 4; i++){
        arg.comm_dim[i] = comm_dim(i);
        arg.comm_coord[i] = comm_coord(i);
        arg.X[i] = X[i];
    }
    kernel_random<<<nblocks,nthreads>>>(state, seed, rng_size, 0, arg);
    #endif
    cudaDeviceSynchronize();
}

RNG::RNG(int rng_sizes, int seedin){
    rng_size = rng_sizes;
    seed = seedin;
    state = NULL;
    node_offset = 0;
    #ifdef MULTI_GPU
    for(int i=0; i<4;i++) X[i]=0;
    node_offset = comm_rank() * rng_sizes;
    #endif
#if defined(XORWOW)
    printfQuda("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
    printfQuda("Using curandStateMRG32k3a\n");
#else
    printfQuda("Using curandStateMRG32k3a\n");
#endif
} 


RNG::RNG(int rng_sizes, int seedin, const int XX[4]){
    rng_size = rng_sizes;
    seed = seedin;
    state = NULL;
    node_offset = 0;
    #ifdef MULTI_GPU
    for(int i=0; i<4;i++) X[i]=XX[i];
    node_offset = comm_rank() * rng_sizes;
    #endif
#if defined(XORWOW)
    printfQuda("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
    printfQuda("Using curandStateMRG32k3a\n");
#else
    printfQuda("Using curandStateMRG32k3a\n");
#endif
} 




/**
    @brief Initialize CURAND RNG states
*/
void RNG::Init(){
	AllocateRNG();
	launch_kernel_random(state, seed, rng_size, node_offset, X);
}		
					

/**
    @brief Allocate Device memory for CURAND RNG states
*/
void RNG::AllocateRNG(){
    if(rng_size>0 && state == NULL){
        state = (cuRNGState*)device_malloc(rng_size * sizeof(cuRNGState));
        CUDA_SAFE_CALL(cudaMemset( state , 0 , rng_size * sizeof(cuRNGState) ));
        printfQuda("Allocated array of random numbers with rng_size: %.2f MB\n", rng_size * sizeof(cuRNGState)/(float)(1048576));
    }
    else{
        errorQuda("Array of random numbers not allocated, array size: %d !\nExiting...\n",rng_size);
    }
}
/**
    @brief Release Device memory for CURAND RNG states
*/
void RNG::Release(){
    if(rng_size>0 && state != NULL){
        device_free(state);
        printfQuda("Free array of random numbers with rng_size: %.2f MB\n", rng_size * sizeof(cuRNGState)/(float)(1048576));
        rng_size = 0;
        state = NULL;
    }
}


/*! @brief Restore CURAND array states initialization */
void RNG::restore(){    
  cudaError_t err = cudaMemcpy(state, backup_state, rng_size * sizeof(cuRNGState), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    host_free(backup_state);
    printfQuda("ERROR: Failed to restore curand rng states array\n");
    errorQuda("Aborting");
  }
  host_free(backup_state);
}
/*! @brief Backup CURAND array states initialization */
void RNG::backup(){ 
  backup_state = (cuRNGState*) safe_malloc(rng_size * sizeof(cuRNGState));   
  cudaError_t err = cudaMemcpy(backup_state, state, rng_size * sizeof(cuRNGState), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    host_free(backup_state);
    printfQuda("ERROR: Failed to backup curand rng states array\n");
    errorQuda("Aborting");
  }
}

#endif // GPU_GAUGE_ALG

}
