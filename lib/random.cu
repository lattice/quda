#include <stdio.h>
#include <string.h>
#include <iostream>
#include <random_quda.h>
#include <cuda.h>
#include <quda_internal.h>

#include <comm_quda.h>
#include <index_helper.cuh>

#define BLOCKSDIVUP(a, b)  (((a)+(b)-1)/(b))
#define CUDA_SAFE_CALL_NO_SYNC( call) {                                 \
    cudaError err = call;                                               \
    if( cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }
#define CUDA_SAFE_CALL( call) CUDA_SAFE_CALL_NO_SYNC(call);


namespace quda {

  dim3 GetBlockDim(size_t threads, size_t size) {
    int blockx = BLOCKSDIVUP(size, threads);
    dim3 blocks(blockx,1,1);
    return blocks;
  }

  struct rngArg {
    int commDim[QUDA_MAX_DIM];
    int commCoord[QUDA_MAX_DIM];
    int X[QUDA_MAX_DIM];
    rngArg(const int X_[]) {
      for (int i=0; i<4; i++) {
        commDim[i] = comm_dim(i);
        commCoord[i] = comm_coord(i);
        X[i] = X_[i];
      }
    }
  };

  /**
     @brief CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param rng_size size of the CURAND RNG state array
     @param arg Metadata needed for computing multi-gpu offsets
  */
  __global__ void kernel_random(cuRNGState *state, int seed, int rng_size, rngArg arg) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < rng_size) {
      // Each thread gets same seed, a different sequence number, no offset
      int x[4];
      getCoords(x, id, arg.X, 0);
      for(int i=0; i<4;i++) x[i] += arg.commCoord[i] * arg.X[i];
      int idd = ((((x[3] * arg.commDim[2] * arg.X[2] + x[2]) * arg.commDim[1] * arg.X[1]) + x[1] ) * arg.commDim[0] * arg.X[0] + x[0]) >> 1 ;
      curand_init(seed, idd, 0, &state[id]);
    }
  }

  /**
     @brief Call CUDA kernel to initialize CURAND RNG states
     @param state CURAND RNG state array
     @param seed initial seed for RNG
     @param rng_size size of the CURAND RNG state array
     @param X array of lattice dimensions
  */
  void launch_kernel_random(cuRNGState *state, int seed, int rng_size, int X[4])
  {
    dim3 nthreads(128,1,1);
    dim3 nblocks = GetBlockDim(nthreads.x, rng_size);
    rngArg arg(X);
    kernel_random<<<nblocks,nthreads>>>(state, seed, rng_size, arg);
    qudaDeviceSynchronize();
  }

  RNG::RNG(int rng_sizes, int seedin, const int XX[4]) :
    seed(seedin), rng_size(rng_sizes)
  {
    state = NULL;
    node_offset = 0;
    for(int i=0; i<4;i++) X[i]=XX[i];
    node_offset = comm_rank() * rng_sizes;
#if defined(XORWOW)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateXORWOW\n");
#elif defined(RG32k3a)
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#else
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Using curandStateMRG32k3a\n");
#endif
  }

  /**
     @brief Initialize CURAND RNG states
  */
  void RNG::Init() {
    AllocateRNG();
    launch_kernel_random(state, seed, rng_size, X);
  }

  /**
     @brief Allocate Device memory for CURAND RNG states
  */
  void RNG::AllocateRNG() {
    if (rng_size>0 && state == NULL) {
      state = (cuRNGState*)device_malloc(rng_size * sizeof(cuRNGState));
      CUDA_SAFE_CALL(cudaMemset( state , 0 , rng_size * sizeof(cuRNGState) ));
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Allocated array of random numbers with rng_size: %.2f MB\n", rng_size * sizeof(cuRNGState)/(float)(1048576));
    } else {
      errorQuda("Array of random numbers not allocated, array size: %d !\nExiting...\n",rng_size);
    }
  }

  /**
     @brief Release Device memory for CURAND RNG states
  */
  void RNG::Release() {
    if(rng_size>0 && state != NULL) {
      device_free(state);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("Free array of random numbers with rng_size: %.2f MB\n", rng_size * sizeof(cuRNGState)/(float)(1048576));
      rng_size = 0;
      state = NULL;
    }
  }

  /*! @brief Restore CURAND array states initialization */
  void RNG::restore() {
    cudaError_t err = cudaMemcpy(state, backup_state, rng_size * sizeof(cuRNGState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      host_free(backup_state);
      errorQuda("Failed to restore curand rng states array\n");
    }
    host_free(backup_state);
  }

  /*! @brief Backup CURAND array states initialization */
  void RNG::backup() {
    backup_state = (cuRNGState*) safe_malloc(rng_size * sizeof(cuRNGState));
    cudaError_t err = cudaMemcpy(backup_state, state, rng_size * sizeof(cuRNGState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      host_free(backup_state);
      errorQuda("Failed to backup curand rng states array\n");
    }
  }

} // namespace quda
