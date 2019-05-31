#pragma once

/**
   @file shared_memory_cache_helper.cuh

   Helper functionality for aiding the use of the shared memory for
   sharing data between threads in a thread block.
 */

namespace quda
{

  /**
     @brief Class which wraps around a shared memory cache for a
     Vector type, where each thread in the thread block stores a
     unique Vector in the cache which any other thread can access.
     Presently, the expectation is that Vector is synonymous with the
     ColorSpinor class, but we could extend this to apply to the
     Matrix class as well.
   */
  template <typename real, typename Vector> class VectorCache
  {

    /**
       @brief This is the handle to the shared memory
       @return Shared memory pointer
     */
    __device__ inline real *cache()
    {
      extern __shared__ int cache_[];
      return reinterpret_cast<real *>(cache_);
    }

public:
    /**
       @brief Save the vector into the 3-d shared memory cache.
       Implicitly store the vector at coordinates given by threadIdx.
       @param[in] a The vector to store in the shared memory cache
     */
    __device__ inline void save(const Vector &a)
    {
      int j = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
#pragma unroll
      for (int i = 0; i < 2 * a.size; i++) {
        cache()[j] = *(reinterpret_cast<const real *>(a.data) + i);
        j += blockDim.z * blockDim.y * blockDim.x;
      }
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The Vector at coordinates (x,y,z)
     */
    __device__ inline Vector load(int x, int y, int z)
    {
      Vector a;
      int j = (z * blockDim.y + y) * blockDim.x + x;
#pragma unroll
      for (int i = 0; i < 2 * a.size; i++) {
        *(reinterpret_cast<real *>(a.data) + i) = cache()[j];
        j += blockDim.z * blockDim.y * blockDim.x;
      }
      return a;
    }

    /**
       @brief Synchronize the cache
    */
    __device__ inline void sync() { __syncthreads(); }
  };

} // namespace quda
