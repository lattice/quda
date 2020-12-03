#pragma once

#include <target_device.h>

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
      QUDA_RT_CONSTS;
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
      QUDA_RT_CONSTS;
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

  template <typename T, int block_size_y = 1, int block_size_z = 1> class SharedMemoryCache
  {
    // maximum number of threads in x given the y and z block sizes
    static constexpr int max_block_size_x = device::max_block_size<block_size_y, block_size_z>();
    // we pad in the x dimension width to ensure that it isn't a multiple of the bank width
    static constexpr int thread_width_x = ((max_block_size_x + device::shared_memory_bank_width() - 1) /
                                           device::shared_memory_bank_width()) * device::shared_memory_bank_width();

    /**
       @brief This is the handle to the shared memory
       @return Shared memory pointer
     */
    __device__ __host__ inline T *cache()
    {
#ifdef __CUDA_ARCH__
      extern __shared__ int cache_[];
#else
      // dummy code path to keep clang happy
      static int *cache_;
#endif
      return reinterpret_cast<T*>(cache_);
    }

  public:
    /**
       @brief Grab the raw base address to shared memory
    */
    __device__ __host__ inline T* data()
    {
      return cache();
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       Implicitly store the vector at coordinates given by threadIdx.
       @param[in] a The vector to store in the shared memory cache
     */
    __device__ __host__ inline void save(const T &a)
    {
      auto tid = device::thread_idx();
      int j = (tid.z * block_size_y + tid.y) * thread_width_x + tid.x;
      cache()[j] = a;
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load(int x = -1, int y = -1, int z = -1)
    {
      auto tid = device::thread_idx();
      x = (x == -1) ? tid.x : x;
      y = (y == -1) ? tid.y : y;
      z = (z == -1) ? tid.z : z;
      int j = (z * block_size_y + y) * thread_width_x + x;
      return cache()[j];
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_x(int x = -1)
    {
      auto tid = device::thread_idx();
      x = (x == -1) ? tid.x : x;
      int j = (tid.z * block_size_y + tid.y) * thread_width_x + x;
      return cache()[j];
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_y(int y = -1)
    {
      auto tid = device::thread_idx();
      y = (y == -1) ? tid.y : y;
      int j = (tid.z * block_size_y + y) * thread_width_x + tid.x;
      return cache()[j];
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_z(int z = -1)
    {
      auto tid = device::thread_idx();
      z = (z == -1) ? tid.z : z;
      int j = (z * block_size_y + tid.y) * thread_width_x + tid.x;
      return cache()[j];
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ inline void sync()
    {
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }
  };

} // namespace quda
