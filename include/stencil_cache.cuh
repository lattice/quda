#pragma once

#include <load_store.h> // for atom_t
#include <kernel_ops.h>
#include <target_device.h>
#include <shared_memory_helper.h>

namespace quda
{

  template <typename T>
  class VanillaSharedMemoryCache
  {

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      char *operator()()
      {
        static char *cache__;
        return reinterpret_cast<char *>(cache__);
      }
    };

    /**
       @brief This is the handle to the shared memory, dynamic specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      __device__ inline char* operator()()
      {
        extern __shared__ char cache__[];
        return reinterpret_cast<char *>(cache__);
      }
    };
  
    /**
       @brief Dummy instantiation for the host compiler
    */
    template <bool is_device, typename dummy = void> struct sync_impl {
      void operator()() { }
    };

    /**
       @brief Synchronize the cache when on the device
    */
    template <typename dummy> struct sync_impl<true, dummy> {
      __device__ inline void operator()() { __syncthreads(); }
    };

    using atom_t = atom_t<T>;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    const int stride;
    atom_t *cache; // the underlying shared memory pointer

  public:
    using value_type = T;

  public:
    /**
       @brief Constructor for SharedMemoryCache.
    */
    __device__ __host__ VanillaSharedMemoryCache(int stride_):
      stride(stride_), cache(reinterpret_cast<atom_t *>(target::dispatch<cache_dynamic>()))
    {
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ void sync() const { target::dispatch<sync_impl>(); }

    __device__ __host__ inline void save(const T &a, int j)
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
#pragma unroll
      for (int i = 0; i < n_element; i++) cache[i * stride + j] = tmp[i];
    }

    __device__ __host__ inline auto load(int j) const
    {
      atom_t tmp[n_element];
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache[i * stride + j];
      T a;
      memcpy((void *)&a, tmp, sizeof(T));
      return a;
    }

  };

} // namespace quda
