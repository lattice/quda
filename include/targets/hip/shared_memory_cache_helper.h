#pragma once

#include <target_device.h>
#include <array.h>

/**
   @file shared_memory_cache_helper.h

   Helper functionality for aiding the use of the shared memory for
   sharing data between threads in a thread block.
 */

namespace quda
{

  /**
     @brief Class which wraps around a shared memory cache for type T,
     where each thread in the thread block stores a unique value in
     the cache which any other thread can access.

     This accessor supports both explicit run-time block size and
     compile-time sizing.

     * For run-time block size, the constructor should be initialied
       with the desired block size.

     * For compile-time block size, no arguments should be passed to
       the constructor, and then the second and third template
       parameters correspond to the y and z dimensions of the block,
       respectively.  The x dimension of the block will be set
       according the maximum number of threads possible, given these
       dimensions.
   */
  template <typename T, int block_size_y_ = 1, int block_size_z_ = 1, bool dynamic_ = true> class SharedMemoryCache
  {
  public:
    using value_type = T;
    static constexpr int block_size_y = block_size_y_;
    static constexpr int block_size_z = block_size_z_;
    static constexpr bool dynamic = dynamic_;

  private:
    /** maximum number of threads in x given the y and z block sizes */
    static constexpr int block_size_x = device::max_block_size<block_size_y, block_size_z>();

    using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    const dim3 block;
    const int stride;
    const unsigned int offset = 0; // dynamic offset in bytes

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool, typename dummy = void> struct cache_dynamic {
      atom_t *operator()(unsigned)
      {
        static atom_t *cache_;
        return reinterpret_cast<atom_t *>(cache_);
      }
    };

    /**
       @brief This is the handle to the shared memory, dynamic specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_dynamic<true, dummy> {
      __device__ inline atom_t *operator()(unsigned int offset)
      {
        extern __shared__ int cache_[];
        return reinterpret_cast<atom_t *>(reinterpret_cast<char *>(cache_) + offset);
      }
    };

    /**
       @brief This is a dummy instantiation for the host compiler
    */
    template <bool is_device, typename dummy = void> struct cache_static {
      atom_t *operator()()
      {
        static atom_t *cache_;
        return reinterpret_cast<atom_t *>(cache_);
      }
    };

    /**
       @brief This is the handle to the shared memory, static specialization
       @return Shared memory pointer
     */
    template <typename dummy> struct cache_static<true, dummy> {
      __device__ inline atom_t *operator()()
      {
        static __shared__ atom_t cache_[n_element * block_size_x * block_size_y * block_size_z];
        return reinterpret_cast<atom_t *>(cache_);
      }
    };

    template <bool dynamic_shared> __device__ __host__ inline std::enable_if_t<dynamic_shared, atom_t *> cache() const
    {
      return target::dispatch<cache_dynamic>(offset);
    }

    template <bool dynamic_shared> __device__ __host__ inline std::enable_if_t<!dynamic_shared, atom_t *> cache() const
    {
      return target::dispatch<cache_static>();
    }

    __device__ __host__ inline void save_detail(const T &a, int x, int y, int z) const
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) cache<dynamic>()[i * stride + j] = tmp[i];
    }

    __device__ __host__ inline T load_detail(int x, int y, int z) const
    {
      atom_t tmp[n_element];
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = cache<dynamic>()[i * stride + j];
      T a;
      memcpy((void *)&a, tmp, sizeof(T));
      return a;
    }

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

  public:
    /**
       @brief constructor for SharedMemory cache.  If no arguments are
       pass, then the dimensions are set according to the templates
       block_size_y and block_size_z, together with the derived
       block_size_x.  Otherwise use the block sizes passed into the
       constructor.

       @param[in] block Block dimensions for the 3-d shared memory object
       @param[in] thread_offset "Perceived" offset from dynamic shared
       memory base pointer (used when we have multiple caches in
       scope).  Need to include block size to actual offset.
    */
    constexpr SharedMemoryCache(dim3 block = dim3(block_size_x, block_size_y, block_size_z),
                                unsigned int thread_offset = 0) :
      block(block), stride(block.x * block.y * block.z), offset(stride * thread_offset)
    {
    }

    /**
       @brief Grab the raw base address to shared memory.
    */
    __device__ __host__ inline auto data() const { return reinterpret_cast<T *>(cache<dynamic>()); }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
     */
    __device__ __host__ inline void save(const T &a, int x = -1, int y = -1, int z = -1) const
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      y = (y == -1) ? tid.y : y;
      z = (z == -1) ? tid.z : z;
      save_detail(a, x, y, z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] x The x index to use
     */
    __device__ __host__ inline void save_x(const T &a, int x = -1) const
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      save_detail(a, x, tid.y, tid.z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] y The y index to use
     */
    __device__ __host__ inline void save_y(const T &a, int y = -1) const
    {
      auto tid = target::thread_idx();
      y = (y == -1) ? tid.y : y;
      save_detail(a, tid.x, y, tid.z);
    }

    /**
       @brief Save the value into the 3-d shared memory cache.
       @param[in] a The value to store in the shared memory cache
       @param[in] z The z index to use
     */
    __device__ __host__ inline void save_z(const T &a, int z = -1) const
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      save_detail(a, tid.x, tid.y, z);
    }

    /**
       @brief Load a value from the shared memory cache
       @param[in] x The x index to use
       @param[in] y The y index to use
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
     */
    __device__ __host__ inline T load(int x = -1, int y = -1, int z = -1) const
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      y = (y == -1) ? tid.y : y;
      z = (z == -1) ? tid.z : z;
      return load_detail(x, y, z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] x The x index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_x(int x = -1) const
    {
      auto tid = target::thread_idx();
      x = (x == -1) ? tid.x : x;
      return load_detail(x, tid.y, tid.z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] y The y index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_y(int y = -1) const
    {
      auto tid = target::thread_idx();
      y = (y == -1) ? tid.y : y;
      return load_detail(tid.x, y, tid.z);
    }

    /**
       @brief Load a vector from the shared memory cache
       @param[in] z The z index to use
       @return The value at coordinates (x,y,z)
    */
    __device__ __host__ inline T load_z(int z = -1) const
    {
      auto tid = target::thread_idx();
      z = (z == -1) ? tid.z : z;
      return load_detail(tid.x, tid.y, z);
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ void sync() const { target::dispatch<sync_impl>(); }

    /**
       @brief Cast operator to allow cache objects to be used where T
       is expected
     */
    __device__ __host__ operator T() const { return load(); }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected.
     */
    __device__ __host__ void operator=(const T &src) const { save(src); }
  };

} // namespace quda

// include overloads
#include "../generic/shared_memory_cache_helper.h"
