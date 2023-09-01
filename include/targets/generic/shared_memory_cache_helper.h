#pragma once

#include <helpers.h>
#include <target_device.h>
#include <shared_memory_helper.h>

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
     the cache which any other thread can access.  The data is stored
     in a coalesced order with element size atom_t<T>.

     The dimensions of the cache is determined by a call to
     D::dims(target::block_dim()), and D defaults to having dimensions
     equal to the block dimensions.

     A byte offset into the shared memory region can be specified with
     the type O, and is given by
     O::shared_mem_size(target::block_dim()) if O is not void.
   */
  template <typename T, typename D = DimsBlock, typename O = void>
  class SharedMemoryCache : SharedMemory<atom_t<T>, SizeDims<D,sizeof(T)/sizeof(atom_t<T>)>, O>
  {
    using Smem = SharedMemory<atom_t<T>, SizeDims<D,sizeof(T)/sizeof(atom_t<T>)>, O>;

  public:
    using value_type = T;
    using dims_type = D;
    using offset_type = O;
    using Smem::shared_mem_size;

  private:
    const dim3 block;
    const int stride;
    using Smem::sharedMem;
    using atom_t = atom_t<T>;
    static_assert(sizeof(T) % 4 == 0, "Shared memory cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    // used to avoid instantiation of load functions if unused, in case T is not a valid return type (e.g. C array)
    template <typename dummy = void> using maybeT = std::conditional_t<std::is_same_v<dummy,void>,T,void>;

    __device__ __host__ inline void save_detail(const T &a, int x, int y, int z) const
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) sharedMem()[i * stride + j] = tmp[i];
    }

    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_detail(int x, int y, int z) const
    {
      atom_t tmp[n_element];
      int j = (z * block.y + y) * block.x + x;
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = sharedMem()[i * stride + j];
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
       @brief Constructor for SharedMemoryCache.
    */
    constexpr SharedMemoryCache() :
      block(D::dims(target::block_dim())), stride(block.x * block.y * block.z)
    {
      // sanity check
      static_assert(shared_mem_size(dim3{32,16,8})==Smem::get_offset(dim3{32,16,8})+SizeDims<D>::size(dim3{32,16,8})*sizeof(T));
    }

    /**
       @brief Grab the raw base address to shared memory.
    */
    __device__ __host__ inline auto data() const {
      return reinterpret_cast<T *>(&sharedMem()[0]);
    }

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
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load(int x = -1, int y = -1, int z = -1) const
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
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_x(int x = -1) const
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
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_y(int y = -1) const
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
    template <typename dummy = void>
    __device__ __host__ inline maybeT<dummy> load_z(int z = -1) const
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
    template <typename dummy = void>
    __device__ __host__ operator maybeT<dummy>() const { return load(); }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected.
     */
    __device__ __host__ void operator=(const T &src) const { save(src); }
  };

  template <typename T, typename D, typename O>
  __device__ __host__ inline T operator+(const SharedMemoryCache<T, D, O> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline T operator+(const T &a, const SharedMemoryCache<T, D, O> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline T operator-(const SharedMemoryCache<T, D, O> &a, const T &b)
  {
    return static_cast<const T &>(a) - b;
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline T operator-(const T &a, const SharedMemoryCache<T, D, O> &b)
  {
    return a - static_cast<const T &>(b);
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline auto operator+=(SharedMemoryCache<T, D, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) + b);
    return a;
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline auto operator-=(SharedMemoryCache<T, D, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) - b);
    return a;
  }

  template <typename T, typename D, typename O>
  __device__ __host__ inline auto conj(const SharedMemoryCache<T, D, O> &a)
  {
    return conj(static_cast<const T &>(a));
  }

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or SharedMemoryCache<T,D,O>
   */
  template <class T>
  struct get_type<T, std::enable_if_t<std::is_same_v<T, SharedMemoryCache<typename T::value_type, typename T::dims_type, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };

} // namespace quda
