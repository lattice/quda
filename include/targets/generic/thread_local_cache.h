#pragma once

#include <helpers.h>
#include <target_device.h>
#include <shared_memory_helper.h>

/**
   @file thread_local_cache.h

   Thread local cache object which may use shared memory for optimization.
   The storage can be a single object or an array of objects.
 */

namespace quda
{

  /**
     @brief Class for threads to store a unique value, or array of values, which can use
     shared memory for optimization purposes.
   */
  template <typename T, int N_ = 0, typename O = void> class ThreadLocalCache :
    SharedMemory<atom_t<T>, SizePerThread<std::max(1,N_)*sizeof(T)/sizeof(atom_t<T>)>, O>
  {
  public:
    using value_type = T;
    static constexpr int N = N_; // size of array, 0 means to behave like T instead of array<T, 1>
    using offset_type = O; // type of object that may also use shared memory at the same time and is located before this one
    static constexpr int len = std::max(1,N); // actual number of elements to store
    using Smem = SharedMemory<atom_t<T>, SizePerThread<std::max(1,N_)*sizeof(T)/sizeof(atom_t<T>)>, O>;
    using Smem::smem;
    using Smem::shared_mem_size;
    using opSmem = op_SharedMemory<T, SizeSmem<Smem>>;
    using dependencies = opSmem;
    using dependentOps = SpecialOps<opSmem>;

  private:
    using atom_t = atom_t<T>;
    static_assert(sizeof(T) % 4 == 0, "Thread local cache does not support sub-word size types");

    // The number of elements of type atom_t that we break T into for optimal shared-memory access
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);

    const int stride;

    //constexpr Smem smem() const { return *dynamic_cast<const Smem*>(this); }

    __device__ __host__ inline void save_detail(const T &a, const int k) const
    {
      atom_t tmp[n_element];
      memcpy(tmp, (void *)&a, sizeof(T));
      int j = target::thread_idx_linear<3>();
#pragma unroll
      for (int i = 0; i < n_element; i++) smem()[(k*n_element + i) * stride + j] = tmp[i];
    }

    __device__ __host__ inline T load_detail(const int k) const
    {
      atom_t tmp[n_element];
      int j = target::thread_idx_linear<3>();
#pragma unroll
      for (int i = 0; i < n_element; i++) tmp[i] = smem()[(k*n_element + i) * stride + j];
      T a;
      memcpy((void *)&a, tmp, sizeof(T));
      return a;
    }

  public:
    /**
       @brief Constructor for ThreadLocalCache.
    */
    constexpr ThreadLocalCache() : stride(target::block_size<3>()) {
      static_assert(shared_mem_size(dim3{8,8,8})==Smem::get_offset(dim3{8,8,8})+SizePerThread<len>::size(dim3{8,8,8})*sizeof(T));
    }

    template <typename ...U>
    constexpr ThreadLocalCache(const SpecialOps<U...> &ops) : Smem(getDependentOps<ThreadLocalCache<T,N,O>>(ops)), stride(target::block_size<3>()) {
      static_assert(shared_mem_size(dim3{8,8,8})==Smem::get_offset(dim3{8,8,8})+SizePerThread<len>::size(dim3{8,8,8})*sizeof(T));
    }

    /**
       @brief Save the value into the thread local cache.  Used when N==0 so cache acts like single object.
       @param[in] a The value to store in the thread local cache
     */
    __device__ __host__ inline void save(const T &a) const {
      static_assert(N == 0);
      save_detail(a, 0);
    }

    /**
       @brief Save the value into an element of the thread local cache.
       @param[in] a The value to store in the thread local cache
       @param[in] k The index to use
     */
    __device__ __host__ inline void save(const T &a, const int k) const { save_detail(a, k); }

    /**
       @brief Load a value from the thread local cache.  Used when N==0 so cache acts like single object.
       @return The value stored in the thread local cache
     */
    __device__ __host__ inline T load() const {
      static_assert(N == 0);
      return load_detail(0);
    }

    /**
       @brief Load a value from an element of the thread local cache
       @param[in] k The index to use
       @return The value stored in the thread local cache at that index
     */
    __device__ __host__ inline T load(const int k) const { return load_detail(k); }

    /**
       @brief Cast operator to allow cache objects to be used where T is expected (when N==0).
     */
    __device__ __host__ operator T() const {
      static_assert(N == 0);
      return load();
    }

    /**
       @brief Assignment operator to allow cache objects to be used on
       the lhs where T is otherwise expected (when N==0).
     */
    __device__ __host__ void operator=(const T &src) const {
      static_assert(N == 0);
      save(src);
    }

    /**
       @brief Subscripting operator returning value at index for convenience.
       @param[in] i The index to use
       @return The value stored in the thread local cache at that index
     */
    __device__ __host__ T operator[](int i) { return load(i); }
  };

  template <typename T, int N, typename O> __device__ __host__ inline T operator+(const ThreadLocalCache<T, N, O> &a, const T &b)
  {
    return static_cast<const T &>(a) + b;
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator+(const T &a, const ThreadLocalCache<T, N, O> &b)
  {
    return a + static_cast<const T &>(b);
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator-(const ThreadLocalCache<T, N, O> &a, const T &b)
  {
    return static_cast<const T &>(a) - b;
  }

  template <typename T, int N, typename O> __device__ __host__ inline T operator-(const T &a, const ThreadLocalCache<T, N, O> &b)
  {
    return a - static_cast<const T &>(b);
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto operator+=(ThreadLocalCache<T, N, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) + b);
    return a;
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto operator-=(ThreadLocalCache<T, N, O> &a, const T &b)
  {
    a.save(static_cast<const T &>(a) - b);
    return a;
  }

  template <typename T, int N, typename O> __device__ __host__ inline auto conj(const ThreadLocalCache<T, N, O> &a)
  {
    return conj(static_cast<const T &>(a));
  }

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or ThreadLocalCache<T,O>
   */
  template <class T>
  struct get_type<T, std::enable_if_t<std::is_same_v<T, ThreadLocalCache<typename T::value_type, T::N, typename T::offset_type>>>> {
    using type = typename T::value_type;
  };

} // namespace quda
