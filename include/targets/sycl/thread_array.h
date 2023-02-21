#pragma once

#include "shared_memory_cache_helper.h"

namespace quda
{

  /**
     @brief Class that provides indexable per-thread storage.  The
     dynamic version maps to using assigning each thread a unique
     window of shared memory.  The static version just uses an stack
     array.
   */

  // default stack version
  template <typename T, int n = 0, bool _ = true> struct thread_array : array<T, n> {
    static_assert(!_);  // always fails, check for use of default version
  };

  // shared memory specialization
  template <typename O> struct thread_array<O,0,(bool)isOpThreadArray<O>> {
    using T = typename O::ElemT;
    static constexpr int n = O::n;
    using array_t = array<T,n>;
    sycl::local_ptr<array_t> array_;
    template <typename ...U, typename ...Ts>
    thread_array(const SpecialOps<U...> *ops, Ts ...t)
    {
      int offset = (target::thread_idx().z * target::block_dim().y + target::thread_idx().y)
	* target::block_dim().x + target::thread_idx().x;
      auto op = getSpecialOp<op_thread_array<T,n>>(ops);
      sycl::local_ptr<void> v(op.smem);
      sycl::local_ptr<array_t> p(v);
      array_ = p + offset;
      if constexpr (sizeof...(Ts) != 0) {
	(*array_) = array_t { t... };
      }
    }
    T &operator[](int i) { return (*array_)[i]; }
    const T &operator[](int i) const { return (*array_)[i]; }
  };


  //template <typename T, int n> struct thread_array<T,n,void> : thread_array_impl<T,n,false> {
  //  thread_array() : thread_array_impl<T,n,false>() {};
  //  //thread_array(const O *ops) : thread_array_impl<T,n,true>(getSpecialOp<only_thread_array<T,n>>(ops)) {};
  //};

  //template <typename T, int n, typename O> struct thread_array<T,n,void> : thread_array_impl<T,n,true> {
    //template <typename ...U, typename Arg> inline SharedMemoryCache(const SpecialOps<U...> *ops, const Arg &arg) :
    // SharedMemoryCacheImpl<typename O::ElemT>(getSpecialOp<SpecialOps<O>>(ops), arg) {}
  //};

  //, bool dynamic = isOpThreadArray<O>
#if 0
#if 1
#if 0
  /**
     @brief Class that provides indexable per-thread storage.  On CUDA
     this maps to using assigning each thread a unique window of
     shared memory using the SharedMemoryCache object.
   */
  template <typename T, int n> struct thread_array {
    //SharedMemoryCache<array<T, n>, 1, 1, false, false> device_array;
    int offset;
    //array<T, n> host_array;
    array<T, n> &array_;

    __device__ __host__ constexpr thread_array() :
      offset((target::thread_idx().z * target::block_dim().y + target::thread_idx().y) * target::block_dim().x
             + target::thread_idx().x),
      array_(target::is_device() ? *(device_array.data() + offset) : host_array)
    {
      array_ = array<T, n>(); // call default constructor
    }

    template <typename... Ts>
    __device__ __host__ constexpr thread_array(T first, const Ts... other) :
      offset((target::thread_idx().z * target::block_dim().y + target::thread_idx().y) * target::block_dim().x
             + target::thread_idx().x),
      array_(target::is_device() ? *(device_array.data() + offset) : host_array)
    {
      array_ = array<T, n> {first, other...};
    }

    __device__ __host__ T &operator[](int i) { return array_[i]; }
    __device__ __host__ const T &operator[](int i) const { return array_[i]; }
  };
#endif

#if 0
  template <typename T, int n>
  struct thread_array {
    SharedMemoryCache<array<T, n>, 1, 1, false, false> device_array;
    int offset;
    vector_type<T, n> host_array;
    vector_type<T, n> &array;

    __device__ __host__ constexpr thread_array() :
      offset((target::thread_idx().z * target::block_dim().y + target::thread_idx().y) * target::block_dim().x + target::thread_idx().x),
      array(target::is_device() ? *(device_array.data() + offset) : host_array)
    {
      array = vector_type<T, n>(); // call default constructor
    }

    __device__ __host__ T& operator[](int i) { return array[i]; }
    __device__ __host__ const T& operator[](int i) const { return array[i]; }
    //__device__ __host__ T& operator[](int i) { return array[0]; }
    //__device__ __host__ const T& operator[](int i) const { return array[0]; }
  };
#else
  template <typename T, int n>
  struct thread_array {
    array<T, n> array_;
    constexpr thread_array()
    {
      array_ = array<T, n>(); // call default constructor
    }
    template <typename O>
    constexpr thread_array(O *ops)
    {
      array_ = array<T, n>(); // call default constructor
    }
    template <typename ...Ts>
    constexpr thread_array(T first, const Ts... other)
    {
      array_ = array<T, n>{first, other...};
    }
    T& operator[](int i) { return array_[i]; }
    const T& operator[](int i) const { return array_[i]; }
  };
#endif

#else

  template <typename T, int n> struct thread_array : array<T, n> {
  };

#endif
#endif

} // namespace quda
