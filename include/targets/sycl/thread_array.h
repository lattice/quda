#include "shared_memory_cache_helper.h"

namespace quda
{

#if 0

  /**
     @brief Class that provides indexable per-thread storage.  On CUDA
     this maps to using assigning each thread a unique window of
     shared memory using the SharedMemoryCache object.
   */
  template <typename T, int n> struct thread_array {
    SharedMemoryCache<array<T, n>, 1, 1, false, false> device_array;
    int offset;
    array<T, n> host_array;
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

} // namespace quda
