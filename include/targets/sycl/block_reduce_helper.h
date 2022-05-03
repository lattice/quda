#pragma once

#include <target_device.h>
#include <reducer.h>
#include <group_reduce.h>

/**
   @file block_reduce_helper.h

   @section This files contains the SYCL implementations
   for warp- and block-level reductions.
 */

using namespace quda;

namespace quda
{

#if 0
  /**
     @brief warp_reduce_param is used as a container for passing
     non-type parameters to specialize warp_reduce through the
     target::dispatch
     @tparam width The number of logical threads taking part in the warp reduction
   */
  template <int width_ = device::warp_size()> struct warp_reduce_param {
    static_assert(width_ <= device::warp_size(), "WarpReduce logical width must not be greater than the warp size");
    static constexpr int width = width_;
  };

  /**
     @brief block_reduce_param is used as a container for passing
     non-type parameters to specialize block_reduce through the
     target::dispatch
     @tparam block_size_x_ The number of threads in the x dimension
     @tparam block_size_y_ The number of threads in the y dimension
     @tparam block_size_z_ The number of threads in the z dimension
     @tparam batched Whether this is a batched reduction or not.  If
     batched, then the block_size_z_ parameter is set equal to the
     batch size.
   */
  template <int block_size_x_, int block_size_y_, int block_size_z_, bool batched> struct block_reduce_param {
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    static constexpr int block_size_z = !batched ? block_size_z_ : 1;
    static constexpr int batch_size = !batched ? 1 : block_size_z_;
  };

  /**
     @brief Dummy generic implementation of warp_reduce
  */
  template <bool is_device> struct warp_reduce {
    template <typename T, typename reducer_t, typename param_t> T operator()(const T &value, bool, reducer_t, param_t)
    {
      return value;
    }
  };

  /**
     @brief Dummy generic implementation of block_reduce
  */
  template <bool is_device> struct block_reduce {
    template <typename T, typename reducer_t, typename param_t>
    T operator()(const T &value, bool, int, bool, reducer_t, param_t)
    {
      return value;
    }
  };
#endif

  /**
     @brief WarpReduce provides a generic interface for performing
     perform reductions at the warp or sub-warp level
     @tparam T The type of the value that we are reducing
     @tparam width The number of logical threads taking part in the warp reduction
  */
  template <typename T, int width> class WarpReduce
  {
    static_assert(width <= device::warp_size(),
		  "WarpReduce logical width must not be greater than the warp size");
    //using param_t = warp_reduce_param<width>;
    //const nreduce = device::warp_size() / width;

  public:
    constexpr WarpReduce() { }

    /**
       @brief Perform a warp-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    inline T Sum(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::Sum unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, false, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    inline T AllSum(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::AllSum unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, true, quda::plus<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    inline T Max(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::Max unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, false, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    inline T AllMax(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::AllMax unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, true, quda::maximum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    inline T Min(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::Min unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, false, quda::minimum<T>(), param_t());
    }

    /**
       @brief Perform a warp-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads within the logical warp)
     */
    inline T AllMin(const T &value)
    {
      //static const __SYCL_CONSTANT_AS char format[] = "WarpReduce::AllMin unimplemented\n";
      //sycl::ext::oneapi::experimental::printf(format);
      return value;
      //return target::dispatch<warp_reduce>(value, true, quda::minimum<T>(), param_t());
    }
  };

  /**
     @brief BlockReduce provides a generic interface for performing
     perform reductions at the block level
     @tparam T The type of the value that we are reducing
     @tparam block_size_x The number of threads in the x dimension
     @tparam block_size_y The number of threads in the y dimension
     @tparam block_size_z The number of threads in the z dimension
     @tparam batched Whether this is a batched reduction or not.  If
     batched, then the block_size_z_ parameter is set equal to the
     batch size.
  */
  template <typename T, int block_size_x, int block_size_y = 1, int block_size_z = 1, bool batched = false>
  class BlockReduce
  {
    static constexpr int batch_size = !batched ? 1 : block_size_z;
    const int batch;

  public:
    constexpr BlockReduce(int batch = 0) : batch(batch) { }

    /**
       @brief Perform a block-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> inline T Sum(const T &value)
    {
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      for(int i=0; i<batch_size; i++) {
	if(i==batch) {
	  blockReduceSum(grp, result, value);
	} else {
	  T dum;
	  auto zero = quda::zero<T>();
	  blockReduceSum(grp, dum, zero);
	}
      }
      return result;
    }

    /**
       @brief Perform a block-wide sum reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllSum(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllSum with batch_size > 1");
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      blockReduceSum(grp, result, value);
      return result;
    }

    /**
       @brief Perform a block-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> __device__ __host__ inline T Max(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do Max with batch_size > 1");
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      blockReduceMax(grp, result, value);
      return result;
    }

    /**
       @brief Perform a block-wide max reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllMax(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllMax with batch_size > 1");
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      blockReduceMax(grp, result, value);
      return result;
    }

    /**
       @brief Perform a block-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in logical thread 0 only)
     */
    template <bool async = true> __device__ __host__ inline T Min(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do Min with batch_size > 1");
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      blockReduceMin(grp, result, value);
      return result;
    }

    /**
       @brief Perform a block-wide min reduction
       @param[in] value Thread-local value to be reduced
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true> __device__ __host__ inline T AllMin(const T &value)
    {
      static_assert(batch_size == 1, "Cannot do AllMin with batch_size > 1");
      if (!async) __syncthreads(); // only synchronize if we are not pipelining
      auto grp = getGroup();
      T result;
      blockReduceMin(grp, result, value);
      return result;
    }

    /**
       @brief Perform a block-wide custom reduction
       @param[in] value Thread-local value to be reduced
       @param[in] r The reduction operation we want to apply
       @return Reduced value (defined in logical thread 0 only)
     */
#if 0
    template <bool async = true, typename U>
    inline T
    ReduceNotSum(const T &value, const quda::maximum<U> &r)
    {
      return Max<async>(value);
    }

    template <bool async = true, typename U>
    inline T
    ReduceNotSum(const T &value, const quda::minimum<U> &r)
    {
      return Min<async>(value);
    }

    template <bool async = true, typename reducer_t>
    inline std::enable_if_t<!reducer_t::do_sum,T>
    Reduce(const T &value, const reducer_t &r)
    {
      return ReduceNotSum<async>(value, typename reducer_t::reducer_t());
    }

    template <bool async = true, typename reducer_t>
    inline std::enable_if_t<reducer_t::do_sum,T>
    Reduce(const T &value, const reducer_t &r)
    {
      return Sum<async>(value);
    }
#endif

    template <bool async = true, typename R>
    inline std::enable_if_t<std::is_same_v<typename R::reducer_t,plus<typename R::reduce_t>>,T>
    Reduce(const T &value, const R &)
    {
      return Sum<async>(value);
    }

    template <bool async = true, typename R>
    inline std::enable_if_t<std::is_same_v<typename R::reducer_t,maximum<typename R::reduce_t>>,T>
    Reduce(const T &value, const R &)
    {
      return Max<async>(value);
    }

    template <bool async = true, typename R>
    inline std::enable_if_t<std::is_same_v<typename R::reducer_t,minimum<typename R::reduce_t>>,T>
    Reduce(const T &value, const R &)
    {
      return Min<async>(value);
    }

#if 0
    /**
       @brief Perform a block-wide custom reduction
       @param[in] value Thread-local value to be reduced
       @param[in] r The reduction operation we want to apply
       @return Reduced value (defined in all threads in the block)
     */
    template <bool async = true, typename R>
    inline T AllReduce(const T &value, const R &r)
    {
      static_assert(batch_size == 1, "Cannot do AllReduce with batch_size > 1");
      auto grp = getGroup();
      T result;
      blockReduce(grp, result, value, r);  // FIXME: not used
      return result;
    }
#endif
  };

} // namespace quda
