#pragma once

#include <target_device.h>
#include <reducer.h>

/**
   @file block_reduce_helper.h

   @section This files contains the CUDA device specializations for
   warp- and block-level reductions, using the CUB library
 */

//using namespace quda;

namespace quda
{

  /**
     @brief The atomic word size we use for a given reduction type.
     This type should be lock-free to guarantee correct behaviour on
     platforms that are not coherent with respect to the host
   */
  template <typename T, typename Enable = void> struct atomic_type;

  template <> struct atomic_type<device_reduce_t> {
    using type = device_reduce_t;
  };

  template <> struct atomic_type<float> {
    using type = float;
  };

  template <typename T> struct atomic_type<T, std::enable_if_t<std::is_same_v<T, array<device_reduce_t, T::N>>>> {
    using type = device_reduce_t;
  };

  template <typename T>
  struct atomic_type<T, std::enable_if_t<std::is_same_v<T, array<array<device_reduce_t, T::value_type::N>, T::N>>>> {
    using type = device_reduce_t;
  };

  template <typename T> struct atomic_type<T, std::enable_if_t<std::is_same_v<T, array<complex<double>, T::N>>>> {
    using type = double;
  };

  template <typename T> struct atomic_type<T, std::enable_if_t<std::is_same_v<T, array<complex<float>, T::N>>>> {
    using type = float;
  };

  template <typename T, typename U> constexpr auto get_reducer(const plus<U> &) { return plus<T>(); }
  template <typename T, typename U> constexpr auto get_reducer(const maximum<U> &) { return maximum<T>(); }
  template <typename T, typename U> constexpr auto get_reducer(const minimum<U> &) { return minimum<T>(); }

  //template <typename T, typename U> constexpr auto get_cg_reducer(const plus<U> &) { return cg::plus<T>(); }
  //template <typename T, typename U> constexpr auto get_cg_reducer(const maximum<U> &) { return cg::greater<T>(); }
  //template <typename T, typename U> constexpr auto get_cg_reducer(const minimum<U> &) { return cg::less<T>(); }


  // pre-declaration of warp_reduce that we wish to specialize
  template <bool> struct warp_reduce;

  /**
     @brief CUDA specialization of warp_reduce, utilizing cooperative groups
  */
  template <> struct warp_reduce<true> {

    /**
       @brief Perform a warp-wide reduction using cooperative groups
       @param[in] value_ thread-local value to be reduced
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       warp (all = false).
       @param[in] r The reduction operation we want to apply
       @return The warp-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    T inline operator()(const T &value_, bool all, const reducer_t &r, const param_t &)
    {
      auto sg = sycl::ext::oneapi::experimental::this_sub_group();
      T value = value_;
#pragma unroll
      for (int offset = device::warp_size() / 2; offset >= 1; offset /= 2) {
	//value = get_reducer<T>(r)(value, tile.shfl_down(value, offset));
	value = r(value, sycl::shift_group_left(sg, value, offset));
      }
      //if (all) value = tile.shfl(value, 0);
      if (all) value = sycl::select_from_group(sg, value, 0);
      return value;
    }

  };


  // pre-declaration of block_reduce that we wish to specialize
  template <bool> struct block_reduce;

  /**
     @brief CUDA specialization of block_reduce, building on the warp_reduce
  */
  template <> struct block_reduce<true> {

    template <int width_> struct warp_reduce_param {
      static constexpr int width = width_;
    };

    /**
       @brief Perform a block-wide reduction
       @param[in] value_ thread-local value to be reduced
       @param[in] async Whether this reduction will be performed
       asynchronously with respect to the calling threads
       @param[in] batch The batch index of the reduction
       @param[in] all Whether we want all threads to have visibility
       to the result (all = true) or just the first thread in the
       block (all = false)
       @param[in] r The reduction operation we want to apply
       @return The block-wide reduced value
     */
    template <typename T, typename reducer_t, typename param_t>
    inline T operator()(const T &value_, bool async, int batch, bool all,
			const reducer_t &r, const param_t &)
    {
      constexpr auto max_items = device::max_block_size() / device::warp_size();
      const auto thread_idx = target::thread_idx_linear<param_t::block_dim>();
      const auto block_size = target::block_size<param_t::block_dim>();
      const auto warp_idx = thread_idx / device::warp_size();
      const auto warp_items = (block_size + device::warp_size() - 1) / device::warp_size();

      // first do warp reduce
      T value = warp_reduce<true>()(value_, false, r, warp_reduce_param<device::warp_size()>());

      if (!all && warp_items == 1) return value; // short circuit for single warp CTA

      // now do reduction between warps
      if (!async) __syncthreads(); // only synchronize if we are not pipelining

      //__shared__ T storage[max_items];
      static_assert(sizeof(T[max_items])<=device::shared_memory_size(), "Block reduce shared mem size too large");
      auto mem = sycl::ext::oneapi::group_local_memory_for_overwrite<T[max_items]>(getGroup());
      auto storage = *mem.get();

      // if first thread in warp, write result to shared memory
      if (thread_idx % device::warp_size() == 0) storage[batch * warp_items + warp_idx] = value;
      __syncthreads();

      // whether to use the first warp or first thread for the final reduction
      constexpr bool final_warp_reduction = true;

      if constexpr (final_warp_reduction) { // first warp completes the reduction (requires first warp is full)
        if (warp_idx == 0) {
          if constexpr (max_items > device::warp_size()) { // never true for max block size 1024, warp = 32
            value = r.init();
            for (auto i = thread_idx; i < warp_items; i += device::warp_size())
              value = r(storage[batch * warp_items + i], value);
          } else { // optimized path where we know the final reduction will fit in a warp
            value = thread_idx < warp_items ? storage[batch * warp_items + thread_idx] : r.init();
          }
          value = warp_reduce<true>()(value, false, r, warp_reduce_param<device::warp_size()>());
        }
      } else { // first thread completes the reduction
        if (thread_idx == 0) {
          for (unsigned int i = 1; i < warp_items; i++) value = r(storage[batch * warp_items + i], value);
        }
      }

      if (all) {
        if (thread_idx == 0) storage[batch * warp_items + 0] = value;
        __syncthreads();
        value = storage[batch * warp_items + 0];
      }

      return value;
    }
  };

} // namespace quda

#include "../generic/block_reduce_helper.h"
