#pragma once

#include <load_store.h> // for atom_t
#include <kernel_ops.h>
#include <target_device.h>
#include <shared_memory_helper.h>

namespace quda
{

  template <typename color_spinor_order_t> class VanillaSharedMemoryCache
  {
    const color_spinor_order_t &color_spinor_order;

    using bulk_t = array<typename color_spinor_order_t::Vector, color_spinor_order_t::M>;
    using norm_t = float;

    using Float = typename color_spinor_order_t::Float;

    static constexpr size_t norm_bytes = isFixed<Float>::value ? sizeof(norm_t) : 0;
    static constexpr size_t bytes = sizeof(bulk_t) + norm_bytes;

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
      __device__ inline char *operator()()
      {
        __align__(128) extern __shared__ char cache__[];
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

    const int stride;

  public:

    void *_bulk_ptr;
    void *_norm_ptr;

    /**
       @brief Constructor for SharedMemoryCache.
    */
    __device__ __host__ VanillaSharedMemoryCache(const color_spinor_order_t &color_spinor_order, int stride_) :
      color_spinor_order(color_spinor_order), stride(stride_)
    {
      char *cache = target::dispatch<cache_dynamic>();
      _bulk_ptr = cache;
      _norm_ptr = cache + stride * sizeof(bulk_t);
    }

    /**
       @brief Synchronize the cache
    */
    __device__ __host__ void sync() const { target::dispatch<sync_impl>(); }

    template <class vector_t> __device__ __host__ vector_t *bulk(int index, int j)
    {
      return &reinterpret_cast<vector_t *>(_bulk_ptr)[index * stride + j];
      // return &reinterpret_cast<vector_t *>(_bulk_ptr[stage])[(warp_id * color_spinor_order_t::M + index) * 32 + lane_id];
    }

    __device__ __host__ norm_t *norm(int j) { return &reinterpret_cast<norm_t *>(_norm_ptr)[j]; }

    __device__ __host__ inline auto load(int j)
    {
      ColorSpinor<typename color_spinor_order_t::real, color_spinor_order_t::Nc, color_spinor_order_t::Ns> color_spinor;

      norm_t nrm = isFixed<Float>::value ? *norm(j) : 0.0;
      using Vector = typename color_spinor_order_t::Vector;
      Vector vecTmp[color_spinor_order_t::M];

#pragma unroll
      for (int i = 0; i < color_spinor_order_t::M; i++) {
        // first load from memory
        vecTmp[i] = *bulk<Vector>(i, j);
      }

      color_spinor_order.unpack(color_spinor.data, vecTmp, nrm);
      return color_spinor;
    }
  };

} // namespace quda
