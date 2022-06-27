#pragma once

#include <cuda/pipeline>
#include <shared_memory_cache_helper.cuh>

namespace quda
{
  template <class color_spinor_order_t, class gauge_order_t, int stages>
  struct StencilCache
  {
    const color_spinor_order_t &color_spinor_order;
    const gauge_order_t &gauge_order;

    using bulk_t = array<typename color_spinor_order_t::Vector, color_spinor_order_t::M>;
    using norm_t = float;
    static constexpr int gauge_M = gauge_order_t::reconLen / gauge_order_t::N;
    using gauge_t = array<typename gauge_order_t::Vector, gauge_M>;

    using Float = typename color_spinor_order_t::Float;

    static constexpr size_t norm_bytes = isFixed<Float>::value ? sizeof(norm_t) : 0;
    static constexpr size_t bytes = sizeof(bulk_t) + norm_bytes + sizeof(gauge_t);

    void *_bulk_ptr[stages];
    void *_norm_ptr[stages];
    void *_gauge_ptr[stages];

    dim3 block;
    dim3 thread;
    const int stride;
    int j;
    int warp_id;
    int lane_id;

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

    template <class vector_t>
    __device__ __host__ vector_t *bulk(int index, int stage) {
      // return &reinterpret_cast<vector_t *>(_bulk_ptr[stage])[index * stride + j];
      return &reinterpret_cast<vector_t *>(_bulk_ptr[stage])[(warp_id * color_spinor_order_t::M + index) * 32 + lane_id];
    }

    __device__ __host__ norm_t *norm(int stage) {
      return &reinterpret_cast<norm_t *>(_norm_ptr[stage])[j];
    }

    template <class vector_t>
    __device__ __host__ vector_t *gauge(int index, int stage) {
      return &reinterpret_cast<vector_t *>(_gauge_ptr[stage])[(warp_id * gauge_M + index) * 32 + lane_id];
    }

    __device__ __host__ inline auto load_color_spinor(int stage)
    {
      ColorSpinor<typename color_spinor_order_t::real, color_spinor_order_t::Nc, color_spinor_order_t::Ns> color_spinor;

      norm_t nrm = isFixed<Float>::value ? *norm(stage) : 0.0;
      using Vector = typename color_spinor_order_t::Vector;
      Vector vecTmp[color_spinor_order_t::M];

#pragma unroll
      for (int i = 0; i < color_spinor_order_t::M; i++) {
        // first load from memory
        vecTmp[i] = *bulk<Vector>(i, stage);
      }

      color_spinor_order.unpack(color_spinor.data, vecTmp, nrm);
      return color_spinor;
    }

    __device__ __host__ inline auto load_color_spinor_half(int stage)
    {
      ColorSpinor<typename color_spinor_order_t::real, color_spinor_order_t::Nc, color_spinor_order_t::Ns / 2> color_spinor;

      norm_t nrm = isFixed<Float>::value ? *norm(stage) : 0.0;
      using GhostVector = typename color_spinor_order_t::GhostVector;
      using Vector = typename color_spinor_order_t::Vector;
      GhostVector vecTmp[color_spinor_order_t::M_ghost];

#pragma unroll
      for (int i = 0; i < color_spinor_order_t::M_ghost; i++) {
        // first load from memory
        vecTmp[i] = *reinterpret_cast<GhostVector *>(bulk<Vector>(i, stage));
      }

      color_spinor_order.unpack_half(color_spinor.data, vecTmp, nrm);
      return color_spinor;
    }

    __device__ __host__ inline auto load_gauge(int stage, int dir, int x, int parity, int inphase = 1.0)
    {
        Matrix<complex<typename gauge_order_t::real>, color_spinor_order_t::Nc> matrix;
        const int M = gauge_order_t::reconLen / gauge_order_t::N;
        using Vector = typename gauge_order_t::Vector;

        Vector vecTmp[gauge_order_t::reconLen];

#pragma unroll
        for (int i = 0; i < M; i++) {
          vecTmp[i] = *gauge<Vector>(i, stage);
        }

        gauge_order.unpack(matrix.data, vecTmp, x, dir, parity, inphase);
        return matrix;
    }

    __device__ __host__ StencilCache(const color_spinor_order_t &color_spinor_order, const gauge_order_t &gauge_order):
      color_spinor_order(color_spinor_order),
      gauge_order(gauge_order),
      block(target::block_dim()),
      thread(target::thread_idx()),
      stride(block.x * block.y * block.z),
      j((thread.z * block.y + thread.y) * block.x + thread.x),
      warp_id(j / 32),
      lane_id(j % 32)
    {
      char *cache = target::dispatch<cache_dynamic>();
#pragma unroll
      for (int stage = 0; stage < stages; stage++) {
        _bulk_ptr[stage] = cache + stride * (stage * bytes);
        _norm_ptr[stage] = cache + stride * (stage * bytes + sizeof(bulk_t));
        _gauge_ptr[stage] = cache + stride * (stage * bytes + sizeof(bulk_t) + norm_bytes);
      }
    }

  };
}

