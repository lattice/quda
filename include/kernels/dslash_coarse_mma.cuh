#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <shared_memory_cache_helper.h>
#include <kernel.h>
#include <warp_collective.h>
#include <dslash_quda.h>
#include <kernels/dslash_coarse.cuh>
#include <mma_tensor_op/gemm.cuh>
#include <complex_quda.h>

#define USE_TENSOR_MEMORY_ACCELERATOR

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
#include <tma_helper.hpp>
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
#endif

namespace quda
{

  template <class mma_t_, bool dslash_, bool clover_, bool dagger_, DslashType type_, typename Float_, typename yFloat_,
            typename ghostFloat, int nSpin_, int nColor_, int nVec_, int bN_, int bM_, int bK_, int block_y_, int block_z_>
  struct DslashCoarseMmaArg : kernel_param<> {
    static constexpr bool dslash = dslash_;
    static constexpr bool clover = clover_;
    static constexpr bool dagger = dagger_;
    static constexpr DslashType type = type_;

    using Float = Float_;
    using yFloat = yFloat_;

    static constexpr int block_dim = block_z_ * block_y_;
    static constexpr int min_blocks = 1;

    using mma_t = mma_t_;

    using real = typename mapper<Float>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nVec = nVec_;
    static constexpr int nDim = 4;
    static constexpr int nFace = 1;

    static constexpr int block_y = block_y_;
    static constexpr int block_z = block_z_;

    static constexpr int bM = bM_;
    static constexpr int bN = bN_;
    static constexpr int bK = bK_;

    static constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    static constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

    using ghost_accessor_t = typename colorspinor::GhostOrder<real, nSpin, nColor, nVec, csOrder, Float, ghostFloat>;
    // disable ghost to reduce arg size
    using field_accessor_t =
      typename colorspinor::FieldOrderCB<real, nSpin, nColor, nVec, csOrder, Float, ghostFloat, true>;
    using gauge_accessor_t = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;

    field_accessor_t out;
    const field_accessor_t inA;
    const field_accessor_t inB;
    const ghost_accessor_t halo;
    const gauge_accessor_t Y;
    const gauge_accessor_t X;

    const real kappa;
    const int parity;         // only use this for single parity fields
    const int nParity;        // number of parities we're working on
    const int_fastdiv X0h;    // X[0]/2
    const int_fastdiv dim[5]; // full lattice dimensions
    const int commDim[4];     // whether a given dimension is partitioned or not
    const int volumeCB;
    int ghostFaceCB[4];

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
    tma_descriptor_t tma_desc_g;
    tma_descriptor_t tma_desc_g_back;
    tma_descriptor_t tma_desc_inA;
#endif

    DslashCoarseMmaArg(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
                       const GaugeField &Y, const GaugeField &X, real kappa, int parity, const ColorSpinorField &halo) :
      kernel_param(dim3(out.SiteSubset() * X.VolumeCB(), block_y, block_z)),
      out(out),
      inA(inA),
      inB(inB),
      halo(halo, nFace),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      X0h(((3 - nParity) * out.X(0)) / 2),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out.VolumeCB() / dim[4])
    {
      if (out.Nvec() != nVec) { errorQuda("out.Nvec() (%d) != nVec (%d)", out.Nvec(), nVec); }
      if (inA.Nvec() != nVec) { errorQuda("inA.Nvec() (%d) != nVec (%d)", inA.Nvec(), nVec); }
      if (inB.Nvec() != nVec) { errorQuda("inB.Nvec() (%d) != nVec (%d)", inB.Nvec(), nVec); }
      // ghostFaceCB does not include the batch (5th) dimension at present
      for (int i = 0; i < 4; i++) ghostFaceCB[i] = halo.getDslashConstant().ghostFaceCB[i];

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
      tma_descriptor_key_t<5> key_g = {
        std::array<size_t, 5>{nColor * nSpin * 2, nColor * nSpin, 8, Y.VolumeCB(), static_cast<size_t>(Y.SiteSubset())},
        std::array<size_t, 5>{bK * 2, bM, 1, 1, 1},
        Y.data<complex<yFloat> *>()
      };
      tma_desc_g = get_tma_descriptor<yFloat, 5>(key_g);
      tma_descriptor_key_t<5> key_g_back = {
        std::array<size_t, 5>{nColor * nSpin * 2, nColor * nSpin, 8, Y.VolumeCB(), static_cast<size_t>(Y.SiteSubset())},
        std::array<size_t, 5>{bM * 2, bK, 1, 1, 1},
        Y.data<complex<yFloat> *>()
      };
      tma_desc_g_back = get_tma_descriptor<yFloat, 5>(key_g_back);
      tma_descriptor_key_t<4> key_inA = {
        std::array<size_t, 4>{nVec * 2, nColor * nSpin, inA.VolumeCB(), static_cast<size_t>(nParity)},
        std::array<size_t, 4>{bN * 2, bK, 1, 1},
        inA.data<complex<Float> *>()
      };
      tma_desc_inA = get_tma_descriptor<Float, 4>(key_inA);
#endif
    }
  };

  /**
     Applies the coarse dslash on a given parity and checkerboard site index
     /out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)

     @param[in] x_cb The checkerboarded site index
     @param[in] parity The site parity
     @param[in] arg Arguments
     @return The result vector
   */
  template <typename Arg>
  __device__ inline auto applyDslashMma(int x_cb, int parity, int m_offset, int n_offset, const Arg &arg)
  {
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    int coord[4];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    constexpr int M = Arg::nSpin * Arg::nColor;
    constexpr int N = Arg::nVec;
    constexpr int K = Arg::nSpin * Arg::nColor;

    constexpr int lda = M;
    constexpr int ldb = N;
    constexpr int ldc = N;

    using mma_t = typename Arg::mma_t;
    using Config = mma::MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

    constexpr bool do_rescale = mma_t::do_rescale();

    static_assert(M % Arg::bM == 0, "M %% Arg::bM != 0.\n");
    static_assert(N % Arg::bN == 0, "N %% Arg::bN != 0.\n");
    static_assert(K % Arg::bK == 0, "K %% Arg::bK != 0.\n");

    __align__(128) extern __shared__ typename mma_t::compute_t smem_ptr[];
    typename Config::SmemObjA smem_obj_a_real(smem_ptr);
    typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * Arg::bK);

    using store_a_t = complex<typename Arg::yFloat>;
    using store_b_t = complex<typename Arg::Float>;
    store_a_t *smem_tmp_a = reinterpret_cast<store_a_t *>(smem_obj_b_imag.ptr + Config::smem_ldb * Arg::bK);
    store_b_t *smem_tmp_b = reinterpret_cast<store_b_t *>(smem_tmp_a + (Arg::bK + mma::get_tmp_pad()) * (Arg::bM + mma::get_tmp_pad()));

    typename Config::Accumulator accumulator((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

    accumulator.zero();

    typename Config::ALoader a_loader;
    typename Config::BLoader b_loader;

    pipeline_t pipe = make_pipeline();

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
    barrier_t *bar = (barrier_t *)(smem_tmp_b + (Arg::bK + mma::get_tmp_pad()) * (Arg::bN + mma::get_tmp_pad()));
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      // Initialize barrier. All `blockDim.x` threads in block participate.
      init(bar, blockDim.x * blockDim.y * blockDim.z);
      // Make initialized barrier visible in async proxy.
      cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();
#endif

    bool forward_exterior[Arg::nDim];
    bool backward_exterior[Arg::nDim];
    int forward_idx[Arg::nDim];
    int backward_idx[Arg::nDim];

#pragma unroll
    for (int d = 0; d < Arg::nDim; d++) // loop over dimension
    {
      forward_exterior[d] = arg.commDim[d] && is_boundary(coord, d, 1, arg);
      backward_exterior[d] = arg.commDim[d] && is_boundary(coord, d, 0, arg);
      forward_idx[d] = linkIndexHop(coord, arg.dim, d, arg.nFace);
      backward_idx[d] = linkIndexHop(coord, arg.dim, d, -arg.nFace);
    }

    auto dslash_forward_producer = [&](int d, float &scale_inv_a, float &scale_inv_b, int k_offset) {
      const int fwd_idx = forward_idx[d];

      if (forward_exterior[d]) {
        if constexpr (doHalo<Arg::type>()) {
          int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

          auto a = arg.Y(Arg::dagger ? d : d + 4, parity, x_cb, 0, 0);
          auto b = arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx, 0, 0, 0);
          constexpr bool a_dagger = false;
          constexpr bool b_dagger = false;

          using store_b_ghost_t = complex<typename decltype(b)::store_type>;
          auto smem_tmp_b_ghost = reinterpret_cast<store_b_ghost_t *>(smem_tmp_b);

          __syncthreads();
          pipe.producer_acquire();
          scale_inv_a = a_loader.template g2tmp<lda, a_dagger>(a, m_offset, k_offset, smem_tmp_a, pipe);
          scale_inv_b = b_loader.template g2tmp<ldb, b_dagger>(b, n_offset, k_offset, smem_tmp_b_ghost, pipe);
          pipe.producer_commit();
        }
      } else if constexpr (doBulk<Arg::type>()) {

        auto a = arg.Y(Arg::dagger ? d : d + 4, parity, x_cb, 0, 0);
        auto b = arg.inA(their_spinor_parity, fwd_idx, 0, 0, 0);

        __syncthreads();

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
        scale_inv_a = a.get_scale_inv();
        scale_inv_b = b.get_scale_inv();
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          // Initiate bulk tensor copy: k_offset * 2 for complex
          cde::cp_async_bulk_tensor_5d_global_to_shared(smem_tmp_a, &arg.tma_desc_g.map, k_offset * 2, m_offset,
              Arg::dagger ? d : d + 4, x_cb, parity, *bar);
          cde::cp_async_bulk_tensor_4d_global_to_shared(smem_tmp_b, &arg.tma_desc_inA.map, n_offset * 2, k_offset,
              fwd_idx, their_spinor_parity, *bar);
        }
#else
        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;
        pipe.producer_acquire();
        scale_inv_a = a_loader.template g2tmp<lda, a_dagger>(a, m_offset, k_offset, smem_tmp_a, pipe);
        scale_inv_b = b_loader.template g2tmp<ldb, b_dagger>(b, n_offset, k_offset, smem_tmp_b, pipe);
        pipe.producer_commit();
#endif

      }
    };

    auto dslash_forward_consumer = [&](int d, float scale_inv_a, float scale_inv_b) -> float {
      float rescale_factor;
      if (forward_exterior[d]) {
        if constexpr (doHalo<Arg::type>()) {
          constexpr bool a_dagger = false;
          constexpr bool b_dagger = false;

          using a_wrapper_t = decltype(arg.Y(0, 0, 0, 0, 0));
          using b_wrapper_t = decltype(arg.halo.Ghost(0, 0, 0, 0, 0, 0, 0));
          using store_b_ghost_t = complex<typename b_wrapper_t::store_type>;
          auto smem_tmp_b_ghost = reinterpret_cast<store_b_ghost_t *>(smem_tmp_b);
          constexpr bool a_fixed = a_wrapper_t::fixed;
          constexpr bool b_fixed = b_wrapper_t::fixed;

          pipe.consumer_wait();
          __syncthreads();
          float a_rescale = a_loader.template tmp2s_rescale<lda, a_dagger, a_fixed, do_rescale>(smem_tmp_a, scale_inv_a, smem_obj_a_real, smem_obj_a_imag);
          float b_rescale = b_loader.template tmp2s_rescale<ldb, b_dagger, b_fixed, do_rescale>(smem_tmp_b_ghost, scale_inv_b, smem_obj_b_real,
              smem_obj_b_imag);
          rescale_factor = a_rescale * b_rescale;
          pipe.consumer_release();
          __syncthreads();
        }
      } else if constexpr (doBulk<Arg::type>()) {

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;

        using a_wrapper_t = decltype(arg.Y(0, 0, 0, 0, 0));
        using b_wrapper_t = decltype(arg.inA(0, 0, 0, 0, 0));
        constexpr bool a_fixed = a_wrapper_t::fixed;
        constexpr bool b_fixed = b_wrapper_t::fixed;

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
        barrier_t::arrival_token token;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          // Arrive on the barrier and tell how many bytes are expected to come in.
          token = cuda::device::barrier_arrive_tx(*bar, 1,
              Arg::bK * 2 * Arg::bM * sizeof(typename Arg::yFloat) + Arg::bN * 2 * Arg::bK * sizeof(typename Arg::Float));
        } else {
          // Other threads just arrive.
          token = bar->arrive();
        }
        // Wait for the data to have arrived.
        bar->wait(std::move(token));
#else
        pipe.consumer_wait();
        __syncthreads();
#endif
        float a_rescale = a_loader.template tmp2s_rescale<lda, a_dagger, a_fixed, do_rescale>(smem_tmp_a, scale_inv_a, smem_obj_a_real, smem_obj_a_imag);
        float b_rescale = b_loader.template tmp2s_rescale<ldb, b_dagger, b_fixed, do_rescale>(smem_tmp_b, scale_inv_b, smem_obj_b_real, smem_obj_b_imag);
        rescale_factor = a_rescale * b_rescale;
#ifndef USE_TENSOR_MEMORY_ACCELERATOR
        pipe.consumer_release();
#endif
        __syncthreads();
      }
      return rescale_factor;
    };

    auto dslash_forward_compute = [&](int d, float rescale_factor) {
      if (forward_exterior[d] && doHalo<Arg::type>() || doBulk<Arg::type>()) {
        if constexpr (do_rescale) {
          accumulator.mma_rescale(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, rescale_factor);
        } else {
          accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
        }
      }
    };

    auto dslash_backward_producer = [&](int d, float &scale_inv_a, float &scale_inv_b, int k_offset) {
      const int back_idx = backward_idx[d];

      if (backward_exterior[d]) {
        if constexpr (doHalo<Arg::type>()) {
          const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

          auto a = arg.Y.Ghost(Arg::dagger ? d + 4 : d, 1 - parity, ghost_idx, 0, 0);
          auto b = arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx, 0, 0, 0);
          constexpr bool a_dagger = true;
          constexpr bool b_dagger = false;

          using store_b_ghost_t = complex<typename decltype(b)::store_type>;
          auto smem_tmp_b_ghost = reinterpret_cast<store_b_ghost_t *>(smem_tmp_b);

          __syncthreads();
          pipe.producer_acquire();
          scale_inv_a = a_loader.template g2tmp<lda, a_dagger>(a, m_offset, k_offset, smem_tmp_a, pipe);
          scale_inv_b = b_loader.template g2tmp<ldb, b_dagger>(b, n_offset, k_offset, smem_tmp_b_ghost, pipe);
          pipe.producer_commit();
        }
      } else if constexpr (doBulk<Arg::type>()) {
        const int gauge_idx = back_idx;

        auto a = arg.Y(Arg::dagger ? d + 4 : d, 1 - parity, gauge_idx, 0, 0);
        auto b = arg.inA(their_spinor_parity, back_idx, 0, 0, 0);

        __syncthreads();
#ifdef USE_TENSOR_MEMORY_ACCELERATOR
        scale_inv_a = a.get_scale_inv();
        scale_inv_b = b.get_scale_inv();
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          // Initiate bulk tensor copy: k_offset * 2 for complex
          cde::cp_async_bulk_tensor_5d_global_to_shared(smem_tmp_a, &arg.tma_desc_g_back.map, m_offset * 2, k_offset,
              Arg::dagger ? d + 4 : d, gauge_idx, 1 - parity, *bar);
          cde::cp_async_bulk_tensor_4d_global_to_shared(smem_tmp_b, &arg.tma_desc_inA.map, n_offset * 2, k_offset,
              back_idx, their_spinor_parity, *bar);
        }
#else
        constexpr bool a_dagger = true;
        constexpr bool b_dagger = false;
        pipe.producer_acquire();
        scale_inv_a = a_loader.template g2tmp<lda, a_dagger>(a, m_offset, k_offset, smem_tmp_a, pipe);
        scale_inv_b = b_loader.template g2tmp<ldb, b_dagger>(b, n_offset, k_offset, smem_tmp_b, pipe);
        pipe.producer_commit();
#endif
      }
    };

    auto dslash_backward_consumer = [&](int d, float scale_inv_a, float scale_inv_b) -> float {
      float rescale_factor;
      if (backward_exterior[d]) {
        if constexpr (doHalo<Arg::type>()) {
          constexpr bool a_dagger = true;
          constexpr bool b_dagger = false;

          using a_wrapper_t = decltype(arg.Y.Ghost(0, 0, 0, 0, 0));
          using b_wrapper_t = decltype(arg.halo.Ghost(0, 0, 0, 0, 0, 0, 0));
          using store_b_ghost_t = complex<typename b_wrapper_t::store_type>;
          auto smem_tmp_b_ghost = reinterpret_cast<store_b_ghost_t *>(smem_tmp_b);
          constexpr bool a_fixed = a_wrapper_t::fixed;
          constexpr bool b_fixed = b_wrapper_t::fixed;

          pipe.consumer_wait();
          __syncthreads();
          float a_rescale = a_loader.template tmp2s_rescale<lda, a_dagger, a_fixed, do_rescale>(smem_tmp_a, scale_inv_a, smem_obj_a_real, smem_obj_a_imag);
          float b_rescale = b_loader.template tmp2s_rescale<ldb, b_dagger, b_fixed, do_rescale>(smem_tmp_b_ghost, scale_inv_b, smem_obj_b_real,
              smem_obj_b_imag);
          rescale_factor = a_rescale * b_rescale;
          pipe.consumer_release();
          __syncthreads();
        }
      } else if constexpr (doBulk<Arg::type>()) {
        constexpr bool a_dagger = true;
        constexpr bool b_dagger = false;

        using a_wrapper_t = decltype(arg.Y(0, 0, 0, 0, 0));
        using b_wrapper_t = decltype(arg.inA(0, 0, 0, 0, 0));
        constexpr bool a_fixed = a_wrapper_t::fixed;
        constexpr bool b_fixed = b_wrapper_t::fixed;

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
        barrier_t::arrival_token token;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
          // Arrive on the barrier and tell how many bytes are expected to come in.
          token = cuda::device::barrier_arrive_tx(*bar, 1,
              Arg::bK * 2 * Arg::bM * sizeof(typename Arg::yFloat) + Arg::bN * 2 * Arg::bK * sizeof(typename Arg::Float));
        } else {
          // Other threads just arrive.
          token = bar->arrive();
        }
        // Wait for the data to have arrived.
        bar->wait(std::move(token));
        // __syncthreads();
#else
        pipe.consumer_wait();
        __syncthreads();
#endif
        float a_rescale = a_loader.template tmp2s_rescale<lda, a_dagger, a_fixed, do_rescale>(smem_tmp_a, scale_inv_a, smem_obj_a_real, smem_obj_a_imag);
        float b_rescale = b_loader.template tmp2s_rescale<ldb, b_dagger, b_fixed, do_rescale>(smem_tmp_b, scale_inv_b, smem_obj_b_real, smem_obj_b_imag);
        rescale_factor = a_rescale * b_rescale;
#ifndef USE_TENSOR_MEMORY_ACCELERATOR
        pipe.consumer_release();
#endif
        __syncthreads();
      }
      return rescale_factor;
    };

      auto dslash_backward_compute = [&](int d, float rescale_factor) {
        if (backward_exterior[d] && doHalo<Arg::type>() || doBulk<Arg::type>()) {
          if constexpr (do_rescale) {
            accumulator.mma_rescale(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, rescale_factor);
          } else {
            accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
          }
        }
      };

      auto clover_producer = [&](float &scale_inv_a, float &scale_inv_b, int k_offset) {
        const int spinor_parity = (arg.nParity == 2) ? parity : 0;

        auto a = arg.X(0, parity, x_cb, 0, 0);
        auto b = arg.inB(spinor_parity, x_cb, 0, 0);
        constexpr bool a_dagger = Arg::dagger;
        constexpr bool b_dagger = false;

        __syncthreads();
        pipe.producer_acquire();
        scale_inv_a = a_loader.template g2tmp<lda, a_dagger>(a, m_offset, k_offset, smem_tmp_a, pipe);
        scale_inv_b = b_loader.template g2tmp<ldb, b_dagger>(b, n_offset, k_offset, smem_tmp_b, pipe);
        pipe.producer_commit();
      };

      auto clover_consumer = [&](float scale_inv_a, float scale_inv_b) -> float {
        constexpr bool a_dagger = Arg::dagger;
        constexpr bool b_dagger = false;

        using a_wrapper_t = decltype(arg.X(0, 0, 0, 0, 0));
        using b_wrapper_t = decltype(arg.inB(0, 0, 0, 0));
        constexpr bool a_fixed = a_wrapper_t::fixed;
        constexpr bool b_fixed = b_wrapper_t::fixed;

        pipe.consumer_wait();
        __syncthreads();
        float a_rescale = a_loader.template tmp2s_rescale<lda, a_dagger, a_fixed, do_rescale>(smem_tmp_a, scale_inv_a, smem_obj_a_real, smem_obj_a_imag);
        float b_rescale = b_loader.template tmp2s_rescale<ldb, b_dagger, b_fixed, do_rescale>(smem_tmp_b, scale_inv_b, smem_obj_b_real, smem_obj_b_imag);
        pipe.consumer_release();
        __syncthreads();
        return a_rescale * b_rescale;
      };

      auto clover_compute
        = [&](float rescale_factor) {
        if constexpr (do_rescale) {
          accumulator.mma_rescale(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, rescale_factor);
        } else {
          accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
        }
      };

      float scale_inv_a;
      float scale_inv_b;

      if constexpr (Arg::dslash) {

        dslash_forward_producer(0, scale_inv_a, scale_inv_b, 0);

        for (int k_offset = 0; k_offset < K; k_offset += Arg::bK) {

          // Forward gather - compute fwd offset for spinor fetch
#pragma unroll
          for (int d = 0; d < Arg::nDim; d++) // loop over dimension
          {
            float rescale_factor = dslash_forward_consumer(d, scale_inv_a, scale_inv_b);
            if (d < 3) {
              dslash_forward_producer(d + 1, scale_inv_a, scale_inv_b, k_offset);
            } else {
              dslash_backward_producer(0, scale_inv_a, scale_inv_b, k_offset);
            }
            dslash_forward_compute(d, rescale_factor);
          } // nDim

          // Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
          for (int d = 0; d < Arg::nDim; d++) {
            float rescale_factor = dslash_backward_consumer(d, scale_inv_a, scale_inv_b);
            if (d < 3) {
              dslash_backward_producer(d + 1, scale_inv_a, scale_inv_b, k_offset);
            } else if (k_offset + Arg::bK < K) {
              dslash_forward_producer(0, scale_inv_a, scale_inv_b, k_offset + Arg::bK);
            } else if constexpr (doBulk<Arg::type>() && Arg::clover) {
              clover_producer(scale_inv_a, scale_inv_b, 0);
            }
            dslash_backward_compute(d, rescale_factor);
          } // nDim
        }

        accumulator.ax(-arg.kappa);
      }

      /**
         Applies the coarse clover matrix on a given parity and
         checkerboard site index
       */
      if constexpr (doBulk<Arg::type>() && Arg::clover) {
        if constexpr (!Arg::dslash) { clover_producer(scale_inv_a, scale_inv_b, 0); }
        for (int k_offset = 0; k_offset < K; k_offset += Arg::bK) {
          float rescale_factor = clover_consumer(scale_inv_a, scale_inv_b);
          if (k_offset + Arg::bK < K) { clover_producer(scale_inv_a, scale_inv_b, k_offset + Arg::bK); }
          clover_compute(rescale_factor);
        }
      }

    return accumulator;
  }

  __device__ __host__ inline void fetch_add(float4 *addr, float4 val)
  {
    addr->x += val.x;
    addr->y += val.y;
    addr->z += val.z;
    addr->w += val.w;
  }

  __device__ __host__ inline void fetch_add(short4 *addr, short4 val)
  {
    addr->x += val.x;
    addr->y += val.y;
    addr->z += val.z;
    addr->w += val.w;
  }

  template <typename T> __device__ __host__ inline void fetch_add(complex<T> *addr, complex<T> val)
  {
    addr->real(addr->real() + val.real());
    addr->imag(addr->imag() + val.imag());
  }

  template <typename T, int n> __device__ __host__ inline void fetch_add(array<T, n> *addr, array<T, n> val)
  {
    for (int i = 0; i < n; i++) { (*addr)[i] += val[i]; }
  }

  struct fetch_add_t {
    template <class T> __device__ __host__ inline void operator()(T *out, T in) { fetch_add(out, in); }
  };

  template <typename Arg> struct CoarseDslashMma {
    const Arg &arg;
    constexpr CoarseDslashMma(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()()
    {
      int parity_x_cb = target::block_idx().x;
      int n_offset = target::block_idx().z * Arg::bN;
      int m_offset = target::block_idx().y * Arg::bM;
      int parity = (arg.nParity == 2) ? parity_x_cb % 2 : arg.parity;
      int x_cb = (arg.nParity == 2) ? parity_x_cb / 2 : parity_x_cb;

      auto out = applyDslashMma(x_cb, parity, m_offset, n_offset, arg);

      constexpr int M = Arg::nSpin * Arg::nColor;
      constexpr int N = Arg::nVec;
      constexpr int ldc = N;

      int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
      auto c = arg.out(my_spinor_parity, x_cb, 0, 0);
      if constexpr (doBulk<Arg::type>()) {
        out.template store<M, N, ldc, false>(c, m_offset, n_offset, assign_t());
      } else {
        out.template store<M, N, ldc, false>(c, m_offset, n_offset, fetch_add_t());
      }
    }
  };

} // namespace quda
