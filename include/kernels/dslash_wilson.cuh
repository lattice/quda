#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <kernels/spinor_reweight.cuh>
#include <stencil_cache.cuh>
#include <tma_helper.hpp>

namespace quda
{

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_, bool distance_pc_ = false>
  struct WilsonArg : DslashArg<Float, nDim> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type F;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, colorspinor::getNative<Float>(nSpin),
                                                    spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool distance_pc = distance_pc_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out[MAX_MULTI_RHS]; /** output vector field set */
    F in[MAX_MULTI_RHS];  /** input vector field set */
    F x[MAX_MULTI_RHS];   /** input vector set when doing xpay */
    Ghost halo_pack;
    Ghost halo;
    const G U;    /** the gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */
    /** parameters for distance preconditioning */
    const real alpha0;
    const int t0;
    const int comm_coord_dim_3;
    const int comm_dim_dim_3;

    tma_descriptor_t tma_desc;

    WilsonArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
              const GaugeField &U, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
              const int *comm_override, double alpha0 = 0.0, int t0 = -1) :
      DslashArg<Float, nDim>(out, in, halo, U, x, parity, dagger, a != 0.0 ? true : false, 1, spin_project,
                             comm_override),
      halo_pack(halo),
      halo(halo),
      U(U),
      a(a),
      alpha0(alpha0),
      t0(t0),
      comm_coord_dim_3(comm_coord(3) * this->dim[3]),
      comm_dim_dim_3(comm_dim(3) * this->dim[3])
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
        this->x[i] = x[i];
      }
    }
  };

  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate struct
     @param[in] s The fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector, typename Cache>
  __device__ __host__ inline void applyWilson(Vector &out, const Arg &arg, Coord &coord, Coord &local_coord, int parity, int idx,
                                              int thread_dim, bool &active, int src_idx, Cache &cache)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    // auto block = target::block_dim();
    // VanillaSharedMemoryCache<Vector> cache(0 ? arg.tb.volume_4d_cb_ex : arg.tb.volume_4d_cb);

    const int t = arg.comm_coord_dim_3 + coord[3];
    const int nt = arg.comm_dim_dim_3;
    real fwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t + 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);
    real bwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t - 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const real fwd_coeff = (d < 3) ? 1.0 : fwd_coeff_3;
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (0 && doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 1, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += fwd_coeff * (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
#if 0
          Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
#else
          Vector in;
          if (0) {
            int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, +1, arg.tb);
            in = cache.load(local_fwd_idx);
          } else {
            bool out_of_block = (local_coord[d] + 1) >= arg.tb.dim[d] && arg.tb.dim[d] < arg.dim[d];
            if (out_of_block) {
              in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
            } else {
              int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, +1, arg.tb);
              in = cache.load(local_fwd_idx);
            }
          }
#endif
          out += fwd_coeff * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const real bwd_coeff = (d < 3) ? 1.0 : bwd_coeff_3;
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (0 && doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 0, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += bwd_coeff * (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
#if 0
          Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
#else
          Vector in;
          if (0) {
            int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, -1, arg.tb);
            in = cache.load(local_fwd_idx);
          } else {
            bool out_of_block = (local_coord[d] - 1) < 0 && arg.tb.dim[d] < arg.dim[d];
            if (out_of_block) {
              in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
            } else {
              int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, -1, arg.tb);
              in = cache.load(local_fwd_idx);
            }
          }
#endif
          out += bwd_coeff * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
    } // nDim
  }

 template <typename Arg>
  __host__ __device__ inline auto get_tb_coords_ex(const Arg &arg, int local_idx, int s, int parity)
  {
    Coord<4> coord;
    Coord<4> local_coord;

    int block_coord[4]; // coordinate of this threadblock in the threadblock grid

    int block_idx = target::block_idx().x;
#pragma unroll
    for (int d = 0; d < 4; d++) {
      block_coord[d] = block_idx % arg.tb.grid_dim[d];
      block_idx /= arg.tb.grid_dim[d];
    }

    int block_offset[4]; // global coordinate offset
#pragma unroll
    for (int d = 0; d < 4; d++) {
      block_offset[d] = block_coord[d] * arg.tb.dim[d];
    }

    int local_parity = (block_offset[0] + block_offset[1] + block_offset[2] + block_offset[3] + parity) % 2;

    local_coord.X = getCoordsCB(local_coord, local_idx, arg.tb.dim_ex, arg.tb.Xex0h, local_parity);
    local_coord.x_cb = local_idx;
#pragma unroll
    for (int d = 0; d < 4; d++) {
      // -1 for the boundary terms, % makes sure we get the around the world terms
      coord[d] = (local_coord[d] - (arg.tb.dim[d] == arg.dim[d] ? 0 : 1) + block_offset[d] + arg.dim[d]) % arg.dim[d];
    }
    int index = 0;
#pragma unroll
    for (int d = 3; d >= 0; d--) {
      index = index * arg.dim[d] + coord[d];
    }
    coord.X = index;
    coord.x_cb = coord.X / 2;
    coord.s = s;
    local_coord.s = s;

    return coord;
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr wilson(const Ftor &ftor) : arg(ftor.arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    // out(x) = M*in = (-D + m) * in(x-mu)
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

#if 0
      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
#else
      bool active = true;
#endif
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)

#if 0
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

      int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
      if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x[src_idx](xs, my_spinor_parity);
        out = x + arg.a * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out[src_idx](xs, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
#else
      // Load all interior color spinor fields
      auto block = target::block_dim();
      VanillaSharedMemoryCache<typename Arg::F> cache(arg.in[src_idx], 0 ? arg.tb.volume_4d_cb_ex : arg.tb.volume_4d_cb);
      int local_idx = target::thread_idx().x;

#if 0
      cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

      while (local_idx < (0 ? arg.tb.volume_4d_cb_ex : arg.tb.volume_4d_cb)) {
        const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;
        Coord<4> coord;
        if (0) {
          // Get the coordinate with all the boundary conditions figured out
          coord = get_tb_coords_ex(arg, local_idx, 0, 1 - (parity + arg.tb.parity_bit) % 2);
        } else {
          Coord<4> local_coord;
          coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, local_idx, 0, 1 - parity, thread_dim, local_coord);
        }
        // cache.save(arg.in[src_idx](coord.x_cb + coord.s * arg.dc.volume_4d_cb, their_spinor_parity), local_idx);
        arg.in[src_idx].cache(cache, pipe, coord.x_cb + coord.s * arg.dc.volume_4d_cb, their_spinor_parity, local_idx);
        local_idx += target::block_dim().x;
      }
      cuda::pipeline_consumer_wait_prior<1>(pipe);
      cache.sync();
#else
      barrier_t *bar = (barrier_t *)(cache._norm_ptr);
  #ifdef __CUDA_ARCH__
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        // Initialize barrier. All `blockDim.x` threads in block participate.
        init(bar, blockDim.x * blockDim.y * blockDim.z);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
      }
  #endif
      // Syncthreads so initialized barrier is visible to all threads.
      cache.sync();
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        int block_idx = target::block_idx().x;
        int block_coord[4]; // coordinate of this threadblock in the threadblock grid
  #pragma unroll
        for (int d = 0; d < 4; d++) {
          block_coord[d] = block_idx % arg.tb.grid_dim[d];
          block_idx /= arg.tb.grid_dim[d];
        }

        int block_offset[4]; // global coordinate offset
  #pragma unroll
        for (int d = 0; d < 4; d++) {
          block_offset[d] = block_coord[d] * arg.tb.dim[d];
        }
        tma_load_gmem_5d(cache._bulk_ptr, &arg.tma_desc.map,
          block_offset[0] / 2 * 16, block_offset[1], block_offset[2], block_offset[3], 0, bar);
      }
  #ifdef __CUDA_ARCH__
      barrier_t::arrival_token token;
      if (target::thread_idx().x == 0 && target::thread_idx().y == 0 && target::thread_idx().z == 0) {
        // Arrive on the barrier and tell how many bytes are expected to come in.
        int bytes = arg.tb.dim[0] * arg.tb.dim[1] * arg.tb.dim[2] * arg.tb.dim[3] / 2 * 96;
        token = cuda::device::barrier_arrive_tx(*bar, 1, bytes);
      } else {
        // Other threads just arrive.
        token = bar->arrive();
      }
      // Wait for the data to have arrived. This also serves as a __syncthreads()
      bar->wait(std::move(token));
  #endif
#endif

      local_idx = target::thread_idx().x;
      while (local_idx < arg.tb.volume_4d_cb) {
        Coord<4> local_coord;
        auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, local_idx, 0, parity, thread_dim, local_coord);

        const int my_spinor_parity = nParity == 2 ? parity : 0;
        Vector out;
        applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, local_coord, parity, idx, thread_dim, active, src_idx, cache);

        int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
        if (xpay) {
          Vector x = arg.x[src_idx](xs, my_spinor_parity);
          out = x + arg.a * out;
        }
        arg.out[src_idx](xs, my_spinor_parity) = out;

        local_idx += target::block_dim().x;
      }
#endif
    }
  };

} // namespace quda
