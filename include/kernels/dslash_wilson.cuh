#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <kernels/spinor_reweight.cuh>
#include <shared_memory_cache_helper.h>

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

    // stencil direction thread parallelization
    static constexpr int dim_threads = 1;
    static constexpr int dir_threads = 2;
    static_assert(dim_threads == 1 || dim_threads == 2 || dim_threads == 4);
    static_assert(dir_threads == 1 || dir_threads == 2);

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, spin_project, spinor_direct_load, false>;

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

    WilsonArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
              const GaugeField &U, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
              const int *comm_override, double alpha0 = 0.0, int t0 = -1) :
      DslashArg<Float, nDim>(out, in, halo, U, x, parity, dagger, a != 0.0 ? true : false, spin_project, comm_override),
      halo_pack(halo),
      halo(halo),
      U(U),
      a(a),
      alpha0(alpha0),
      t0(t0),
      comm_coord_dim_3(comm_coord(3) * this->dc.X[3]),
      comm_dim_dim_3(comm_dim(3) * this->dc.X[3])
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
  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilson(Vector &out, const Arg &arg, Coord &coord, int parity, int idx,
                                              int thread_dim, bool &active, int src_idx, int dim_idx, int dir_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    const int t = arg.comm_coord_dim_3 + coord[3];
    const int nt = arg.comm_dim_dim_3;
    real fwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t + 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);
    real bwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t - 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);

#pragma unroll
    for (int d0 = 0; d0 < 4; d0 += Arg::dim_threads) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      int d = d0 + dim_idx;

      if (!(dir_idx % Arg::dir_threads)) { // Forward gather - compute fwd offset for vector fetch
        const real fwd_coeff = (d < 3) ? 1.0 : fwd_coeff_3;
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost = coord.inBoundary(d, 1) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) : idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 1, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += fwd_coeff * (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
          Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += fwd_coeff * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }

      if (!((1 - dir_idx) % Arg::dir_threads)) { // Backward gather - compute back offset for spinor and gauge fetch
        const real bwd_coeff = (d < 3) ? 1.0 : bwd_coeff_3;
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = coord.inBoundary(d, 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) : idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 0, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += bwd_coeff * (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
          Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += bwd_coeff * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
    } // nDim
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    constexpr wilson(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    // out(x) = M*in = (-D + m) * in(x-mu)
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity_dim_dir)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, 4>;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                          // which dimension is thread working on (fused kernel only)
      
      int dir_idx = parity_dim_dir % Arg::dir_threads;
      int parity_dim = parity_dim_dir / Arg::dir_threads;
      int dim_idx = Arg::dim_threads == 1 ? 0 : parity_dim % Arg::dim_threads;
      int parity = parity_dim / Arg::dim_threads;
      int dim_dir_idx = dim_idx * Arg::dir_threads + dir_idx;

      // for full fields set parity from z thread index else use arg setting
      if (nParity == 1) parity = arg.parity;

      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx, dim_idx, dir_idx);

      if constexpr (Arg::dim_threads > 1 || Arg::dir_threads > 1) {
        SharedMemoryCache<Vector> cache;
        if (dim_dir_idx > 0) cache.save(out);
        cache.sync();
        if (dim_dir_idx == 0) {
          out += cache.load_z(target::thread_idx().z + 1); // remainder of x dim
          for (int dim = 1; dim < Arg::dim_threads; dim++) {
            for (int dir = 0; dir < Arg::dir_threads; dir++) {
              out += cache.load_z(2 * dim + dir);
            }
          }
        }
      }

      // only thread 0 in dim/dir handles writing
      if ((Arg::dim_threads == 1 && Arg::dir_threads == 1) || dim_dir_idx == 0) {
        int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x = arg.x[src_idx](xs, my_spinor_parity);
          out = x + arg.a * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out[src_idx](xs, my_spinor_parity);
          out = x + (xpay ? arg.a * out : out);
        }

        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
      }
    }
  };

} // namespace quda
