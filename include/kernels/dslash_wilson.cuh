#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <shared_memory_cache_helper.cuh>

namespace quda
{

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_>
  struct WilsonArg : DslashArg<Float, nDim> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out;        /** output vector field */
    const F in;   /** input vector field */
    const F in_pack; /** input vector field used in packing to be able to independently resetGhost */
    const F x;    /** input vector when doing xpay */
    const G U;    /** the gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */

    WilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
              const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      DslashArg<Float, nDim>(in, U, parity, dagger, a != 0.0 ? true : false, 1, spin_project, comm_override),
      out(out),
      in(in),
      in_pack(in),
      x(x),
      U(U),
      a(a)
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in, x);        // check all orders match
      checkPrecision(out, in, x, U); // check all precisions match
      checkLocation(out, in, x, U);  // check all locations match
      if (!in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
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
  __device__ __host__ inline void applyWilson(Vector &out, const Arg &arg, Coord &coord, Coord &local_coord, int parity, int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    auto block = target::block_dim();
    SharedMemoryCache<Vector> cache({static_cast<unsigned int>(arg.tb.volume_4d_cb), block.y, block.z});

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.in.Ghost(d, 1, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          out += (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
          bool out_of_block = (local_coord[d] + 1) >= arg.tb.dim[d] && arg.tb.dim[d] < arg.dim[d];
          Vector in;
          if (out_of_block) {
            in = arg.in(fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
          } else {
            int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, +1, arg.tb);
            in = cache.load_x(local_fwd_idx);
          }
          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);

        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.in.Ghost(d, 0, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          out += (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
          bool out_of_block = (local_coord[d] - 1) < 0 && arg.tb.dim[d] < arg.dim[d];
          Vector in;
          if (out_of_block) {
            in = arg.in(back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
          } else {
            int local_fwd_idx = thread_blocking_get_neighbor_index_cb(local_coord, d, -1, arg.tb);
            in = cache.load_x(local_fwd_idx);
          }
          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);

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
    __device__ __host__ __forceinline__ void operator()(int idx, int, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active = true;
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)

      // Load all interior color spinor fields
      auto block = target::block_dim();
      SharedMemoryCache<Vector> cache({static_cast<unsigned int>(arg.tb.volume_4d_cb), block.y, block.z});
      int local_idx = target::thread_idx().x;

      while (local_idx < arg.tb.volume_4d_cb) {
        const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;
        Coord<4> local_coord;
        auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, local_idx, 0, parity, thread_dim, local_coord);
        cache.save_x(arg.in(coord.x_cb + coord.s * arg.dc.volume_4d_cb, their_spinor_parity), local_idx);
        local_idx += target::block_dim().x;
      }
      cache.sync();

      local_idx = target::thread_idx().x;
      while (local_idx < arg.tb.volume_4d_cb) {
        Coord<4> local_coord;
        auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, local_idx, 0, parity, thread_dim, local_coord);

        const int my_spinor_parity = nParity == 2 ? parity : 0;
        Vector out;
        applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, local_coord, parity, idx, thread_dim, active);

        int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
        if (xpay) {
          Vector x = arg.x(xs, my_spinor_parity);
          out = x + arg.a * out;
        }
        arg.out(xs, my_spinor_parity) = out;

        local_idx += target::block_dim().x;
      }
    }
  };

} // namespace quda
