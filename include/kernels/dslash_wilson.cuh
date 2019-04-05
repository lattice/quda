#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>

namespace quda
{

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_> struct WilsonArg : DslashArg<Float> {
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
    const F x;    /** input vector when doing xpay */
    const G U;    /** the gauge field */
    const real a; /** xpay scale facotor - can be -kappa or -kappa^2 */

    WilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
        DslashArg<Float>(in, U, parity, dagger, a != 0.0 ? true : false, 1, comm_override),
        out(out),
        in(in),
        U(U),
        x(x),
        a(a)
    {
      if (!out.isNative() || !x.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };

  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate
     @param[in] x_cb The checker-boarded site index (at present this is a 4-d index only)
     @param[in] s The fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilson(
      Vector &out, Arg &arg, int coord[nDim], int x_cb, int s, int parity, int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 2> HalfVector;
    typedef Matrix<complex<real>, nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (nDim == 5 ? (x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB<nDim>(coord, d, +1, arg.dc);
        const int gauge_idx = (nDim == 5 ? x_cb % arg.dc.volume_4d_cb : x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
              ghostFaceIndex<1, nDim>(coord, arg.dim, d, arg.nFace) :
              idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.in.Ghost(d, 1, ghost_idx + s * arg.dc.ghostFaceCB[d], their_spinor_parity);
          if (d == 3) in *= arg.t_proj_scale; // put this in the Ghost accessor and merge with any rescaling?

          out += (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
          Vector in = arg.in(fwd_idx + s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB<nDim>(coord, d, -1, arg.dc);
        const int gauge_idx = (nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
              ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace) :
              idx;

          const int gauge_ghost_idx = (nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.in.Ghost(d, 0, ghost_idx + s * arg.dc.ghostFaceCB[d], their_spinor_parity);
          if (d == 3) in *= arg.t_proj_scale;

          out += (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
          Vector in = arg.in(back_idx + s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
    } // nDim
  }

  // out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void wilson(Arg &arg, int idx, int s, int parity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 4> Vector;

    bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                          // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;
    applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(
        out, arg, coord, x_cb, s, parity, idx, thread_dim, active);

    int xs = x_cb + s * arg.dc.volume_4d_cb;
    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(xs, my_spinor_parity);
      out = x + arg.a * out;
    } else if (kernel_type != INTERIOR_KERNEL && active) {
      Vector x = arg.out(xs, my_spinor_parity);
      out = x + (xpay ? arg.a * out : out);
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
  }

  // CPU kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void wilsonCPU(Arg arg)
  {

    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        wilson<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 0, parity);
      } // 4-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void wilsonGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: wilson<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 0, 0); break;
    case 1: wilson<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 0, 1); break;
    }
  }

} // namespace quda
