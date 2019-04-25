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
     @brief Parameter structure for driving the covariatnt derivative operator
  */
  template <typename Float, int nColor, QudaReconstructType reconstruct_> struct CovDevArg : DslashArg<Float> {
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out;      /** output vector field */
    const F in; /** input vector field */
    const G U;  /** the gauge field */
    int mu;     /** The direction in which to apply the derivative */

    CovDevArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int mu, int parity, bool dagger,
              const int *comm_override) :

      DslashArg<Float>(in, U, parity, dagger, false, 1, spin_project, comm_override),
      out(out),
      in(in),
      U(U),
      mu(mu)
    {
      if (!out.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor(in)=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };

  /**
     Applies the off-diagonal part of the covariant derivative operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] U The gauge field
     @param[in] coord Site coordinate
     @param[in] x_cb The checker-boarded site index. This is a 4-d index only
     @param[in] parity The site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)

  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, int mu,
            typename Arg, typename Vector>
  __device__ __host__ inline void applyCovDev(Vector &out, Arg &arg, int coord[nDim], int x_cb, int parity, int idx,
                                              int thread_dim, bool &active)
  {

    typedef typename mapper<Float>::type real;
    typedef Matrix<complex<real>, nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    const int d = mu % 4;

    if (mu < 4) { // Forward gather - compute fwd offset for vector fetch

      const int fwd_idx = getNeighborIndexCB<nDim>(coord, d, +1, arg.dc);
      const bool ghost = (coord[d] + 1 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

      const Link U = arg.U(d, x_cb, parity);

      if (doHalo<kernel_type>(d) && ghost) {

        const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
        const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);

        out += U * in;

      } else if (doBulk<kernel_type>() && !ghost) {

        const Vector in = arg.in(fwd_idx, their_spinor_parity);
        out += U * in;
      }

    } else { // Backward gather - compute back offset for spinor and gauge fetch

      const int back_idx = getNeighborIndexCB<nDim>(coord, d, -1, arg.dc);
      const int gauge_idx = back_idx;

      const bool ghost = (coord[d] - 1 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

      if (doHalo<kernel_type>(d) && ghost) {

        const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
        const Link U = arg.U.Ghost(d, ghost_idx, 1 - parity);
        const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);

        out += conj(U) * in;
      } else if (doBulk<kernel_type>() && !ghost) {

        const Link U = arg.U(d, gauge_idx, 1 - parity);
        const Vector in = arg.in(back_idx, their_spinor_parity);

        out += conj(U) * in;
      }
    } // Forward/backward derivative
  }

  // out(x) = M*in
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void covDev(Arg &arg, int idx, int parity)
  {

    using real = typename mapper<Float>::type;
    using Vector = ColorSpinor<real, nColor, 4>;

    // is thread active (non-trival for fused kernel only)
    bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true;

    // which dimension is thread working on (fused kernel only)
    int thread_dim;

    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type, Arg>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;

    switch (arg.mu) { // ensure that mu is known to compiler for indexing in applyCovDev (avoid register spillage)
    case 0:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 0>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 1:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 1>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 2:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 2>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 3:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 3>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 4:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 4>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 5:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 5>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 6:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 6>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    case 7:
      applyCovDev<Float, nDim, nColor, nParity, dagger, kernel_type, 7>(out, arg, coord, x_cb, parity, idx, thread_dim,
                                                                        active);
      break;
    }

    if (kernel_type != INTERIOR_KERNEL) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out += x;
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // GPU Kernel for applying the covariant derivative operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void covDevGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: covDev<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 0); break;
    case 1: covDev<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 1); break;
    }
  }
} // namespace quda
