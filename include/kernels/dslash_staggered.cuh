#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>

namespace quda {
 /**
     @brief Parameter structure for driving the Staggered Dslash operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_u_, QudaReconstructType reconstruct_l_, bool improved_>
  struct StaggeredArg : DslashArg<Float> {
    typedef typename mapper<Float>::type real;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    static constexpr bool use_inphase = improved_ ? false : true;
    using GU = typename gauge_mapper<Float, reconstruct_u, 18, QUDA_STAGGERED_PHASE_MILC, gauge_direct_load, ghost, use_inphase>::type;
    using GL = typename gauge_mapper<Float, reconstruct_l, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, use_inphase>::type;

    F out;        /** output vector field */
    const F in;   /** input vector field */
    const F x;    /** input vector when doing xpay */
    const GU U;   /** the gauge field */
    const GL L;   /** the long gauge field */

    const real a; /** xpay scale factor */
    const real fat_link_max; // MWTODO: FIXME: This a workaround should probably be in the accessor ...
    const real tboundary;
    static constexpr bool improved = improved_;
    StaggeredArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a,
                 const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      DslashArg<Float>(in, U, parity, dagger, a == 0.0 ? false : true, improved_ ? 3 : 1, comm_override),
      out(out),
      in(in, improved_ ? 3 : 1),
      U(U),
      L(L),
      fat_link_max(improved_ ? U.Precision() == QUDA_HALF_PRECISION ? U.LinkMax() : 1.0 : 1.0),
      tboundary(U.TBoundary()),
      x(x),
      a(a)
    {
      if (!out.isNative() || !x.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };


 // MWTODO: THis should probably be placed in the correct header
  template <typename Float, typename I> __device__ __host__ inline int StaggeredPhase(const int coords[], const I X[], int d, int dir, Float tboundary)
  {
    Float sign; // = static_cast<Float>(1.0);
    // int coords[4];
    // getCoords(coords, x, X, parity);
    switch (d) {
    case 0: sign = (coords[3]) % 2 == 0 ? static_cast<Float>(1.0) : static_cast<Float>(-1.0); break;
    case 1: sign = (coords[3] + coords[0]) % 2 == 0 ? static_cast<Float>(1.0) : static_cast<Float>(-1.0); break;
    case 2: sign = (coords[3] + coords[1] + coords[0]) % 2 == 0 ? static_cast<Float>(1.0) : static_cast<Float>(-1.0); break;
    case 3: sign = (coords[3] + dir >= X[3] || coords[3] + dir < 0) ? tboundary : static_cast<Float>(1.0); break;
    default: sign = static_cast<Float>(1.0);
    }
    return sign;
  }

  /**
       Applies the off-diagonal part of the Staggered / Asqtad operator

       @param[out] out The out result field
       @param[in] U The gauge field
       @param[in] in The input field
       @param[in] parity The site parity
       @param[in] x_cb The checkerboarded site index
     */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg, typename Vector>
  __device__ __host__ inline void applyStaggered(Vector &out, Arg &arg, int coord[nDim], int x_cb, int parity, int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<Float>::type real;
    typedef Matrix<complex<real>, nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension
      
      // standard - forward direction
      {
        const bool ghost = (coord[d] + 1 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndexStaggered<1>(coord, arg.dim, d, 1);
          const Link U = arg.improved ? arg.U(d, x_cb, parity) : arg.U(d, x_cb, parity, StaggeredPhase<Float>(coord, arg.dim, d, +1, arg.tboundary));
          Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
          in *= arg.fat_link_max;
          out += (U * in);
        } else if (doBulk<kernel_type>() && !ghost) {
          const int fwd_idx = linkIndexP1(coord, arg.dim, d);
          const Link U = arg.improved ? arg.U(d, x_cb, parity) : arg.U(d, x_cb, parity, StaggeredPhase<Float>(coord, arg.dim, d, +1, arg.tboundary));
          Vector in = arg.in(fwd_idx, their_spinor_parity);
          in *= arg.fat_link_max;
          out += (U * in);
        }
      }

      // improved - forward direction
      if (arg.improved) {
        const bool ghost = (coord[d] + 3 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndexStaggered<1>(coord, arg.dim, d, arg.nFace);
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
          out += L * in;
        } else if (doBulk<kernel_type>() && !ghost) {
          const int fwd3_idx = linkIndexP3(coord, arg.dim, d);
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in(fwd3_idx, their_spinor_parity);
          out += L * in;
        }
      }

      {
        // Backward gather - compute back offset for spinor and gauge fetch
        const bool ghost = (coord[d] - 1 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx2 = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
          const int ghost_idx = arg.improved ? ghostFaceIndexStaggered<0>(coord, arg.dim, d, 3) : ghost_idx2;
          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const Link U = arg.improved ? arg.U.Ghost(d, ghost_idx2, 1 - parity)
                                      : arg.U.Ghost(d, ghost_idx2, 1 - parity, StaggeredPhase<Float>(coord, arg.dim, d, -1, arg.tboundary));
          Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          in *= arg.fat_link_max;
          out -= (conj(U) * in);
        } else if (doBulk<kernel_type>() && !ghost) {
          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const int gauge_idx = back_idx;
          const Link U
              = arg.improved ? arg.U(d, gauge_idx, 1 - parity) : arg.U(d, gauge_idx, 1 - parity, StaggeredPhase<Float>(coord, arg.dim, d, -1, arg.tboundary));
          Vector in = arg.in(back_idx, their_spinor_parity);
          in *= arg.fat_link_max;
          out -= (conj(U) * in);
        }
      }

      // improved - backward direction
      if (arg.improved) {
        const bool ghost = (coord[d] - 3 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if (doHalo<kernel_type>(d) && ghost) {
          // when updating replace arg.nFace with 1 here
          const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
          const Link L = arg.L.Ghost(d, ghost_idx, 1 - parity);
          const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          out -= conj(L) * in;
        } else if (doBulk<kernel_type>() && !ghost) {
          const int back3_idx = linkIndexM3(coord, arg.dim, d);
          const int gauge_idx = back3_idx;
          const Link L = arg.L(d, gauge_idx, 1 - parity);
          const Vector in = arg.in(back3_idx, their_spinor_parity);
          out -= conj(L) * in;
        }
      }
    } // nDim
  }

  // out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void staggered(Arg &arg, int idx, int parity)
  {
    using real = typename mapper<Float>::type;
    using Vector = ColorSpinor<real, nColor, 1>;

    bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                                  // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = arg.improved ? getCoords<nDim, QUDA_4D_PC, kernel_type, Arg, 3>(coord, arg, idx, parity, thread_dim)
                            : getCoords<nDim, QUDA_4D_PC, kernel_type, Arg, 1>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;

    Vector out;

    applyStaggered<Float, nDim, nColor, nParity, dagger, kernel_type>(out, arg, coord, x_cb, parity, idx, thread_dim, active);

    if (dagger) { out = real(-1) * out; }

    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      out = arg.a * x - out;
    } else if (kernel_type != INTERIOR_KERNEL) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x + (xpay ? real(-1) * out : out); // MWTODO: verify
    }
    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // GPU Kernel for applying the staggered operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void staggeredGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    	case 0: staggered<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 0); break;
    	case 1: staggered<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 1); break;
    }
  }
}
