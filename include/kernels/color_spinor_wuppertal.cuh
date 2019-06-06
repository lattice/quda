#ifndef USE_LEGACY_DSLASH

#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>

namespace quda {

  template <typename Float, int nSpin, int nColor, QudaReconstructType reconstruct_> struct WuppertalSmearingArg : DslashArg<Float> {

    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out;                // output vector field
    const F in;           // input vector field
    const G U;            // the gauge field
    const real a;        // a parameter
    const real b;        // b parameter

    WuppertalSmearingArg(ColorSpinorField &out, const ColorSpinorField &in, int parity, const GaugeField &U,
                       Float a, Float b, const int *comm_override)
      :
        DslashArg<Float>(in, U, parity, false, true, 1, false, comm_override),
        out(out),
        in(in),
        U(U), a(a), b(b)
    {
      if (in.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };


  /**
     Computes out = sum_mu U_mu(x)in(x+d) + U^\dagger_mu(x-d)in(x-d)
     @param[out] out The out result field
     @param[in] U The gauge field
     @param[in] in The input field
     @param[in] x_cb The checkerboarded site index
     @param[in] parity The site parity
  */
  template <typename Float, int nDim, int nColor, int nParity, KernelType kernel_type,
            typename Arg, typename Vector>
  __device__ __host__ inline void computeNeighborSum(Vector &out, Arg &arg, int coord[nDim], int x_cb, int parity, int thread_dim, bool &active) {

    using real = typename mapper<Float>::type;
    using Link = typedef Matrix<complex<real>, nColor>;
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

#pragma unroll
    for (int dir=0; dir<3; dir++) { // loop over spatial directions

      // Forward gather - compute fwd offset for vector fetch
      const bool ghost = (coord[dir] + 1 >= arg.dim[dir]) && isActive<kernel_type>(active, thread_dim, dir, coord, arg);

      if ( doHalo<kernel_type>(d) && ghost ) {

        const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, dir, arg.nFace);
        const Link U = arg.U(dir, x_cb, parity);
        const Vector in = arg.in.Ghost(dir, 1, ghost_idx, their_spinor_parity);

        out += U * in;

      } else if (doBulk<kernel_type>() && !ghost){
        //Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = linkIndexP1(coord, arg.dim, dir);
        const Link U = arg.U(dir, x_cb, parity);
        const Vector in = arg.in(fwd_idx, their_spinor_parity);

        out += U * in;
      }

      // Backward gather - compute back offset for spinor and gauge fetch
      const int back_idx = linkIndexM1(coord, arg.dim, dir);
      const int gauge_idx = back_idx;
      const bool ghost = (coord[dir] - 1 < 0) && isActive<kernel_type>(active, thread_dim, dir, coord, arg);

      if (doHalo<kernel_type>(dir) && ghost) {

        const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, dir, arg.nFace);

        const Link U = arg.U.Ghost(dir, ghost_idx, 1 - parity);
        const Vector in = arg.in.Ghost(dir, 0, ghost_idx, their_spinor_parity);

        out += conj(U) * in;
      } else if (doBulk<kernel_type>() && !ghost) {

        const Link U = arg.U(dir, gauge_idx, 1 - parity);
        const Vector in = arg.in(back_idx, their_spinor_parity);

        out += conj(U) * in;
      }
    }
  }

  //out(x) = A in(x) + B computeNeighborSum(out, x)
  template <typename Float, int nDim, int nSpin, int nColor, int nParity, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void computeWuppertalStep(Arg &arg, int idx, int parity)
  {
    using real = typename mapper<Float>::type;
    using Vector = ColorSpinor<real, nColor, nSpin>;
    Vector out;

    // is thread active (non-trival for fused kernel only)
    bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true;

    // which dimension is thread working on (fused kernel only)
    int thread_dim;

    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type, Arg>(coord, arg, idx, parity, thread_dim);

    computeNeighborSum<Float, nDim, nColor, nParity, kernel_type>(out, arg, coord, x_cb, parity, thread_dim, active);

    Vector in;
    arg.in.load((real*)in.data, x_cb, parity);
    out = arg.a*in + arg.b*out;

    arg.out(x_cb, parity) = out;
  }


  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nSpin, int nColor, int nParity, KernelType kernel_type, typename Arg>
  __global__ void wuppertalStepGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;
    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: computeWuppertalStep<Float, nDim, nColor, nParity, kernel_type>(arg, x_cb, 0); break;
    case 1: computeWuppertalStep<Float, nDim, nColor, nParity, kernel_type>(arg, x_cb, 1); break;
    }
  }


} // namespace quda

#endif
