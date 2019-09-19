#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool asymmetric_>
  struct TwistedMassArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    static constexpr bool asymmetric = asymmetric_; /** whether we are applying the asymmetric operator or not */
    real a;          /** this is the scaling factor */
    real b;          /** this is the twist factor */
    real c;          /** dummy parameter to allow us to reuse applyWilsonTM for non-degenerate operator */
    real a_inv;      /** inverse scaling factor - used to allow early xpay inclusion */
    real b_inv;      /** inverse twist factor - used to allow early xpay inclusion */

    TwistedMassArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b, bool xpay,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
      a(a),
      b(dagger ? -b : b), // if dagger flip the twist
      c(0.0),
      a_inv(1.0 / (a * (1 + b * b))),
      b_inv(dagger ? b : -b)
    {
      // set parameters for twisting in the packing kernel
      if (dagger && !asymmetric) {
        DslashArg<Float, nDim>::twist_a = this->a;
        DslashArg<Float, nDim>::twist_b = this->b;
      }
    }
  };

  /**
     @brief Applies the off-diagonal part of the Wilson operator
     premultiplied by twist rotation - this is required for applying
     the symmetric preconditioned twisted-mass dagger operator.

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate
     @param[in] x_cb The checker-boarded site index
     @param[in] s Fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <int nParity, bool dagger, int twist, KernelType kernel_type, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilsonTM(Vector &out, Arg &arg, int coord[Arg::nDim], int x_cb, int s,
                                                int parity, int idx, int thread_dim, bool &active)
  {
    static_assert(twist == 1 || twist == 2, "twist template must equal 1 or 2"); // ensure singlet or doublet
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

#pragma unroll
    for (int d = 0; d < Arg::nDim; d++) { // loop over dimension
      {                              // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        constexpr int proj_dir = dagger ? +1 : -1;
        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dim, d, arg.nFace) :
            idx;

          Link U = arg.U(d, x_cb, parity);
          HalfVector in = arg.in.Ghost(d, 1, ghost_idx + s * arg.dc.ghostFaceCB[d], their_spinor_parity);
          if (d == 3) in *= arg.t_proj_scale; // put this in the Ghost accessor and merge with any rescaling?

          out += (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, x_cb, parity);
          Vector in;
          if (twist == 1) {
            in = arg.in(fwd_idx + s * arg.dc.volume_4d_cb, their_spinor_parity);
            in = arg.a * (in + arg.b * in.igamma(4)); // apply A^{-1} to in
          } else {                                    // twisted doublet
            Vector in0 = arg.in(fwd_idx + 0 * arg.dc.volume_4d_cb, their_spinor_parity);
            Vector in1 = arg.in(fwd_idx + 1 * arg.dc.volume_4d_cb, their_spinor_parity);
            if (s == 0)
              in = arg.a * (in0 + arg.b * in0.igamma(4) + arg.c * in1);
            else
              in = arg.a * (in1 - arg.b * in1.igamma(4) + arg.c * in0);
          }

          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = back_idx;
        constexpr int proj_dir = dagger ? -1 : +1;
        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dim, d, arg.nFace) :
            idx;

          Link U = arg.U.Ghost(d, ghost_idx, 1 - parity);
          HalfVector in = arg.in.Ghost(d, 0, ghost_idx + s * arg.dc.ghostFaceCB[d], their_spinor_parity);
          if (d == 3) in *= arg.t_proj_scale;

          out += (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - parity);
          Vector in;
          if (twist == 1) {
            in = arg.in(back_idx + s * arg.dc.volume_4d_cb, their_spinor_parity);
            in = arg.a * (in + arg.b * in.igamma(4)); // apply A^{-1} to in
          } else {                                    // twisted doublet
            Vector in0 = arg.in(back_idx + 0 * arg.dc.volume_4d_cb, their_spinor_parity);
            Vector in1 = arg.in(back_idx + 1 * arg.dc.volume_4d_cb, their_spinor_parity);
            if (s == 0)
              in = arg.a * (in0 + arg.b * in0.igamma(4) + arg.c * in1);
            else
              in = arg.a * (in1 - arg.b * in1.igamma(4) + arg.c * in0);
          }

          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
    } // nDim
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct twistedMassPreconditioned : dslash_default {

    Arg &arg;
    constexpr twistedMassPreconditioned(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation
    constexpr int twist_pack() const { return (!Arg::asymmetric && dagger) ? 1 : 0; }

    /**
       @brief Apply the preconditioned twisted-mass dslash
       - no xpay: out(x) = M*in = a*(1+i*b*gamma_5)D * in
       - with xpay:  out(x) = M*in = x + a*(1+i*b*gamma_5)D * in
    */
    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      int coord[Arg::nDim];
      int x_cb = getCoords<QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;

      Vector out;

      if (!dagger || Arg::asymmetric) // defined in dslash_wilson.cuh
        applyWilson<nParity, dagger, kernel_type>(out, arg, coord, x_cb, 0, parity, idx, thread_dim, active);
      else // special dslash for symmetric dagger
        applyWilsonTM<nParity, dagger, 1, kernel_type>(out, arg, coord, x_cb, 0, parity, idx, thread_dim, active);

      if (xpay && kernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(x_cb, my_spinor_parity);
        if (!dagger || Arg::asymmetric) {
          out += arg.a_inv * (x + arg.b_inv * x.igamma(4)); // apply inverse twist which is undone below
        } else {
          out += x; // just directly add since twist already applied in the dslash
        }
      } else if (kernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out(x_cb, my_spinor_parity);
        out += x;
      }

      if (isComplete<kernel_type>(arg, coord) && active) {
        if (!dagger || Arg::asymmetric) out = arg.a * (out + arg.b * out.igamma(4)); // apply A^{-1} to D*in
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
    }

  };

} // namespace quda
