#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel

namespace quda
{

  /**
     @brief Parameter structure for driving the covariatnt derivative operator
  */
  template <typename Float, int nDim, QudaReconstructType reconstruct_>
  struct StaggeredQSmearArg : DslashArg<Float, nDim> {//DslashArg has kernel_type, see dslash_helper.cuh
    static constexpr int nColor = 3;
    static constexpr int nSpin  = 1;
    static constexpr bool spin_project       = false;
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
    const F x;    /** input vector field for xpay*/
    const G U;    /** the gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */
    const real b; /** used by Wuppetal smearing kernel */
    int dir;      /** The direction from which to omit the derivative */

    StaggeredQSmearArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double a, double b,
               const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :

      DslashArg<Float, nDim>(in, U, parity, dagger, a != 0.0 ? true : false, 1, false, comm_override),
      out(out),
      in(in),
      in_pack(in),
      x(x),
      U(U),
      a(a),
      b(b),
      dir(dir)
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in, x);        // check all orders match
      checkPrecision(out, in, x, U); // check all precisions match
      checkLocation(out, in, x, U);  // check all locations match
      if (!in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor(in)=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
      if (dir < 3 || dir > 4) errorQuda("Unsupported laplace direction %d (must be 3 or 4)", dir);
    }
  };

  /**
     Applies the off-diagonal part of the covariant derivative operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] U The gauge field
     @param[in] coord Site coordinate struct
     @param[in] parity The site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)

  */
  template <int nParity, bool dagger, KernelType kernel_type, int dir, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyStaggeredQSmear(Vector &out, Arg &arg, Coord &coord, int parity,
                                               int, int thread_dim, bool &active)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef Matrix<complex<real>, Arg::nColor> Link;

#pragma unroll
    for (int d = 0; d < Arg::nDim; d++) { // loop over dimension
      if (d != dir) {
        {
          // Forward gather - compute fwd offset for vector fetch
          const bool ghost = (coord[d] + 2 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);//1=>2
	  
          if (doHalo<kernel_type>(d) && ghost) {//?
#if 0	    
            const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);//?
            const Link U = arg.U(d, coord.x_cb, parity);//?
            const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);//?

            out += U * in;
#endif            
          } else if (doBulk<kernel_type>() && !ghost) {//doBulk

            const int _2hop_fwd_idx = linkIndexP2(coord, arg.dim, d);
            const int _1hop_fwd_idx = linkIndexP1(coord, arg.dim, d);            
            const Link U_1hop_link  = arg.U(d, _1hop_fwd_idx, 1-parity);

            const Vector in_2hop = arg.in(_2hop_fwd_idx, their_spinor_parity);
            const Vector tmp     = U_1hop_link * in;           
            const Link U         = arg.U(d, coord.x_cb, parity);            
            out += U * tmp;
          }
        }
        {
          // Backward gather - compute back offset for spinor and gauge fetch

          const int _1hop_back_idx = linkIndexM1(coord, arg.dim, d);
          const int _2hop_back_idx = linkIndexM2(coord, arg.dim, d);          
          const int _1hop_gauge_idx= _1hop_back_idx;
          const int _2hop_gauge_idx= _2hop_back_idx;          

          const bool ghost = (coord[d] - 2 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);//1=>2

          if (doHalo<kernel_type>(d) && ghost) {
#if 0
            // const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

            const Link U = arg.U.Ghost(d, ghost_idx, 1 - parity);
            const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
	    
            out += conj(U) * in;
#endif            
          } else if (doBulk<kernel_type>() && !ghost) {//?

            const Link U_2hop_link = arg.U(d, _2hop_gauge_idx, parity);
            const Vector in_2hop   = arg.in(_2hop_back_idx, parity);
            const Vector tmp       = conj(U_1hop_link) * in;               
                    
            const Link U_1hop_link = arg.U(d, _1hop_gauge_idx, 1 - parity);
            out += conj(U_1hop_link) * tmp;
          }
        }
      }
    }
  }
  
  // out(x) = M*in
  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct staggered_qsmear : dslash_default {

    Arg &arg;
    constexpr staggered_qsmear(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // is thread active (non-trival for fused kernel only)
      bool active = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true;

      // which dimension is thread working on (fused kernel only)
      int thread_dim;

      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      // We instantiate two kernel types:
      // case 4 is an operator in all x,y,z,t dimensions
      // case 3 is a spatial operator only, the t dimension is omitted.
      switch (arg.dir) {
      case 3: applyStaggeredQSmear<nParity, dagger, mykernel_type, 3>(out, arg, coord, parity, idx, thread_dim, active); break;
      case 4:
      default:
        applyStaggeredQSmear<nParity, dagger, mykernel_type, -1>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      }

      if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(coord.x_cb, my_spinor_parity);
        out = arg.a * out + arg.b * x;
      } else if (mykernel_type != INTERIOR_KERNEL) {
        Vector x = arg.out(coord.x_cb, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
