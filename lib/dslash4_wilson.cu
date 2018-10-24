#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>

/**
   This is the basic gauged Wilson operator

   TODO
   - gauge fix support
   - fused halo kernel
   - commDim vs ghostDim
   - host dslash halo
   - all the policies
   - ghost texture support in accessors

*/

namespace quda {

#include <dslash_index.cuh> // FIXME - remove
#include <dslash_events.cuh>
#include <dslash_policy.cuh>

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct WilsonArg : DslashArg<Float> {
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float,4,nColor,spin_project,spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float,reconstruct,18,QUDA_STAGGERED_PHASE_NO,gauge_direct_load,ghost>::type G;

    F out;                // output vector field
    const F in;           // input vector field
    const F x;            // input vector when doing xpay
    const G U;            // the gauge field

    WilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
              double kappa, const ColorSpinorField &x, int parity, bool dagger)
      : DslashArg<Float>(in, U, kappa, parity, dagger), out(out), in(in), U(U), x(x)
    {
      if (!out.isNative() || !x.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());

      // FIXME: need to reset ghost pointers since Ghost() returns ghost_buf but this is only presently set with exchangeGhost
      void *ghost[8];
      for (int dim=0; dim<4; dim++) {
        for (int dir=0; dir<2; dir++) {
          // really pointing to from_face_dim_dir_d[bufferIndex][d][dir] / from_face_dim_dir_h[bufferIndex][d][dir]
          ghost[2*dim+dir] = (Float*)((char*)in.Ghost2() + in.GhostOffset(dim,dir)*in.GhostPrecision());
        }
      }
      this->in.resetGhost(in, ghost);
    }
  };


  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in] U The gauge field
     @param[in] kappa Kappa value
     @param[in] in The input field
     @param[in] parity The site parity
     @param[in] x_cb The checker-boarded site index
  */
  template <typename Float, int nDim, int nColor, bool dagger, KernelType kernel_type, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilson(Vector &out, Arg &arg, int coord[nDim], int x_cb, int parity, int nParity) {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,2> HalfVector;
    typedef Matrix<complex<real>,nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1-parity : 0;

#pragma unroll
    for (int d = 0; d<nDim; d++) // loop over dimension
      {
        { // Forward gather - compute fwd offset for vector fetch
          const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
          constexpr int proj_dir = dagger ? +1 : -1;
          const bool ghost = ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) );

          if ( doHalo<kernel_type>(d) && ghost) {
            const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

            Link U = arg.U(d, x_cb, parity);
            HalfVector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
            if (d == 3) in *= arg.t_proj_scale; // put this in the Ghost accessor and merge with any rescaling?

            out += (U * in).reconstruct(d, proj_dir);
          } else if ( doBulk<kernel_type>() && !ghost ) {

            Link U = arg.U(d, x_cb, parity);
            Vector in = arg.in(fwd_idx, their_spinor_parity);

            out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }

        { // Backward gather - compute back offset for spinor and gauge fetch
          const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
          const int gauge_idx = back_idx;
          constexpr int proj_dir = dagger ? -1 : +1;
          const bool ghost = ( arg.commDim[d] && (coord[d] - arg.nFace < 0) );

          if ( doHalo<kernel_type>(d) && ghost) {
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

            Link U = arg.U.Ghost(d, ghost_idx, 1-parity);
            HalfVector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
            if (d == 3) in *= arg.t_proj_scale;

            out += (conj(U) * in).reconstruct(d, proj_dir);
          } else if ( doBulk<kernel_type>() && !ghost ) {

            Link U = arg.U(d, gauge_idx, 1-parity);
            Vector in = arg.in(back_idx, their_spinor_parity);

            out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }
      } //nDim

  }

  //out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void wilson(Arg &arg, int idx, int parity, int nParity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,4> Vector;

    int x_cb;
    int coord[nDim];
    if (kernel_type == INTERIOR_KERNEL) {
      x_cb = idx;
      getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);
    } else {
      // FIXME not for fused halo
      const int face_volume = arg.threads >> 1;
      const int face_num = idx >= face_volume;
      const int face_idx = idx - face_num*face_volume;
      int X;
      coordsFromFaceIndex<4,QUDA_4D_PC,kernel_type,1>(X, x_cb, coord, face_idx, face_num, arg);
    }

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;

    applyWilson<Float,nDim,nColor,dagger,kernel_type>(out, arg, coord, x_cb, parity, nParity);

    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      out = x + arg.kappa * out;
    } else if (kernel_type != INTERIOR_KERNEL) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x + (xpay ? arg.kappa * out : out);
    }

    arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void wilsonCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, parity, arg.nParity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void wilsonGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from y thread index else use arg setting
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;

#if 0
    wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, parity, arg.nParity);
#else
    switch(arg.nParity) { // better code if parity is explicit
    case 1:
      switch(parity) {
      case 0: wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, 0, 1); break;
      case 1: wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, 1, 1); break;
      }
      break;
    case 2:
      switch(parity) {
      case 0: wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, 0, 2); break;
      case 1: wilson<Float,nDim,nColor,dagger,xpay,kernel_type>(arg, x_cb, 1, 2); break;
      }
      break;
    }
#endif
  }

  template <typename Float, int nDim, int nColor, typename Arg>
  class Wilson : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    using Dslash<Float>::launch;

  public:

    Wilson(Arg &arg, const ColorSpinorField &meta)
      : Dslash<Float>(arg, meta), arg(arg), meta(meta)
    {  }

    virtual ~Wilson() { }

    template <bool dagger, bool xpay>
    inline void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        switch (arg.kernel_type) {
        case   INTERIOR_KERNEL: wilsonCPU<Float,nDim,nColor,dagger,xpay,INTERIOR_KERNEL  >(arg); break;
        case EXTERIOR_KERNEL_X: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_X>(arg); break;
        case EXTERIOR_KERNEL_Y: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Y>(arg); break;
        case EXTERIOR_KERNEL_Z: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Z>(arg); break;
        case EXTERIOR_KERNEL_T: wilsonCPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_T>(arg); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      } else {
        switch(arg.kernel_type) {
        case INTERIOR_KERNEL:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,INTERIOR_KERNEL,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_X:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_X,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Y:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Y,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_Z:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_Z,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_T:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_T,Arg>, tp, arg, stream); break;
        case EXTERIOR_KERNEL_ALL:
          launch(wilsonGPU<Float,nDim,nColor,dagger,xpay,EXTERIOR_KERNEL_ALL,Arg>, tp, arg, stream); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      }
    }

    void apply(const cudaStream_t &stream) {
      arg.t_proj_scale = getKernelPackT() ? 1.0 : 2.0;
      if (arg.xpay) arg.dagger ? apply<true, true>(stream) : apply<false, true>(stream);
      else          arg.dagger ? apply<true,false>(stream) : apply<false,false>(stream);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger)
  {
    constexpr int nDim = 4;
    WilsonArg<Float,nColor,recon> arg(out, in, U, kappa, x, parity, dagger);
    Wilson<Float,nDim,nColor,WilsonArg<Float,nColor,recon> > wilson(arg, in);

    TimeProfile profile("dummy");
    DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)), in.VolumeCB(), in.GhostFace(), profile);
    policy.apply(0);
    //wilson.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity, dagger);
#if 0
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity, dagger);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                     double kappa, const ColorSpinorField &x, int parity, bool dagger)
  {
    if (in.Ncolor() == 3) {
      ApplyWilson<Float,3>(out, in, U, kappa, x, parity, dagger);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  //Apply the Wilson operator
  //out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger, Dslash4Type type)
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilson<double>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilson<float>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilson<short>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyWilson<char>(out, in, U, kappa, x, parity, dagger);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
  }


} // namespace quda
