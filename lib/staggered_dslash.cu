#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <stencil.h>
#include <color_spinor.h>
#include <worker.h>
#include <tune_quda.h>

/**
   This is a staggered Dirac operator
*/

namespace quda {

  /**
     @brief Parameter structure for driving the Staggered Dslash operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_u_, QudaReconstructType reconstruct_l_, bool improved_>
  struct StaggeredArg : DslashArg<Float> {
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F= typename colorspinor_mapper<Float,nSpin,nColor,spin_project,spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    using GU = typename gauge_mapper<Float,reconstruct_u,18,QUDA_STAGGERED_PHASE_MILC,gauge_direct_load,ghost>::type;
    using GL = typename gauge_mapper<Float,reconstruct_l,18,QUDA_STAGGERED_PHASE_NO,gauge_direct_load,ghost>::type;

    F out;                // output vector field
    const F in;           // input vector field
    const F x;            // input vector when doing xpay
    const GU U;            // the gauge field
    const GL L;            // the long gauge field

    const Float a;
    static constexpr bool improved = improved_;
    StaggeredArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
        Float a, const ColorSpinorField &x, int parity, bool dagger,  const int *comm_override)
      : DslashArg<Float>(in, U, 0.0, parity, dagger, a == 0.0 ? false : true, comm_override), out(out), in(in), U(U), L(L), x(x), a(a) 
    {
      if (!out.isNative() || !x.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };




  /**
     Applies the off-diagonal part of the Laplace operator

     @param[out] out The out result field
     @param[in] U The gauge field
     @param[in] kappa Kappa value
     @param[in] in The input field
     @param[in] parity The site parity
     @param[in] x_cb The checkerboarded site index
   */
  template <typename Float, int nDim, int nColor, typename Vector, typename Arg>
  __device__ __host__ inline void ApplyDslashStaggered(Vector &out, Arg &arg, int x_cb, int parity) {
    typedef Matrix<complex<Float>,nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;
  // const int their_spinor_parity =  1-parity ;


    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;


#pragma unroll
    for (int d = 0; d<4; d++) {// loop over dimension{
      //         Float sign = (d == 0 && ((coords[3] ) & 1) != 0) ||
      // ( d == 1 && ((coords[0] + coords[3] ) & 1) != 0) ||
      // ( d == 2 && ((coords[0] + coords[1] + coords[3] ) & 1) != 0) ? -1.0 : 1.0;
      //Forward gather - compute fwd offset for vector fetch
      {
      const int fwd_idx = linkIndexP1(coord, arg.dim, d);

      if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
  const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
  const Link U = arg.U(d, x_cb, parity);
  const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
  out += U * in;
  } else {
  const Link U = arg.U(d, x_cb, parity);
  // U[6] *= sign;
  // U[7] *= sign;
  // U[8] *= sign;
  const Vector in = arg.in(fwd_idx, their_spinor_parity);
          out += U * in;
        }
      }
      if(arg.improved){
        const int fwd3_idx = linkIndexP3(coord, arg.dim, d);
        if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
          const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
        out += L * in;
        } else {
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in(fwd3_idx, their_spinor_parity);
          out += L * in;
        }  
      }
#if 1
      {
      //Backward gather - compute back offset for spinor and gauge fetch
      const int back_idx = linkIndexM1(coord, arg.dim, d);
      const int gauge_idx = back_idx;

      if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
  const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
  const Link U = arg.U.Ghost(d, ghost_idx, 1-parity);
  const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          out -= conj(U) * in;
      } else {
  const Link U = arg.U(d, gauge_idx, 1-parity);
  const Vector in = arg.in(back_idx, their_spinor_parity);
          out -= conj(U) * in;
        }
      }
      if(arg.improved){
        //Backward gather - compute back offset for spinor and gauge fetch
        const int back3_idx = linkIndexM3(coord, arg.dim, d);
        const int gauge_idx = back3_idx;

        if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
          const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
          const Link L = arg.L.Ghost(d, ghost_idx, 1-parity);
          const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          out -= conj(L) * in;
        } else {
          const Link L = arg.L(d, gauge_idx, 1-parity);
          const Vector in = arg.in(back3_idx, their_spinor_parity);
          out -= conj(L) * in;
        }
      }
      #endif
    } //nDim

  }

  //out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, typename Arg>
  __device__ __host__ inline void dslashStaggered(Arg &arg, int x_cb, int parity)
  {
    using Vector = ColorSpinor<Float,nColor,1>;
    Vector out;

    ApplyDslashStaggered<Float,nDim,nColor>(out, arg, x_cb, parity);

    if (arg.xpay) {
      Vector x = arg.x(x_cb, parity);
      out = arg.a * x -out ;
    }
    if (arg.dagger){
      out = Float(-1)*out;
    }
    arg.out(x_cb, arg.nParity == 2 ? parity : 0) = out;
  }

  // CPU kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  void dslashStaggeredCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
        dslashStaggered<Float,nDim,nColor>(arg, x_cb, parity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  __global__ void  dslashStaggeredGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    // for full fields set parity from y thread index else use arg setting
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    dslashStaggered<Float,nDim,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nDim, int nColor, typename Arg>
  class DslashStaggered : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return (2*nDim*(8*nColor*nColor)-2*nColor + (arg.xpay ? 2*2*nColor : 0) )*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + 2*nDim*arg.in.Bytes() + arg.nParity*2*nDim*arg.U.Bytes()*meta.VolumeCB() +
  (arg.xpay ? arg.x.Bytes() : 0);
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    DslashStaggered(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux,",comm=");
      strcat(aux,comm);
#endif
      if (arg.xpay) strcat(aux,",xpay");
    }
    virtual ~DslashStaggered() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
  dslashStaggeredCPU<Float,nDim,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  dslashStaggeredGPU<Float,nDim,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


// //MW TODO 20181219
#if 0
  template <typename Float, int nColor, QudaReconstructType recon_u, QudaReconstructType recon_l, bool improved>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger, bool improved,
                            const int *comm_override, TimeProfile &profile) 
  {
    constexpr int nDim = 4;
    StaggeredArg<Float,nColor,recon_u,recon_l,improved> arg(out, in, U, kappa, x, parity, dagger, comm_override);
    Staggered<Float,nDim,nColor, decltype(arg) > > staggered(arg, out, in);

    DslashPolicyTune<decltype(staggered)> policy(staggered, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }
#else
  template <typename Float, int nColor, QudaReconstructType recon_u, QudaReconstructType recon_l, bool improved>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile) 
  {
    constexpr int nDim = 4;

      // DslashArg<Float,nColor,recon_u, recon_l, improved, true, false> arg(out, in, U, L, a, x, parity);
      StaggeredArg<Float,nColor,recon_u,recon_l,improved> arg(out, in, U, L, a, x, parity, dagger, comm_override); 
      DslashStaggered<Float,nDim,nColor,decltype(arg) > dslash(arg, in);
      dslash.apply(0);
    
  }
#endif

  // template on the gauge reconstruction
  template <typename Float, int nColor>
    void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger, bool improved,
                            const int *comm_override, TimeProfile &profile) 
  {
    if(improved){
    if (L.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO, true>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
    } else if (L.Reconstruct()== QUDA_RECONSTRUCT_13) {
      ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, true >(out, in, U, L, a, x, parity, dagger, comm_override, profile);
    } else if (L.Reconstruct()== QUDA_RECONSTRUCT_9) {
      ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_9, true>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }
  else{
      if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO,false>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_NO,false>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      errorQuda("Recon 8 not implemented for standard staggered.\n");
       // ApplyDslashStaggered<Float,nColor,QUDA_RECONSTRUCT_8, QUDA_RECONSTRUCT_NO, false>(out, in, U, L, a, x, parity);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }  
  }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger, bool improved,
                            const int *comm_override, TimeProfile &profile) 
  {
    if (in.Ncolor() == 3) {
      ApplyDslashStaggered<Float,3>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }



  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger, bool improved,
                            const int *comm_override, TimeProfile &profile)      
  {
    
#ifdef GPU_STAGGERED_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all precisions match
    checkPrecision(out, in, U, L);

    // check all locations match
    checkLocation(out, in, U, L);

    // const int nFace = 1;
    // in.exchangeGhost((QudaParity)(1-parity), nFace, 0); // last parameter is dummy

    // if (dslash::aux_worker) dslash::aux_worker->apply(0);
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyDslashStaggered<double>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyDslashStaggered<float>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } 
    //else if (U.Precision() == QUDA_HALF_PRECISION) {
    //   ApplyDslashStaggered<short, true>(out, in, U, L,  a, x, parity);
    // } 
    else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }  
#else
    errorQuda("Staggered dslash has not been built");
#endif
}

} // namespace quda
