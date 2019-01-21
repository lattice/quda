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

namespace quda {
#include <dslash_events.cuh>
#include <dslash_policy.cuh>
}

/**
   This is a staggered Dirac operator
*/

namespace quda {


//MWTODO: This should be merged in the generic ghostFaceIndex function

 /**
     Compute the checkerboarded index into the ghost field
     corresponding to full (local) site index x[]
     @param x local site
     @param X local lattice dimensions
     @param dim dimension
     @param depth of ghost
  */
  template <int dir, typename I>
  __device__ __host__ inline int mwghostFaceIndex(const int x[], const I X[], int dim, int nFace) {
    int index = 0;
    switch(dim) {
    case 0:
      switch(dir) {
      case 0:
 index = ((x[0]+nFace-1)*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1])+x[2]*X[1] + x[1])>>1;
  break;
      case 1:
  index = ((x[0]-X[0]+nFace)*X[4]*X[3]*X[2]*X[1] + x[4]*X[3]*X[2]*X[1] + x[3]*(X[2]*X[1]) + x[2]*X[1] + x[1])>>1;
  break;
      }
      break;
    case 1:
      switch(dir) {
      case 0:
  index = ((x[1]+nFace-1)*X[4]*X[3]*X[2]*X[0] + x[4]*X[3]*X[2]*X[0] + x[3]*X[2]*X[0]+x[2]*X[0]+x[0])>>1;
  break;
      case 1:
  index = ((x[1]-X[1]+nFace)*X[4]*X[3]*X[2]*X[0] +x[4]*X[3]*X[2]*X[0]+ x[3]*X[2]*X[0] + x[2]*X[0] + x[0])>>1;
  break;
      }
      break;
    case 2:
      switch(dir) {
      case 0:
  index = ((x[2]+nFace-1)*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
  break;
      case 1:
  index = ((x[2]-X[2]+nFace)*X[4]*X[3]*X[1]*X[0] + x[4]*X[3]*X[1]*X[0] + x[3]*X[1]*X[0] + x[1]*X[0] + x[0])>>1;
  break;
      }
      break;
    case 3:
      switch(dir) {
      case 0:
  index = ((x[3]+nFace-1)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0]+x[0])>>1;
  break;
      case 1:
  index  = ((x[3]-X[3]+nFace)*X[4]*X[2]*X[1]*X[0] + x[4]*X[2]*X[1]*X[0] + x[2]*X[1]*X[0]+x[1]*X[0] + x[0])>>1;
  break;
      }
      break;
    }
    return index;
  }

  /**
     @brief Parameter structure for driving the Staggered Dslash operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_u_, QudaReconstructType reconstruct_l_, bool improved_>
  struct StaggeredArg : DslashArg<Float> {
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F = typename colorspinor_mapper<Float,nSpin,nColor,spin_project,spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    //TODO: recon 9/13 seems to break with gauge_direct_load = false
    static constexpr bool gauge_direct_load = true; // false means texture load
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
      : DslashArg<Float>(in, U, 0.0, parity, dagger, a == 0.0 ? false : true, improved_ ? 3 : 1, comm_override), out(out), in(in, improved_ ? 3 :1 ), U(U), L(L), x(x), a(a) 
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
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg, typename Vector>
     __device__ __host__ inline void applyStaggered(Vector &out, Arg &arg, int coord[nDim], int x_cb,
      int parity, int idx, int thread_dim, bool &active) {
      typedef typename mapper<Float>::type real;
      typedef Matrix<complex<real>,nColor> Link;
      const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;


#ifdef XONLY
#pragma unroll
    for (int d = 0; d<1; d++) {// loop over dimension{
#else
#pragma unroll
    for (int d = 0; d<4; d++) {// loop over dimension{
#endif 

      //Forward gather - compute fwd offset for vector fetch
#ifndef XONLY2
      
      // standard - forward direction
      {
        const bool ghost = (coord[d] + 1 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if ( doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, 1);
          const Link U = arg.U(d, x_cb, parity);
          const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
          out += U * in;

          // printf("in %f %f %f %f %f %f\n",in.data[0].real(),in.data[0].imag(),in.data[1].real(),in.data[1].imag(),out.data[2].real(),out.data[2].imag());
        }
        else if ( doBulk<kernel_type>() && !ghost ) {
          const int fwd_idx = linkIndexP1(coord, arg.dim, d);
          const Link U = arg.U(d, x_cb, parity);
          const Vector in = arg.in(fwd_idx, their_spinor_parity);
          out += U * in;
           // printf("in %f %f %f %f %f %f\n",in.data[0].real(),in.data[0].imag(),in.data[1].real(),in.data[1].imag(),out.data[2].real(),out.data[2].imag());

        }
      }


      // improved - forward direction
      if(arg.improved){
        const bool ghost = (coord[d] + 3 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
        const int fwd3_idx = linkIndexP3(coord, arg.dim, d);
        if ( doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);
          out += L * in;
        // printf("Halo (%i %i %i %i), idx %i x_cb %i ghost %i nbr_idx1 %i \t in %f %f %f %f %f %f\n",coord[0],coord[1],coord[2],coord[3], idx, x_cb, ghost_idx, nbr_idx3,in.data[1].real(),in.data[1].imag(),in2.data[1].real(),in2.data[1].imag(),out.data[1].real(),out.data[1].imag());
        } 
        else if ( doBulk<kernel_type>() && !ghost ) {
          const Link L = arg.L(d, x_cb, parity);
          const Vector in = arg.in(fwd3_idx, their_spinor_parity);
          out += L * in;
        }  
      }
#endif
// #endif
#ifndef XONLY2
      {
      //Backward gather - compute back offset for spinor and gauge fetch

        const bool ghost = (coord[d] - 1 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if ( doHalo<kernel_type>(d) && ghost) {
        // MW - check indexing into GhostFace here
          const int ghost_idx2 = mwghostFaceIndex<0>(coord, arg.dim, d, 1);
          const int ghost_idx = arg.improved ? mwghostFaceIndex<0>(coord, arg.dim, d, 3) : ghost_idx2; 
          const Link U = arg.U.Ghost(d, ghost_idx2, 1-parity);
          const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          out -= conj(U) * in;
        }
        else  if ( doBulk<kernel_type>() && !ghost ) {
          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const int gauge_idx = back_idx;
          const Link U = arg.U(d, gauge_idx, 1-parity);
          const Vector in = arg.in(back_idx, their_spinor_parity);
          out -= conj(U) * in;
        }
      }
#endif
      // #ifndef XONLY
      if(arg.improved){
        //Backward gather - compute back offset for spinor and gauge fetch
        
        const bool ghost = (coord[d] - 3 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if ( doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
          const Link L = arg.L.Ghost(d, ghost_idx, 1-parity);
          const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);
          out -= conj(L) * in;
        }
        else if ( doBulk<kernel_type>() && !ghost ) {
          const int back3_idx = linkIndexM3(coord, arg.dim, d);
          const int gauge_idx = back3_idx;
          const Link L = arg.L(d, gauge_idx, 1-parity);
          const Vector in = arg.in(back3_idx, their_spinor_parity);
          out -= conj(L) * in;
        }
      }
// #endif
    } //nDim

  }

  //out(x) = M*in = (-D + m) * in(x-mu)
 template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void staggered(Arg &arg, int idx, int parity)
  {
    using real = typename mapper<Float>::type;
    using Vector = ColorSpinor<real,nColor,1>;

    bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim; // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = arg. improved ? getCoords<nDim,QUDA_4D_PC,kernel_type, Arg, 3>(coord, arg, idx, parity, thread_dim) : getCoords<nDim,QUDA_4D_PC,kernel_type, Arg, 1>(coord, arg, idx, parity, thread_dim);
    // coord[4] = 0;
    //MWTODO -> coord[4]
    
    const int my_spinor_parity = nParity == 2 ? parity : 0;

    Vector out;

    applyStaggered<Float,nDim,nColor,nParity,dagger,kernel_type>(out, arg, coord, x_cb, parity, idx, thread_dim, active);

// printf("NEW1 Out cb %i %i \t %f %f %f %f %f %f\n",x_cb, my_spinor_parity, out.data[0].real(),out.data[0].imag(),out.data[1].real(),out.data[1].imag(),out.data[2].real(),out.data[2].imag());
   
    //MWTODO: clean up here
    // if (xpay) {
    //   Vector x = arg.x(x_cb, my_spinor_parity);
    //   out = arg.a * x -out ;
    // }
    if (dagger){
      out = real(-1)*out;
    }


    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      out = arg.a * x -out ;
    } else if (kernel_type != INTERIOR_KERNEL ) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x +  ( xpay ? real(-1)*out : out ); //MWTODO: verify
      //MWTODO - aadd xpay
    }
   // printf("NEW2 Out cb %i %i \t %f %f %f %f %f %f\n",x_cb, my_spinor_parity, out.data[0].real(),out.data[0].imag(),out.data[1].real(),out.data[1].imag(),out.data[2].real(),out.data[2].imag());
    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }



 // GPU Kernel for applying the staggered operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void staggeredGPU(Arg arg)
   {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from y thread index else use arg setting
    int parity = nParity == 2 ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;

    switch(parity) {
    case 0: staggered<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, 0); break;
    case 1: staggered<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, 1); break;
    // dslashStaggered<Float,nDim,nColor>(arg, x_cb, parity);
  }
}


  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct StaggeredLaunch {
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      dslash.launch(staggeredGPU<Float,nDim,nColor,nParity,dagger,xpay,kernel_type,Arg>, tp, arg, stream);
    }
  };


  template <typename Float, int nDim, int nColor, typename Arg>
  class Staggered : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

//TODO: fix flop / byte count?

/*
    long long flops() const
    {
      return (2*nDim*(8*nColor*nColor)-2*nColor + (arg.xpay ? 2*2*nColor : 0) )*arg.nParity*(long long)in.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + 2*nDim*arg.in.Bytes() + arg.nParity*2*nDim*arg.U.Bytes()*in.VolumeCB() +
  (arg.xpay ? arg.x.Bytes() : 0);
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }
*/
  public:
    Staggered(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
    : Dslash<Float>(arg, out, in), arg(arg), in(in) {  }

    virtual ~Staggered() { }

    void apply(const cudaStream_t &stream) {
      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        // dslashStaggeredCPU<Float,nDim,nColor>(arg);
        errorQuda("Staggered Dslash not implemented on CPU");
      } else {
#if 0
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        dslashStaggeredGPU<Float,nDim,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
# else
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash<Float>::setParam(arg);
        if (arg.xpay) Dslash<Float>::template instantiate<StaggeredLaunch,nDim,nColor, true>(tp, arg, stream);
        else          Dslash<Float>::template instantiate<StaggeredLaunch,nDim,nColor,false>(tp, arg, stream);
#endif
      }
    }

  TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
};


  template <typename Float, int nColor, QudaReconstructType recon_u, QudaReconstructType recon_l, bool improved>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                            double a, const ColorSpinorField &x, int parity, bool dagger,
                            const int *comm_override, TimeProfile &profile) 
  {
    constexpr int nDim = 4; //MWTODO: this probably should be 5 for mrhs Dslash
    StaggeredArg<Float,nColor,recon_u,recon_l,improved> arg(out, in, U, L, a, x, parity, dagger, comm_override); 
    Staggered<Float,nDim,nColor,decltype(arg) > staggered(arg, out, in);

    DslashPolicyTune<decltype(staggered)> policy(staggered, const_cast<cudaColorSpinorField*>(static_cast<const cudaColorSpinorField*>(&in)),
                                              in.VolumeCB(), in.GhostFaceCB(), profile);
    //TODO: launch policy
    //staggered.apply(0);
    policy.apply(0);

    checkCudaError();
  }

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

    if (dslash::aux_worker) dslash::aux_worker->apply(0);
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyDslashStaggered<double>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyDslashStaggered<float>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } 
    else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyDslashStaggered<short>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);

    } 
    else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }  
#else
    errorQuda("Staggered dslash has not been built");
#endif
}

} // namespace quda
