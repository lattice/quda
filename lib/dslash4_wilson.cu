#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <dslash_quda.h>
#include <color_spinor.h>
#include <worker.h>

/**
   This is the basic gauged Wilson operator
*/

namespace quda {

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct WilsonArg {
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float,4,nColor,spin_project,spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float,reconstruct,18,QUDA_STAGGERED_PHASE_NO,gauge_direct_load,ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out;                // output vector field
    const F in;           // input vector field
    const F x;            // input vector when doing xpay
    const G U;            // the gauge field
    const real kappa;     // kappa parameter = 1/(8+m)
    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const int_fastdiv X0h;
    const int_fastdiv dim[5];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume

    const bool dagger;    // dagger
    const bool xpay;      // whether we are doing xpay or not

    const DslashConstant dc; // pre-computed dslash constants for optimized indexing

    WilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
              double kappa, const ColorSpinorField &x, int parity, bool dagger)
      : out(out), in(in), U(U), kappa(kappa), x(x), parity(parity), nParity(in.SiteSubset()), nFace(1),
	X0h(nParity == 2 ? in.X(0)/2 : in.X(0)), dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
        volumeCB(in.VolumeCB()), dagger(dagger), xpay(kappa == 0.0 ? false : true), dc(in.getDslashConstant())
    {
      if (!out.isNative() || !x.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
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
  template <typename Float, int nDim, int nColor, bool dagger, typename Vector, typename Arg>
  __device__ __host__ inline void applyWilson(Vector &out, Arg &arg, int x_cb, int parity, int nParity) {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,2> HalfVector;
    typedef Matrix<complex<real>,nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1-parity : 0;

    int coord[nDim];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

#pragma unroll
    for (int d = 0; d<nDim; d++) // loop over dimension
      {
        { // Forward gather - compute fwd offset for vector fetch
          const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
          constexpr int proj_dir = dagger ? +1 : -1;

          if ( 0 && arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
            const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

            const Link U = arg.U(d, x_cb, parity);
            const HalfVector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);

            out += (U * in).reconstruct(d, proj_dir);
          } else {

            const Link U = arg.U(d, x_cb, parity);
            const Vector in = arg.in(fwd_idx, their_spinor_parity);

            out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }

        { //Backward gather - compute back offset for spinor and gauge fetch
          const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
          const int gauge_idx = back_idx;
          constexpr int proj_dir = dagger ? -1 : +1;

          if ( 0 && arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

            const Link U = arg.U.Ghost(d, ghost_idx, 1-parity);
            const HalfVector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);

            out += (conj(U) * in).reconstruct(d, proj_dir);
          } else {

            const Link U = arg.U(d, gauge_idx, 1-parity);
            const Vector in = arg.in(back_idx, their_spinor_parity);

            out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }
      } //nDim

  }

  //out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, typename Arg>
  __device__ __host__ inline void wilson(Arg &arg, int x_cb, int parity, int nParity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,4> Vector;
    Vector out;
    const int my_spinor_parity = nParity == 2 ? parity : 0;

    applyWilson<Float,nDim,nColor,dagger>(out, arg, x_cb, parity, nParity);

    if (xpay) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      out = x + arg.kappa * out;
    }
    arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, typename Arg>
  void wilsonCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
        wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, parity, arg.nParity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, bool dagger, bool xpay, typename Arg>
  __global__ void wilsonGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    // for full fields set parity from y thread index else use arg setting
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;

#if 0
    wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, parity, arg.nParity);
#else
    switch(arg.nParity) { // better code if parity is explicit
    case 1:
      switch(parity) {
      case 0: wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, 0, 1); break;
      case 1: wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, 1, 1); break;
      }
      break;
    case 2:
      switch(parity) {
      case 0: wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, 0, 2); break;
      case 1: wilson<Float,nDim,nColor,dagger,xpay>(arg, x_cb, 1, 2); break;
      }
      break;
    }
#endif
  }

  template <typename Float, int nDim, int nColor, typename Arg>
  class Wilson : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      int mv_flops = (8 * meta.Ncolor() - 2) * meta.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = meta.Nspin() == 4 ? 2 : 1;
      int ghost_flops = (num_mv_multiply * mv_flops + 2*meta.Ncolor()*meta.Nspin());
      int xpay_flops = 2 * 2 * meta.Ncolor() * meta.Nspin(); // multiply and add per real component
      int num_dir = 2*nDim;

      long long flops_ = 0;

      {
        long long sites = meta.Volume();
        flops_ = (num_dir*(meta.Nspin()/4)*meta.Ncolor()*meta.Nspin() +   // spin project (=0 for staggered)
                  num_dir*num_mv_multiply*mv_flops +                   // SU(3) matrix-vector multiplies
                  ((num_dir-1)*2*meta.Ncolor()*meta.Nspin())) * sites;   // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites;
      }
      return flops_;
    }

    long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * meta.Precision();
      bool isFixed = (meta.Precision() == sizeof(short) || meta.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * meta.Ncolor() * meta.Nspin() * meta.Precision() + (isFixed ? sizeof(float) : 0);
      int proj_spinor_bytes = meta.Ncolor() * meta.Nspin() * meta.Precision() + (isFixed ? sizeof(float) : 0);
      int num_dir = 2 * nDim; // set to 4 dimensions since we take care of 5-d fermions in derived classes where necessary

      long long bytes_=0;
      {
        long long sites = meta.Volume();
        bytes_ = (num_dir*gauge_bytes + ((num_dir-2)*spinor_bytes + 2*proj_spinor_bytes) + spinor_bytes)*sites;
        if (arg.xpay) bytes_ += spinor_bytes;
      }
      return bytes_;
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    Wilson(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
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
      if (arg.dagger) strcat(aux, ",Dagger");
      if (arg.xpay) strcat(aux,",xpay");
    }
    virtual ~Wilson() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        if (arg.xpay) arg.dagger ?
                        wilsonCPU<Float,nDim,nColor, true,true>(arg) :
                        wilsonCPU<Float,nDim,nColor,false,true>(arg);
        else          arg.dagger ?
                        wilsonCPU<Float,nDim,nColor, true,false>(arg) :
                        wilsonCPU<Float,nDim,nColor,false,false>(arg);
      } else {
        if (arg.xpay) arg.dagger ?
                        wilsonGPU<Float,nDim,nColor, true,true> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
                        wilsonGPU<Float,nDim,nColor,false,true> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
        else          arg.dagger ?
                        wilsonGPU<Float,nDim,nColor, true,false> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg) :
                        wilsonGPU<Float,nDim,nColor,false,false> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor, QudaReconstructType recon>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger)
  {
    constexpr int nDim = 4;
    WilsonArg<Float,nColor,recon> arg(out, in, U, kappa, x, parity, dagger);
    Wilson<Float,nDim,nColor,WilsonArg<Float,nColor,recon> > wilson(arg, in);
    wilson.apply(0);
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                   double kappa, const ColorSpinorField &x, int parity, bool dagger)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyWilson<Float,nColor,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity, dagger);
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

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
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

    const int nFace = 1;
    in.exchangeGhost((QudaParity)(1-parity), nFace, 0); // last parameter is dummy

    if (dslash::aux_worker) dslash::aux_worker->apply(0);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyWilson<double>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyWilson<float>(out, in, U, kappa, x, parity, dagger);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyWilson<short>(out, in, U, kappa, x, parity, dagger);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }

    in.bufferIndex = (1 - in.bufferIndex);
  }


} // namespace quda
