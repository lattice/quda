#include <transfer.h>
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

  template <typename Float, int Ns, int Nc, QudaReconstructType gRecon>
  struct WuppertalSmearingArg {
    typedef typename colorspinor_mapper<Float,Ns,Nc>::type F;
    typedef typename gauge_mapper<Float,gRecon>::type G;

    F out;                // output vector field
    const F in;           // input vector field
    const G U;            // the gauge field
    const Float A;        // A parameter
    const Float B;        // B parameter
    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const int dim[5];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume

    WuppertalSmearingArg(ColorSpinorField &out, const ColorSpinorField &in, int parity, const GaugeField &U,
                       Float A, Float B)
      : out(out), in(in), U(U), A(A), B(B), parity(parity), nParity(in.SiteSubset()), nFace(1),
        dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(in.VolumeCB())
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
  template <typename Float, int Nc, typename Vector, typename Arg>
  __device__ __host__ inline void computeNeighborSum(Vector &out, Arg &arg, int x_cb, int parity) {

    typedef Matrix<complex<Float>,Nc> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

#pragma unroll
    for (int dir=0; dir<3; dir++) { // loop over spatial directions

      //Forward gather - compute fwd offset for vector fetch
      const int fwd_idx = linkIndexP1(coord, arg.dim, dir);

      if ( arg.commDim[dir] && (coord[dir] + arg.nFace >= arg.dim[dir]) ) {
        const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, dir, arg.nFace);

        const Link U = arg.U(dir, x_cb, parity);
	const Vector in = arg.in.Ghost(dir, 1, ghost_idx, their_spinor_parity);

        out += U * in;
      } else {
        const Link U = arg.U(dir, x_cb, parity);
	const Vector in = arg.in(fwd_idx, their_spinor_parity);

        out += U * in;
      }

      //Backward gather - compute back offset for spinor and gauge fetch
      const int back_idx = linkIndexM1(coord, arg.dim, dir);
      const int gauge_idx = back_idx;

      if ( arg.commDim[dir] && (coord[dir] - arg.nFace < 0) ) {
        const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, dir, arg.nFace);

        const Link U = arg.U.Ghost(dir, ghost_idx, 1-parity);
	const Vector in = arg.in.Ghost(dir, 0, ghost_idx, their_spinor_parity);

        out += conj(U) * in;
      } else {
        const Link U = arg.U(dir, gauge_idx, 1-parity);
	const Vector in = arg.in(back_idx, their_spinor_parity);

        out += conj(U) * in;
      }
    }
  }

  //out(x) = A in(x) + B computeNeighborSum(out, x)
  template <typename Float, int Ns, int Nc, typename Arg>
  __device__ __host__ inline void computeWupperalStep(Arg &arg, int x_cb, int parity)
  {
    typedef ColorSpinor<Float,Nc,Ns> Vector;
    Vector out;

    computeNeighborSum<Float,Nc>(out, arg, x_cb, parity);

    Vector in;
    arg.in.load((Float*)in.data, x_cb, parity);
    out = arg.A*in + arg.B*out;
    
    arg.out(x_cb, parity) = out;
  }

  // CPU kernel for applying a wuppertal smearing step to a vector
  template <typename Float, int Ns, int Nc, typename Arg>
  void wuppertalStepCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
        computeWupperalStep<Float,Ns,Nc>(arg, x_cb, parity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying a wuppertal smearing step to a vector
  template <typename Float, int Ns, int Nc, typename Arg>
  __global__ void wuppertalStepGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    // for full fields set parity from y thread index else use arg setting
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;
    parity = (arg.nParity == 2) ? parity : arg.parity;

    computeWupperalStep<Float,Ns,Nc>(arg, x_cb, parity);
  }

  template <typename Float, int Ns, int Nc, typename Arg>
  class WuppertalSmearing : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return (2*3*Ns*Nc*(8*Nc-2) + 2*3*Nc*Ns )*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + (2*3+1)*arg.in.Bytes() + arg.nParity*2*3*arg.U.Bytes()*meta.VolumeCB();
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    WuppertalSmearing(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
    }
    virtual ~WuppertalSmearing() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        wuppertalStepCPU<Float,Ns,Nc>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        wuppertalStepGPU<Float,Ns,Nc> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  template<typename Float, int Ns, int Nc, QudaReconstructType gRecon>
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity,
		     const GaugeField& U, double A, double B)
  {
    WuppertalSmearingArg<Float,Ns,Nc,gRecon> arg(out, in, parity, U, A, B);
    WuppertalSmearing<Float,Ns,Nc,WuppertalSmearingArg<Float,Ns,Nc,gRecon> > wuppertal(arg, in);
    wuppertal.apply(0);
  }

  // template on the gauge reconstruction
  template<typename Float, int Ns, int Nc>
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity,
		     const GaugeField& U, double A, double B)
  {
    if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      wuppertalStep<Float,Ns,Nc,QUDA_RECONSTRUCT_NO>(out, in, parity, U, A, B);
    } else if(U.Reconstruct() == QUDA_RECONSTRUCT_12) {
      wuppertalStep<Float,Ns,Nc,QUDA_RECONSTRUCT_12>(out, in, parity, U, A, B);
    } else if(U.Reconstruct() == QUDA_RECONSTRUCT_8) {
      wuppertalStep<Float,Ns,Nc,QUDA_RECONSTRUCT_8>(out, in, parity, U, A, B);
    } else {
      errorQuda("Reconstruction type %d of origin gauge field not supported", U.Reconstruct());
    }
  }


  // template on the number of colors
  template<typename Float, int Ns>
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity,
		     const GaugeField& U, double A, double B)
  {
    if (out.Ncolor() != in.Ncolor()) {
      errorQuda("Orign and destination fields must have the same number of colors\n");
    }

    if (out.Ncolor() == 3 ) {
      wuppertalStep<Float,Ns,3>(out, in, parity, U, A, B);
    } else {
      errorQuda(" is not implemented for Ncolor!=3");
    }
  }

  // template on the number of spins
  template<typename Float>
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity,
		     const GaugeField& U, double A, double B)
  {
    if(out.Nspin() != in.Nspin()) {
      errorQuda("Orign and destination fields must have the same number of spins\n");
    }

    if (out.Nspin() == 4 ){
      wuppertalStep<Float,4>(out, in, parity, U, A, B);
    }else if (in.Nspin() == 1 ){
      wuppertalStep<Float,1>(out, in, parity, U, A, B);
    }else{
      errorQuda("Nspin %d not supported", out.Nspin());
    }
  }

  /**
     Apply a generic Wuppertal smearing step
     Computes out(x) = A*in(x)  + B*\sum_mu (U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
     @param[out] out The out result field
     @param[in] in The in spinor field
     @param[in] U The gauge field
     @param[in] A The scaling factor for in(x)
     @param[in] B The scaling factor for \sum_mu (U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  */
  // template on the precision
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity,
		     const GaugeField& U, double A, double B)
  {
    if (in.V() == out.V()) {
      errorQuda("Orign and destination fields must be different pointers");
    }

    // check precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    const int nFace = 1;
    in.exchangeGhost((QudaParity)(1-parity), nFace, 0); // last parameter is dummy

    if (out.Precision() == QUDA_SINGLE_PRECISION){
      wuppertalStep<float>(out, in, parity, U, A, B);
    } else if(out.Precision() == QUDA_DOUBLE_PRECISION) {
      wuppertalStep<double>(out, in, parity, U, A, B);
    } else {
      errorQuda("Precision %d not supported", out.Precision());
    }
    return;
  }

  /**
     Apply a standard Wuppertal smearing step
     Computes out(x) = 1/(1+6*alpha)*(in(x)  + alpha*\sum_mu (U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)))
     @param[out] out The out result field
     @param[in] in The in spinor field
     @param[in] U The gauge field
     @param[in] alpha The smearing parameter
  */
  void wuppertalStep(ColorSpinorField &out, const ColorSpinorField &in, int parity, const GaugeField& U, double alpha)
  {
    wuppertalStep(out, in, parity, U, 1./(1.+6.*alpha), alpha/(1.+6.*alpha));
  }
} // namespace quda
