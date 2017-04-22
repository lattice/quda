#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <stencil.h>

/**
   This is a basic gauged Laplace operator
*/

namespace quda {

  /**
     @brief Parameter structure for driving the Laplace operator
   */
  template <typename Float, int nColor, QudaReconstructType reconstruct, bool xpay>
  struct LaplaceArg {
    typedef typename colorspinor_order_mapper<Float,QUDA_FLOAT2_FIELD_ORDER,1,nColor>::type F;
    typedef typename gauge_mapper<Float,reconstruct>::type G;

    F out;                // output vector field
    const F in;           // input vector field
    const F x;           // input vector field
    G U;            // the gauge field
    const Float kappa;    // kappa parameter = 1/2m
    const int parity;     // only use this for single parity fields
    const int nParity;    // number of parities we're working on
    const int nFace;      // hard code to 1 for now
    const int dim[4];     // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;   // checkerboarded volume

    __host__ __device__ static constexpr bool isXpay() { return xpay; }

    LaplaceArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
	       Float kappa, const ColorSpinorField *x, int parity)
      : out(out), in(in), U(U), kappa(kappa), x(xpay ? *x : in), parity(parity), nParity(in.SiteSubset()), nFace(1),
	dim{ (3-nParity) * in.X(0), in.X(1), in.X(2), in.X(3) },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(in.VolumeCB())
    {
      if (in.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER || !U.isNative())
      errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
    }
  };

  /**
     Applies the Laplace operator

     @param out The result - kappa * Laplace in
     @param U The gauge field
     @param kappa Kappa value
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  extern __shared__ float s[];
  template <typename Float, int nDim, int nColor, typename Arg>
  __device__ __host__ inline void applyLaplace(complex<Float> out[], Arg &arg, int x_cb, int parity) {
    typedef Matrix<complex<Float>,nColor> Link;

    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[4];
    getCoords(coord, x_cb, arg.dim, parity);

    for (int d = 0; d<nDim; d++) // loop over dimension
    {
      //Forward gather - compute fwd offset for vector fetch
      const int fwd_idx = linkIndexP1(coord, arg.dim, d);

      if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

	const Link U = arg.U(d, x_cb, parity);

	complex<Float> in[nColor]; // forward vector ghost
	arg.in.loadGhost((Float*)in, ghost_idx, d, 1, their_spinor_parity);

	for (int i=0; i<nColor; i++) for (int j=0; j<nColor; j++) out[i] += U(i,j) * in[j];
	  
      } else {

	const Link U = arg.U(d, x_cb, parity);

	complex<Float> in[nColor];
	arg.in.load((Float*)in, fwd_idx, their_spinor_parity);

	for (int i=0; i<nColor; i++) for (int j=0; j<nColor; j++) out[i] += U(i,j) * in[j];

      }

      //Backward gather - compute back offset for spinor and gauge fetch
      const int back_idx = linkIndexM1(coord, arg.dim, d);
      const int gauge_idx = back_idx;

      if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

	Link U;
	arg.U.loadGhost((Float*)U.data, ghost_idx, d, 1-parity);

	complex<Float> in[nColor];
	arg.in.loadGhost((Float*)in, ghost_idx, d, 0, their_spinor_parity);

	for (int i=0; i<nColor; i++) for (int j=0; j<nColor; j++) out[i] += conj(U(j,i)) * in[j];

      } else {
	
	const Link U = arg.U(d, gauge_idx, 1-parity);

	complex<Float> in[nColor];
	arg.in.load((Float*)in, back_idx, their_spinor_parity);

	for (int i=0; i<nColor; i++) for (int j=0; j<nColor; j++) out[i] += conj(U(j,i)) * in[j];

      }
    } //nDim

  }


  //out(x) = M*in = (-D + m) * in(x-mu)
  template <typename Float, int nDim, int nColor, typename Arg>
  __device__ __host__ inline void laplace(Arg &arg, int x_cb, int parity)
  {
    complex <Float> out[nColor];
    for (int c=0; c<nColor; c++) out[c] = 0.0;

    applyLaplace<Float,nDim,nColor>(out, arg, x_cb, parity);

    complex<Float> x[nColor];
    if (arg.isXpay()) {
      arg.x.load((Float*)x, x_cb, parity);
      for (int i=0; i<nColor; i++) out[i] = x[i] + arg.kappa * out[i];
    }
    arg.out.save((Float*)out, x_cb, parity);
  }

  // CPU kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  void laplaceCPU(Arg arg)
  {

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	laplace<Float,nDim,nColor>(arg, x_cb, parity);
      } // 4-d volumeCB
    } // parity

  }

  // GPU Kernel for applying the Laplace operator to a vector
  template <typename Float, int nDim, int nColor, typename Arg>
  __global__ void laplaceGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;

    // for full fields set parity from y thread index else use arg setting
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    laplace<Float,nDim,nColor>(arg, x_cb, parity);
  }

  template <typename Float, int nDim, int nColor, typename Arg>
  class Laplace : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return (2*nDim*(8*nColor*nColor)-2*nColor)*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return arg.out.Bytes() + 2*nDim*arg.in.Bytes() + arg.nParity*(2*nDim*arg.U.Bytes());
    }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }
    unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / arg.nParity; }

  public:
    Laplace(Arg &arg, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), meta(meta)
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
      if (arg.isXpay()) strcat(aux,",xpay");
    }
    virtual ~Laplace() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	laplaceCPU<Float,nDim,nColor>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	laplaceGPU<Float,nDim,nColor> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };


  template <typename Float, int nColor, QudaReconstructType recon>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    constexpr int nDim = 4;
    if (x) {
      LaplaceArg<Float,nColor,recon,true> arg(out, in, U, kappa, x, parity);
      Laplace<Float,nDim,nColor,LaplaceArg<Float,nColor,recon,true> > laplace(arg, in);
      laplace.apply(0);
    } else {
      LaplaceArg<Float,nColor,recon,false> arg(out, in, U, kappa, x, parity);
      Laplace<Float,nDim,nColor,LaplaceArg<Float,nColor,recon,false> > laplace(arg, in);
      laplace.apply(0);
    }
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    if (U.Reconstruct()== QUDA_RECONSTRUCT_NO) {
      ApplyLaplace<Float,3,QUDA_RECONSTRUCT_NO>(out, in, U, kappa, x, parity);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_12) {
      ApplyLaplace<Float,3,QUDA_RECONSTRUCT_12>(out, in, U, kappa, x, parity);
    } else if (U.Reconstruct()== QUDA_RECONSTRUCT_8) {
      ApplyLaplace<Float,3,QUDA_RECONSTRUCT_8>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
    void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		      double kappa, const ColorSpinorField *x, int parity)
  {
    if (in.Ncolor() == 3) {
      ApplyLaplace<Float,3>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

  //Apply the Laplace operator
  //out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
		    double kappa, const ColorSpinorField *x, int parity)		    
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (out.Precision() != in.Precision() || U.Precision() != in.Precision())
      errorQuda("Precision mismatch out=%d in=%d U=%d", out.Precision(), in.Precision(), U.Precision());
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());
    
    // check all locations match
    Location(out, in, U);

    in.exchangeGhost((QudaParity)(1-parity), 0); // last parameter is dummy

    if (dslash::aux_worker) dslash::aux_worker->apply(0);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyLaplace<double>(out, in, U, kappa, x, parity);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyLaplace<float>(out, in, U, kappa, x, parity);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }


} // namespace quda
