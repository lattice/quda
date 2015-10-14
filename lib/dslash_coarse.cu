#include <multigrid.h>
#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>

namespace quda {

#ifdef GPU_MULTIGRID

  template <typename Float, typename F, typename G>
  struct CoarseDslashArg {
    F out;
    const F inA;
    const F inB;
    const G Y;
    const G X;
    Float kappa;
    int parity; // only use this for single parity fields
    int nFace;  // hard code to 1 for now
    int dim[4];   // full lattice dimensions
    int commDim[4]; // whether a given dimension is partitioned or not

    CoarseDslashArg(F &out, const F &inA, const F &inB, const G &Y, const G &X, Float kappa, int parity)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity), nFace(1) { 
      for (int i=0; i<4; i++) {
	dim[i] = inA.X(i);
	commDim[i] = comm_dim_partitioned(i);
      }
      dim[0] = (inA.Nparity() == 1) ? 2 * dim[0] : dim[0];
    }
  };

  /**
     Applies the coarse dslash on a given parity and checkerboard site index

     @param out The result -2 * kappa * Dslash in
     @param Y The coarse gauge field
     @param kappa Kappa value
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __device__ __host__ inline void dslash(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity) {
    const int their_spinor_parity = (arg.inA.Nparity() == 2) ? (parity+1)&1 : 0;

    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    for(int d = 0; d < nDim; d++) { //Ndim
      //Forward link - compute fwd offset for spinor fetch
      {
	/*if (coord[d] + arg.nFace >= arg.in.X(d) && arg.commDim[d]) {
	// load from ghost
	} else {
	linkIndexP1(coord, arg.X(), d);
	}*/
	int fwd_idx = linkIndexP1(coord, arg.dim, d);

	complex<Float> in[Ns*Nc];
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
	  in[s*Nc+c] = arg.inA(their_spinor_parity, fwd_idx, s, c);

	for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      Float sign = (s_row == s_col) ? 1.0 : -1.0;
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		out[s_row*Nc+c_row] += sign*arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col)
		  * in[s_col*Nc+c_col]; //arg.inA(their_spinor_parity, fwd_idx, s_col, c_col);
	      } //Color column
	    } //Spin column
	  } //Color row
	} //Spin row
      }

      //Backward link - compute back offset for spinor and gauge fetch
      {
	/*if (coord[d] - arg.nFace < 0) {
	// load from ghost
	} else {
	linkIndexM1(coord, arg.X(), d);
	}*/
	int back_idx = linkIndexM1(coord, arg.dim, d);

	complex<Float> in[Ns*Nc];
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
	  in[s*Nc+c] = arg.inA(their_spinor_parity, back_idx, s, c);

	for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		out[s_row*Nc+c_row] += conj(arg.Y(d,(parity+1)&1, back_idx, s_col, s_row, c_col, c_row))
		  * in[s_col*Nc+c_col]; //arg.inA(their_spinor_parity, back_idx, s_col, c_col);
	      } //Color column
	    } //Spin column
	  } //Color row
	} //Spin row
      } //nDim
    }

    // apply kappa
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] *= -(Float)2.0*arg.kappa;
  }

  /**
     Applies the coarse clover matrix on a given parity and
     checkerboard site index

     @param out The result out += X * in
     @param X The coarse clover field
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <typename Float, typename F, typename G, int Ns, int Nc>
  __device__ __host__ inline void clover(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity) {
    const int spinor_parity = (arg.inB.Nparity() == 2) ? parity : 0;

    complex<Float> in[Ns*Nc];
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
      in[s*Nc+c] = arg.inB(spinor_parity, x_cb, s, c);

    // apply clover term
    for(int s = 0; s < Ns; s++) { //Spin out
      for(int c = 0; c < Nc; c++) { //Color out
	//This term is now incorporated into the matrix X.
	//out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	  for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	    //Factor of 2*kappa now incorporated in X
	    //out(parity,x_cb,s,c) -= 2*kappa*X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
	    out[s*Nc+c] += arg.X(0, parity, x_cb, s, s_col, c, c_col)*in[s_col*Nc+c_col];
	  } //Color in
	} //Spin in
      } //Color out
    } //Spin out
  }

  // CPU kernel for applying the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  void coarseDslash(CoarseDslashArg<Float,F,G> arg) {

    //#pragma omp parallel for 
    for (int parity= 0; parity < arg.inA.Nparity(); parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.inA.Nparity() == 2) ? parity : arg.parity;

      for(int x_cb = 0; x_cb < arg.inA.VolumeCB(); x_cb++) { //Volume
	complex <Float> out[Ns*Nc];// = { };
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] = 0.0;
	dslash<Float,F,G,nDim,Ns,Nc>(out, arg, x_cb, parity);
	clover<Float,F,G,Ns,Nc>(out, arg, x_cb, parity);

	const int my_spinor_parity = (arg.inA.Nparity() == 2) ? parity : 0;
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
	  arg.out(my_spinor_parity, x_cb, s, c) = out[s*Nc+c];
      }//VolumeCB
    } // parity
    
  }

  // FIXME need to instrument parity for gauge fields to be set from
  // parity parity while also set to ignore parity for single parity
  // quark fields

  // GPU Kernel for applying the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __global__ void coarseDslashKernel(CoarseDslashArg<Float,F,G> arg) {

    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.inA.VolumeCB()) return;

    // for full fields then set parity from y thread index else use arg setting
    int parity = (blockDim.y == 2) ? threadIdx.y : arg.parity;

    complex<Float> out[Ns*Nc];// = { };
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] = 0.0;

    dslash<Float,F,G,nDim,Ns,Nc>(out, arg, x_cb, parity);
    clover<Float,F,G,Ns,Nc>(out, arg, x_cb, parity);

    const int my_spinor_parity = (blockDim.y == 2) ? parity : 0;
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) 
      arg.out(my_spinor_parity, x_cb, s, c) = out[s*Nc+c];
  }

  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  class CoarseDslash : public Tunable {

  protected:
    CoarseDslashArg<Float,F,G> &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const
    {
      return arg.out.Bytes() + 8*arg.inA.Bytes() + arg.inB.Bytes() + arg.inA.Nparity()*(8*arg.Y.Bytes() + arg.X.Bytes());
    }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.inA.VolumeCB(); }

    bool advanceTuneParam(TuneParam &param) const 
    {
      bool rtn = Tunable::advanceTuneParam(param);
      param.block.y = arg.inA.Nparity();
      return rtn;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = arg.inA.Nparity();
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = arg.inA.Nparity();
    }


  public:
    CoarseDslash(CoarseDslashArg<Float,F,G> &arg, const ColorSpinorField &meta)
      : arg(arg), meta(meta) { }
    virtual ~CoarseDslash() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	coarseDslash<Float,F,G,nDim,Ns,Nc>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	coarseDslashKernel<Float,F,G,nDim,Ns,Nc> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

  };


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,  const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;
    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessorA(const_cast<ColorSpinorField&>(inA));
    F inAccessorB(const_cast<ColorSpinorField&>(inB));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    CoarseDslashArg<Float,F,G> arg(outAccessor, inAccessorA, inAccessorB, yAccessor, xAccessor, (Float)kappa, parity);
    CoarseDslash<Float,F,G,4,coarseSpin,coarseColor> dslash(arg, inA);
    dslash.apply(0);
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {
    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",inA.Nspin());

    if (inA.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder fOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (Y.FieldOrder() == QUDA_FLOAT2_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_FLOAT2_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else if (Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_QDP_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported field order %d\n", Y.FieldOrder());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {
    if (inA.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_FLOAT2_FIELD_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported field order %d\n", inA.FieldOrder());
    }
  }

#endif // GPU_MULTIGRID

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {
#ifdef GPU_MULTIGRID
    if (Y.Precision() != inA.Precision() || Y.Precision() != inB.Precision() ||
	X.Precision() != Y.Precision() || Y.Precision() != out.Precision())
      errorQuda("Unsupported precision mix");

    if (inA.V() == out.V()) errorQuda("Aliasing pointers");
    if (out.Precision() != inA.Precision() ||
	Y.Precision() != inA.Precision() ||
	X.Precision() != inA.Precision()) 
      errorQuda("Precision mismatch out=%d inA=%d inB=%dY=%d X=%d", 
		out.Precision(), inA.Precision(), inB.Precision(), Y.Precision(), X.Precision());

    // check all locations match
    Location(out, inA, inB, Y, X);

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, inA, inB, Y, X, kappa, parity);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }//ApplyCoarse

} // namespace quda
