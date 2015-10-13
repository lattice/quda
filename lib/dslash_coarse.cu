#include <multigrid.h>
#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>

namespace quda {

#ifdef GPU_MULTIGRID

  template <typename Float, typename F, typename G>
  struct CoarseDslashArg {
    F out;
    const F in;
    const G Y;
    const G X;
    Float kappa;
    int parity;

    int nFace; // hard code to 1 for now

    CoarseDslashArg(F &out, const F &in, const G &Y, const G &X, Float kappa, int parity)
      : out(out), in(in), Y(Y), X(X), kappa(kappa), parity(parity), nFace(1) { }
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
    template <typename Float, typename F, typename G, int nDim>
    __device__ __host__ inline void dslash( CoarseDslashArg<Float,F,G> &arg, int x_cb) {
    const int Nc = arg.in.Ncolor();
    const int Ns = arg.in.Nspin();
    const int parity = arg.parity;

    int coord[nDim];
    arg.in.LatticeIndex(coord,parity*arg.in.VolumeCB()+x_cb);

    for(int s = 0; s < Ns; s++) for(int c = 0; c < Nc; c++) arg.out(parity, x_cb, s, c) = (Float)0.0;

    for(int d = 0; d < nDim; d++) { //Ndim
      //Forward link - compute fwd offset for spinor fetch
      int coordTmp = coord[d];
      coord[d] = (coord[d] + 1)%arg.in.X(d);
      int fwd_idx = 0;
      for(int dim = nDim-1; dim >= 0; dim--) fwd_idx = arg.in.X(dim) * fwd_idx + coord[dim];
      coord[d] = coordTmp;

      for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	  for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	    Float sign = (s_row == s_col) ? 1.0 : -1.0;
	    for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	      arg.out(parity, x_cb, s_row, c_row) += sign*arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col)
		* arg.in((parity+1)&1, fwd_idx/2, s_col, c_col);
	    } //Color column
	  } //Spin column
	} //Color row
      } //Spin row

      //Backward link - compute back offset for spinor and gauge fetch
      int back_idx = 0;
      coord[d] = (coordTmp - 1 + arg.in.X(d))% (arg.in.X(d));
      for(int dim = nDim-1; dim >= 0; dim--) back_idx = arg.in.X(d) * back_idx + coord[dim];
      coord[d] = coordTmp;

      for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	  for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	    for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	      arg.out(parity, x_cb, s_row, c_row) += conj(arg.Y(d,(parity+1)&1, back_idx/2, s_col, s_row, c_col, c_row))
		* arg.in((parity+1)&1, back_idx/2, s_col, c_col);
	    } //Color column
	  } //Spin column
	} //Color row
      } //Spin row
    } //nDim

    // apply kappa
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) arg.out(parity, x_cb, s, c) *= -(Float)2.0*arg.kappa;
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
  template <typename Float, typename F, typename G>
  __device__ __host__ inline void clover(CoarseDslashArg<Float,F,G> &arg, int x_cb) {
    const int Nc = arg.in.Ncolor();
    const int Ns = arg.in.Nspin();
    const int parity = arg.parity;

    // apply clover term
    for(int s = 0; s < Ns; s++) { //Spin out
      for(int c = 0; c < Nc; c++) { //Color out
	//This term is now incorporated into the matrix X.
	//out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	  for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	    //Factor of 2*kappa now incorporated in X
	    //out(parity,x_cb,s,c) -= 2*kappa*X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
	    arg.out(parity,x_cb,s,c) += arg.X(0, parity, x_cb, s, s_col, c, c_col)*arg.in(parity,x_cb,s_col,c_col);
	  } //Color in
	} //Spin in
      } //Color out
    } //Spin out
  }

  // CPU kernel for applying the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim>
  void coarseDslash(CoarseDslashArg<Float,F,G> arg) {

    //#pragma omp parallel for 
    for(int x_cb = 0; x_cb < arg.in.VolumeCB(); x_cb++) { //Volume
      dslash<Float,F,G,nDim>(arg, x_cb);
      clover<Float,F,G>(arg, x_cb);
    }//VolumeCB
    
  }

  // GPU Kernel for applying the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim>
  __global__ void coarseDslashKernel(CoarseDslashArg<Float,F,G> arg) {

    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.in.VolumeCB()) return;

    dslash<Float,F,G,nDim>(arg, x_cb);
    clover<Float,F,G>(arg, x_cb);
  }

  template <typename Float, typename F, typename G, int nDim>
  class CoarseDslash : public Tunable {

  protected:
    CoarseDslashArg<Float,F,G> &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + 8*arg.in.Bytes() + 8*arg.Y.Bytes() + arg.X.Bytes(); }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.in.Volume()/2; } // fine parity is the block y dimension

  public:
    CoarseDslash(CoarseDslashArg<Float,F,G> &arg, const ColorSpinorField &meta)
      : arg(arg), meta(meta) { }
    virtual ~CoarseDslash() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	coarseDslash<Float,F,G,nDim>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	coarseDslashKernel<Float,F,G,nDim> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

  };


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;
    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessor(const_cast<ColorSpinorField&>(in));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    CoarseDslashArg<Float,F,G> arg(outAccessor, inAccessor, yAccessor, xAccessor, (Float)kappa, parity);
    CoarseDslash<Float,F,G,4> dslash(arg, in);
    dslash.apply(0);
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    if (in.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",in.Nspin());

    if (in.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, in, Y, X, kappa, parity);
    } else if (in.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, in, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder fOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (Y.FieldOrder() == QUDA_FLOAT2_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_FLOAT2_GAUGE_ORDER>(out, in, Y, X, kappa, parity);
    } else if (Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_QDP_GAUGE_ORDER>(out, in, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported field order %d\n", Y.FieldOrder());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_FLOAT2_FIELD_ORDER>(out, in, Y, X, kappa, parity);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(out, in, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported field order %d\n", in.FieldOrder());
    }
  }

#endif // GPU_MULTIGRID

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
#ifdef GPU_MULTIGRID
    if (Y.Precision() != in.Precision() || X.Precision() != Y.Precision() || Y.Precision() != out.Precision())
      errorQuda("Unsupported precision mix");

    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (out.Precision() != in.Precision() ||
	Y.Precision() != in.Precision() ||
	X.Precision() != in.Precision()) 
      errorQuda("Precision mismatch out=%d in=%d Y=%d X=%d", 
		out.Precision(), in.Precision(), Y.Precision(), X.Precision());

    // check all locations match
    Location(out, in, Y, X);

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, in, Y, X, kappa, parity);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, in, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }//ApplyCoarse

} // namespace quda
