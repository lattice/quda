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
    int nParity; // number of parities we're working on
    int volumeCB;
    int dim[5];   // full lattice dimensions
    int commDim[4]; // whether a given dimension is partitioned or not
    int nFace;  // hard code to 1 for now

    CoarseDslashArg(F &out, const F &inA, const F &inB, const G &Y, const G &X,
		    Float kappa, int parity, const ColorSpinorField &meta)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity),
	nParity(meta.SiteSubset()), volumeCB(meta.VolumeCB()), nFace(1) {
      for (int i=0; i<4; i++) {
	dim[i] = meta.X(i);
	commDim[i] = comm_dim_partitioned(i);
      }
      dim[0] = (nParity == 1) ? 2 * dim[0] : dim[0];
      dim[4] = 1; // ghost index expects a fifth dimension
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
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc>
  __device__ __host__ inline void dslash(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity, int s_row, int color_block) {
    const int their_spinor_parity = (arg.nParity == 2) ? (parity+1)&1 : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

    for(int d = 0; d < nDim; d++) { //Ndim
      //Forward link - compute fwd offset for spinor fetch
      {
	const int fwd_idx = linkIndexP1(coord, arg.dim, d);
	if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	  int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

	  for(int color_local = 0; color_local < Mc; color_local++) { //Color row
	    int c_row = color_block + color_local; // global color index
	    int row = s_row*Nc + c_row;
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      Float sign = (s_row == s_col) ? 1.0 : -1.0;
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		int col = s_col*Nc + c_col;
		out[color_local] += sign*(arg.Y(d, parity, x_cb, row, col)) * arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx, s_col, c_col);
	      }
	    }
	  }
	} else {

	  for(int color_local = 0; color_local < Mc; color_local++) { //Color row
	    int c_row = color_block + color_local; // global color index
	    int row = s_row*Nc + c_row;
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      Float sign = (s_row == s_col) ? 1.0 : -1.0;
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		int col = s_col*Nc + c_col;
		out[color_local] += sign*(arg.Y(d, parity, x_cb, row, col)) * arg.inA(their_spinor_parity, fwd_idx, s_col, c_col);
	      }
	    }
	  }
	}

      }

      //Backward link - compute back offset for spinor and gauge fetch
      {
	const int back_idx = linkIndexM1(coord, arg.dim, d);
	const int gauge_idx = back_idx;
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c_row = color_block + color_local;
	    int row = s_row*Nc + c_row;
	    for (int s_col=0; s_col<Ns; s_col++)
	      for (int c_col=0; c_col<Nc; c_col++) {
		int col = s_col*Nc + c_col;
		out[color_local] += conj(arg.Y.Ghost(d, (parity+1)&1, ghost_idx, col, row)) * arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx, s_col, c_col);
	      }
	  }
	} else {
	  for(int color_local = 0; color_local < Mc; color_local++) {
	    int c_row = color_block + color_local;
	    int row = s_row*Nc + c_row;
	    for(int s_col = 0; s_col < Ns; s_col++)
	      for(int c_col = 0; c_col < Nc; c_col++) {
		int col = s_col*Nc + c_col;
		out[color_local] += conj(arg.Y(d, (parity+1)&1, gauge_idx, col, row)) * arg.inA(their_spinor_parity, back_idx, s_col, c_col);
	      }
	  }

	}

      } //nDim
    }

    // apply kappa
    for (int color_local=0; color_local<Mc; color_local++) out[color_local] *= -(Float)2.0*arg.kappa;
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
  template <typename Float, typename F, typename G, int Ns, int Nc, int Mc>
  __device__ __host__ inline void clover(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity, int s, int color_block) {
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;

    // M is number of colors per thread
    for(int color_local = 0; color_local < Mc; color_local++) {//Color out
      int c = color_block + color_local; // global color index
      int row = s*Nc + c;
      for(int s_col = 0; s_col < Ns; s_col++) //Spin in
	for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	  //Factor of 2*kappa and diagonal addition now incorporated in X
	  int col = s_col*Nc + c_col;
	  out[color_local] += arg.X(0, parity, x_cb, row, col) * arg.inB(spinor_parity, x_cb, s_col, c_col);
	}
    }

  }

  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc>
  __device__ __host__ inline void coarseDslash(CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity, int s, int color_block)
  {
    complex <Float> out[Mc];
    for (int c=0; c<Mc; c++) out[c] = 0.0;
    dslash<Float,F,G,nDim,Ns,Nc,Mc>(out, arg, x_cb, parity, s, color_block);
    clover<Float,F,G,Ns,Nc,Mc>(out, arg, x_cb, parity, s, color_block);

    const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
    for (int color_local=0; color_local<Mc; color_local++) {
      int c = color_block + color_local; // global color index
      arg.out(my_spinor_parity, x_cb, s, c) = out[color_local];
    }
  }

  // CPU kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc>
  void coarseDslash(CoarseDslashArg<Float,F,G> arg)
  {
    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      //#pragma omp parallel for
      for(int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { //Volume
	for (int s=0; s<2; s++) {
	  for (int color_block=0; color_block<Nc; color_block+=Mc) { // Mc=Nc means all colors in a thread
	    coarseDslash<Float,F,G,nDim,Ns,Nc,Mc>(arg, x_cb, parity, s, color_block);
	  }
	}
      }//VolumeCB
    } // parity
    
  }

  // GPU Kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc>
  __global__ void coarseDslashKernel(CoarseDslashArg<Float,F,G> arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    // for full fields then set parity from y thread index else use arg setting
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int sM = blockDim.z*blockIdx.z + threadIdx.z;
    int s = sM / (Nc/Mc);
    int color_block = (sM % (Nc/Mc)) * Mc;

    coarseDslash<Float,F,G,nDim,Ns,Nc,Mc>(arg, x_cb, parity, s, color_block);
  }

  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc>
  class CoarseDslash : public Tunable {

  protected:
    CoarseDslashArg<Float,F,G> &arg;
    const ColorSpinorField &meta;

    long long flops() const
    {
      return ((2*nDim+1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*arg.nParity*arg.volumeCB;
    }
    long long bytes() const
    {
      return arg.out.Bytes() + 8*arg.inA.Bytes() + arg.inB.Bytes() + arg.nParity*(8*arg.Y.Bytes() + arg.X.Bytes());
    }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volumeCB; }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = Tunable::advanceBlockDim(param);
      param.block.y = block.y; param.block.z = block.z;
      param.grid.y = grid.y; param.grid.z = grid.z;

      if (ret) { // we advanced the block.x so we're done
	return true;
      } else { // block.x (spacetime) was reset

	if (param.block.y == 1 && arg.nParity == 2) { // advance parity
	  param.block.y = arg.nParity;
	  param.grid.y = 1;
	  return true;
	} else {
	  // reset parity
	  param.block.y = 1;
	  param.grid.y = arg.nParity;

	  // let's try to advance spin/block-color
	  while(param.block.z <= 2* (Nc/Mc)) {
	    param.block.z++;
	    if ( (2*(Nc/Mc)) % param.block.z == 0) {
	      param.grid.z = (2 * (Nc/Mc)) / param.block.z;
	      break;
	    }
	  }

	  // we can advance spin/block-color since this is valid
	  if (param.block.z <= 2 * (Nc/Mc)) { //
	    return true;
	  } else { // we have run off the end so let's reset
	    param.block.z = 1;
	    param.grid.z = 2 * (Nc/Mc);
	    return false;
	  }

	}
      }
    }

    bool advanceTuneParam(TuneParam &param) const 
    {
      bool rtn = Tunable::advanceTuneParam(param);
      return rtn;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = arg.nParity;
      param.block.z = 1;
      param.grid.z = 2*(Nc/Mc);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = 1;
      param.grid.y = arg.nParity;
      param.block.z = 1;
      param.grid.z = 2*(Nc/Mc);
    }


  public:
    CoarseDslash(CoarseDslashArg<Float,F,G> &arg, const ColorSpinorField &meta)
      : arg(arg), meta(meta) {
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
    }
    virtual ~CoarseDslash() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	coarseDslash<Float,F,G,nDim,Ns,Nc,Mc>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

  };


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin, QudaFieldLocation location>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,  const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;

    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessorA(const_cast<ColorSpinorField&>(inA));
    F inAccessorB(const_cast<ColorSpinorField&>(inB));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    CoarseDslashArg<Float,F,G> arg(outAccessor, inAccessorA, inAccessorB, yAccessor, xAccessor, (Float)kappa, parity, inA);

    const int colors_per_thread = 2;
    CoarseDslash<Float,F,G,4,coarseSpin,coarseColor,colors_per_thread> dslash(arg, inA);
    dslash.apply(0);
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,  const GaugeField &Y, const GaugeField &X,
		   double kappa, int parity) {
    if (inA.Location() == QUDA_CUDA_FIELD_LOCATION) {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CUDA_FIELD_LOCATION>(out, inA, inB, Y, X, kappa, parity);
    } else {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CPU_FIELD_LOCATION>(out, inA, inB, Y, X, kappa, parity);
    }
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {
    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",inA.Nspin());

    if (inA.Ncolor() == 2) {
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 4) {
      ApplyCoarse<Float,csOrder,gOrder,4,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 8) {
      ApplyCoarse<Float,csOrder,gOrder,8,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 12) {
      ApplyCoarse<Float,csOrder,gOrder,12,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 16) {
      ApplyCoarse<Float,csOrder,gOrder,16,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 20) {
      ApplyCoarse<Float,csOrder,gOrder,20,2>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.Ncolor() == 24) {
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Y.FieldOrder() == QUDA_FLOAT2_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_FLOAT2_FIELD_ORDER, QUDA_FLOAT2_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else if (inA.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,QUDA_QDP_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity);
    } else {
      errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());
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

    int dummy = 0; // ignored
    inA.exchangeGhost((QudaParity)(1-parity), dummy);

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
