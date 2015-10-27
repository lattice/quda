#include <multigrid.h>
#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>

#define LEGACY_SPINOR
#define LEGACY_GAUGE

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

    bool staggered_coarse_dslash;//staggered coarse dslash has more sparse structure.

    CoarseDslashArg(F &out, const F &inA, const F &inB, const G &Y, const G &X,
		    Float kappa, int parity, const ColorSpinorField &meta, bool is_staggered)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity),
	nParity(meta.SiteSubset()), volumeCB(meta.VolumeCB()), nFace(1), staggered_coarse_dslash(is_staggered) {
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
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __device__ __host__ inline void dslash(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity) {
    const int their_spinor_parity = (arg.nParity == 2) ? (parity+1)&1 : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
    coord[4] = 0;

    for(int d = 0; d < nDim; d++) { //Ndim
      //Forward link - compute fwd offset for spinor fetch
      {
	complex<Float> in[Ns*Nc];
	complex<Float> Y[Ns*Nc][Ns*Nc];
	int fwd_idx = linkIndexP1(coord, arg.dim, d);
#ifdef LEGACY_SPINOR
	if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	  int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
	  for (int s=0; s<Ns; s++)
	    for (int c=0; c<Nc; c++)
	      in[s*Nc+c] = arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx, s, c);
	} else {
	  for (int s=0; s<Ns; s++)
	    for (int c=0; c<Nc; c++)
	      in[s*Nc+c] = arg.inA(their_spinor_parity, fwd_idx, s, c);
	}

#else
	if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	  int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
	  arg.inA.loadGhost(reinterpret_cast<Float*>(in), ghost_idx, d, 1, their_spinor_parity);
	} else {
	  arg.inA.load(reinterpret_cast<Float*>(in), fwd_idx, their_spinor_parity);
	}
#endif // LEGACY_SPINOR

#ifdef LEGACY_GAUGE
	for (int s_row=0; s_row<Ns; s_row++)
	  for (int c_row=0; c_row<Nc; c_row++)
	    for (int s_col=0; s_col<Ns; s_col++)
	      for (int c_col=0; c_col<Nc; c_col++)
		Y[s_row*Nc+c_row][s_col*Nc+c_col] = arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col);
#else
	arg.Y.load(reinterpret_cast<Float*>(Y), x_cb, d, parity);
#endif // LEGACY_GAUGE

	for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      Float sign = (s_row == s_col) ? 1.0 : -1.0;
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		//out[s_row*Nc+c_row] += sign*arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col)
		//* in[s_col*Nc+c_col]; //arg.inA(their_spinor_parity, fwd_idx, s_col, c_col);
		out[s_row*Nc+c_row] += sign*(Y[s_row*Nc+c_row][s_col*Nc+c_col]) * in[s_col*Nc+c_col];
	      } //Color column
	    } //Spin column
	  } //Color row
	} //Spin row
      }

      //Backward link - compute back offset for spinor and gauge fetch
      {
	complex<Float> in[Ns*Nc];
	complex<Float> Y[Ns*Nc][Ns*Nc];
	int gauge_idx;
	int back_idx = linkIndexM1(coord, arg.dim, d);
#ifdef LEGACY_SPINOR
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
	  for (int s=0; s<Ns; s++)
	    for (int c=0; c<Nc; c++)
	      in[s*Nc+c] = arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx, s, c);
	} else {
	  for (int s=0; s<Ns; s++)
	    for (int c=0; c<Nc; c++)
	      in[s*Nc+c] = arg.inA(their_spinor_parity, back_idx, s, c);
	}
#else
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
	  arg.inA.loadGhost(reinterpret_cast<Float*>(in), ghost_idx, d, 0, their_spinor_parity);
	} else {
	  arg.inA.load(reinterpret_cast<Float*>(in), back_idx, their_spinor_parity);
	}
#endif // LEGACY_SPINOR

	gauge_idx = back_idx;
#ifdef LEGACY_GAUGE
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
	  for (int s_row=0; s_row<Ns; s_row++)
	    for (int c_row=0; c_row<Nc; c_row++)
	      for (int s_col=0; s_col<Ns; s_col++)
		for (int c_col=0; c_col<Nc; c_col++) {
		  Y[s_row*Nc+c_row][s_col*Nc+c_col] = arg.Y.Ghost(d, (parity+1)&1, ghost_idx, s_row, s_col, c_row, c_col);
		}

	} else {
	  for (int s_row=0; s_row<Ns; s_row++)
	    for (int c_row=0; c_row<Nc; c_row++)
	      for (int s_col=0; s_col<Ns; s_col++)
		for (int c_col=0; c_col<Nc; c_col++)
		  Y[s_row*Nc+c_row][s_col*Nc+c_col] = arg.Y(d, (parity+1)&1, gauge_idx, s_row, s_col, c_row, c_col);
	}
#else
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  // load from ghost
	  int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
	  arg.Y.loadGhost(reinterpret_cast<Float*>(Y), ghost_idx, d, (parity+1)&1);
        } else {
	  arg.Y.load(reinterpret_cast<Float*>(Y), gauge_idx, d, (parity+1)&1);
	}
#endif // LEGACY_GAUGE

	for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	//out[s_row*Nc+c_row] += conj(arg.Y(d,(parity+1)&1, gauge_idx, s_col, s_row, c_col, c_row))
	//	  * in[s_col*Nc+c_col]; //arg.inA(their_spinor_parity, back_idx, s_col, c_col);
		out[s_row*Nc+c_row] += conj(Y[s_col*Nc+c_col][s_row*Nc+c_row]) * in[s_col*Nc+c_col];
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
     A.S.: staggered coarse dslash has more sparse structure
     Applies the coarse dslash on a given parity and checkerboard site index

     @param out The result -2 * kappa * Dslash in
     @param Y The coarse gauge field
     @param kappa Kappa value
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __device__ __host__ inline void ks_dslash(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity) {
    const int their_spinor_parity = (arg.nParity == 2) ? (parity+1)&1 : 0;

    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    for(int d = 0; d < nDim; d++) { //Ndim
      //Forward link - compute fwd offset for spinor fetch
      {
	complex<Float> in[Ns*Nc];
	complex<Float> Y[Ns*Nc][Ns*Nc];
#ifdef LEGACY_SPINOR
	int fwd_idx = linkIndexP1(coord, arg.dim, d);
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
	  in[s*Nc+c] = arg.inA(their_spinor_parity, fwd_idx, s, c);
#else
	int fwd_idx = linkIndexP1(coord, arg.dim, d);
	arg.inA.load(reinterpret_cast<Float*>(in), fwd_idx, their_spinor_parity);
#endif // LEGACY_SPINOR

#ifdef LEGACY_GAUGE
	for (int s_row=0; s_row<Ns; s_row++)
	  for (int c_row=0; c_row<Nc; c_row++)
	    for (int s_col=0; s_col<Ns; s_col++)
	      for (int c_col=0; c_col<Nc; c_col++)
		Y[s_row*Nc+c_row][s_col*Nc+c_col] = arg.Y(d, (parity+1)&1, x_cb, s_row, s_col, c_row, c_col);
#else
	//if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	// load from ghost
	//} else {
	arg.Y.load(reinterpret_cast<Float*>(Y), x_cb, d, (parity+1)&1);
	  //}
#endif // LEGACY_GAUGE

        for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	  for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		out[0*Nc+c_row] -= Y[0*Nc+c_row][1*Nc+c_col] * in[1*Nc+c_col]; //arg.inA(their_spinor_parity, fwd_idx, s_col, c_col);
		out[1*Nc+c_row] -= Y[1*Nc+c_row][0*Nc+c_col] * in[0*Nc+c_col]; //arg.inA(their_spinor_parity, fwd_idx, s_col, c_col);
	  } //Color column
	} //Color row
      }

      //Backward link - compute back offset for spinor and gauge fetch
      {
	complex<Float> in[Ns*Nc];
	complex<Float> Y[Ns*Nc][Ns*Nc];
	int gauge_idx;
	int back_idx = linkIndexM1(coord, arg.dim, d);
#ifdef LEGACY_SPINOR
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++)
	  in[s*Nc+c] = arg.inA(their_spinor_parity, back_idx, s, c);
#else
	arg.inA.load(reinterpret_cast<Float*>(in), back_idx, their_spinor_parity);
#endif // LEGACY_SPINOR

	gauge_idx = back_idx;
#ifdef LEGACY_GAUGE
	for (int s_row=0; s_row<Ns; s_row++)
	  for (int c_row=0; c_row<Nc; c_row++)
	    for (int s_col=0; s_col<Ns; s_col++)
	      for (int c_col=0; c_col<Nc; c_col++)
		Y[s_row*Nc+c_row][s_col*Nc+c_col] = arg.Y(d, (parity+1)&1, gauge_idx, s_row, s_col, c_row, c_col);
#else
	//if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	// load from ghost
        //} else {
	arg.Y.load(reinterpret_cast<Float*>(Y), gauge_idx, d, (parity+1)&1);
	  //}
#endif // LEGACY_GAUGE

        for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	  for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	      out[0*Nc+c_row] += conj(Y[1*Nc+c_col][0*Nc+c_row]) * in[1*Nc+c_col]; //arg.inA(their_spinor_parity, back_idx, s_col, c_col);
              out[1*Nc+c_row] += conj(Y[0*Nc+c_col][1*Nc+c_row]) * in[0*Nc+c_col]; //arg.inA(their_spinor_parity, back_idx, s_col, c_col);
	  } //Color column
	} //Color row
      } //nDim
    }
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
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;

    complex<Float> in[Ns*Nc];
    complex<Float> X[Ns*Nc][Ns*Nc];
#ifdef LEGACY_SPINOR
    for (int s=0; s<Ns; s++)
      for (int c=0; c<Nc; c++)
	in[s*Nc+c] = arg.inB(spinor_parity, x_cb, s, c);
#else
    arg.inB.load(reinterpret_cast<Float*>(in), x_cb, spinor_parity);
#endif // LEGACY_SPINOR

#ifdef LEGACY_GAUGE
    for (int s_row=0; s_row<Ns; s_row++)
      for (int c_row=0; c_row<Nc; c_row++)
	for (int s_col=0; s_col<Ns; s_col++)
	  for (int c_col=0; c_col<Nc; c_col++)
	    X[s_row*Nc+c_row][s_col*Nc+c_col] = arg.X(0, parity, x_cb, s_row, s_col, c_row, c_col);
#else
    arg.X.load(reinterpret_cast<Float*>(X), x_cb, 0, parity);
#endif

    // apply clover term
    for(int s = 0; s < Ns; s++) { //Spin out
      for(int c = 0; c < Nc; c++) { //Color out
	//This term is now incorporated into the matrix X.
	//out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	  for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	    //Factor of 2*kappa now incorporated in X
	    out[s*Nc+c] += X[s*Nc+c][s_col*Nc+c_col] *in[s_col*Nc+c_col];
	  } //Color in
	} //Spin in
      } //Color out
    } //Spin out
  }

  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __device__ __host__ inline void coarseDslash(CoarseDslashArg<Float,F,G> &arg, int x_cb, int parity)
  {
    complex <Float> out[Ns*Nc];
    for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] = 0.0;

    if(!arg.staggered_coarse_dslash) 
      dslash<Float,F,G,nDim,Ns,Nc>(out, arg, x_cb, parity);
    else
      ks_dslash<Float,F,G,nDim,Ns,Nc>(out, arg, x_cb, parity);

    clover<Float,F,G,Ns,Nc>(out, arg, x_cb, parity);

    const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
#ifdef LEGACY_SPINOR
    for (int s=0; s<Ns; s++)
      for (int c=0; c<Nc; c++)
	arg.out(my_spinor_parity, x_cb, s, c) = out[s*Nc+c];
#else
    arg.out.save(reinterpret_cast<Float*>(out), x_cb, my_spinor_parity);
#endif
  }

  // CPU kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  void coarseDslash(CoarseDslashArg<Float,F,G> arg)
  {
    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for(int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { //Volume
        coarseDslash<Float,F,G,nDim,Ns,Nc>(arg, x_cb, parity);
      }//VolumeCB
    } // parity
    
  }

  // GPU Kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  __global__ void coarseDslashKernel(CoarseDslashArg<Float,F,G> arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    // for full fields then set parity from y thread index else use arg setting
    int parity = (blockDim.y == 2) ? threadIdx.y : arg.parity;

    coarseDslash<Float,F,G,nDim,Ns,Nc>(arg, x_cb, parity);
  }

  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc>
  class CoarseDslash : public Tunable {

  protected:
    CoarseDslashArg<Float,F,G> &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const
    {
      return arg.out.Bytes() + 8*arg.inA.Bytes() + arg.inB.Bytes() + arg.nParity*(8*arg.Y.Bytes() + arg.X.Bytes());
    }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volumeCB; }

    bool advanceTuneParam(TuneParam &param) const 
    {
      bool rtn = Tunable::advanceTuneParam(param);
      param.block.y = arg.nParity;
      return rtn;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = arg.nParity;
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = arg.nParity;
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
	coarseDslash<Float,F,G,nDim,Ns,Nc>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	coarseDslashKernel<Float,F,G,nDim,Ns,Nc> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

  };


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin, QudaFieldLocation location>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,  const GaugeField &Y, const GaugeField &X,
		   double kappa, bool is_staggered, int parity) {
#ifdef LEGACY_SPINOR
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
#else
    typedef typename colorspinor_order_mapper<Float,csOrder,coarseSpin,coarseColor>::type F;
#endif // LEGACY_SPINOR

#ifdef LEGACY_GAUGE
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;
#else
    typedef typename gauge_order_mapper<Float,gOrder,coarseSpin*coarseColor>:: type G;
#endif // LEGACY_GAUGE

    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessorA(const_cast<ColorSpinorField&>(inA));
    F inAccessorB(const_cast<ColorSpinorField&>(inB));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    CoarseDslashArg<Float,F,G> arg(outAccessor, inAccessorA, inAccessorB, yAccessor, xAccessor, (Float)kappa, parity, inA, is_staggered);
    CoarseDslash<Float,F,G,4,coarseSpin,coarseColor> dslash(arg, inA);
    dslash.apply(0);
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,  const GaugeField &Y, const GaugeField &X,
		   double kappa, bool is_staggered, int parity) {
    if (inA.Location() == QUDA_CUDA_FIELD_LOCATION) {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CUDA_FIELD_LOCATION>(out, inA, inB, Y, X, kappa,is_staggered, parity);
    } else {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CPU_FIELD_LOCATION>(out, inA, inB, Y, X, kappa,is_staggered, parity);
    }
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, bool is_staggered, int parity) {
    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",inA.Nspin());

    if (inA.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, inA, inB, Y, X, kappa,is_staggered, parity);
    } else if (inA.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, inA, inB, Y, X, kappa,is_staggered, parity);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, bool is_staggered, int parity) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Y.FieldOrder() == QUDA_FLOAT2_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_FLOAT2_FIELD_ORDER, QUDA_FLOAT2_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, is_staggered, parity);
    } else if (inA.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,QUDA_QDP_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, is_staggered, parity);
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
		   const GaugeField &Y, const GaugeField &X, double kappa, bool is_staggered, int parity) {
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
      ApplyCoarse<double>(out, inA, inB, Y, X, kappa, is_staggered, parity);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, inA, inB, Y, X, kappa, is_staggered, parity);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }//ApplyCoarse

} // namespace quda
