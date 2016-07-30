/** 
  Parameter struct for generic blas kernel
*/
template <typename SpinorX, typename SpinorY, typename SpinorZ, 
  typename SpinorW, typename Functor>
struct BlasArg {
  SpinorX X;
  SpinorY Y;
  SpinorZ Z;
  SpinorW W;
  Functor f;
  const int length;
  BlasArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, Functor f, int length) 
  : X(X), Y(Y), Z(Z), W(W), f(f), length(length) { ; }
};

/**
   Generic blas kernel with four loads and up to four stores.
 */
template <typename FloatN, int M, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename Functor>
  __global__ void blasKernel(BlasArg<SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  arg.f.init();

  while (i < arg.length) {
    FloatN x[M], y[M], z[M], w[M];
    arg.X.load(x, i);
    arg.Y.load(y, i);
    arg.Z.load(z, i);
    arg.W.load(w, i);

#pragma unroll
    for (int j=0; j<M; j++) arg.f(x[j], y[j], z[j], w[j]);

    arg.X.save(x, i);
    arg.Y.save(y, i);
    arg.Z.save(z, i);
    arg.W.save(w, i);
    i += gridSize;
  }
}

template <typename FloatN, int M, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename Functor>
class BlasCuda : public Tunable {

private:
  mutable BlasArg<SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X_h, *Y_h, *Z_h, *W_h;
  char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h;
  const size_t *bytes_;
  const size_t *norm_bytes_;

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  virtual bool advanceSharedBytes(TuneParam &param) const
  {
    TuneParam next(param);
    advanceBlockDim(next); // to get next blockDim
    int nthreads = next.block.x * next.block.y * next.block.z;
    param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ?
      sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
    return false;
  }

public:
  BlasCuda(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor &f, 
	   int length, const size_t *bytes, const size_t *norm_bytes) :
  arg(X, Y, Z, W, f, length), X_h(0), Y_h(0), Z_h(0), W_h(0), 
    Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), bytes_(bytes), norm_bytes_(norm_bytes) { }

  virtual ~BlasCuda() { }

  inline TuneKey tuneKey() const { 
    return TuneKey(blasStrings.vol_str, typeid(arg.f).name(), blasStrings.aux_str);
  }

  inline void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    blasKernel<FloatN,M> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
  }

  void preTune() {
    arg.X.save(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.save(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.save(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.save(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
  }

  void postTune() {
    arg.X.load(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.load(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.load(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.load(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
  }

  long long flops() const { return arg.f.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }
  long long bytes() const { 
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
    return arg.f.streams()*bytes*arg.length; }
  int tuningIter() const { return 3; }
};

double2 make_Float2(const std::complex<double> &a) {
  return make_double2( real(a), imag(a) );
}

float2 make_Float2(const std::complex<float> &a) {
  return make_float2( real(a), imag(a) );
}

std::complex<double> make_Complex(const double2 &a) {
  return std::complex<double>(a.x, a.y);
}

std::complex<float> make_Complex(const float2 &a) {
  return std::complex<float>(a.x, a.y);
}


/**
   Generic blas kernel with four loads and up to four stores.

   FIXME - this is hacky due to the lack of std::complex support in
   CUDA.  The functors are defined in terms of FloatN vectors, whereas
   the operator() accessor returns std::complex<Float>
  */
template <typename Float2, int writeX, int writeY, int writeZ, int writeW, 
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, 
  typename Functor>
void genericBlas(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor f) {

  for (int parity=0; parity<X.Nparity(); parity++) {
    for (int x=0; x<X.VolumeCB(); x++) {
      for (int s=0; s<X.Nspin(); s++) {
	for (int c=0; c<X.Ncolor(); c++) {
	  Float2 X2 = make_Float2( X(parity, x, s, c) );
	  Float2 Y2 = make_Float2( Y(parity, x, s, c) );
	  Float2 Z2 = make_Float2( Z(parity, x, s, c) );
	  Float2 W2 = make_Float2( W(parity, x, s, c) );
	  f(X2, Y2, Z2, W2);
	  if (writeX) X(parity, x, s, c) = make_Complex(X2);
	  if (writeY) Y(parity, x, s, c) = make_Complex(Y2);
	  if (writeZ) Z(parity, x, s, c) = make_Complex(Z2);
	  if (writeW) W(parity, x, s, c) = make_Complex(W2);
	}
      }
    }
  }
}

template<typename, int N> struct vector { };
template<> struct vector<double, 2> { typedef double2 type; };
template<> struct vector<float, 2> { typedef float2 type; };

template <typename Float, int nSpin, int nColor, QudaFieldOrder order, 
  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, 
		   ColorSpinorField &w, Functor f) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Y(y), Z(z), W(w);
  typedef typename vector<Float,2>::type Float2;
  genericBlas<Float2,writeX,writeY,writeZ,writeW>(X, Y, Z, W, f);
}

template <typename Float, int nSpin, QudaFieldOrder order, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Ncolor() == 2) {
    genericBlas<Float,nSpin,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 3) {
    genericBlas<Float,nSpin,3,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 4) {
    genericBlas<Float,nSpin,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 8) {
    genericBlas<Float,nSpin,8,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 12) {
    genericBlas<Float,nSpin,12,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 16) {
    genericBlas<Float,nSpin,16,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 20) {
    genericBlas<Float,nSpin,20,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 24) {
    genericBlas<Float,nSpin,24,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 32) {
    genericBlas<Float,nSpin,32,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else {
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
}

template <typename Float, QudaFieldOrder order, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Nspin() == 4) {
    genericBlas<Float,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Nspin() == 2) {
    genericBlas<Float,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    genericBlas<Float,1,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
}

template <typename Float, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    genericBlas<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,Functor>
      (x, y, z, w, f);
  } else {
    errorQuda("Not implemeneted");
  }
}


/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
  int writeX, int writeY, int writeZ, int writeW>
  void blasCuda(const double2 &a, const double2 &b, const double2 &c,
		ColorSpinorField &x, ColorSpinorField &y, 
		ColorSpinorField &z, ColorSpinorField &w) {

  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);

  if (Location(x, y, z, w) == QUDA_CUDA_FIELD_LOCATION) {
    // FIXME this condition should be outside of the Location test but
    // Even and Odd must be implemented for cpu fields first
    if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      blasCuda<Functor,writeX,writeY,writeZ,writeW>
	(a, b, c, x.Even(), y.Even(), z.Even(), w.Even());
      blasCuda<Functor,writeX,writeY,writeZ,writeW>
	(a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd());
      return;
    }

    if (!x.isNative()) {
      warningQuda("Device blas on non-native fields is not supported\n");
      return;
    }

    blasStrings.vol_str = x.VolString();
    blasStrings.aux_str = x.AuxString();

    size_t bytes[] = {x.Bytes(), y.Bytes(), z.Bytes(), w.Bytes()};
    size_t norm_bytes[] = {x.NormBytes(), y.NormBytes(), z.NormBytes(), w.NormBytes()};

  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
    const int M = 1;
    Spinor<double2,double2,M,writeX,0> X(x);
    Spinor<double2,double2,M,writeY,1> Y(y);
    Spinor<double2,double2,M,writeZ,2> Z(z);
    Spinor<double2,double2,M,writeW,3> W(w);
    Functor<double2, double2> f(a,b,c);
    BlasCuda<double2,M,
      Spinor<double2,double2,M,writeX,0>, Spinor<double2,double2,M,writeY,1>,
      Spinor<double2,double2,M,writeZ,2>, Spinor<double2,double2,M,writeW,3>,
      Functor<double2, double2> > blas(X, Y, Z, W, f, x.Length()/(2*M), bytes, norm_bytes);
    blas.apply(*blasStream);
#else
    errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    if (x.Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = 1;
      Spinor<float4,float4,M,writeX,0> X(x);
      Spinor<float4,float4,M,writeY,1> Y(y);
      Spinor<float4,float4,M,writeZ,2> Z(z);
      Spinor<float4,float4,M,writeW,3> W(w);
      Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float4,M,
	Spinor<float4,float4,M,writeX,0>, Spinor<float4,float4,M,writeY,1>,
	Spinor<float4,float4,M,writeZ,2>, Spinor<float4,float4,M,writeW,3>,
	Functor<float2, float4> > blas(X, Y, Z, W, f, x.Length()/(4*M), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else if (x.Nspin()==2 || x.Nspin()==1) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
      const int M = 1;
      Spinor<float2,float2,M,writeX,0> X(x);
      Spinor<float2,float2,M,writeY,1> Y(y);
      Spinor<float2,float2,M,writeZ,2> Z(z);
      Spinor<float2,float2,M,writeW,3> W(w);
      Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float2,M,
	Spinor<float2,float2,M,writeX,0>, Spinor<float2,float2,M,writeY,1>,
	Spinor<float2,float2,M,writeZ,2>, Spinor<float2,float2,M,writeW,3>,
	Functor<float2, float2> > blas(X, Y, Z, W, f, x.Length()/(2*M), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x.Nspin()); }
  } else {
    if (x.Ncolor() != 3) { errorQuda("nColor = %d is not supported", x.Ncolor()); }
    if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
	Spinor<float4,short4,6,writeX,0> X(x);
	Spinor<float4,short4,6,writeY,1> Y(y);
	Spinor<float4,short4,6,writeZ,2> Z(z);
	Spinor<float4,short4,6,writeW,3> W(w);
	Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float4, 6, 
	  Spinor<float4,short4,6,writeX,0>, Spinor<float4,short4,6,writeY,1>,
	  Spinor<float4,short4,6,writeZ,2>, Spinor<float4,short4,6,writeW,3>,
	  Functor<float2, float4> > blas(X, Y, Z, W, f, y.Volume(), bytes, norm_bytes);
	blas.apply(*blasStream);
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
	Spinor<float2,short2,3,writeX,0> X(x);
	Spinor<float2,short2,3,writeY,1> Y(y);
	Spinor<float2,short2,3,writeZ,2> Z(z);
	Spinor<float2,short2,3,writeW,3> W(w);
	Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float2, 3,
	  Spinor<float2,short2,3,writeX,0>, Spinor<float2,short2,3,writeY,1>,
	  Spinor<float2,short2,3,writeZ,2>, Spinor<float2,short2,3,writeW,3>,
	  Functor<float2, float2> > blas(X, Y, Z, W, f, y.Volume(), bytes, norm_bytes);
	blas.apply(*blasStream);
#else
	errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else {
	errorQuda("nSpin=%d is not supported\n", x.Nspin());
      }
      blas::bytes += Functor<double2,double2>::streams()*(unsigned long long)x.Volume()*sizeof(float);
    }
  } else { // fields on the cpu
    using namespace quda::colorspinor;
    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      Functor<double2, double2> f(a, b, c);
      genericBlas<double, writeX, writeY, writeZ, writeW>(x, y, z, w, f);
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      Functor<float2, float2> 
	f(make_float2(a.x,a.y), make_float2(b.x,b.y), make_float2(c.x,c.y) );
      genericBlas<float, writeX, writeY, writeZ, writeW>(x, y, z, w, f);
    } else {
      errorQuda("Not implemented");
    }
  }

  bytes += Functor<double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  flops += Functor<double2,double2>::flops()*(unsigned long long)x.RealLength();
  
  checkCudaError();
}
  
