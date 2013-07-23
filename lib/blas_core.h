/**
   Generic blas kernel with four loads and up to four stores.
 */
template <typename FloatN, int M, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename Functor>
__global__ void blasKernel(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, Functor f, 
			   int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    FloatN x[M], y[M], z[M], w[M];
    X.load(x, i);
    Y.load(y, i);
    Z.load(z, i);
    W.load(w, i);

#pragma unroll
    for (int j=0; j<M; j++) f(x[j], y[j], z[j], w[j]);

    X.save(x, i);
    Y.save(y, i);
    Z.save(z, i);
    W.save(w, i);
    i += gridSize;
  }
}

template <typename FloatN, int M, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename Functor>
class BlasCuda : public Tunable {

private:
  SpinorX &X;
  SpinorY &Y;
  SpinorZ &Z;
  SpinorW &W;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X_h, *Y_h, *Z_h, *W_h;
  char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h;

  Functor &f;
  const int length;

  int sharedBytesPerThread() const { return 0; }
  int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

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
	   int length) :
  X(X), Y(Y), Z(Z), W(W), f(f), X_h(0), Y_h(0), Z_h(0), W_h(0), 
  Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), length(length)
    { ; }
  virtual ~BlasCuda() { }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << blasConstants.x[0] << "x";
    vol << blasConstants.x[1] << "x";
    vol << blasConstants.x[2] << "x";
    vol << blasConstants.x[3];    
    aux << "stride=" << blasConstants.stride << ",prec=" << X.Precision();
    return TuneKey(vol.str(), typeid(f).name(), aux.str());
  }  

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, blasTuning, verbosity);
    blasKernel<FloatN,M> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>
      (X, Y, Z, W, f, length);
  }

  void preTune() {
    size_t bytes = X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*X.Stride();
    size_t norm_bytes = (X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*length : 0;
    X.save(&X_h, &Xnorm_h, bytes, norm_bytes);
    Y.save(&Y_h, &Ynorm_h, bytes, norm_bytes);
    Z.save(&Z_h, &Znorm_h, bytes, norm_bytes);
    W.save(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  void postTune() {
    size_t bytes = X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*X.Stride();
    size_t norm_bytes = (X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*length : 0;
    X.load(&X_h, &Xnorm_h, bytes, norm_bytes);
    Y.load(&Y_h, &Ynorm_h, bytes, norm_bytes);
    Z.load(&Z_h, &Znorm_h, bytes, norm_bytes);
    W.load(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  long long flops() const { return f.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*length*M; }
  long long bytes() const { 
    size_t bytes = X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (X.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
    return f.streams()*bytes*length; }
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

  for (int x=0; x<X.Volume(); x++) {
    for (int s=0; s<X.Nspin(); s++) {
      for (int c=0; c<X.Ncolor(); c++) {
	Float2 X2 = make_Float2( X(x, s, c) );
	Float2 Y2 = make_Float2( Y(x, s, c) );
	Float2 Z2 = make_Float2( Z(x, s, c) );
	Float2 W2 = make_Float2( W(x, s, c) );
	f(X2, Y2, Z2, W2);
	if (writeX) X(x, s, c) = make_Complex(X2);
	if (writeY) Y(x, s, c) = make_Complex(Y2);
	if (writeZ) Z(x, s, c) = make_Complex(Z2);
	if (writeW) W(x, s, c) = make_Complex(W2);
      }
    }
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

    for (int d=0; d<QUDA_MAX_DIM; d++) blasConstants.x[d] = x.X()[d];
    blasConstants.stride = x.Stride();

  // FIXME: use traits to encapsulate register type for shorts -
  // will reduce template type parameters from 3 to 2

    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      const int M = 1;
      Spinor<double2,double2,double2,M,writeX,0> X(x);
      Spinor<double2,double2,double2,M,writeY,1> Y(y);
      Spinor<double2,double2,double2,M,writeZ,2> Z(z);
      Spinor<double2,double2,double2,M,writeW,3> W(w);
      Functor<double2, double2> f(a,b,c);
      BlasCuda<double2,M,
	Spinor<double2,double2,double2,M,writeX,0>, Spinor<double2,double2,double2,M,writeY,1>,
	Spinor<double2,double2,double2,M,writeZ,2>, Spinor<double2,double2,double2,M,writeW,3>,
	Functor<double2, double2> > blas(X, Y, Z, W, f, x.Length()/(2*M));
      blas.apply(*blasStream);
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      const int M = 1;
      if (x.Nspin() == 4) {
	Spinor<float4,float4,float4,M,writeX,0> X(x);
	Spinor<float4,float4,float4,M,writeY,1> Y(y);
	Spinor<float4,float4,float4,M,writeZ,2> Z(z);
	Spinor<float4,float4,float4,M,writeW,3> W(w);
	Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float4,M,
	  Spinor<float4,float4,float4,M,writeX,0>, Spinor<float4,float4,float4,M,writeY,1>, 
	  Spinor<float4,float4,float4,M,writeZ,2>, Spinor<float4,float4,float4,M,writeW,3>, 
	  Functor<float2, float4> > blas(X, Y, Z, W, f, x.Length()/(4*M));
	blas.apply(*blasStream);
      } else {
	Spinor<float2,float2,float2,M,writeX,0> X(x);
	Spinor<float2,float2,float2,M,writeY,1> Y(y);
	Spinor<float2,float2,float2,M,writeZ,2> Z(z);
	Spinor<float2,float2,float2,M,writeW,3> W(w);
	Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float2,M,
	  Spinor<float2,float2,float2,M,writeX,0>, Spinor<float2,float2,float2,M,writeY,1>, 
	  Spinor<float2,float2,float2,M,writeZ,2>, Spinor<float2,float2,float2,M,writeW,3>, 
	  Functor<float2, float2> > blas(X, Y, Z, W, f, x.Length()/(2*M));
	blas.apply(*blasStream);
      }
    } else {
      if (x.Ncolor() != 3) { errorQuda("Not supported"); }
      if (x.Nspin() == 4){ //wilson
	Spinor<float4,float4,short4,6,writeX,0> X(x);
	Spinor<float4,float4,short4,6,writeY,1> Y(y);
	Spinor<float4,float4,short4,6,writeZ,2> Z(z);
	Spinor<float4,float4,short4,6,writeW,3> W(w);
	Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float4, 6, 
	  Spinor<float4,float4,short4,6,writeX,0>, Spinor<float4,float4,short4,6,writeY,1>, 
	  Spinor<float4,float4,short4,6,writeZ,2>, Spinor<float4,float4,short4,6,writeW,3>, 
	  Functor<float2, float4> > blas(X, Y, Z, W, f, y.Volume());
	blas.apply(*blasStream);
      } else if (x.Nspin() == 1) {//staggered
	Spinor<float2,float2,short2,3,writeX,0> X(x);
	Spinor<float2,float2,short2,3,writeY,1> Y(y);
	Spinor<float2,float2,short2,3,writeZ,2> Z(z);
	Spinor<float2,float2,short2,3,writeW,3> W(w);
	Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
	BlasCuda<float2, 3,
	  Spinor<float2,float2,short2,3,writeX,0>, Spinor<float2,float2,short2,3,writeY,1>,
	  Spinor<float2,float2,short2,3,writeZ,2>, Spinor<float2,float2,short2,3,writeW,3>,
	  Functor<float2, float2> > blas(X, Y, Z, W, f, y.Volume());
	blas.apply(*blasStream);
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
      bytes += Functor<double2,double2>::streams()*(unsigned long long)x.Volume()*sizeof(float);
    }
  } else { // fields on the cpu
    using namespace quda::colorspinor;
    if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      FieldOrder<double> *X = createOrder<double>(x);
      FieldOrder<double> *Y = createOrder<double>(y);
      FieldOrder<double> *Z = createOrder<double>(z);
      FieldOrder<double> *W = createOrder<double>(w);
      Functor<double2, double2> f(a, b, c);
      genericBlas<double2, writeX, writeY, writeZ, writeW>(*X, *Y, *Z, *W, f);
      delete X;
      delete Y;
      delete Z;
      delete W;
    } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
      FieldOrder<float> *X = createOrder<float>(x);
      FieldOrder<float> *Y = createOrder<float>(y);
      FieldOrder<float> *Z = createOrder<float>(z);
      FieldOrder<float> *W = createOrder<float>(w);
      Functor<float2, float2> 
	f(make_float2(a.x,a.y), make_float2(b.x,b.y), make_float2(c.x,c.y) );
      genericBlas<float2, writeX, writeY, writeZ, writeW>(*X, *Y, *Z, *W, f);
      delete X;
      delete Y;
      delete Z;
      delete W;
    } else {
      errorQuda("Not implemented");
    }
  }

  bytes += Functor<double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  flops += Functor<double2,double2>::flops()*(unsigned long long)x.RealLength();
  
  checkCudaError();
}
  
