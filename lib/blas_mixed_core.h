namespace mixed {

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
	   int length) :
  arg(X, Y, Z, W, f, length), X_h(0), Y_h(0), Z_h(0), W_h(0), 
  Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0)
    { ; }
  virtual ~BlasCuda() { }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << blasConstants.x[0] << "x";
    vol << blasConstants.x[1] << "x";
    vol << blasConstants.x[2] << "x";
    vol << blasConstants.x[3];    
    aux << "stride=" << blasConstants.stride << ",prec=" << arg.X.Precision();
    return TuneKey(vol.str(), typeid(arg.f).name(), aux.str());
  }  

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    blasKernel<FloatN,M> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
  }

  void preTune() {
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.X.Stride();
    size_t norm_bytes = (arg.X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    size_t ybytes = arg.Y.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.Y.Stride();
    size_t ynorm_bytes = (arg.Y.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    arg.X.save(&X_h, &Xnorm_h, bytes, norm_bytes);
    arg.Y.save(&Y_h, &Ynorm_h, ybytes, ynorm_bytes);
    arg.Z.save(&Z_h, &Znorm_h, bytes, norm_bytes);
    arg.W.save(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  void postTune() {
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.X.Stride();
    size_t norm_bytes = (arg.X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    size_t ybytes = arg.Y.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.Y.Stride();
    size_t ynorm_bytes = (arg.Y.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    arg.X.load(&X_h, &Xnorm_h, bytes, norm_bytes);
    arg.Y.load(&Y_h, &Ynorm_h, ybytes, ynorm_bytes);
    arg.Z.load(&Z_h, &Znorm_h, bytes, norm_bytes);
    arg.W.load(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  long long flops() const { return arg.f.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }
  long long bytes() const { 
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
    return arg.f.streams()*bytes*arg.length; }
};

/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
  int writeX, int writeY, int writeZ, int writeW>
void blasCuda(const double2 &a, const double2 &b, const double2 &c,
	      cudaColorSpinorField &x, cudaColorSpinorField &y, 
	      cudaColorSpinorField &z, cudaColorSpinorField &w) {
  checkLength(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);

  if (!x.isNative()) {
    warningQuda("Blas on non-native fields is not supported\n");
    return;
  }

  for (int d=0; d<QUDA_MAX_DIM; d++) blasConstants.x[d] = x.X()[d];
  blasConstants.stride = x.Stride();

  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    mixed::blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Even(), y.Even(), z.Even(), w.Even());
    mixed::blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd());
    return;
  }

  // FIXME: use traits to encapsulate register type for shorts -
  // will reduce template type parameters from 3 to 2

  if (x.Nspin() != 1) errorQuda("Not supported");

  const int M = 3;
  if (x.Precision() == QUDA_SINGLE_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
    Spinor<double2,double2,float2,M,writeX,0> X(x);
    Spinor<double2,double2,double2,M,writeY,1> Y(y);
    Spinor<double2,double2,float2,M,writeZ,2> Z(z);
    Spinor<double2,double2,float2,M,writeW,3> W(w);
    Functor<double2, double2> f(a, b, c);
    BlasCuda<double2, M,
      Spinor<double2,double2,float2,M,writeX,0>, Spinor<double2,double2,double2,M,writeY,1>,
      Spinor<double2,double2,float2,M,writeZ,2>, Spinor<double2,double2,float2,M,writeW,3>,
      Functor<double2, double2> > blas(X, Y, Z, W, f, y.Volume());
    blas.apply(*blasStream);
  } else if (x.Precision() == QUDA_HALF_PRECISION && y.Precision() == QUDA_DOUBLE_PRECISION) {
    Spinor<double2,double2,short2,M,writeX,0> X(x);
    Spinor<double2,double2,double2,M,writeY,1> Y(y);
    Spinor<double2,double2,short2,M,writeZ,2> Z(z);
    Spinor<double2,double2,short2,M,writeW,3> W(w);
    Functor<double2, double2> f(a, b, c);
    BlasCuda<double2, M,
      Spinor<double2,double2,short2,M,writeX,0>, Spinor<double2,double2,double2,M,writeY,1>,
      Spinor<double2,double2,short2,M,writeZ,2>, Spinor<double2,double2,short2,M,writeW,3>,
      Functor<double2, double2> > blas(X, Y, Z, W, f, y.Volume());
    blas.apply(*blasStream);
  } else if (y.Precision() == QUDA_SINGLE_PRECISION) {
    Spinor<float2,float2,short2,M,writeX,0> X(x);
    Spinor<float2,float2,float2,M,writeY,1> Y(y);
    Spinor<float2,float2,short2,M,writeZ,2> Z(z);
    Spinor<float2,float2,short2,M,writeW,3> W(w);
    Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
    BlasCuda<float2, 3,
      Spinor<float2,float2,short2,M,writeX,0>, Spinor<float2,float2,float2,M,writeY,1>,
      Spinor<float2,float2,short2,M,writeZ,2>, Spinor<float2,float2,short2,M,writeW,3>,
      Functor<float2, float2> > blas(X, Y, Z, W, f, y.Volume());
    blas.apply(*blasStream);
  } else {
    errorQuda("Not implemented for this precision combination");
  }

  blas_bytes += Functor<double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  blas_flops += Functor<double2,double2>::flops()*(unsigned long long)x.RealLength();

  checkCudaError();
}

}
