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

/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
  int writeX, int writeY, int writeZ, int writeW>
inline void blasCuda(const double2 &a, const double2 &b, const double2 &c,
		     cudaColorSpinorField &x, cudaColorSpinorField &y, 
		     cudaColorSpinorField &z, cudaColorSpinorField &w) {

  static TimeProfile head("head");

  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);

  if (!x.isNative()) {
    warningQuda("Blas on non-native fields is not supported\n");
    return;
  }

  blasStrings.vol_str = x.VolString();
  blasStrings.aux_str = x.AuxString();

  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Even(), y.Even(), z.Even(), w.Even());
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd());
    return;
  }

  // FIXME: use traits to encapsulate register type for shorts -
  // will reduce template type parameters from 3 to 2

  size_t bytes[] = {x.Bytes(), y.Bytes(), z.Bytes(), w.Bytes()};
  size_t norm_bytes[] = {x.NormBytes(), y.NormBytes(), z.NormBytes(), w.NormBytes()};

  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
    const int M = 1;
    Spinor<double2,double2,double2,M,writeX,0> X(x);
    Spinor<double2,double2,double2,M,writeY,1> Y(y);
    Spinor<double2,double2,double2,M,writeZ,2> Z(z);
    Spinor<double2,double2,double2,M,writeW,3> W(w);
    Functor<double2, double2> f(a,b,c);
    BlasCuda<double2,M,
      Spinor<double2,double2,double2,M,writeX,0>, Spinor<double2,double2,double2,M,writeY,1>,
      Spinor<double2,double2,double2,M,writeZ,2>, Spinor<double2,double2,double2,M,writeW,3>,
      Functor<double2, double2> > blas(X, Y, Z, W, f, x.Length()/(2*M), bytes, norm_bytes);
    blas.apply(*blasStream);
#else
    errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    const int M = 1;
    if (x.Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      Spinor<float4,float4,float4,M,writeX,0> X(x);
      Spinor<float4,float4,float4,M,writeY,1> Y(y);
      Spinor<float4,float4,float4,M,writeZ,2> Z(z);
      Spinor<float4,float4,float4,M,writeW,3> W(w);
      Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float4,M,
	Spinor<float4,float4,float4,M,writeX,0>, Spinor<float4,float4,float4,M,writeY,1>, 
	Spinor<float4,float4,float4,M,writeZ,2>, Spinor<float4,float4,float4,M,writeW,3>, 
	Functor<float2, float4> > blas(X, Y, Z, W, f, x.Length()/(4*M), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else {
#ifdef GPU_STAGGERED_DIRAC
      Spinor<float2,float2,float2,M,writeX,0> X(x);
      Spinor<float2,float2,float2,M,writeY,1> Y(y);
      Spinor<float2,float2,float2,M,writeZ,2> Z(z);
      Spinor<float2,float2,float2,M,writeW,3> W(w);
      Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float2,M,
	Spinor<float2,float2,float2,M,writeX,0>, Spinor<float2,float2,float2,M,writeY,1>, 
	Spinor<float2,float2,float2,M,writeZ,2>, Spinor<float2,float2,float2,M,writeW,3>, 
	Functor<float2, float2> > blas(X, Y, Z, W, f, x.Length()/(2*M), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    }
  } else {
    if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      Spinor<float4,float4,short4,6,writeX,0> X(x);
      Spinor<float4,float4,short4,6,writeY,1> Y(y);
      Spinor<float4,float4,short4,6,writeZ,2> Z(z);
      Spinor<float4,float4,short4,6,writeW,3> W(w);
      Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float4, 6, 
	Spinor<float4,float4,short4,6,writeX,0>, Spinor<float4,float4,short4,6,writeY,1>, 
	Spinor<float4,float4,short4,6,writeZ,2>, Spinor<float4,float4,short4,6,writeW,3>, 
	Functor<float2, float4> > blas(X, Y, Z, W, f, y.Volume(), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
      Spinor<float2,float2,short2,3,writeX,0> X(x);
      Spinor<float2,float2,short2,3,writeY,1> Y(y);
      Spinor<float2,float2,short2,3,writeZ,2> Z(z);
      Spinor<float2,float2,short2,3,writeW,3> W(w);
      Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      BlasCuda<float2, 3,
	Spinor<float2,float2,short2,3,writeX,0>, Spinor<float2,float2,short2,3,writeY,1>,
	Spinor<float2,float2,short2,3,writeZ,2>, Spinor<float2,float2,short2,3,writeW,3>,
	Functor<float2, float2> > blas(X, Y, Z, W, f, y.Volume(), bytes, norm_bytes);
      blas.apply(*blasStream);
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    blas_bytes += Functor<double2,double2>::streams()*(unsigned long long)x.Volume()*sizeof(float);
  }

  blas_bytes += Functor<double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  blas_flops += Functor<double2,double2>::flops()*(unsigned long long)x.RealLength();

  checkCudaError();
}

