/**
   Generic blas kernel with four loads and up to four stores.
 */
template <typename FloatN, int M, int writeX, int writeY, int writeZ, int writeW, 
  typename InputX, typename InputY, typename InputZ, typename InputW, 
  typename OutputX, typename OutputY, typename OutputZ, typename OutputW, typename Functor>
__global__ void blasKernel(InputX X, InputY Y, InputZ Z, InputW W, Functor f, 
			   OutputX XX, OutputY YY, OutputZ ZZ, OutputW WW, int length) {
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

    if (writeX) XX.save(x, i);
    if (writeY) YY.save(y, i);
    if (writeZ) ZZ.save(z, i);
    if (writeW) WW.save(w, i);
    i += gridSize;
  }
}

template <typename FloatN, int M, int writeX, int writeY, int writeZ, int writeW, 
  typename InputX, typename InputY, typename InputZ, typename InputW, 
  typename OutputX, typename OutputY, typename OutputZ, typename OutputW, typename Functor>
class BlasCuda : public Tunable {

private:
  InputX &X;
  InputY &Y;
  InputZ &Z;
  InputW &W;
  OutputX &XX;
  OutputY &YY;
  OutputZ &ZZ;
  OutputW &WW;

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
  BlasCuda(InputX &X, InputY &Y, InputZ &Z, InputW &W, Functor &f, 
	   OutputX &XX, OutputY &YY, OutputZ &ZZ, OutputW &WW, int length) :
  X(X), Y(Y), Z(Z), W(W), f(f), XX(XX), YY(YY), ZZ(ZZ), WW(WW), length(length)
    { ; }
  virtual ~BlasCuda() { }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << blasConstants.x[0] << "x";
    vol << blasConstants.x[1] << "x";
    vol << blasConstants.x[2] << "x";
    vol << blasConstants.x[3];    
    aux << "stride=" << blasConstants.stride << ",prec=" << XX.Precision();
    return TuneKey(vol.str(), typeid(f).name(), aux.str());
  }  

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, blasTuning, verbosity);
    blasKernel<FloatN,M,writeX,writeY,writeZ,writeW>
      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>
      (X, Y, Z, W, f, XX, YY, ZZ, WW, length);
  }

  void preTune() {
    size_t bytes = XX.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*XX.Stride();
    size_t norm_bytes = (XX.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*length : 0;
    if (writeX) XX.save(&X_h, &Xnorm_h, bytes, norm_bytes);
    if (writeY) YY.save(&Y_h, &Ynorm_h, bytes, norm_bytes);
    if (writeZ) ZZ.save(&Z_h, &Znorm_h, bytes, norm_bytes);
    if (writeW) WW.save(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  void postTune() {
    size_t bytes = XX.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*XX.Stride();
    size_t norm_bytes = (XX.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*length : 0;
    if (writeX) XX.load(&X_h, &Xnorm_h, bytes, norm_bytes);
    if (writeY) YY.load(&Y_h, &Ynorm_h, bytes, norm_bytes);
    if (writeZ) ZZ.load(&Z_h, &Znorm_h, bytes, norm_bytes);
    if (writeW) WW.load(&W_h, &Wnorm_h, bytes, norm_bytes);
  }

  long long flops() const { return f.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*length*M; }
  long long bytes() const { 
    size_t bytes = XX.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (XX.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
    return f.streams()*bytes*length; }
};

/**
   Driver for generic blas routine with four loads and two store.
 */
template <template <typename Float, typename FloatN> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void blasCuda(const int kernel, const double2 &a, const double2 &b, const double2 &c,
	      cudaColorSpinorField &x, cudaColorSpinorField &y, 
	      cudaColorSpinorField &z, cudaColorSpinorField &w) {
  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);

  for (int d=0; d<QUDA_MAX_DIM; d++) blasConstants.x[d] = x.X()[d];
  blasConstants.stride = x.Stride();

  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (kernel, a, b, c, x.Even(), y.Even(), z.Even(), w.Even());
    blasCuda<Functor,writeX,writeY,writeZ,writeW>
      (kernel, a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd());
    return;
  }

  Tunable *blas = 0;
  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    const int M = 1;
    SpinorTexture<double2,double2,double2,M,0> xTex(x);
    SpinorTexture<double2,double2,double2,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<double2,double2,double2,M,1>(y);    
    SpinorTexture<double2,double2,double2,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<double2,double2,double2,M,2>(z);    
    SpinorTexture<double2,double2,double2,M,3> wTex;
    if (x.V() != w.V()) wTex = SpinorTexture<double2,double2,double2,M,3>(w);    
    Spinor<double2,double2,double2,M> X(x);
    Spinor<double2,double2,double2,M> Y(y);
    Spinor<double2,double2,double2,M> Z(z);
    Spinor<double2,double2,double2,M> W(w);
    Functor<double2, double2> f(a,b,c);
    blas = new BlasCuda<double2,M,writeX,writeY,writeZ,writeW,
      SpinorTexture<double2,double2,double2,M,0>, SpinorTexture<double2,double2,double2,M,1>,
      SpinorTexture<double2,double2,double2,M,2>, SpinorTexture<double2,double2,double2,M,3>,
      Spinor<double2,double2,double2,M>, Spinor<double2,double2,double2,M>, 
      Spinor<double2,double2,double2,M>, Spinor<double2,double2,double2,M>, Functor<double2, double2> >
      (xTex, yTex, zTex, wTex, f, X, Y, Z, W, x.Length()/(2*M));
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    const int M = 1;
    SpinorTexture<float4,float4,float4,M,0> xTex(x);
    SpinorTexture<float4,float4,float4,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,float4,M,1>(y);
    SpinorTexture<float4,float4,float4,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,float4,M,2>(z);
    SpinorTexture<float4,float4,float4,M,3> wTex;
    if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,float4,M,3>(w);
    Spinor<float4,float4,float4,M> X(x);
    Spinor<float4,float4,float4,M> Y(y);
    Spinor<float4,float4,float4,M> Z(z);
    Spinor<float4,float4,float4,M> W(w);
    Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
    blas = new BlasCuda<float4,M,writeX,writeY,writeZ,writeW,
      SpinorTexture<float4,float4,float4,M,0>, SpinorTexture<float4,float4,float4,M,1>, 
      SpinorTexture<float4,float4,float4,M,2>, SpinorTexture<float4,float4,float4,M,3>, 
      Spinor<float4,float4,float4,M>, Spinor<float4,float4,float4,M>, 
      Spinor<float4,float4,float4,M>, Spinor<float4,float4,float4,M>, Functor<float2, float4> >
      (xTex, yTex, zTex, wTex, f, X, Y, Z, W, x.Length()/(4*M));
  } else {
    if (x.Nspin() == 4){ //wilson
      SpinorTexture<float4,float4,short4,6,0> xTex(x);
      SpinorTexture<float4,float4,short4,6,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,short4,6,1>(y);
      SpinorTexture<float4,float4,short4,6,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,short4,6,2>(z);
      SpinorTexture<float4,float4,short4,6,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,short4,6,3>(w);
      Spinor<float4,float4,short4,6> xStore(x);
      Spinor<float4,float4,short4,6> yStore(y);
      Spinor<float4,float4,short4,6> zStore(z);
      Spinor<float4,float4,short4,6> wStore(w);
      Functor<float2, float4> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      blas = new BlasCuda<float4, 6, writeX, writeY, writeZ, writeW, 
	SpinorTexture<float4,float4,short4,6,0>, SpinorTexture<float4,float4,short4,6,1>, 
	SpinorTexture<float4,float4,short4,6,2>, SpinorTexture<float4,float4,short4,6,3>, 
	Spinor<float4,float4,short4,6>, Spinor<float4,float4,short4,6>, 
	Spinor<float4,float4,short4,6>,	Spinor<float4,float4,short4,6>, Functor<float2, float4> >
	(xTex, yTex, zTex, wTex, f, xStore, yStore, zStore, wStore, y.Volume());
    } else if (x.Nspin() == 1) {//staggered
      SpinorTexture<float2,float2,short2,3,0> xTex(x);
      SpinorTexture<float2,float2,short2,3,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float2,float2,short2,3,1>(y);
      SpinorTexture<float2,float2,short2,3,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float2,float2,short2,3,2>(z);
      SpinorTexture<float2,float2,short2,3,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float2,float2,short2,3,3>(w);
      Spinor<float2,float2,short2,3> xStore(x);
      Spinor<float2,float2,short2,3> yStore(y);
      Spinor<float2,float2,short2,3> zStore(z);
      Spinor<float2,float2,short2,3> wStore(w);
      Functor<float2, float2> f(make_float2(a.x, a.y), make_float2(b.x, b.y), make_float2(c.x, c.y));
      blas = new BlasCuda<float2, 3,writeX,writeY,writeZ,writeW,
	SpinorTexture<float2,float2,short2,3,0>, SpinorTexture<float2,float2,short2,3,1>,
	SpinorTexture<float2,float2,short2,3,2>, SpinorTexture<float2,float2,short2,3,3>,
	Spinor<float2,float2,short2,3>, Spinor<float2,float2,short2,3>,
	Spinor<float2,float2,short2,3>, Spinor<float2,float2,short2,3>, Functor<float2, float2> >
	(xTex, yTex, zTex, wTex, f, xStore, yStore, zStore, wStore, y.Volume());
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    quda::blas_bytes += Functor<double2,double2>::streams()*x.Volume()*sizeof(float);
  }
  quda::blas_bytes += Functor<double2,double2>::streams()*x.RealLength()*x.Precision();
  quda::blas_flops += Functor<double2,double2>::flops()*x.RealLength();

  blas->apply(*blasStream);
  delete blas;

  checkCudaError();
}

