/**
  Parameter struct for generic blas kernel
*/
// NXZ is dimension of input vectors: X,Z
// NYW is dimension of in-output vectors: Y,W
// need to check all loops. Not everything here needs to be templated
// some NXZ, NYW should be be runtime args

template <int NXZ, int NYW, typename SpinorX, typename SpinorY, typename SpinorZ,
  typename SpinorW, typename Functor>
struct MultBlasArg {
  SpinorX X[NXZ];
  SpinorY Y[NYW];
  SpinorZ Z[NXZ];
  SpinorW W[NYW];
  Functor f;
  const int length;
  const int out_dim;
  const int in_dim;
  MultBlasArg(SpinorX X[NXZ], SpinorY Y[NYW], SpinorZ Z[NXZ], SpinorW W[NYW], Functor f, int length)
  :  f(f), length(length), in_dim(NXZ), out_dim(NYW){

  for(int i=0; i<in_dim; ++i){
    this->X[i] = X[i];
    this->Z[i] = Z[i];
  }
  for(int i=0; i<out_dim; ++i){
    this->Y[i] = Y[i];
    this->W[i] = W[i];
  }
}
};

/**
Generic blas kernel with four loads and up to four stores.
*/
template <typename FloatN, int M, int NXZ, int NYW, typename SpinorX, typename SpinorY,
typename SpinorZ, typename SpinorW, typename Functor>
__global__ void multblasKernel(MultBlasArg<NXZ, NYW, SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSizex = gridDim.x*blockDim.x;
  unsigned int gridSizey = gridDim.y*blockDim.y;
  unsigned int k = blockIdx.y*blockDim.y + threadIdx.y;

  arg.f.init();

  while (k < arg.out_dim){
    while (i < arg.length) {
      FloatN x[M], y[M], z[M], w[M];

      arg.Y[k].load(y, i);
      arg.W[k].load(w, i);

      for (unsigned int in_idx=0; i < arg.in_dim; in_idx++ ){
        arg.X[in_idx].load(x, i);
        arg.Z[in_idx].load(z, i);


        #pragma unroll
        for (int j=0; j<M; j++) arg.f(x[j], y[j], z[j], w[j], i, k);
      }
      // arg.X[out_idx].save(x, i);
      arg.Y[k].save(y, i);
      // arg.Z[out_idx].save(z, i);
      arg.W[k].save(w, i);


      i += gridSizex;
    }
    k += gridSizey;
  }
}

namespace detail
{
    template<unsigned... digits>
      struct to_chars { static const char value[]; };

    template<unsigned... digits>
      const char to_chars<digits...>::value[] = {('0' + digits)..., 0};

    template<unsigned rem, unsigned... digits>
      struct explode : explode<rem / 10, rem % 10, digits...> {};

    template<unsigned... digits>
      struct explode<0, digits...> : to_chars<digits...> {};
}

template<unsigned num>
struct num_to_string : detail::explode<num / 10, num % 10> {};


template <int NXZ, int NYW, typename FloatN, int M, typename SpinorX, typename SpinorY,
  typename SpinorZ, typename SpinorW, typename Functor>
class MultBlasCuda : public Tunable {

private:
  mutable MultBlasArg<NXZ, NYW,SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  // host pointer used for backing up fields when tuning
  char *X_h[NXZ], *Y_h[NYW], *Z_h[NXZ], *W_h[NYW];
  char *Xnorm_h[NXZ], *Ynorm_h[NYW], *Znorm_h[NXZ], *Wnorm_h[NYW];
  const size_t **bytes_;
  const size_t **norm_bytes_;

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
  MultBlasCuda(SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Functor &f,
	   int length,  size_t **bytes,  size_t **norm_bytes) :
  arg(X, Y, Z, W, f, length), X_h(), Y_h(), Z_h(), W_h(),
    Xnorm_h(), Ynorm_h(), Znorm_h(), Wnorm_h(), bytes_(const_cast<const size_t**>(bytes)), norm_bytes_(const_cast<const size_t**>(norm_bytes)) { }

  virtual ~MultBlasCuda() { }

  inline TuneKey tuneKey() const {
    char name[TuneKey::name_n];
    strcpy(name, num_to_string<NXZ>::value);
    strcat(name, num_to_string<NYW>::value);
    strcat(name, typeid(arg.f).name());
    return TuneKey(blasStrings.vol_str, name, blasStrings.aux_str);
  }

  inline void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

    multblasKernel<FloatN,M> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
  }

  bool advanceGridDim(TuneParam &param) const {
      bool rtn = Tunable::advanceGridDim(param);
    if (NYW > deviceProp.maxGridSize[1]) errorQuda("N=%d is greater than the maximum support grid size", NYW);
    param.grid.y = NYW;
    return rtn;
  }

  void preTune() {
    for(int i=0; i<NXZ; ++i){
      arg.X[i].save(&X_h[i], &Xnorm_h[i], bytes_[i][0], norm_bytes_[i][0]);
      arg.Z[i].save(&Z_h[i], &Znorm_h[i], bytes_[i][2], norm_bytes_[i][2]);
    }
    for(int i=0; i<NYW; ++i){
      arg.Y[i].save(&Y_h[i], &Ynorm_h[i], bytes_[i][1], norm_bytes_[i][1]);
      arg.W[i].save(&W_h[i], &Wnorm_h[i], bytes_[i][3], norm_bytes_[i][3]);
    }
  }

  void postTune() {
    for(int i=0; i<NXZ; ++i){
      arg.X[i].load(&X_h[i], &Xnorm_h[i], bytes_[i][0], norm_bytes_[i][0]);
      arg.Z[i].load(&Z_h[i], &Znorm_h[i], bytes_[i][2], norm_bytes_[i][2]);
    }
    for(int i=0; i<NYW; ++i){
      arg.Y[i].load(&Y_h[i], &Ynorm_h[i], bytes_[i][1], norm_bytes_[i][1]);
      arg.W[i].load(&W_h[i], &Wnorm_h[i], bytes_[i][3], norm_bytes_[i][3]);
    }
  }




  long long flops() const { return arg.f.flops()*vec_length<FloatN>::value*arg.length*M; }
  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.X[0].Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.Y[0].Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    return ((arg.f.streams()-2)*base_bytes + 2*extra_bytes)*arg.length;
  }
  int tuningIter() const { return 3; }
};

template <typename RegType, typename StoreType, typename yType, int M,
	  template <int,int,typename,typename> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void multblasCuda(const Complex *a, const double2 &b, const double2 &c,
CompositeColorSpinorField& x, CompositeColorSpinorField& y,
  CompositeColorSpinorField& z, CompositeColorSpinorField& w,
         int length) {
           const int N=1;
  // FIXME implement this as a single kernel
  if (x[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    std::vector<cudaColorSpinorField> xp, yp, zp, wp;
    std::vector<ColorSpinorField*> xpp, ypp, zpp, wpp;
    xp.reserve(N); yp.reserve(N); zp.reserve(N); wp.reserve(N);
    xpp.reserve(N); ypp.reserve(N); zpp.reserve(N); wpp.reserve(N);

    for (int i=0; i<N; i++) {
      xp.push_back(x[i]->Even()); yp.push_back(y[i]->Even()); zp.push_back(z[i]->Even());
      wp.push_back(w[i]->Even());
      xpp.push_back(&xp[i]); ypp.push_back(&yp[i]); zpp.push_back(&zp[i]); wpp.push_back(&wp[i]);
    }

    multblasCuda<RegType, StoreType, yType, M, Functor, writeX, writeY, writeZ, writeW>
      (a, b, c, xpp, ypp, zpp, wpp, length);

    for (int i=0; i<N; ++i) {
      xp.push_back(x[i]->Odd()); yp.push_back(y[i]->Odd()); zp.push_back(z[i]->Odd());
      wp.push_back(w[i]->Odd());
      xpp.push_back(&xp[i]); ypp.push_back(&yp[i]); zpp.push_back(&zp[i]); wpp.push_back(&wp[i]);
    }

    multblasCuda<RegType, StoreType, yType, M, Functor, writeX, writeY, writeZ, writeW>
       (a, b, c, xpp, ypp, zpp, wpp, length);

    return;
  }

  // for (int i=0; i<N; i++) {
  //   checkSpinor(*x[i],*y[i]); checkSpinor(*x[i],*z[i]); checkSpinor(*x[i],*w[i]);
  //   if (!x[i]->isNative()) {
  //     warningQuda("Reductions on non-native fields are not supported\n");
  //     return;
  //   }
  // }

  blasStrings.vol_str = x[0]->VolString();
  strcpy(blasStrings.aux_tmp, x[0]->AuxString());
  if (typeid(StoreType) != typeid(yType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, y[0]->AuxString());
  }

  size_t **bytes = new size_t*[N], **norm_bytes = new size_t*[N];
  for (int i=0; i<N; i++) {
    bytes[i] = new size_t[4]; norm_bytes[i] = new size_t[4];
    bytes[i][0] = x[i]->Bytes(); bytes[i][1] = y[i]->Bytes(); bytes[i][2] = z[i]->Bytes();
    bytes[i][3] = w[i]->Bytes();
    norm_bytes[i][0] = x[i]->NormBytes(); norm_bytes[i][1] = y[i]->NormBytes(); norm_bytes[i][2] = z[i]->NormBytes();
    norm_bytes[i][3] = w[i]->NormBytes();
  }

  Spinor<RegType,StoreType,M,writeX,0> X[N];
  Spinor<RegType,    yType,M,writeY,1> Y[N];
  Spinor<RegType,StoreType,M,writeZ,2> Z[N];
  Spinor<RegType,StoreType,M,writeW,3> W[N];

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  //MWFIXME
  for (int i=0; i<N; i++) { X[i].set(*x[i]); Y[i].set(*y[i]); Z[i].set(*z[i]); W[i].set(*w[i]);}

  Functor<1,1,Float2, RegType> f( a, (Float2)vec2(b), (Float2)vec2(c));

  MultBlasCuda<1,1,RegType,M,
    Spinor<RegType,StoreType,M,writeX,0>,
    Spinor<RegType,    yType,M,writeY,1>,
    Spinor<RegType,StoreType,M,writeZ,2>,
    Spinor<RegType,StoreType,M,writeW,3>,
    decltype(f) >
    blas(X, Y, Z, W, f, length, bytes, norm_bytes);
  blas.apply(*blasStream);

  blas::bytes += blas.bytes();
  blas::flops += blas.flops();

  checkCudaError();
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
void genericMultBlas(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor f) {

  for (int parity=0; parity<X.Nparity(); parity++) {
    for (int x=0; x<X.VolumeCB(); x++) {
      for (int s=0; s<X.Nspin(); s++) {
	for (int c=0; c<X.Ncolor(); c++) {
	  Float2 X2 = make_Float2<Float2>( X(parity, x, s, c) );
	  Float2 Y2 = make_Float2<Float2>( Y(parity, x, s, c) );
	  Float2 Z2 = make_Float2<Float2>( Z(parity, x, s, c) );
	  Float2 W2 = make_Float2<Float2>( W(parity, x, s, c) );
    f(X2, Y2, Z2, W2, 1 , 1);
    // if (writeX) X(parity, x, s, c) = make_Complex(X2);
    if (writeX) errorQuda("writeX not supported in multblas.");
    if (writeY) Y(parity, x, s, c) = make_Complex(Y2);
	  // if (writeZ) Z(parity, x, s, c) = make_Complex(Z2);
    if (writeX) errorQuda("writeZ not supported in multblas.");
	  if (writeW) W(parity, x, s, c) = make_Complex(W2);
	}
      }
    }
  }
}

template <typename Float, typename yFloat, int nSpin, int nColor, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
		   ColorSpinorField &w, Functor f) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Z(z), W(w);
  colorspinor::FieldOrderCB<yFloat,nSpin,nColor,1,order> Y(y);
  typedef typename vector<yFloat,2>::type Float2;
  genericMultBlas<Float2,writeX,writeY,writeZ,writeW>(X, Y, Z, W, f);
}

template <typename Float, typename yFloat, int nSpin, QudaFieldOrder order,
	  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Ncolor() == 2) {
    genericMultBlas<Float,yFloat,nSpin,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 3) {
    genericMultBlas<Float,yFloat,nSpin,3,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 4) {
    genericMultBlas<Float,yFloat,nSpin,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 8) {
    genericMultBlas<Float,yFloat,nSpin,8,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 12) {
    genericMultBlas<Float,yFloat,nSpin,12,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 16) {
    genericMultBlas<Float,yFloat,nSpin,16,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 20) {
    genericMultBlas<Float,yFloat,nSpin,20,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 24) {
    genericMultBlas<Float,yFloat,nSpin,24,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 32) {
    genericMultBlas<Float,yFloat,nSpin,32,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else {
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
}

template <typename Float, typename yFloat, QudaFieldOrder order, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Nspin() == 4) {
    genericMultBlas<Float,yFloat,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Nspin() == 2) {
    genericMultBlas<Float,yFloat,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    genericMultBlas<Float,yFloat,1,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
}

template <typename Float, typename yFloat, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    genericMultBlas<Float,yFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,Functor>
      (x, y, z, w, f);
  } else {
    errorQuda("Not implemeneted");
  }
}
