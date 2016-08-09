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
  : X(X), Y(Y), Z(Z), W(W), f(f), length(length), in_dim(NXZ), out_dim(NYW){

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
        for (int j=0; j<M; j++) arg.f(x[j], y[j], z[j], w[j]);
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
  MultBlasCuda(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor &f,
	   int length, const size_t *bytes, const size_t *norm_bytes) :
  arg(X, Y, Z, W, f, length), X_h(), Y_h(), Z_h(), W_h(),
    Xnorm_h(), Ynorm_h(), Znorm_h(), Wnorm_h(), bytes_(bytes), norm_bytes_(norm_bytes) { }

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
      arg.X[i].save(&X_h[i], &Xnorm_h[i], bytes_[0], norm_bytes_[0]);
      arg.Z[i].save(&Z_h[i], &Znorm_h[i], bytes_[2], norm_bytes_[2]);
    }
    for(int i=0; i< NYW; ++i){
      arg.Y[i].save(&Y_h[i], &Ynorm_h[i], bytes_[1], norm_bytes_[1]);
      arg.W[i].save(&W_h[i], &Wnorm_h[i], bytes_[3], norm_bytes_[3]);
    }
  }

  void postTune() {
    for(int i=0; i<NXZ; ++i){
      arg.X[i].load(&X_h[i], &Xnorm_h[i], bytes_[0], norm_bytes_[0]);
      arg.Z[i].load(&Z_h[i], &Znorm_h[i], bytes_[2], norm_bytes_[2]);
    }
    for(int i=0; i< NYW; ++i){
      arg.Y[i].load(&Y_h[i], &Ynorm_h[i], bytes_[1], norm_bytes_[1]);
      arg.W[i].load(&W_h[i], &Wnorm_h[i], bytes_[3], norm_bytes_[3]);
    }
  }




  long long flops() const { return arg.f.flops()*vec_length<FloatN>::value*arg.length*M; }
  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X.Precision()*vec_length<FloatN>::value*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y.Precision()*vec_length<FloatN>::value*M;
    if (arg.Y.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    return ((arg.f.streams()-2)*base_bytes + 2*extra_bytes)*arg.length;
  }
  int tuningIter() const { return 3; }
};

template <typename RegType, typename StoreType, typename yType, int M,
	  template <int,int,typename,typename> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void multblasCuda(const Complex *a, const double2 &b, const double2 &c,
	      ColorSpinorField &x, ColorSpinorField &y,
	      ColorSpinorField &z, ColorSpinorField &w, int length) {
  // FIXME implement this as a single kernel
  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    multblasCuda<RegType,StoreType,yType,M,Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Even(), y.Even(), z.Even(), w.Even(), length);
    multblasCuda<RegType,StoreType,yType,M,Functor,writeX,writeY,writeZ,writeW>
      (a, b, c, x.Odd(), y.Odd(), z.Odd(), w.Odd(), length);
    return;
  }

  checkLength(x, y); checkLength(x, z); checkLength(x, w);

  if (!x.isNative()) {
    warningQuda("Device blas on non-native fields is not supported\n");
    return;
  }

  blasStrings.vol_str = x.VolString();
  strcpy(blasStrings.aux_tmp, x.AuxString());
  if (typeid(StoreType) != typeid(yType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, y.AuxString());
  }

  size_t bytes[] = {x.Bytes(), y.Bytes(), z.Bytes(), w.Bytes()};
  size_t norm_bytes[] = {x.NormBytes(), y.NormBytes(), z.NormBytes(), w.NormBytes()};

  Spinor<RegType,StoreType,M,writeX,0> X(x);
  Spinor<RegType,    yType,M,writeY,1> Y(y);
  Spinor<RegType,StoreType,M,writeZ,2> Z(z);
  Spinor<RegType,StoreType,M,writeW,3> W(w);

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  //MWFIXME
  Functor<1,1,Float2, RegType> f( a, (Float2)vec2(b), (Float2)vec2(c));

  MultBlasCuda<1,1,RegType,M,
    decltype(X), decltype(Y), decltype(Z), decltype(W),
    decltype(f) >
    blas(X.Components(), Y.Components(), Z.Components(), W.Components(), f, length, bytes, norm_bytes);
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
