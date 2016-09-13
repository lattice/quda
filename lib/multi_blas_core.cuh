/**
   @brief Parameter struct for generic multi-blas kernel.

   Need to check all loops. Not everything here needs to be templated
   some NXZ, NYW should be be runtime args

   @tparam NXZ is dimension of input vectors: X,Z
   @tparam NYW is dimension of in-output vectors: Y,W
   @tparam SpinorX Type of input spinor for x argument
   @tparam SpinorY Type of input spinor for y argument
   @tparam SpinorZ Type of input spinor for z argument
   @tparam SpinorW Type of input spinor for w argument
   @tparam Functor Functor used to operate on data
*/
template <int NXZ, int NYW, typename SpinorX, typename SpinorY, typename SpinorZ,
	  typename SpinorW, typename Functor>
struct MultiBlasArg {
  SpinorX X[NXZ];
  SpinorY Y[NYW];
  SpinorZ Z[NXZ];
  SpinorW W[NYW];
  Functor f;
  const int length;
  const int out_dim;
  const int in_dim;
  MultiBlasArg(SpinorX X[NXZ], SpinorY Y[NYW], SpinorZ Z[NXZ], SpinorW W[NYW], Functor f, int length)
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

template<int k_, int NXZ, int NYW, typename FloatN, int M, typename Arg>
__device__ inline void compute(Arg &arg, int idx) {

  constexpr int k = k_ < NYW ? k_ : 0; // silence out-of-bounds compiler warning

  while (idx < arg.length) {

    FloatN x[M], y[M], z[M], w[M];
    arg.Y[k].load(y, idx);
    arg.W[k].load(w, idx);

#pragma unroll
    for (int l=0; l < NXZ; l++) {
      arg.X[l].load(x, idx);
      arg.Z[l].load(z, idx);

#pragma unroll
      for (int j=0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], k, l);
    }
    arg.Y[k].save(y, idx);
    arg.W[k].save(w, idx);

    idx += gridDim.x*blockDim.x;
  }
}

/**
   @brief Generic multi-blas kernel with four loads and up to four stores.

   @param[in,out] arg Argument struct with required meta data
   (input/output fields, functor, etc.)
*/
template <typename FloatN, int M, int NXZ, int NYW, typename SpinorX, typename SpinorY,
typename SpinorZ, typename SpinorW, typename Functor>
__global__ void multiblasKernel(MultiBlasArg<NXZ, NYW, SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg) {

  // use i to loop over elements in kernel
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  arg.f.init();
  if (k >= NYW) return;

  switch(k) {
    case 0: compute<0,NXZ,NYW,FloatN,M>(arg,i); break;
    case 1: compute<1,NXZ,NYW,FloatN,M>(arg,i); break;
    case 2: compute<2,NXZ,NYW,FloatN,M>(arg,i); break;
    case 3: compute<3,NXZ,NYW,FloatN,M>(arg,i); break;
    case 4: compute<4,NXZ,NYW,FloatN,M>(arg,i); break;
    case 5: compute<5,NXZ,NYW,FloatN,M>(arg,i); break;
    case 6: compute<6,NXZ,NYW,FloatN,M>(arg,i); break;
    case 7: compute<7,NXZ,NYW,FloatN,M>(arg,i); break;
    case 8: compute<8,NXZ,NYW,FloatN,M>(arg,i); break;
    case 9: compute<9,NXZ,NYW,FloatN,M>(arg,i); break;
    case 10: compute<10,NXZ,NYW,FloatN,M>(arg,i); break;
    case 11: compute<11,NXZ,NYW,FloatN,M>(arg,i); break;
    case 12: compute<12,NXZ,NYW,FloatN,M>(arg,i); break;
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
class MultiBlasCuda : public TunableVectorY {

private:
  mutable MultiBlasArg<NXZ, NYW,SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  // host pointer used for backing up fields when tuning
  char *X_h[NXZ], *Y_h[NYW], *Z_h[NXZ], *W_h[NYW];
  char *Xnorm_h[NXZ], *Ynorm_h[NYW], *Znorm_h[NXZ], *Wnorm_h[NYW];
  const size_t **bytes_;
  const size_t **norm_bytes_;

  bool tuneSharedBytes() const { return false; }

public:
  MultiBlasCuda(SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Functor &f,
		int length,  size_t **bytes,  size_t **norm_bytes)
    : TunableVectorY(NYW), arg(X, Y, Z, W, f, length), X_h(), Y_h(), Z_h(), W_h(),
      Xnorm_h(), Ynorm_h(), Znorm_h(), Wnorm_h(),
      bytes_(const_cast<const size_t**>(bytes)), norm_bytes_(const_cast<const size_t**>(norm_bytes)) { }

  virtual ~MultiBlasCuda() { }

  inline TuneKey tuneKey() const {
    char name[TuneKey::name_n];
    strcpy(name, num_to_string<NXZ>::value);
    strcat(name, num_to_string<NYW>::value);
    strcat(name, typeid(arg.f).name());
    return TuneKey(blasStrings.vol_str, name, blasStrings.aux_tmp);
  }

  inline void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    multiblasKernel<FloatN,M,NXZ,NYW> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
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

template <int NXZ, int NYW, typename RegType, typename StoreType, typename yType, int M,
	  template <int,int,typename,typename> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void multiblasCuda(const Complex *a, const double2 &b, const double2 &c,
		   CompositeColorSpinorField& x, CompositeColorSpinorField& y,
		   CompositeColorSpinorField& z, CompositeColorSpinorField& w,
		   int length) {

  // FIXME implement this as a single kernel
  if (x[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    std::vector<cudaColorSpinorField> xp, yp, zp, wp;
    std::vector<ColorSpinorField*> xpp, ypp, zpp, wpp;
    xp.reserve(NXZ); yp.reserve(NYW); zp.reserve(NXZ); wp.reserve(NYW);
    xpp.reserve(NXZ); ypp.reserve(NYW); zpp.reserve(NXZ); wpp.reserve(NYW);

    for (int i=0; i<NXZ; i++) {
      xp.push_back(x[i]->Even());zp.push_back(z[i]->Even());
      xpp.push_back(&xp[i]); zpp.push_back(&zp[i]);
    }
    for (int i=0; i<NYW; i++) {
      yp.push_back(y[i]->Even()); wp.push_back(w[i]->Even());
      ypp.push_back(&yp[i]); ; wpp.push_back(&wp[i]);
    }

    multiblasCuda<NXZ, NYW, RegType, StoreType, yType, M, Functor, writeX, writeY, writeZ, writeW>
      (a, b, c, xpp, ypp, zpp, wpp, length);

    for (int i=0; i<NXZ; ++i) {
      xp.push_back(x[i]->Odd()); zp.push_back(z[i]->Odd());
      xpp.push_back(&xp[i]);  zpp.push_back(&zp[i]);
    }
    for (int i=0; i<NYW; ++i) {
      yp.push_back(y[i]->Odd()); wp.push_back(w[i]->Odd());
      ypp.push_back(&yp[i]); wpp.push_back(&wp[i]);
    }

    multiblasCuda<NXZ, NYW, RegType, StoreType, yType, M, Functor, writeX, writeY, writeZ, writeW>
       (a, b, c, xpp, ypp, zpp, wpp, length);

    return;
  }

  // for (int i=0; i<N; i++) {
  //   checkSpinor(*x[i],*y[i]); checkSpinor(*x[i],*z[i]); checkSpinor(*x[i],*w[i]);
  // }

  blasStrings.vol_str = x[0]->VolString();
  strcpy(blasStrings.aux_tmp, x[0]->AuxString());
  if (typeid(StoreType) != typeid(yType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, y[0]->AuxString());
  }

  const int N = NXZ > NYW ? NXZ : NYW;
  size_t **bytes = new size_t*[N], **norm_bytes = new size_t*[N];
  for (int i=0; i<N; i++) {
    bytes[i] = new size_t[4]; norm_bytes[i] = new size_t[4];
  }
  for (int i=0; i<NXZ; i++) {
    bytes[i][0] = x[i]->Bytes();  bytes[i][2] = z[i]->Bytes();
    norm_bytes[i][0] = x[i]->NormBytes(); norm_bytes[i][2] = z[i]->NormBytes();
  }
  for (int i=0; i<NYW; i++) {
    bytes[i][1] = y[i]->Bytes(); bytes[i][3] = w[i]->Bytes();
    norm_bytes[i][1] = y[i]->NormBytes(); norm_bytes[i][3] = w[i]->NormBytes();
  }

  Spinor<RegType,StoreType,M,writeX,0> X[NXZ];
  Spinor<RegType,    yType,M,writeY,1> Y[NYW];
  Spinor<RegType,StoreType,M,writeZ,2> Z[NXZ];
  Spinor<RegType,StoreType,M,writeW,3> W[NYW];

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  //MWFIXME
  for (int i=0; i<NXZ; i++) { X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i])); Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));}
  for (int i=0; i<NYW; i++) { Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i])); W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));}
  Functor<NXZ,NYW,Float2, RegType> f( a, (Float2)vec2(b), (Float2)vec2(c));

  MultiBlasCuda<NXZ,NYW,RegType,M,
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
void genericMultiBlas(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor f) {

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
    if (writeX) errorQuda("writeX not supported in multiblas.");
    if (writeY) Y(parity, x, s, c) = make_Complex(Y2);
	  // if (writeZ) Z(parity, x, s, c) = make_Complex(Z2);
    if (writeX) errorQuda("writeZ not supported in multiblas.");
	  if (writeW) W(parity, x, s, c) = make_Complex(W2);
	}
      }
    }
  }
}

template <typename Float, typename yFloat, int nSpin, int nColor, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultiBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
		   ColorSpinorField &w, Functor f) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Z(z), W(w);
  colorspinor::FieldOrderCB<yFloat,nSpin,nColor,1,order> Y(y);
  typedef typename vector<yFloat,2>::type Float2;
  genericMultiBlas<Float2,writeX,writeY,writeZ,writeW>(X, Y, Z, W, f);
}

template <typename Float, typename yFloat, int nSpin, QudaFieldOrder order,
	  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultiBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Ncolor() == 2) {
    genericMultiBlas<Float,yFloat,nSpin,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 3) {
    genericMultiBlas<Float,yFloat,nSpin,3,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 4) {
    genericMultiBlas<Float,yFloat,nSpin,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 8) {
    genericMultiBlas<Float,yFloat,nSpin,8,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 12) {
    genericMultiBlas<Float,yFloat,nSpin,12,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 16) {
    genericMultiBlas<Float,yFloat,nSpin,16,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 20) {
    genericMultiBlas<Float,yFloat,nSpin,20,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 24) {
    genericMultiBlas<Float,yFloat,nSpin,24,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 32) {
    genericMultiBlas<Float,yFloat,nSpin,32,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else {
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
}

template <typename Float, typename yFloat, QudaFieldOrder order, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultiBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Nspin() == 4) {
    genericMultiBlas<Float,yFloat,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Nspin() == 2) {
    genericMultiBlas<Float,yFloat,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    genericMultiBlas<Float,yFloat,1,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
}

template <typename Float, typename yFloat, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericMultiBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    genericMultiBlas<Float,yFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,Functor>
      (x, y, z, w, f);
  } else {
    errorQuda("Not implemeneted");
  }
}
