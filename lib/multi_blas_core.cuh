/**
   @brief Parameter struct for generic multi-blas kernel.
   @tparam NXZ is dimension of input vectors: X,Z
   @tparam NYW is dimension of in-output vectors: Y,W
   @tparam SpinorX Type of input spinor for x argument
   @tparam SpinorY Type of input spinor for y argument
   @tparam SpinorZ Type of input spinor for z argument
   @tparam SpinorW Type of input spinor for w argument
   @tparam Functor Functor used to operate on data
*/
template <int NXZ, typename SpinorX, typename SpinorY, typename SpinorZ,
	  typename SpinorW, typename Functor>
struct MultiBlasArg {
  const int NYW;
  SpinorX X[NXZ];
  SpinorY Y[MAX_MULTI_BLAS_N];
  SpinorZ Z[NXZ];
  SpinorW W[MAX_MULTI_BLAS_N];
  Functor f;
  const int length;

  MultiBlasArg(SpinorX X[NXZ], SpinorY Y[], SpinorZ Z[NXZ], SpinorW W[], Functor f, int NYW, int length)
    :  NYW(NYW), f(f), length(length) {

    for(int i=0; i<NXZ; ++i){
      this->X[i] = X[i];
      this->Z[i] = Z[i];
    }
    for(int i=0; i<NYW; ++i){
      this->Y[i] = Y[i];
      this->W[i] = W[i];
    }
  }
};


// storage for matrix coefficients
#define MAX_MATRIX_SIZE 4096
static __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
static __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
static __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

static signed char *Amatrix_h;
static signed char *Bmatrix_h;
static signed char *Cmatrix_h;

template<int k, int NXZ, typename FloatN, int M, typename Arg>
__device__ inline void compute(Arg &arg, int idx, int parity) {

  while (idx < arg.length) {

    FloatN x[M], y[M], z[M], w[M];
    arg.Y[k].load(y, idx, parity);
    arg.W[k].load(w, idx, parity);

#pragma unroll
    for (int l=0; l < NXZ; l++) {
      arg.X[l].load(x, idx, parity);
      arg.Z[l].load(z, idx, parity);

#pragma unroll
      for (int j=0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], k, l);
    }
    arg.Y[k].save(y, idx, parity);
    arg.W[k].save(w, idx, parity);

    idx += gridDim.x*blockDim.x;
  }
}

/**
   @brief Generic multi-blas kernel with four loads and up to four stores.
   @param[in,out] arg Argument struct with required meta data
   (input/output fields, functor, etc.)
*/
template <typename FloatN, int M, int NXZ, typename SpinorX, typename SpinorY,
typename SpinorZ, typename SpinorW, typename Functor>
__global__ void multiblasKernel(MultiBlasArg<NXZ,SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg) {

  // use i to loop over elements in kernel
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int parity = blockIdx.z;

  arg.f.init();
  if (k >= arg.NYW) return;

  switch(k) {
  case  0: compute< 0,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 2
  case  1: compute< 1,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 3
  case  2: compute< 2,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 4
  case  3: compute< 3,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 5
  case  4: compute< 4,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 6
  case  5: compute< 5,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 7
  case  6: compute< 6,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 8
  case  7: compute< 7,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 9
  case  8: compute< 8,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 10
  case  9: compute< 9,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 11
  case 10: compute<10,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 12
  case 11: compute<11,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 13
  case 12: compute<12,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 14
  case 13: compute<13,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 15
  case 14: compute<14,NXZ,FloatN,M>(arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 16
  case 15: compute<15,NXZ,FloatN,M>(arg,i,parity); break;
#endif //16
#endif //15
#endif //14
#endif //13
#endif //12
#endif //11
#endif //10
#endif //9
#endif //8
#endif //7
#endif //6
#endif //5
#endif //4
#endif //3
#endif //2
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


template <int NXZ, typename FloatN, int M, typename SpinorX, typename SpinorY,
  typename SpinorZ, typename SpinorW, typename Functor>
class MultiBlasCuda : public TunableVectorY {

private:
  const int NYW;
  mutable MultiBlasArg<NXZ,SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg;
  const int nParity;

  // host pointers used for backing up fields when tuning
  // don't curry into the Spinors to minimize parameter size
  char *Y_h[MAX_MULTI_BLAS_N], *W_h[MAX_MULTI_BLAS_N], *Ynorm_h[MAX_MULTI_BLAS_N], *Wnorm_h[MAX_MULTI_BLAS_N];
  std::vector<ColorSpinorField*> &y, &w;

  bool tuneSharedBytes() const { return false; }

public:
  MultiBlasCuda(SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Functor &f,
		int NYW, int length, int nParity, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &w)
    : TunableVectorY(NYW), NYW(NYW), arg(X, Y, Z, W, f, NYW, length/nParity),
      nParity(nParity), Y_h(), W_h(), Ynorm_h(), Wnorm_h(), y(y), w(w) { ; }

  virtual ~MultiBlasCuda() { }

  inline TuneKey tuneKey() const {
    char name[TuneKey::name_n];
    strcpy(name, num_to_string<NXZ>::value);
    strcat(name, std::to_string(NYW).c_str());
    strcat(name, typeid(arg.f).name());
    return TuneKey(blasStrings.vol_str, name, blasStrings.aux_tmp);
  }

  inline void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    multiblasKernel<FloatN,M,NXZ> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
  }

  void preTune() {
    for(int i=0; i<NYW; ++i){
      arg.Y[i].backup(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
      arg.W[i].backup(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
    }
  }

  void postTune() {
    for(int i=0; i<NYW; ++i){
      arg.Y[i].restore(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
      arg.W[i].restore(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
    }
  }

  void initTuneParam(TuneParam &param) const {
    TunableVectorY::initTuneParam(param);
    param.grid.z = nParity;
  }

  void defaultTuneParam(TuneParam &param) const {
    TunableVectorY::defaultTuneParam(param);
    param.grid.z = nParity;
  }

  long long flops() const { return arg.f.flops()*vec_length<FloatN>::value*(long)arg.length*nParity*M; }

  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.X[0].Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.Y[0].Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    return ((arg.f.streams()-2)*base_bytes + 2*extra_bytes)*arg.length*nParity;
  }

  int tuningIter() const { return 3; }
};

template <typename T>
struct coeff_array {
  const T *data;
  const bool use_const;
  coeff_array() : data(nullptr), use_const(false) { }
  coeff_array(const T *data, bool use_const) : data(data), use_const(use_const) { }
};

template <int NXZ, typename RegType, typename StoreType, typename yType, int M,
	  template <int,typename,typename> class Functor,
	  int writeX, int writeY, int writeZ, int writeW, typename T>
void multiblasCuda(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
		   std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
		   std::vector<ColorSpinorField*> &z, std::vector<ColorSpinorField*> &w,
		   int length) {

  const int NYW = y.size();

  const int N = NXZ > NYW ? NXZ : NYW;
  if (N > MAX_MULTI_BLAS_N) errorQuda("Spinor vector length exceeds max size (%d > %d)", N, MAX_MULTI_BLAS_N);

  if (NXZ*NYW*sizeof(Complex) > MAX_MATRIX_SIZE)
    errorQuda("A matrix exceeds max size (%lu > %d)", NXZ*NYW*sizeof(Complex), MAX_MATRIX_SIZE);

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;

  // FIXME - if NXZ=1 no need to copy entire array
  // FIXME - do we really need strided access here?
  if (a.data && a.use_const) {
    Float2 A[MAX_MATRIX_SIZE/sizeof(Float2)];
    // since the kernel doesn't know the width of them matrix at compile
    // time we stride it and copy the padded matrix to GPU
    for (int i=0; i<NXZ; i++) for (int j=0; j<NYW; j++)
      A[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));

    cudaMemcpyToSymbolAsync(Amatrix_d, A, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
    Amatrix_h = reinterpret_cast<signed char*>(const_cast<T*>(a.data));
  }

  if (b.data && b.use_const) {
    Float2 B[MAX_MATRIX_SIZE/sizeof(Float2)];
    // since the kernel doesn't know the width of them matrix at compile
    // time we stride it and copy the padded matrix to GPU
    for (int i=0; i<NXZ; i++) for (int j=0; j<NYW; j++)
      B[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));

    cudaMemcpyToSymbolAsync(Bmatrix_d, B, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
    Bmatrix_h = reinterpret_cast<signed char*>(const_cast<T*>(b.data));
  }

  if (c.data && c.use_const) {
    Float2 C[MAX_MATRIX_SIZE/sizeof(Float2)];
    // since the kernel doesn't know the width of them matrix at compile
    // time we stride it and copy the padded matrix to GPU
    for (int i=0; i<NXZ; i++) for (int j=0; j<NYW; j++)
      C[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));

    cudaMemcpyToSymbolAsync(Cmatrix_d, C, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
    Cmatrix_h = reinterpret_cast<signed char*>(const_cast<T*>(c.data));
  }

  // for (int i=0; i<N; i++) {
  //   checkLength(*x[i],*y[i]); checkLength(*x[i],*z[i]); checkLength(*x[i],*w[i]);
  // }

  blasStrings.vol_str = x[0]->VolString();
  strcpy(blasStrings.aux_tmp, x[0]->AuxString());
  if (typeid(StoreType) != typeid(yType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, y[0]->AuxString());
  }

  multi::SpinorTexture<RegType,StoreType,M,0> X[NXZ];
  multi::Spinor<RegType,    yType,M,writeY,1> Y[MAX_MULTI_BLAS_N];
  multi::SpinorTexture<RegType,StoreType,M,2> Z[NXZ];
  multi::Spinor<RegType,StoreType,M,writeW,3> W[MAX_MULTI_BLAS_N];

  //MWFIXME
  for (int i=0; i<NXZ; i++) { X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i])); Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));}
  for (int i=0; i<NYW; i++) { Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i])); W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));}

  // if block caxpy is an 'outer product of caxpy' where 'x'

  Functor<NXZ,Float2, RegType> f(a, b, c, NYW);

  MultiBlasCuda<NXZ,RegType,M,
		multi::SpinorTexture<RegType,StoreType,M,0>,
		multi::Spinor<RegType,    yType,M,writeY,1>,
		multi::SpinorTexture<RegType,StoreType,M,2>,
		multi::Spinor<RegType,StoreType,M,writeW,3>,
		decltype(f) >
    blas(X, Y, Z, W, f, NYW, length, x[0]->SiteSubset(), y, w);
  blas.apply(*getStream());

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
    if (writeZ) errorQuda("writeZ not supported in multiblas.");
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
