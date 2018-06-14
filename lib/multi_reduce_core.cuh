__host__ __device__ inline double set(double &x) { return x;}
__host__ __device__ inline double2 set(double2 &x) { return x;}
__host__ __device__ inline double3 set(double3 &x) { return x;}
__host__ __device__ inline void sum(double &a, double &b) { a += b; }
__host__ __device__ inline void sum(double2 &a, double2 &b) { a.x += b.x; a.y += b.y; }
__host__ __device__ inline void sum(double3 &a, double3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; }

#ifdef QUAD_SUM
__host__ __device__ inline double set(doubledouble &a) { return a.head(); }
__host__ __device__ inline double2 set(doubledouble2 &a) { return make_double2(a.x.head(),a.y.head()); }
__host__ __device__ inline double3 set(doubledouble3 &a) { return make_double3(a.x.head(),a.y.head(),a.z.head()); }
__host__ __device__ inline void sum(double &a, doubledouble &b) { a += b.head(); }
__host__ __device__ inline void sum(double2 &a, doubledouble2 &b) { a.x += b.x.head(); a.y += b.y.head(); }
__host__ __device__ inline void sum(double3 &a, doubledouble3 &b) { a.x += b.x.head(); a.y += b.y.head(); a.z += b.z.head(); }
#endif

//#define WARP_MULTI_REDUCE

__device__ static unsigned int count = 0;
__shared__ static bool isLastBlockDone;

#include <launch_kernel.cuh>


/**
   @brief Parameter struct for generic multi-blas kernel.
   @tparam NXZ is dimension of input vectors: X,Z,V
   @tparam NYW is dimension of in-output vectors: Y,W
   @tparam SpinorX Type of input spinor for x argument
   @tparam SpinorY Type of input spinor for y argument
   @tparam SpinorZ Type of input spinor for z argument
   @tparam SpinorW Type of input spinor for w argument
   @tparam SpinorW Type of input spinor for v argument
   @tparam Reducer Functor used to operate on data
*/
template <int NXZ, typename ReduceType, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename Reducer>
struct MultiReduceArg : public ReduceArg<vector_type<ReduceType,NXZ> > {

  const int NYW;
  SpinorX X[NXZ];
  SpinorY Y[MAX_MULTI_BLAS_N];
  SpinorZ Z[NXZ];
  SpinorW W[MAX_MULTI_BLAS_N];
  Reducer  r;
  const int length;
 MultiReduceArg(SpinorX X[NXZ], SpinorY Y[], SpinorZ Z[NXZ], SpinorW W[], Reducer r, int NYW, int length)
   : NYW(NYW), r(r), length(length) {

    for (int i=0; i<NXZ; ++i)
    {
      this->X[i] = X[i];
      this->Z[i] = Z[i];
    }

    for (int i=0; i<NYW; ++i)
    {
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

// 'sum' should be an array of length NXZ...?
template<int k, int NXZ, typename FloatN, int M, typename ReduceType, typename Arg>
__device__ inline void compute(vector_type<ReduceType,NXZ> &sum, Arg &arg, int idx, int parity) {

  constexpr int kmod = k; // there's an old warning about silencing an out-of-bounds compiler warning,
                          // but I never seem to get it, and I'd need to compare against NYW anyway,
                          // which we can't really get at here b/c it's not a const, and I don't want 
                          // to fix that. It works fine based on the switch in the function below.

  while (idx < arg.length) {

    FloatN x[M], y[M], z[M], w[M];

    arg.Y[kmod].load(y, idx, parity);
    arg.W[kmod].load(w, idx, parity);

    // Each NYW owns its own thread.
    // The NXZ's are all in the same thread block,
    // so they can share the same memory.
#pragma unroll
    for (int l=0; l < NXZ; l++) {
      arg.X[l].load(x, idx, parity);
      arg.Z[l].load(z, idx, parity);

      arg.r.pre();

#pragma unroll
      for (int j=0; j<M; j++) arg.r(sum[l], x[j], y[j], z[j], w[j], k, l);

      arg.r.post(sum[l]);
    }

    arg.Y[kmod].save(y, idx, parity);
    arg.W[kmod].save(w, idx, parity);

    idx += gridDim.x*blockDim.x;
 }

}

#ifdef WARP_MULTI_REDUCE
template<typename ReduceType, typename FloatN, int M, int NXZ,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
#else
  template<int block_size, typename ReduceType, typename FloatN, int M, int NXZ,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
#endif
  __global__ void multiReduceKernel(MultiReduceArg<NXZ,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int parity = blockIdx.z;

  if (k >= arg.NYW) return; // safe since k are different thread blocks

  vector_type<ReduceType,NXZ> sum;

  switch(k) {
  case  0: compute< 0,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 2
  case  1: compute< 1,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 3
  case  2: compute< 2,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 4
  case  3: compute< 3,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 5
  case  4: compute< 4,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 6
  case  5: compute< 5,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 7
  case  6: compute< 6,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 8
  case  7: compute< 7,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 9
  case  8: compute< 8,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 10
  case  9: compute< 9,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 11
  case 10: compute<10,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 12
  case 11: compute<11,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 13
  case 12: compute<12,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 14
  case 13: compute<13,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 15
  case 14: compute<14,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
#if MAX_MULTI_BLAS_N >= 16
  case 15: compute<15,NXZ,FloatN,M,ReduceType>(sum,arg,i,parity); break;
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

#ifdef WARP_MULTI_REDUCE
  ::quda::warp_reduce<vector_type<ReduceType,NXZ> >(arg, sum, arg.NYW*parity + k);
#else
  ::quda::reduce<block_size, vector_type<ReduceType,NXZ> >(arg, sum, arg.NYW*parity + k);
#endif

} // multiReduceKernel

template<typename doubleN, typename ReduceType, typename FloatN, int M, int NXZ,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
  void multiReduceLaunch(doubleN result[],
			 MultiReduceArg<NXZ,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,Reducer> &arg,
			 const TuneParam &tp, const cudaStream_t &stream) {

  if(tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);
  
  // ESW: this is where the multireduce kernel is called...?
#ifdef WARP_MULTI_REDUCE
  multiReduceKernel<ReduceType,FloatN,M,NXZ><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
  LAUNCH_KERNEL_LOCAL_PARITY(multiReduceKernel, tp, stream, arg, ReduceType, FloatN, M, NXZ);
#endif
  
#if (defined(_MSC_VER) && defined(_WIN64) || defined(__LP64__))
  if(deviceProp.canMapHostMemory){
    qudaEventRecord(*getReduceEvent(), stream);
    while(cudaSuccess != qudaEventQuery(*getReduceEvent())) {}
  } else
#endif
    { qudaMemcpy(getHostReduceBuffer(), getMappedHostReduceBuffer(), tp.grid.z*sizeof(ReduceType)*NXZ*arg.NYW, cudaMemcpyDeviceToHost); }

  // need to transpose for same order with vector thread reduction
  for (int i=0; i<NXZ; i++) {
    for (int j=0; j<arg.NYW; j++) {
      result[i*arg.NYW+j] = set(((ReduceType*)getHostReduceBuffer())[j*NXZ+i]);
      if (tp.grid.z==2) result[i*arg.NYW+j] = set(((ReduceType*)getHostReduceBuffer())[NXZ*arg.NYW+j*NXZ+i]);
    }
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

template<int NXZ, typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX,
  typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
  class MultiReduceCuda : public Tunable {

 private:
  const int NYW; 
  mutable MultiReduceArg<NXZ,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,Reducer> arg;
  doubleN *result;
  int nParity;

  // host pointer used for backing up fields when tuning
  // don't curry into the Spinors to minimize parameter size
  char *Y_h[MAX_MULTI_BLAS_N], *W_h[MAX_MULTI_BLAS_N], *Ynorm_h[MAX_MULTI_BLAS_N], *Wnorm_h[MAX_MULTI_BLAS_N];
  std::vector<ColorSpinorField*> &y, &w;

  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  virtual bool advanceSharedBytes(TuneParam &param) const
  {
    TuneParam next(param);
    advanceBlockDim(next); // to get next blockDim
    int nthreads = next.block.x * next.block.y * next.block.z;
    param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ? sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
    return false;
  }

  // we only launch thread blocks up to size 512 since the autoner
  // tuner favours smaller blocks and this helps with compile time
  unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / 2; }

public:
  MultiReduceCuda(doubleN result[], SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[],
		  Reducer &r, int NYW, int length, int nParity, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &w)
    : NYW(NYW), arg(X, Y, Z, W, r, NYW, length/nParity), nParity(nParity), result(result),
      Y_h(), W_h(), Ynorm_h(), Wnorm_h(), y(y), w(w) { }

  inline TuneKey tuneKey() const {
    char name[TuneKey::name_n];
    strcpy(name, num_to_string<NXZ>::value);
    strcat(name, std::to_string(NYW).c_str());
    strcat(name, typeid(arg.r).name());
    return TuneKey(blasStrings.vol_str, name, blasStrings.aux_tmp);
  }

  void apply(const cudaStream_t &stream){
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    multiReduceLaunch<doubleN,ReduceType,FloatN,M,NXZ>(result,arg,tp,stream);
  }

  // Should these be NYW?
#ifdef WARP_MULTI_REDUCE
  /**
     @brief This is a specialized variant of the reducer that only
     assigns an individial warp within a thread block to a given row
     of the reduction.  It's typically slower than CTA-wide reductions
     and spreading the y dimension across blocks rather then within
     the blocks so left disabled.
   */
  bool advanceBlockDim(TuneParam &param) const {
    if (param.block.y < NYW) {
      param.block.y++;
      param.grid.y = (NYW + param.block.y - 1) / param.block.y;
      return true;
    } else {
      param.block.y = 1;
      param.grid.y = NYW;
      return false;
    }
  }
#endif

  bool advanceGridDim(TuneParam &param) const {
    bool rtn = Tunable::advanceGridDim(param);
    if (NYW > deviceProp.maxGridSize[1]) errorQuda("N=%d is greater than the maximum support grid size", NYW);
    return rtn;
  }

  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.block.y = 1;
    param.grid.y = NYW;
    param.grid.z = nParity;
  }

  void defaultTuneParam(TuneParam &param) const {
    Tunable::defaultTuneParam(param);
    param.block.y = 1;
    param.grid.y = NYW;
    param.grid.z = nParity;
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

  // Need to check this!
  // The NYW seems right?
    long long flops() const { return NYW*NXZ*arg.r.flops()*vec_length<FloatN>::value*(long long)arg.length*nParity*M; }
  long long bytes() const {
    size_t bytes = NYW*NXZ*arg.X[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.X[0].Precision() == QUDA_HALF_PRECISION) bytes += NYW*NXZ*sizeof(float);
    return arg.r.streams()*bytes*arg.length*nParity; }
  int tuningIter() const { return 3; }
};

template <typename T>
struct coeff_array {
  const T *data;
  const bool use_const;
  coeff_array() : data(nullptr), use_const(false) { }
  coeff_array(const T *data, bool use_const) : data(data), use_const(use_const) { }
};



template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename yType,
	  int M, int NXZ, template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write, typename T>
  void multiReduceCuda(doubleN result[], const reduce::coeff_array<T> &a, const reduce::coeff_array<T> &b, const reduce::coeff_array<T> &c,
			  std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			  std::vector<ColorSpinorField*>& z, std::vector<ColorSpinorField*>& w,
			  int length) {

  const int NYW = y.size();

  int nParity = x[0]->SiteSubset();
  memset(result, 0, NXZ*NYW*sizeof(doubleN));

  const int N_MAX = NXZ > NYW ? NXZ : NYW;
  const int N_MIN = NXZ < NYW ? NXZ : NYW;

  static_assert(MAX_MULTI_BLAS_N*MAX_MULTI_BLAS_N <= QUDA_MAX_MULTI_REDUCE, "MAX_MULTI_BLAS_N^2 exceeds maximum number of reductions");
  static_assert(MAX_MULTI_BLAS_N <= 16, "MAX_MULTI_BLAS_N exceeds maximum size 16");
  if (N_MAX > MAX_MULTI_BLAS_N) errorQuda("Spinor vector length exceeds max size (%d > %d)", N_MAX, MAX_MULTI_BLAS_N);

  if (NXZ*NYW*sizeof(Complex) > MAX_MATRIX_SIZE)
    errorQuda("A matrix exceeds max size (%lu > %d)", NXZ*NYW*sizeof(Complex), MAX_MATRIX_SIZE);

  for (int i=0; i<N_MIN; i++) {
    checkSpinor(*x[i],*y[i]); checkSpinor(*x[i],*z[i]); checkSpinor(*x[i],*w[i]); 
    if (!x[i]->isNative()) {
      warningQuda("Reductions on non-native fields are not supported\n");
      return;
    }
  }

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

  blasStrings.vol_str = x[0]->VolString();
  strcpy(blasStrings.aux_tmp, x[0]->AuxString());
  if (typeid(StoreType) != typeid(yType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, y[0]->AuxString());
  }

  multi::SpinorTexture<RegType,StoreType,M,0> X[NXZ];
  multi::Spinor<RegType,yType,M,write::Y,1> Y[MAX_MULTI_BLAS_N];
  multi::SpinorTexture<RegType,StoreType,M,2> Z[NXZ];
  multi::Spinor<RegType,StoreType,M,write::W,3> W[MAX_MULTI_BLAS_N];

  for (int i=0; i<NXZ; i++) {
    X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i]));
    Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));
  }
  for (int i=0; i<NYW; i++) {
    Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i]));
    W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));
  }

  // since the block dot product and the block norm use the same functors, we need to distinguish them
  bool is_norm = false;
  if (NXZ==NYW) {
    is_norm = true;
    for (int i=0; i<NXZ; i++) {
      if (x[i]->V() != y[i]->V() || x[i]->V() != z[i]->V() || x[i]->V() != w[i]->V()) {
	is_norm = false;
	break;
      }
    }
  }
  if (is_norm) strcat(blasStrings.aux_tmp, ",norm");

  Reducer<NXZ, ReduceType, Float2, RegType> r(a, b, c, NYW);

  MultiReduceCuda<NXZ,doubleN,ReduceType,RegType,M,
		  multi::SpinorTexture<RegType,StoreType,M,0>,
		  multi::Spinor<RegType,yType,M,write::Y,1>,
		  multi::SpinorTexture<RegType,StoreType,M,2>,
		  multi::Spinor<RegType,StoreType,M,write::W,3>,
		  decltype(r) >
    reduce(result, X, Y, Z, W, r, NYW, length, x[0]->SiteSubset(), y, w);
    reduce.apply(*blas::getStream());

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return;
}
