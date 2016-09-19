//#define WARP_MULTI_REDUCE

template <int N, typename ReduceType, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  struct MultiReduceArg : public ReduceArg<ReduceType> {

  SpinorX X[N];
  SpinorY Y[N];
  SpinorZ Z[N];
  SpinorW W[N];
  SpinorV V[N];
  Reducer r;
  const int length;
  MultiReduceArg(SpinorX X[N], SpinorY Y[N], SpinorZ Z[N], SpinorW W[N], SpinorV V[N], Reducer r, int length)
    : r(r), length(length) {

    for(int i=0; i<N; ++i){
      this->X[i] = X[i];
      this->Y[i] = Y[i];
      this->Z[i] = Z[i];
      this->W[i] = W[i];
      this->V[i] = V[i];
    }
  }
};

template<int src, int N, typename FloatN, int M, typename ReduceType, typename Arg>
  __device__ inline void compute(ReduceType &sum, Arg &arg, int idx) {

  constexpr int k = src < N ? src : 0; // silence out-of-bounds compiler warning

  while (idx < arg.length) {

    FloatN x[M], y[M], z[M], w[M], v[M];

    arg.X[k].load(x, idx);
    arg.Y[k].load(y, idx);
    arg.Z[k].load(z, idx);
    arg.W[k].load(w, idx);
    arg.V[k].load(v, idx);

    arg.r.pre();

#pragma unroll
    for (int j=0; j<M; j++) arg.r(sum, x[j], y[j], z[j], w[j], v[j]);

    arg.r.post(sum);

    arg.X[k].load(x, idx);
    arg.Y[k].load(y, idx);
    arg.Z[k].load(z, idx);
    arg.W[k].load(w, idx);
    arg.V[k].load(v, idx);

    idx += gridDim.x*blockDim.x;
 }

}

#ifdef WARP_MULTI_REDUCE
template<int N, typename ReduceType, typename FloatN, int M,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
#else
  template<int block_size, int N, typename ReduceType, typename FloatN, int M,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
#endif
  __global__ void multiReduceKernel(MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int src_idx = blockIdx.y*blockDim.y + threadIdx.y;

  ReduceType sum;
  ::quda::zero(sum);

  if (src_idx < N) {
    switch(src_idx) {
    case 0: compute<0,N,FloatN,M>(sum,arg,i); break;
    case 1: compute<1,N,FloatN,M>(sum,arg,i); break;
    case 2: compute<2,N,FloatN,M>(sum,arg,i); break;
    case 3: compute<3,N,FloatN,M>(sum,arg,i); break;
    case 4: compute<4,N,FloatN,M>(sum,arg,i); break;
    }
  }

#ifdef WARP_MULTI_REDUCE
  ::quda::warp_reduce<ReduceType>(arg, sum, src_idx);
#else
  ::quda::reduce<block_size, ReduceType>(arg, sum, src_idx);
#endif

} // multiReduceKernel

template<int N, typename doubleN, typename ReduceType, typename FloatN, int M,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  void multiReduceLaunch(doubleN result[],
			 MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> &arg,
			 const TuneParam &tp, const cudaStream_t &stream){

  if(tp.grid.x > REDUCE_MAX_BLOCKS)
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, REDUCE_MAX_BLOCKS);

#ifdef WARP_MULTI_REDUCE
  multiReduceKernel<N,ReduceType,FloatN,M><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
  LAUNCH_KERNEL(multiReduceKernel, tp, stream, arg, N, ReduceType, FloatN, M);
#endif

#if (defined(_MSC_VER) && defined(_WIN64) || defined(__LP64__))
  if(deviceProp.canMapHostMemory){
    cudaEventRecord(reduceEnd, stream);
    while(cudaSuccess != cudaEventQuery(reduceEnd)) {}
  } else
#endif
    { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType)*N, cudaMemcpyDeviceToHost); }

  for(int i=0; i<N; ++i) result[i] = set(((ReduceType*)h_reduce)[i]);
}

namespace detail
{
  template<unsigned... digits> struct to_chars { static const char value[]; };
  template<unsigned... digits> const char to_chars<digits...>::value[] = {('0' + digits)..., 0};
  template<unsigned rem, unsigned... digits> struct explode : explode<rem / 10, rem % 10, digits...> {};
  template<unsigned... digits> struct explode<0, digits...> : to_chars<digits...> {};
}

template<unsigned num>
struct num_to_string : detail::explode<num / 10, num % 10> {};

template<int N, typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX,
  typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  class MultiReduceCuda : public Tunable {

 private:
  mutable MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg;
  doubleN *result;

  // host pointer used for backing up fields when tuning
  char *X_h[N], *Y_h[N], *Z_h[N], *W_h[N], *V_h[N];
  char *Xnorm_h[N], *Ynorm_h[N], *Znorm_h[N], *Wnorm_h[N], *Vnorm_h[N];
  const size_t **bytes_;
  const size_t **norm_bytes_;

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

 public:
 MultiReduceCuda(doubleN result[], SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], SpinorV V[],
		 Reducer &r, int length, size_t **bytes, size_t **norm_bytes) :
  arg(X, Y, Z, W, V, r, length), result(result),
    X_h(), Y_h(), Z_h(), W_h(), V_h(), Xnorm_h(),
    Ynorm_h(), Znorm_h(), Wnorm_h(), Vnorm_h(),
    bytes_(const_cast<const size_t**>(bytes)), norm_bytes_(const_cast<const size_t**>(norm_bytes)) { }

  inline TuneKey tuneKey() const {
    char name[TuneKey::name_n];
    strcpy(name, num_to_string<N>::value);
    strcat(name, typeid(arg.r).name());
    return TuneKey(blasStrings.vol_str, name, blasStrings.aux_str);
  }

  unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock; }

  void apply(const cudaStream_t &stream){
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    multiReduceLaunch<N,doubleN,ReduceType,FloatN,M>(result,arg,tp,stream);
  }

#ifdef WARP_MULTI_REDUCE
  bool advanceBlockDim(TuneParam &param) const {
    if (param.block.y < N) {
      param.block.y++;
      param.grid.y = (N + param.block.y - 1) / param.block.y;
      return true;
    } else {
      param.block.y = 1;
      param.grid.y = N;
      return false;
    }
  }
#endif

  bool advanceGridDim(TuneParam &param) const {
    bool rtn = Tunable::advanceGridDim(param);
    if (N > deviceProp.maxGridSize[1]) errorQuda("N=%d is greater than the maximum support grid size", N);
    return rtn;
  }

  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.block.y = 1;
    param.grid.y = N;
  }

  void defaultTuneParam(TuneParam &param) const {
    Tunable::defaultTuneParam(param);
    param.block.y = 1;
    param.grid.y = N;
  }

  void preTune() {
    for(int i=0; i<N; ++i){
      arg.X[i].backup(&X_h[i], &Xnorm_h[i], bytes_[i][0], norm_bytes_[i][0]);
      arg.Y[i].backup(&Y_h[i], &Ynorm_h[i], bytes_[i][1], norm_bytes_[i][1]);
      arg.Z[i].backup(&Z_h[i], &Znorm_h[i], bytes_[i][2], norm_bytes_[i][2]);
      arg.W[i].backup(&W_h[i], &Wnorm_h[i], bytes_[i][3], norm_bytes_[i][3]);
      arg.V[i].backup(&V_h[i], &Vnorm_h[i], bytes_[i][4], norm_bytes_[i][4]);
    }
  }

  void postTune() {
    for(int i=0; i<N; ++i){
      arg.X[i].restore(&X_h[i], &Xnorm_h[i], bytes_[i][0], norm_bytes_[i][0]);
      arg.Y[i].restore(&Y_h[i], &Ynorm_h[i], bytes_[i][1], norm_bytes_[i][1]);
      arg.Z[i].restore(&Z_h[i], &Znorm_h[i], bytes_[i][2], norm_bytes_[i][2]);
      arg.W[i].restore(&W_h[i], &Wnorm_h[i], bytes_[i][3], norm_bytes_[i][3]);
      arg.V[i].restore(&V_h[i], &Vnorm_h[i], bytes_[i][4], norm_bytes_[i][4]);
    }
  }

  // Need to check this!
  long long flops() const { return N*arg.r.flops()*vec_length<FloatN>::value*arg.length*M; }
  long long bytes() const {
    size_t bytes = N*arg.X[0].Precision()*vec_length<FloatN>::value*M;
    if (arg.X[0].Precision() == QUDA_HALF_PRECISION) bytes += N*sizeof(float);
    return arg.r.streams()*bytes*arg.length; }
  int tuningIter() const { return 3; }
};


template <int N, typename doubleN, typename ReduceType, typename RegType, typename StoreType,
  int M, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV>
  void multiReduceCuda(doubleN result[], const double2 &a, const double2 &b,
			  std::vector<cudaColorSpinorField*>& x, std::vector<cudaColorSpinorField*>& y,
			  std::vector<cudaColorSpinorField*>& z, std::vector<cudaColorSpinorField*>& w,
			  std::vector<cudaColorSpinorField*>& v, int length) {

  if (x[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    doubleN evenResult[N], oddResult[N];
    std::vector<cudaColorSpinorField> xp, yp, zp, wp, vp;
    std::vector<cudaColorSpinorField*> xpp, ypp, zpp, wpp, vpp;
    xp.reserve(N); yp.reserve(N); zp.reserve(N); wp.reserve(N); vp.reserve(N);
    xpp.reserve(N); ypp.reserve(N); zpp.reserve(N); wpp.reserve(N); vpp.reserve(N);

    for (int i=0; i<N; i++) {
      xp.push_back(x[i]->Even()); yp.push_back(y[i]->Even()); zp.push_back(z[i]->Even());
      wp.push_back(w[i]->Even()); vp.push_back(v[i]->Even());
      xpp.push_back(&xp[i]); ypp.push_back(&yp[i]); zpp.push_back(&zp[i]); wpp.push_back(&wp[i]); vpp.push_back(&vp[i]);
    }

    multiReduceCuda<N, doubleN, ReduceType, RegType, StoreType, M, Reducer, writeX, writeY, writeZ, writeW, writeV>
      (evenResult, a, b, xpp, ypp, zpp, wpp, vpp, length);

    for (int i=0; i<N; ++i) {
      xp.push_back(x[i]->Odd()); yp.push_back(y[i]->Odd()); zp.push_back(z[i]->Odd());
      wp.push_back(w[i]->Odd()); vp.push_back(v[i]->Odd());
      xpp.push_back(&xp[i]); ypp.push_back(&yp[i]); zpp.push_back(&zp[i]); wpp.push_back(&wp[i]); vpp.push_back(&vp[i]);
    }

    multiReduceCuda<N, doubleN, ReduceType, RegType, StoreType, M, Reducer, writeX, writeY, writeZ, writeW, writeV>
      (oddResult, a, b, xpp, ypp, zpp, wpp, vpp, length);

    for (int i=0; i<N; ++i) result[i] = evenResult[i] + oddResult[i];
    return;
  }

  memset(result, 0, N*sizeof(doubleN));

  for (int i=0; i<N; i++) {
    checkSpinor(*x[i],*y[i]); checkSpinor(*x[i],*z[i]); checkSpinor(*x[i],*w[i]); checkSpinor(*x[i],*v[i]);
    if (!x[i]->isNative()) {
      warningQuda("Reductions on non-native fields are not supported\n");
      return;
    }
  }

  blasStrings.vol_str = x[0]->VolString();
  blasStrings.aux_str = x[0]->AuxString();

  size_t **bytes = new size_t*[N], **norm_bytes = new size_t*[N];
  for (int i=0; i<N; i++) {
    bytes[i] = new size_t[5]; norm_bytes[i] = new size_t[5];
    bytes[i][0] = x[i]->Bytes(); bytes[i][1] = y[i]->Bytes(); bytes[i][2] = z[i]->Bytes();
    bytes[i][3] = w[i]->Bytes(); bytes[i][4] = v[i]->Bytes();
    norm_bytes[i][0] = x[i]->NormBytes(); norm_bytes[i][1] = y[i]->NormBytes(); norm_bytes[i][2] = z[i]->NormBytes();
    norm_bytes[i][3] = w[i]->NormBytes(); norm_bytes[i][4] = v[i]->NormBytes();
  }

  Spinor<RegType,StoreType,M,writeX,0> X[N];
  Spinor<RegType,StoreType,M,writeY,1> Y[N];
  Spinor<RegType,StoreType,M,writeZ,2> Z[N];
  Spinor<RegType,StoreType,M,writeW,3> W[N];
  Spinor<RegType,StoreType,M,writeV,4> V[N];

  for (int i=0; i<N; i++) { X[i].set(*x[i]); Y[i].set(*y[i]); Z[i].set(*z[i]); W[i].set(*w[i]); V[i].set(*v[i]); }

  typedef typename ::quda::scalar<RegType>::type Float;
  typedef typename ::quda::vector<Float,2>::type Float2;
  typedef ::quda::vector<Float,2> vec2;
  Reducer<ReduceType, Float2, RegType> r((Float2)vec2(a), (Float2)vec2(b));

  MultiReduceCuda<N,doubleN,ReduceType,RegType,M,
    Spinor<RegType,StoreType,M,writeX,0>,Spinor<RegType,StoreType,M,writeY,1>,
    Spinor<RegType,StoreType,M,writeZ,2>,Spinor<RegType,StoreType,M,writeW,3>,
    Spinor<RegType,StoreType,M,writeV,4>,Reducer<ReduceType, Float2, RegType> >
    reduce(result, X, Y, Z, W, V, r, length, bytes, norm_bytes);
  reduce.apply(*blas::getStream());

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  for (int i=0; i<N; i++) { delete []bytes[i]; delete []norm_bytes[i]; }
  delete []bytes;
  delete []norm_bytes;

  return;
}

template<int N, typename doubleN, typename ReduceType,
  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
  void multiReduceCuda(doubleN result[], const double2& a, const double2& b,
		       std::vector<cudaColorSpinorField*>& x, std::vector<cudaColorSpinorField*>& y,
		       std::vector<cudaColorSpinorField*>& z, std::vector<cudaColorSpinorField*>& w,
		       std::vector<cudaColorSpinorField*>& v){

  int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

  if (x[0]->Precision() == QUDA_DOUBLE_PRECISION) {
    if (x[0]->Nspin() == 4 || x[0]->Nspin() == 2) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
      const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
      if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
      multiReduceCuda<N,doubleN,ReduceType,double2,double2,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
      const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
      multiReduceCuda<N,doubleN,ReduceType,double2,double2,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  } else if (x[0]->Precision() == QUDA_SINGLE_PRECISION) {
    if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
      multiReduceCuda<N,doubleN,ReduceType,float4,float4,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, reduce_length/(4*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if(x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
      const int M = siteUnroll ? 3 : 1;
      if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
      multiReduceCuda<N,doubleN,ReduceType,float2,float2,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, reduce_length/(2*M));
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  } else { // half precision
    if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = 6;
      multiReduceCuda<N,doubleN,ReduceType,float4,short4,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, x[0]->Volume());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else if(x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
      const int M = 3;
      multiReduceCuda<N,doubleN,ReduceType,float2,short2,M,Reducer,writeX,writeY,writeZ,writeW,writeV>
	(result, a, b, x, y, z, w, v, x[0]->Volume());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
    } else { errorQuda("nSpin=%d is not supported\n", x[0]->Nspin()); }
  }

  // now do multi-node reduction
  const int Nreduce = N*(sizeof(doubleN)/sizeof(double));
  reduceDoubleArray((double*)result, Nreduce);

  return;
}
