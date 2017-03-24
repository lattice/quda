__host__ __device__ inline double set(double &x) { return x;}
__host__ __device__ inline double2 set(double2 &x) { return x;}
__host__ __device__ inline double3 set(double3 &x) { return x;}
__host__ __device__ inline double4 set(double4 &x) { return x;}
__host__ __device__ inline void sum(double &a, double &b) { a += b; }
__host__ __device__ inline void sum(double2 &a, double2 &b) { a.x += b.x; a.y += b.y; }
__host__ __device__ inline void sum(double3 &a, double3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__host__ __device__ inline void sum(double4 &a, double4 &b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

#ifdef QUAD_SUM
__host__ __device__ inline double set(doubledouble &a) { return a.head(); }
__host__ __device__ inline double2 set(doubledouble2 &a) { return make_double2(a.x.head(),a.y.head()); }
__host__ __device__ inline double3 set(doubledouble3 &a) { return make_double3(a.x.head(),a.y.head(),a.z.head()); }
__host__ __device__ inline void sum(double &a, doubledouble &b) { a += b.head(); }
__host__ __device__ inline void sum(double2 &a, doubledouble2 &b) { a.x += b.x.head(); a.y += b.y.head(); }
__host__ __device__ inline void sum(double3 &a, doubledouble3 &b) { a.x += b.x.head(); a.y += b.y.head(); a.z += b.z.head(); }
#endif

__device__ static unsigned int count = 0;
__shared__ static bool isLastBlockDone;

#include <launch_kernel.cuh>

template <typename ReduceType, typename SpinorX, typename SpinorY,
typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
struct ReductionArg : public ReduceArg<ReduceType> {
  SpinorX X;
  SpinorY Y;
  SpinorZ Z;
  SpinorW W;
  SpinorV V;
  Reducer r;
  const int length;
  ReductionArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, SpinorV V, Reducer r, int length)
    : X(X), Y(Y), Z(Z), W(W), V(V), r(r), length(length) { ; }
};

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename FloatN, int M, typename SpinorX,
	  typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
__global__ void reduceKernel(ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  ::quda::zero(sum);

  while (i < arg.length) {
    FloatN x[M], y[M], z[M], w[M], v[M];
    arg.X.load(x, i, parity);
    arg.Y.load(y, i, parity);
    arg.Z.load(z, i, parity);
    arg.W.load(w, i, parity);
    arg.V.load(v, i, parity);

    arg.r.pre();

#pragma unroll
    for (int j=0; j<M; j++) arg.r(sum, x[j], y[j], z[j], w[j], v[j]);

    arg.r.post(sum);

    arg.X.save(x, i, parity);
    arg.Y.save(y, i, parity);
    arg.Z.save(z, i, parity);
    arg.W.save(w, i, parity);
    arg.V.save(v, i, parity);

    i += gridSize;
  }

  ::quda::reduce<block_size, ReduceType>(arg, sum, parity);
}


/**
   Generic reduction kernel launcher
*/
template <typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX,
	  typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
doubleN reduceLaunch(ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> &arg,
		     const TuneParam &tp, const cudaStream_t &stream) {
  if (tp.grid.x > REDUCE_MAX_BLOCKS)
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, REDUCE_MAX_BLOCKS);

  LAUNCH_KERNEL(reduceKernel,tp,stream,arg,ReduceType,FloatN,M);

  if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
    if(deviceProp.canMapHostMemory) {
      cudaEventRecord(reduceEnd, stream);
      while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
    } else
#endif
      { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType), cudaMemcpyDeviceToHost); }
  }
  doubleN cpu_sum = set(((ReduceType*)h_reduce)[0]);
  if (tp.grid.y==2) sum(cpu_sum, ((ReduceType*)h_reduce)[1]); // add other parity if needed
  return cpu_sum;
}


template <typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX,
	  typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
class ReduceCuda : public Tunable {

private:
  mutable ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg;
  doubleN &result;
  const int nParity;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X_h, *Y_h, *Z_h, *W_h, *V_h;
  char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h, *Vnorm_h;
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
  ReduceCuda(doubleN &result, SpinorX &X, SpinorY &Y, SpinorZ &Z,
	     SpinorW &W, SpinorV &V, Reducer &r, int length, int nParity,
	     const size_t *bytes, const size_t *norm_bytes) :
    arg(X, Y, Z, W, V, r, length/nParity),
    result(result), nParity(nParity), X_h(0), Y_h(0), Z_h(0), W_h(0), V_h(0),
    Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), Vnorm_h(0),
    bytes_(bytes), norm_bytes_(norm_bytes) { }
  virtual ~ReduceCuda() { }

  inline TuneKey tuneKey() const {
    return TuneKey(blasStrings.vol_str, typeid(arg.r).name(), blasStrings.aux_tmp);
  }

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    result = reduceLaunch<doubleN,ReduceType,FloatN,M>(arg, tp, stream);
  }

  void preTune() {
    arg.X.backup(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.backup(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.backup(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.backup(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
    arg.V.backup(&V_h, &Vnorm_h, bytes_[4], norm_bytes_[4]);
  }

  void postTune() {
    arg.X.restore(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.restore(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.restore(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.restore(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
    arg.V.restore(&V_h, &Vnorm_h, bytes_[4], norm_bytes_[4]);
  }

  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.grid.y = nParity;
  }

  void defaultTuneParam(TuneParam &param) const {
    Tunable::defaultTuneParam(param);
    param.grid.y = nParity;
  }

  long long flops() const { return arg.r.flops()*vec_length<FloatN>::value*arg.length*nParity*M; }

  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X.Precision()*vec_length<FloatN>::value*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y.Precision()*vec_length<FloatN>::value*M;
    if (arg.Y.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
    return ((arg.r.streams()-2)*base_bytes + 2*extra_bytes)*arg.length*nParity;
  }

  int tuningIter() const { return 3; }
};


template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename zType,
	  int M, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeX, int writeY, int writeZ, int writeW, int writeV>
doubleN reduceCuda(const double2 &a, const double2 &b,
		   ColorSpinorField &x, ColorSpinorField &y,
		   ColorSpinorField &z, ColorSpinorField &w,
		   ColorSpinorField &v, int length) {

  checkLength(x, y); checkLength(x, z); checkLength(x, w); checkLength(x, v);

  if (!x.isNative()) {
    warningQuda("Device reductions on non-native fields is not supported\n");
    doubleN value;
    ::quda::zero(value);
    return value;
  }

  blasStrings.vol_str = x.VolString();
  strcpy(blasStrings.aux_tmp, x.AuxString());
  if (typeid(StoreType) != typeid(zType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, z.AuxString());
  }

  size_t bytes[] = {x.Bytes(), y.Bytes(), z.Bytes(), w.Bytes()};
  size_t norm_bytes[] = {x.NormBytes(), y.NormBytes(), z.NormBytes(), w.NormBytes()};

  Spinor<RegType,StoreType,M,writeX,0> X(x);
  Spinor<RegType,StoreType,M,writeY,1> Y(y);
  Spinor<RegType,    zType,M,writeZ,2> Z(z);
  Spinor<RegType,StoreType,M,writeW,3> W(w);
  Spinor<RegType,StoreType,M,writeV,4> V(v);

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  Reducer<ReduceType, Float2, RegType> r((Float2)vec2(a), (Float2)vec2(b));
  doubleN value;

  ReduceCuda<doubleN,ReduceType,RegType,M,
    Spinor<RegType,StoreType,M,writeX,0>,
    Spinor<RegType,StoreType,M,writeY,1>,
    Spinor<RegType,    zType,M,writeZ,2>,
    Spinor<RegType,StoreType,M,writeW,3>,
    Spinor<RegType,StoreType,M,writeV,4>,
    Reducer<ReduceType, Float2, RegType> >
    reduce(value, X, Y, Z, W, V, r, length, x.SiteSubset(), bytes, norm_bytes);
  reduce.apply(*(blas::getStream()));

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return value;
}

/**
   Generic reduce kernel with four loads and up to four stores.
   FIXME - this is hacky due to the lack of std::complex support in
   CUDA.  The functors are defined in terms of FloatN vectors, whereas
   the operator() accessor returns std::complex<Float>
  */
template <typename ReduceType, typename Float2, int writeX, int writeY, int writeZ,
  int writeW, int writeV, typename SpinorX, typename SpinorY, typename SpinorZ,
  typename SpinorW, typename SpinorV, typename Reducer>
ReduceType genericReduce(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Reducer r) {

  ReduceType sum;
  ::quda::zero(sum);

  for (int parity=0; parity<X.Nparity(); parity++) {
    for (int x=0; x<X.VolumeCB(); x++) {
      r.pre();
      for (int s=0; s<X.Nspin(); s++) {
	for (int c=0; c<X.Ncolor(); c++) {
	  Float2 X2 = make_Float2<Float2>( X(parity, x, s, c) );
	  Float2 Y2 = make_Float2<Float2>( Y(parity, x, s, c) );
	  Float2 Z2 = make_Float2<Float2>( Z(parity, x, s, c) );
	  Float2 W2 = make_Float2<Float2>( W(parity, x, s, c) );
	  Float2 V2 = make_Float2<Float2>( V(parity, x, s, c) );
	  r(sum, X2, Y2, Z2, W2, V2);
	  if (writeX) X(parity, x, s, c) = make_Complex(X2);
	  if (writeY) Y(parity, x, s, c) = make_Complex(Y2);
	  if (writeZ) Z(parity, x, s, c) = make_Complex(Z2);
	  if (writeW) W(parity, x, s, c) = make_Complex(W2);
	  if (writeV) V(parity, x, s, c) = make_Complex(V2);
	}
      }
      r.post(sum);
    }
  }

  return sum;
}

template<typename, int N> struct vector { };
template<> struct vector<double, 2> { typedef double2 type; };
template<> struct vector<float, 2> { typedef float2 type; };

template <typename ReduceType, typename Float, typename zFloat, int nSpin, int nColor, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
			   ColorSpinorField &w, ColorSpinorField &v, R r) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Y(y), W(w), V(v);
  colorspinor::FieldOrderCB<zFloat,nSpin,nColor,1,order> Z(z);
  typedef typename vector<zFloat,2>::type Float2;
  return genericReduce<ReduceType,Float2,writeX,writeY,writeZ,writeW,writeV>(X, Y, Z, W, V, r);
}

template <typename ReduceType, typename Float, typename zFloat, int nSpin, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
			   ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  if (x.Ncolor() == 2) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,2,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 3) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,3,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 4) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,4,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 6) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,6,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 8) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,8,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 12) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,12,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 16) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,16,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 20) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,20,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 24) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,24,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 32) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,32,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 72) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,72,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 576) {
    value = genericReduce<ReduceType,Float,zFloat,nSpin,576,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else {
    ::quda::zero(value);
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
  return value;
}

template <typename ReduceType, typename Float, typename zFloat, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  ::quda::zero(value);
  if (x.Nspin() == 4) {
    value = genericReduce<ReduceType,Float,zFloat,4,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Nspin() == 2) {
    value = genericReduce<ReduceType,Float,zFloat,2,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    value = genericReduce<ReduceType,Float,zFloat,1,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
  return value;
}

template <typename doubleN, typename ReduceType, typename Float, typename zFloat,
	  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
doubleN genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
		      ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  ::quda::zero(value);
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    value = genericReduce<ReduceType,Float,zFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,writeV,R>
      (x, y, z, w, v, r);
  } else {
    warningQuda("CPU reductions not implemeneted for %d field order", x.FieldOrder());
  }
  return set(value);
}
