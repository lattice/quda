__host__ __device__ double set(double &x) { return x;}
__host__ __device__ double2 set(double2 &x) { return x;}
__host__ __device__ double3 set(double3 &x) { return x;}

#ifdef QUAD_SUM
__host__ __device__ double set(doubledouble &a) { return a.head(); }
__host__ __device__ double2 set(doubledouble2 &a) { return make_double2(a.x.head(),a.y.head()); }
__host__ __device__ double3 set(doubledouble3 &a) { return make_double3(a.x.head(),a.y.head(),a.z.head()); }
#endif

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

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
template <int block_size, typename ReduceType, typename ReduceSimpleType,
	  typename FloatN, int M, typename SpinorX, typename SpinorY,
	  typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
__global__ void reduceKernel(ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  ::quda::zero(sum);

  while (i < arg.length) {
    FloatN x[M], y[M], z[M], w[M], v[M];
    arg.X.load(x, i);
    arg.Y.load(y, i);
    arg.Z.load(z, i);
    arg.W.load(w, i);
    arg.V.load(v, i);

    arg.r.pre();

#pragma unroll
    for (int j=0; j<M; j++) arg.r(sum, x[j], y[j], z[j], w[j], v[j]);

    arg.r.post(sum);

    arg.X.save(x, i);
    arg.Y.save(y, i);
    arg.Z.save(z, i);
    arg.W.save(w, i);
    arg.V.save(v, i);

    i += gridSize;
  }

  ::quda::reduce<block_size, ReduceType>(arg, sum);
}


/**
   Generic reduction kernel launcher
*/
template <typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN,
  int M, typename SpinorX, typename SpinorY, typename SpinorZ,
  typename SpinorW, typename SpinorV, typename Reducer>
doubleN reduceLaunch(ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> &arg,
		     const TuneParam &tp, const cudaStream_t &stream) {
  if (tp.grid.x > REDUCE_MAX_BLOCKS)
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, REDUCE_MAX_BLOCKS);

  LAUNCH_KERNEL(reduceKernel,tp,stream,arg,ReduceType,ReduceSimpleType,FloatN,M);

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
  if(deviceProp.canMapHostMemory) {
    cudaEventRecord(reduceEnd, stream);
    while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
  } else
#endif
    { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType), cudaMemcpyDeviceToHost); }

  doubleN cpu_sum = set(((ReduceType*)h_reduce)[0]);

  return cpu_sum;
}


template <typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN,
  int M, typename SpinorX, typename SpinorY, typename SpinorZ,
  typename SpinorW, typename SpinorV, typename Reducer>
class ReduceCuda : public Tunable {

private:
  mutable ReductionArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg;
  doubleN &result;

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
	     SpinorW &W, SpinorV &V, Reducer &r, int length,
	     const size_t *bytes, const size_t *norm_bytes) :
    arg(X, Y, Z, W, V, r, length),
    result(result), X_h(0), Y_h(0), Z_h(0), W_h(0), V_h(0),
    Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), Vnorm_h(0),
    bytes_(bytes), norm_bytes_(norm_bytes) { }
  virtual ~ReduceCuda() { }

  inline TuneKey tuneKey() const {
    return TuneKey(blasStrings.vol_str, typeid(arg.r).name(), blasStrings.aux_tmp);
  }

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    result = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,FloatN,M>(arg, tp, stream);
  }

  void preTune() {
    arg.X.save(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.save(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.save(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.save(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
    arg.V.save(&V_h, &Vnorm_h, bytes_[4], norm_bytes_[4]);
  }

  void postTune() {
    arg.X.load(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.load(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.load(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.load(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
    arg.V.load(&V_h, &Vnorm_h, bytes_[4], norm_bytes_[4]);
  }

  long long flops() const { return arg.r.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }

  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (arg.Y.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
    return ((arg.r.streams()-2)*base_bytes + 2*extra_bytes)*arg.length;
  }

  int tuningIter() const { return 3; }
};

double2 make_Float2(double x, double y) {
  return make_double2( x, y );
}

float2 make_Float2(float x, float y) {
  return make_float2( x, y );
}

double2 make_Float2(const std::complex<double> &a) {
  return make_double2( real(a), imag(a) );
}

float2 make_Float2(const std::complex<float> &a) {
  return make_float2( real(a), imag(a) );
}

std::complex<double> make_Complex(const double2 &a) {
  return std::complex<double>(a.x, a.y);
}

std::complex<float> make_Complex(const float2 &a) {
  return std::complex<float>(a.x, a.y);
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
	  Float2 X2 = make_Float2( X(parity, x, s, c) );
	  Float2 Y2 = make_Float2( Y(parity, x, s, c) );
	  Float2 Z2 = make_Float2( Z(parity, x, s, c) );
	  Float2 W2 = make_Float2( W(parity, x, s, c) );
	  Float2 V2 = make_Float2( V(parity, x, s, c) );
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

template <typename ReduceType, typename Float, int nSpin, int nColor, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
			   ColorSpinorField &w, ColorSpinorField &v, R r) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Y(y), Z(z), W(w), V(v);
  typedef typename vector<Float,2>::type Float2;
  return genericReduce<ReduceType,Float2,writeX,writeY,writeZ,writeW,writeV>(X, Y, Z, W, V, r);
}

template <typename ReduceType, typename Float, int nSpin, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
			   ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  if (x.Ncolor() == 2) {
    value = genericReduce<ReduceType,Float,nSpin,2,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 3) {
    value = genericReduce<ReduceType,Float,nSpin,3,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 4) {
    value = genericReduce<ReduceType,Float,nSpin,4,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 6) {
    value = genericReduce<ReduceType,Float,nSpin,6,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 8) {
    value = genericReduce<ReduceType,Float,nSpin,8,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 12) {
    value = genericReduce<ReduceType,Float,nSpin,12,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 16) {
    value = genericReduce<ReduceType,Float,nSpin,16,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 20) {
    value = genericReduce<ReduceType,Float,nSpin,20,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 24) {
    value = genericReduce<ReduceType,Float,nSpin,24,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 32) {
    value = genericReduce<ReduceType,Float,nSpin,32,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 72) {
    value = genericReduce<ReduceType,Float,nSpin,72,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Ncolor() == 576) {
    value = genericReduce<ReduceType,Float,nSpin,576,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else {
    ::quda::zero(value);
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
  return value;
}

template <typename ReduceType, typename Float, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
  ReduceType genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  ::quda::zero(value);
  if (x.Nspin() == 4) {
    value = genericReduce<ReduceType,Float,4,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
  } else if (x.Nspin() == 2) {
    value = genericReduce<ReduceType,Float,2,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    value = genericReduce<ReduceType,Float,1,order,writeX,writeY,writeZ,writeW,writeV,R>(x, y, z, w, v, r);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
  return value;
}

template <typename doubleN, typename ReduceType, typename Float,
	  int writeX, int writeY, int writeZ, int writeW, int writeV, typename R>
doubleN genericReduce(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
		      ColorSpinorField &w, ColorSpinorField &v, R r) {
  ReduceType value;
  ::quda::zero(value);
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    value = genericReduce<ReduceType,Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,writeV,R>
      (x, y, z, w, v, r);
  } else {
    errorQuda("Not implemeneted");
  }
  return set(value);
}
