//EXPERIMENTAL:
template <typename ReduceType, typename SpinorS, typename SpinorP,
typename SpinorZ, typename SpinorR, typename SpinorX, typename SpinorW, 
typename SpinorQ, typename Reducer>
struct ReductionArgExp : public ReduceArg<ReduceType> {//defined in cub
  SpinorS S;
  SpinorP P;
  SpinorZ Z;
  SpinorR R;
  SpinorX X;
  SpinorW W;
  SpinorQ Q;

  Reducer reduce_functor;

  const int length;
  ReductionArgExp(SpinorS S, SpinorP P, SpinorZ Z, SpinorR R, SpinorX X, SpinorW W, SpinorQ Q, Reducer r, int length)
    : S(S), P(P), Z(Z), R(R), X(X), W(W), Q(Q), reduce_functor(r), length(length) { ; }
};

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename FloatN, int M_, typename SpinorS,
	  typename SpinorP, typename SpinorZ, typename SpinorR, typename SpinorX, typename SpinorW, typename SpinorQ, typename Reducer>
__global__ void reduceKernelExp(ReductionArgExp<ReduceType,SpinorS,SpinorP,SpinorZ,SpinorR,SpinorX,SpinorW,SpinorQ,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  ::quda::zero(sum);

  while (i < arg.length) {
    FloatN s[M_], p[M_], z[M_], r[M_], x[M_],w[M_], q[M_];
    arg.S.load(s, i, parity);
    arg.P.load(p, i, parity);
    arg.Z.load(z, i, parity);
    arg.R.load(r, i, parity);
    arg.X.load(x, i, parity);

    arg.W.load(w, i, parity);
    arg.Q.load(q, i, parity);

    arg.reduce_functor.pre();

#pragma unroll
    for (int j=0; j<M_; j++) arg.reduce_functor(sum, s[j], p[j], z[j], r[j], x[j], w[j], q[j]);

    arg.reduce_functor.post(sum);

    arg.S.save(s, i, parity);
    arg.P.save(p, i, parity);
    arg.Z.save(z, i, parity);
    arg.R.save(r, i, parity);
    arg.X.save(x, i, parity);

    arg.W.save(w, i, parity);
    arg.Q.save(q, i, parity);

    i += gridSize;
  }

  ::quda::reduce<block_size, ReduceType>(arg, sum, parity);
}


/**
   Generic reduction kernel launcher
*/
template <typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorS,
	  typename SpinorP, typename SpinorZ, typename SpinorR, typename SpinorX, typename SpinorW, 
          typename SpinorQ, typename Reducer>
doubleN reduceLaunchExp(ReductionArgExp<ReduceType,SpinorS,SpinorP,SpinorZ,SpinorR,SpinorX,SpinorW,SpinorQ,Reducer> &arg,
		     const TuneParam &tp, const cudaStream_t &stream) {
  if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
     errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

  LAUNCH_KERNEL(reduceKernelExp,tp,stream,arg,ReduceType,FloatN,M_);

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


template <typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorS,
	  typename SpinorP, typename SpinorZ, typename SpinorR, typename SpinorX, typename SpinorW, typename SpinorQ, typename Reducer>
class ReduceCudaExp : public Tunable {

private:
  mutable ReductionArgExp<ReduceType,SpinorS,SpinorP,SpinorZ,SpinorR,SpinorX,SpinorW,SpinorQ, Reducer> arg;
  doubleN &result;
  const int nParity;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *S_h, *P_h, *Z_h, *R_h, *X_h, *W_h, *Q_h;
  char *Snorm_h, *Pnorm_h, *Znorm_h, *Rnorm_h, *Xnorm_h, *Wnorm_h, *Qnorm_h;
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
  ReduceCudaExp(doubleN &result, SpinorS &S, SpinorP &P, SpinorZ &Z,
	     SpinorR &R, SpinorX &X, SpinorW &W, SpinorQ &Q, Reducer &r, int length, int nParity,
	     const size_t *bytes, const size_t *norm_bytes) :
    arg(S, P, Z, R, X, W, Q, r, length/nParity),
    result(result), nParity(nParity), S_h(0), P_h(0), Z_h(0), R_h(0), X_h(0), W_h(0), Q_h(0),  
    Snorm_h(0), Pnorm_h(0), Znorm_h(0), Rnorm_h(0), Xnorm_h(0), Wnorm_h(0), Qnorm_h(0),
    bytes_(bytes), norm_bytes_(norm_bytes) { }
  virtual ~ReduceCudaExp() { }

  inline TuneKey tuneKey() const {
    return TuneKey(blasStrings.vol_str, typeid(arg.reduce_functor).name(), blasStrings.aux_tmp);
  }

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    result = reduceLaunchExp<doubleN,ReduceType,FloatN,M_>(arg, tp, stream);
  }

  void preTune() {
    arg.S.backup(&S_h, &Snorm_h, bytes_[0], norm_bytes_[0]);
    arg.P.backup(&P_h, &Pnorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.backup(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.R.backup(&R_h, &Rnorm_h, bytes_[3], norm_bytes_[3]);
    arg.X.backup(&X_h, &Xnorm_h, bytes_[4], norm_bytes_[4]);
    arg.W.backup(&W_h, &Wnorm_h, bytes_[5], norm_bytes_[5]);
    arg.Q.backup(&Q_h, &Qnorm_h, bytes_[6], norm_bytes_[6]);
  }

  void postTune() {
    arg.S.restore(&S_h, &Snorm_h, bytes_[0], norm_bytes_[0]);
    arg.P.restore(&P_h, &Pnorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.restore(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.R.restore(&R_h, &Rnorm_h, bytes_[3], norm_bytes_[3]);
    arg.X.restore(&X_h, &Xnorm_h, bytes_[4], norm_bytes_[4]);
    arg.W.restore(&W_h, &Wnorm_h, bytes_[5], norm_bytes_[5]);
    arg.Q.restore(&Q_h, &Qnorm_h, bytes_[6], norm_bytes_[6]);
  }

  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.grid.y = nParity;
  }

  void defaultTuneParam(TuneParam &param) const {
    Tunable::defaultTuneParam(param);
    param.grid.y = nParity;
  }

  long long flops() const { return arg.reduce_functor.flops()*vec_length<FloatN>::value*arg.length*nParity*M_; }

  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X.Precision()*vec_length<FloatN>::value*M_;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.R.Precision()*vec_length<FloatN>::value*M_;
    if (arg.R.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
    return ((arg.reduce_functor.streams()-2)*base_bytes + 2*extra_bytes)*arg.length*nParity;
  }

  int tuningIter() const { return 3; }
};


template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename qType,
	  int M_, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeS,int writeP,int writeZ,int writeR, int writeX, int writeW,int writeQ>
doubleN reduceCudaExp(const double2 &a, const double2 &b, ColorSpinorField &s, 
		   ColorSpinorField &p, ColorSpinorField &z, ColorSpinorField &r,
                   ColorSpinorField &x, ColorSpinorField &w, ColorSpinorField &q, 
		   int length) {

  checkLength(x, p); checkLength(x, r); checkLength(x, s);
  checkLength(x, w); checkLength(x, q); checkLength(x, z); 

  if (!x.isNative()) {
    warningQuda("Device reductions on non-native fields is not supported\n");
    doubleN value;
    ::quda::zero(value);
    return value;
  }

  blasStrings.vol_str = x.VolString();
  strcpy(blasStrings.aux_tmp, x.AuxString());
  if (typeid(StoreType) != typeid(qType)) {
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, z.AuxString());
  }

  size_t bytes[] = {s.Bytes(), p.Bytes(), z.Bytes(), r.Bytes(), x.Bytes(), w.Bytes(), q.Bytes()};
  size_t norm_bytes[] = {s.NormBytes(), p.NormBytes(), z.NormBytes(), r.NormBytes(), x.NormBytes(), w.NormBytes(), q.NormBytes()};

  Spinor<RegType,StoreType,M_,writeS,0> S(s);
  Spinor<RegType,StoreType,M_,writeP,1> P(p);
  Spinor<RegType,StoreType,M_,writeZ,2> Z(z);
  Spinor<RegType,StoreType,M_,writeR,3> R(r);
  Spinor<RegType,StoreType,M_,writeX,4> X(x);
  Spinor<RegType,StoreType,M_,writeW,5> W(w);
  Spinor<RegType,    qType,M_,writeQ,6> Q(q);//qType is same as zTypr

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  Reducer<ReduceType, Float2, RegType> reducer_((Float2)vec2(a), (Float2)vec2(b));
  doubleN value;

  int partitions = (x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset());

  ReduceCudaExp<doubleN,ReduceType,RegType,M_,
  Spinor<RegType,StoreType,M_,writeS,0>,
  Spinor<RegType,StoreType,M_,writeP,1>,
  Spinor<RegType,StoreType,M_,writeZ,2>,
  Spinor<RegType,StoreType,M_,writeR,3>,
  Spinor<RegType,StoreType,M_,writeX,4>,
  Spinor<RegType,StoreType,M_,writeW,5>,
  Spinor<RegType,    qType,M_,writeQ,6>,
    Reducer<ReduceType, Float2, RegType> >
    reduce(value, S, P, Z, R, X, W, Q, reducer_, length, partitions, bytes, norm_bytes);
  reduce.apply(*(blas::getStream()));

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return value;
}


