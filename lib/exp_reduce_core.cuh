//EXPERIMENTAL:
template <typename ReduceType, typename SpinorX, typename SpinorP,
typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, 
typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
struct ReductionArgExp : public ReduceArg<ReduceType> {//defined in cub
  SpinorX X;
  SpinorP P;
  SpinorU U;
  SpinorR R;
  SpinorS S;
  SpinorM M;
  SpinorQ Q;
  SpinorW W;
  SpinorN N;
  SpinorZ Z;

  Reducer reduce_functor;

  const int length;
  ReductionArgExp(SpinorX X, SpinorP P, SpinorU U, SpinorR R, SpinorS S, SpinorM M, SpinorQ Q, SpinorW W, SpinorN N, SpinorZ Z, Reducer r, int length)
    : X(X), P(P), U(U), R(R), S(S), M(M), Q(Q), W(W), N(N), Z(Z), reduce_functor(r), length(length) { ; }
};

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename FloatN, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
__global__ void reduceKernelExp(ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  ::quda::zero(sum);

  while (i < arg.length) {
    FloatN x[M_], p[M_], u[M_], r[M_], s[M_],m[M_], q[M_], w[M_], n[M_], z[M_];
    arg.X.load(x, i, parity);
    arg.P.load(p, i, parity);
    arg.U.load(u, i, parity);
    arg.R.load(r, i, parity);
    arg.S.load(s, i, parity);

    arg.M.load(m, i, parity);
    arg.Q.load(q, i, parity);
    arg.W.load(w, i, parity);
    arg.N.load(n, i, parity);
    arg.Z.load(z, i, parity);


    arg.reduce_functor.pre();

#pragma unroll
    for (int j=0; j<M_; j++) arg.reduce_functor(sum, x[j], p[j], u[j], r[j], s[j], m[j], q[j], w[j], n[j], z[j]);

    arg.reduce_functor.post(sum);

    arg.X.save(x, i, parity);
    arg.P.save(p, i, parity);
    arg.U.save(u, i, parity);
    arg.R.save(r, i, parity);
    arg.S.save(s, i, parity);

    arg.M.save(m, i, parity);
    arg.Q.save(q, i, parity);
    arg.W.save(w, i, parity);
    arg.N.save(n, i, parity);
    arg.Z.save(z, i, parity);

    i += gridSize;
  }

  ::quda::reduce<block_size, ReduceType>(arg, sum, parity);
}


/**
   Generic reduction kernel launcher
*/
template <typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, 
          typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
doubleN reduceLaunchExp(ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ, Reducer> &arg,
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


template <typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
class ReduceCudaExp : public Tunable {

private:
  mutable ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ, Reducer> arg;
  doubleN &result;
  const int nParity;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X_h, *P_h, *U_h, *R_h, *S_h, *M_h, *Q_h, *W_h, *N_h, *Z_h;
  char *Xnorm_h, *Pnorm_h, *Unorm_h, *Rnorm_h, *Snorm_h, *Mnorm_h, *Qnorm_h, *Wnorm_h, *Nnorm_h, *Znorm_h;
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
  ReduceCudaExp(doubleN &result, SpinorX &X, SpinorP &P, SpinorU &U,
	     SpinorR &R, SpinorS &S, SpinorM &M, SpinorQ &Q, SpinorW &W, SpinorN &N, SpinorZ &Z, Reducer &r, int length, int nParity,
	     const size_t *bytes, const size_t *norm_bytes) :
    arg(X, P, U, R, S, M, Q, W, N, Z, r, length/nParity),
    result(result), nParity(nParity), X_h(0), P_h(0), U_h(0), R_h(0), S_h(0), M_h(0), Q_h(0), W_h(0), N_h(0), Z_h(0), 
    Xnorm_h(0), Pnorm_h(0), Unorm_h(0), Rnorm_h(0), Snorm_h(0), Mnorm_h(0), Qnorm_h(0), Wnorm_h(0), Nnorm_h(0), Znorm_h(0),
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
    arg.X.backup(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.P.backup(&P_h, &Pnorm_h, bytes_[1], norm_bytes_[1]);
    arg.U.backup(&U_h, &Unorm_h, bytes_[2], norm_bytes_[2]);
    arg.R.backup(&R_h, &Rnorm_h, bytes_[3], norm_bytes_[3]);
    arg.S.backup(&S_h, &Snorm_h, bytes_[4], norm_bytes_[4]);
    arg.M.backup(&M_h, &Mnorm_h, bytes_[5], norm_bytes_[5]);
    arg.Q.backup(&Q_h, &Qnorm_h, bytes_[6], norm_bytes_[6]);
    arg.W.backup(&W_h, &Wnorm_h, bytes_[7], norm_bytes_[7]);
    arg.N.backup(&N_h, &Nnorm_h, bytes_[8], norm_bytes_[8]);
    arg.Z.backup(&Z_h, &Znorm_h, bytes_[9], norm_bytes_[9]);
  }

  void postTune() {
    arg.X.restore(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.P.restore(&P_h, &Pnorm_h, bytes_[1], norm_bytes_[1]);
    arg.U.restore(&U_h, &Unorm_h, bytes_[2], norm_bytes_[2]);
    arg.R.restore(&R_h, &Rnorm_h, bytes_[3], norm_bytes_[3]);
    arg.S.restore(&S_h, &Snorm_h, bytes_[4], norm_bytes_[4]);
    arg.M.restore(&M_h, &Mnorm_h, bytes_[5], norm_bytes_[5]);
    arg.Q.restore(&Q_h, &Qnorm_h, bytes_[6], norm_bytes_[6]);
    arg.W.restore(&W_h, &Wnorm_h, bytes_[7], norm_bytes_[7]);
    arg.N.restore(&N_h, &Nnorm_h, bytes_[8], norm_bytes_[8]);
    arg.Z.restore(&Z_h, &Znorm_h, bytes_[9], norm_bytes_[9]);
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


template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename zType,
	  int M_, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeX,int writeP,int writeU,int writeR, int writeS, int writeM,int writeQ,int writeW,int writeN, int writeZ>
doubleN reduceCudaExp(const double2 &a, const double2 &b, ColorSpinorField &x, 
		   ColorSpinorField &p, ColorSpinorField &u, ColorSpinorField &r,
                   ColorSpinorField &s, ColorSpinorField &m, ColorSpinorField &q, 
                   ColorSpinorField &w, ColorSpinorField &n, ColorSpinorField &z,
		   int length) {

  checkLength(x, p); checkLength(x, u); checkLength(x, r); checkLength(x, s);
  checkLength(x, m); checkLength(x, q); checkLength(x, w); checkLength(x, n);
  checkLength(x, z);

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

  size_t bytes[] = {x.Bytes(), p.Bytes(), u.Bytes(), r.Bytes(), s.Bytes(), m.Bytes(), q.Bytes(), w.Bytes(), n.Bytes(), z.Bytes()};
  size_t norm_bytes[] = {x.NormBytes(), p.NormBytes(), u.NormBytes(), r.NormBytes(), s.NormBytes(), m.NormBytes(), q.NormBytes(), w.NormBytes(), n.NormBytes(), z.NormBytes()};

  Spinor<RegType,StoreType,M_,writeX,0> X(x);
  Spinor<RegType,StoreType,M_,writeP,1> P(p);
  Spinor<RegType,StoreType,M_,writeU,2> U(u);
  Spinor<RegType,StoreType,M_,writeR,3> R(r);
  Spinor<RegType,StoreType,M_,writeS,4> S(s);
  Spinor<RegType,StoreType,M_,writeM,5> M(m);
  Spinor<RegType,StoreType,M_,writeQ,6> Q(q);
  Spinor<RegType,StoreType,M_,writeW,7> W(w);
  Spinor<RegType,StoreType,M_,writeN,8> N(n);
  Spinor<RegType,    zType,M_,writeZ,9> Z(z);//zType same as StoreType, just not to forget

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  Reducer<ReduceType, Float2, RegType> reducer_((Float2)vec2(a), (Float2)vec2(b));
  doubleN value;

  int partitions = (x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset());

  ReduceCudaExp<doubleN,ReduceType,RegType,M_,
  Spinor<RegType,StoreType,M_,writeX,0>,
  Spinor<RegType,StoreType,M_,writeP,1>,
  Spinor<RegType,StoreType,M_,writeU,2>,
  Spinor<RegType,StoreType,M_,writeR,3>,
  Spinor<RegType,StoreType,M_,writeS,4>,
  Spinor<RegType,StoreType,M_,writeM,5>,
  Spinor<RegType,StoreType,M_,writeQ,6>,
  Spinor<RegType,StoreType,M_,writeW,7>,
  Spinor<RegType,StoreType,M_,writeN,8>,
  Spinor<RegType,    zType,M_,writeZ,9>,
    Reducer<ReduceType, Float2, RegType> >
    reduce(value, X, P, U, R, S, M, Q, W, N, Z, reducer_, length, partitions, bytes, norm_bytes);
  reduce.apply(*(blas::getStream()));

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return value;
}


