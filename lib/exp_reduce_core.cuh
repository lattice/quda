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
template <int block_size, typename ReduceType, typename FloatN, int Nreduce, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
__global__ void reduceKernelExp(ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum[Nreduce];
#pragma unroll
  for (int j=0; j<Nreduce;j++) ::quda::zero(sum[j]);

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

  ::quda::array_reduce<Nreduce, block_size, ReduceType>(arg, sum, parity);
}


/**
   Generic reduction kernel launcher
*/
template <int Nreduce, typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, 
          typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
void reduceLaunchExp(doubleN cpu_sum[Nreduce], ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ, Reducer> &arg,
		     const TuneParam &tp, const cudaStream_t &stream) {
  if (Nreduce < 0 or Nreduce > 8) errorQuda("Incorrect size of the reduce array: %d.\n", Nreduce);
  if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
     errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

  LAUNCH_KERNEL(reduceKernelExp,tp,stream,arg,ReduceType,FloatN,Nreduce,M_);

  if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
    if(deviceProp.canMapHostMemory) {
      cudaEventRecord(reduceEnd, stream);
      while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
    } else
#endif
    { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType)*Nreduce, cudaMemcpyDeviceToHost); }
  }

#pragma unroll
  for (int j=0; j<Nreduce;j++) cpu_sum[j] = set(((ReduceType*)h_reduce)[0+j*tp.grid.y]);
  if (tp.grid.y==2) {
#pragma unroll 
    for (int j=0; j<Nreduce;j++) sum(cpu_sum[j], ((ReduceType*)h_reduce)[1+j*tp.grid.y]); // add other parity if needed
  }
  return;
}


template <int Nreduce, typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX,
	  typename SpinorP, typename SpinorU, typename SpinorR, typename SpinorS, typename SpinorM, typename SpinorQ, typename SpinorW, typename SpinorN, typename SpinorZ, typename Reducer>
class ReduceCudaExp : public Tunable {

private:
  mutable ReductionArgExp<ReduceType,SpinorX,SpinorP,SpinorU,SpinorR,SpinorS,SpinorM,SpinorQ,SpinorW,SpinorN,SpinorZ, Reducer> arg;
  doubleN *result;
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
  ReduceCudaExp(doubleN result[Nreduce], SpinorX &X, SpinorP &P, SpinorU &U,
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
    reduceLaunchExp<Nreduce,doubleN,ReduceType,FloatN,M_>(result, arg, tp, stream);
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


template <int Nreduce, typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename zType,
	  int M_, template <int Nreduce_, typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeX,int writeP,int writeU,int writeR, int writeS, int writeM,int writeQ,int writeW,int writeN, int writeZ>
void reduceCudaExp(doubleN reduce_buffer[Nreduce], const double2 &a, const double2 &b, ColorSpinorField &x, 
		   ColorSpinorField &p, ColorSpinorField &u, ColorSpinorField &r,
                   ColorSpinorField &s, ColorSpinorField &m, ColorSpinorField &q, 
                   ColorSpinorField &w, ColorSpinorField &n, ColorSpinorField &z,
		   int length) {

  checkLength(x, p); checkLength(x, u); checkLength(x, r); checkLength(x, s);
  checkLength(x, m); checkLength(x, q); checkLength(x, w); checkLength(x, n);
  checkLength(x, z);

  if (!x.isNative()) {
    warningQuda("Device reductions on non-native fields is not supported\n");
    for(int j = 0; j < Nreduce; j++) ::quda::zero(reduce_buffer[j]);
    return;
  }

  blasStrings.vol_str = x.VolString();
  strcpy(blasStrings.aux_tmp, x.AuxString());
  if (typeid(StoreType) != typeid(zType)) {//?
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
  Reducer<Nreduce, ReduceType, Float2, RegType> reducer_((Float2)vec2(a), (Float2)vec2(b));

  int partitions = (x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset());

  ReduceCudaExp<Nreduce,doubleN,ReduceType,RegType,M_,
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
    Reducer<Nreduce, ReduceType, Float2, RegType> >
    reduce(reduce_buffer, X, P, U, R, S, M, Q, W, N, Z, reducer_, length, partitions, bytes, norm_bytes);
  reduce.apply(*(blas::getStream()));

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return;
}



//////COMPOSITE

template <typename ReduceType, typename SpinorX1, typename SpinorR1, typename SpinorW1, typename SpinorQ1, typename SpinorD1,
typename SpinorH1, typename SpinorZ1, typename SpinorP1, typename SpinorU1, typename SpinorG1, 
typename SpinorX2, typename SpinorR2, typename SpinorW2, typename SpinorQ2, typename SpinorD2, typename SpinorH2, typename SpinorZ2, typename SpinorP2, typename SpinorU2, typename SpinorG2, typename Reducer>
struct ReductionComponentwiseArgExp : public ReduceArg<ReduceType> {//defined in cub
  SpinorX1 X1;
  SpinorR1 R1;
  SpinorW1 W1;
  SpinorQ1 Q1;
  SpinorD1 D1;
  SpinorH1 H1;
  SpinorZ1 Z1;
  SpinorP1 P1;
  SpinorU1 U1;
  SpinorG1 G1;

  SpinorX2 X2;
  SpinorR2 R2;
  SpinorW2 W2;
  SpinorQ2 Q2;
  SpinorD2 D2;
  SpinorH2 H2;
  SpinorZ2 Z2;
  SpinorP2 P2;
  SpinorU2 U2;
  SpinorG2 G2;

  Reducer reduce_functor;

  const int length;
  ReductionComponentwiseArgExp(SpinorX1 X1, SpinorR1 R1, SpinorW1 W1, SpinorQ1 Q1, SpinorD1 D1, SpinorH1 H1, SpinorZ1 Z1, SpinorP1 P1, SpinorU1 U1, SpinorG1 G1, SpinorX2 X2, SpinorR2 R2, SpinorW2 W2, SpinorQ2 Q2, SpinorD2 D2, SpinorH2 H2, SpinorZ2 Z2, SpinorP2 P2, SpinorU2 U2, SpinorG2 G2, Reducer r, int length)
    : X1(X1), R1(R1), W1(W1), Q1(Q1), D1(D1), H1(H1), Z1(Z1), P1(P1), U1(U1), G1(G1), X2(X2), R2(R2), W2(W2), Q2(Q2), D2(D2), H2(H2), Z2(Z2), P2(P2), U2(U2), G2(G2), reduce_functor(r), length(length) { ; }
};

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename FloatN, int Nreduce, int M_, typename SpinorX1, typename SpinorR1, typename SpinorW1, typename SpinorQ1, typename SpinorD1,
	  typename SpinorH1, typename SpinorZ1, typename SpinorP1, typename SpinorU1, typename SpinorG1, typename SpinorX2, typename SpinorR2, typename SpinorW2, typename SpinorQ2, typename SpinorD2, typename SpinorH2, typename SpinorZ2, typename SpinorP2, typename SpinorU2, typename SpinorG2, typename Reducer>
__global__ void reduceComponentwiseKernelExp(ReductionComponentwiseArgExp<ReduceType,SpinorX1,SpinorR1,SpinorW1,SpinorQ1,SpinorD1,SpinorH1,SpinorZ1,SpinorP1,SpinorU1,SpinorG1,
SpinorX2,SpinorR2,SpinorW2,SpinorQ2,SpinorD2,SpinorH2,SpinorZ2,SpinorP2,SpinorU2,SpinorG2,Reducer> arg) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum[Nreduce];
#pragma unroll
  for (int j=0; j<Nreduce;j++) ::quda::zero(sum[j]);

  while (i < arg.length) {
    FloatN x1[M_], r1[M_], w1[M_], d1[M_], q1[M_], h1[M_],z1[M_], p1[M_], u1[M_], g1[M_];
    FloatN x2[M_], r2[M_], w2[M_], d2[M_], q2[M_], h2[M_],z2[M_], p2[M_], u2[M_], g2[M_];

    arg.X1.load(x1, i, parity);
    arg.R1.load(r1, i, parity);
    arg.W1.load(w1, i, parity);
    arg.Q1.load(q1, i, parity);
    arg.D1.load(d1, i, parity);

    arg.H1.load(h1, i, parity);
    arg.Z1.load(z1, i, parity);
    arg.P1.load(p1, i, parity);
    arg.U1.load(u1, i, parity);
    arg.G1.load(g1, i, parity);

    arg.X2.load(x2, i, parity);
    arg.R2.load(r2, i, parity);
    arg.W2.load(w2, i, parity);
    arg.Q2.load(q2, i, parity);
    arg.D2.load(d2, i, parity);

//    arg.H1.load(h1, i, parity);
    arg.Z2.load(z2, i, parity);
    arg.P2.load(p2, i, parity);
    arg.U2.load(u2, i, parity);
//    arg.G1.load(g1, i, parity);


    arg.reduce_functor.pre();

#pragma unroll
    for (int j=0; j<M_; j++) arg.reduce_functor(sum, x1[j], r1[j], w1[j], q1[j], d1[j], h1[j], z1[j], p1[j], u1[j], g1[j], x2[j], r2[j], w2[j], q2[j], d2[j], h2[j], z2[j], p2[j], u2[j], g2[j]);

    arg.reduce_functor.post(sum);

    arg.X1.save(x1, i, parity);
    arg.R1.save(r1, i, parity);
    arg.W1.save(w1, i, parity);
    arg.Q1.save(q1, i, parity);
    arg.D1.save(d1, i, parity);

    arg.H1.save(h1, i, parity);
    arg.Z1.save(z1, i, parity);
    arg.P1.save(p1, i, parity);
    arg.U1.save(u1, i, parity);
    arg.G1.save(g1, i, parity);

    arg.X2.save(x2, i, parity);
    arg.R2.save(r2, i, parity);
    arg.W2.save(w2, i, parity);
    arg.Q2.save(q2, i, parity);
//    arg.D2.save(d2, i, parity);

//    arg.H1.save(h1, i, parity);
    arg.Z2.save(z2, i, parity);
    arg.P2.save(p2, i, parity);
    arg.U2.save(u2, i, parity);
//    arg.G1.save(g1, i, parity);

    i += gridSize;
  }

  ::quda::array_reduce<Nreduce, block_size, ReduceType>(arg, sum, parity);
}


/**
   Generic reduction kernel launcher
*/
template <int Nreduce, typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX1, typename SpinorR1, typename SpinorW1, typename SpinorQ1, typename SpinorD1,
	  typename SpinorH1, typename SpinorZ1, typename SpinorP1, typename SpinorU1, typename SpinorG1, 
          typename SpinorX2, typename SpinorR2, typename SpinorW2, typename SpinorQ2, typename SpinorD2, typename SpinorH2, typename SpinorZ2, typename SpinorP2, typename SpinorU2, typename SpinorG2, typename Reducer>
void reduceLaunchExp(doubleN cpu_sum[Nreduce], ReductionComponentwiseArgExp<ReduceType,SpinorX1,SpinorR1,SpinorW1,SpinorQ1,SpinorD1,SpinorH1,SpinorZ1,SpinorP1,SpinorU1,SpinorG1, SpinorX2,SpinorR2,SpinorW2,SpinorQ2,SpinorD2,SpinorH2,SpinorZ2,SpinorP2,SpinorU2,SpinorG2, Reducer> &arg, const TuneParam &tp, const cudaStream_t &stream) {
  if (Nreduce < 0 or Nreduce > 8) errorQuda("Incorrect size of the reduce array: %d.\n", Nreduce);
  if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
     errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

  LAUNCH_KERNEL(reduceComponentwiseKernelExp,tp,stream,arg,ReduceType,FloatN,Nreduce,M_);

  if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
    if(deviceProp.canMapHostMemory) {
      cudaEventRecord(reduceEnd, stream);
      while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
    } else
#endif
    { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType)*Nreduce, cudaMemcpyDeviceToHost); }
  }

#pragma unroll
  for (int j=0; j<Nreduce;j++) cpu_sum[j] = set(((ReduceType*)h_reduce)[0+j*tp.grid.y]);
  if (tp.grid.y==2) {
#pragma unroll 
    for (int j=0; j<Nreduce;j++) sum(cpu_sum[j], ((ReduceType*)h_reduce)[1+j*tp.grid.y]); // add other parity if needed
  }
  return;
}


template <int Nreduce, typename doubleN, typename ReduceType, typename FloatN, int M_, typename SpinorX1, typename SpinorR1, typename SpinorW1, typename SpinorQ1, typename SpinorD1, 
          typename SpinorH1, typename SpinorZ1, typename SpinorP1, typename SpinorU1, typename SpinorG1,  
	  typename SpinorX2, typename SpinorR2, typename SpinorW2, typename SpinorQ2, typename SpinorD2, typename SpinorH2, typename SpinorZ2, typename SpinorP2, typename SpinorU2, typename SpinorG2, typename Reducer>
class ReduceComponentwiseCudaExp : public Tunable {

private:
  mutable ReductionComponentwiseArgExp<ReduceType,SpinorX1,SpinorR1,SpinorW1,SpinorQ1,SpinorD1,SpinorH1,SpinorZ1,SpinorP1,SpinorU1,SpinorG1, SpinorX2,SpinorR2,SpinorW2,SpinorQ2,SpinorD2,SpinorH2,SpinorZ2,SpinorP2,SpinorU2,SpinorG2, Reducer> arg;
  doubleN *result;
  const int nParity;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X1_h, *R1_h, *W1_h, *Q1_h, *D1_h, *H1_h, *Z1_h, *P1_h, *U1_h, *G1_h;
  char *X2_h, *R2_h, *W2_h, *Q2_h, *D2_h, *H2_h, *Z2_h, *P2_h, *U2_h, *G2_h;
  char *X1norm_h, *R1norm_h, *W1norm_h, *Q1norm_h, *D1norm_h, *H1norm_h, *Z1norm_h, *P1norm_h, *U1norm_h, *G1norm_h;
  char *X2norm_h, *R2norm_h, *W2norm_h, *Q2norm_h, *D2norm_h, *H2norm_h, *Z2norm_h, *P2norm_h, *U2norm_h, *G2norm_h;
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
  ReduceComponentwiseCudaExp(doubleN result[Nreduce], SpinorX1 &X1, SpinorR1 &R1, SpinorW1 &W1,
	     SpinorQ1 &Q1, SpinorD1 &D1, SpinorH1 &H1, SpinorZ1 &Z1, SpinorP1 &P1, SpinorU1 &U1, SpinorG1 &G1, SpinorX2 &X2, SpinorR2 &R2, SpinorW2 &W2, SpinorQ2 &Q2, SpinorD2 &D2, SpinorH2 &H2, SpinorZ2 &Z2, SpinorP2 &P2, SpinorU2 &U2, SpinorG2 &G2, Reducer &r, int length, int nParity,
	     const size_t *bytes, const size_t *norm_bytes) :
    arg(X1,R1,W1,Q1,D1,H1,Z1,P1,U1,G1,X2,R2,W2,Q2,D2,H2,Z2,P2,U2,G2, r, length/nParity),
    result(result), nParity(nParity), X1_h(0), R1_h(0), W1_h(0), Q1_h(0), D1_h(0), H1_h(0), Z1_h(0), P1_h(0), U1_h(0), G1_h(0), 
    X2_h(0), R2_h(0), W2_h(0), Q2_h(0), D2_h(0), H2_h(0), Z2_h(0), P2_h(0), U2_h(0), G2_h(0),
    X1norm_h(0), R1norm_h(0), W1norm_h(0), Q1norm_h(0), D1norm_h(0), H1norm_h(0), Z1norm_h(0), P1norm_h(0), U1norm_h(0), G1norm_h(0),
    X2norm_h(0), R2norm_h(0), W2norm_h(0), Q2norm_h(0), D2norm_h(0), H2norm_h(0), Z2norm_h(0), P2norm_h(0), U2norm_h(0), G2norm_h(0),
    bytes_(bytes), norm_bytes_(norm_bytes) { }
  virtual ~ReduceComponentwiseCudaExp() { }

  inline TuneKey tuneKey() const {
    return TuneKey(blasStrings.vol_str, typeid(arg.reduce_functor).name(), blasStrings.aux_tmp);
  }

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    reduceLaunchExp<Nreduce,doubleN,ReduceType,FloatN,M_>(result, arg, tp, stream);
  }

  void preTune() {
    arg.X1.backup(&X1_h, &X1norm_h, bytes_[ 0], norm_bytes_[ 0]);
    arg.R1.backup(&R1_h, &R1norm_h, bytes_[ 1], norm_bytes_[ 1]);
    arg.W1.backup(&W1_h, &W1norm_h, bytes_[ 2], norm_bytes_[ 2]);
    arg.Q1.backup(&Q1_h, &Q1norm_h, bytes_[ 3], norm_bytes_[ 3]);
    arg.D1.backup(&D1_h, &D1norm_h, bytes_[ 4], norm_bytes_[ 4]);
    arg.H1.backup(&H1_h, &H1norm_h, bytes_[ 5], norm_bytes_[ 5]);
    arg.Z1.backup(&Z1_h, &Z1norm_h, bytes_[ 6], norm_bytes_[ 6]);
    arg.P1.backup(&P1_h, &P1norm_h, bytes_[ 7], norm_bytes_[ 7]);
    arg.U1.backup(&U1_h, &U1norm_h, bytes_[ 8], norm_bytes_[ 8]);
    arg.G1.backup(&G1_h, &G1norm_h, bytes_[ 9], norm_bytes_[ 9]);

    arg.X2.backup(&X2_h, &X2norm_h, bytes_[10], norm_bytes_[10]);
    arg.R2.backup(&R2_h, &R2norm_h, bytes_[11], norm_bytes_[11]);
    arg.W2.backup(&W2_h, &W2norm_h, bytes_[12], norm_bytes_[12]);
    arg.Q2.backup(&Q2_h, &Q2norm_h, bytes_[13], norm_bytes_[13]);
    arg.D2.backup(&D2_h, &D2norm_h, bytes_[14], norm_bytes_[14]);
    arg.H2.backup(&H2_h, &H2norm_h, bytes_[15], norm_bytes_[15]);
    arg.Z2.backup(&Z2_h, &Z2norm_h, bytes_[16], norm_bytes_[16]);
    arg.P2.backup(&P2_h, &P2norm_h, bytes_[17], norm_bytes_[17]);
    arg.U2.backup(&U2_h, &U2norm_h, bytes_[18], norm_bytes_[18]);
    arg.G2.backup(&G2_h, &G2norm_h, bytes_[19], norm_bytes_[19]);
  }

  void postTune() {
    arg.X1.restore(&X1_h, &X1norm_h, bytes_[ 0], norm_bytes_[ 0]);
    arg.R1.restore(&R1_h, &R1norm_h, bytes_[ 1], norm_bytes_[ 1]);
    arg.W1.restore(&W1_h, &W1norm_h, bytes_[ 2], norm_bytes_[ 2]);
    arg.Q1.restore(&Q1_h, &Q1norm_h, bytes_[ 3], norm_bytes_[ 3]);
    arg.D1.restore(&D1_h, &D1norm_h, bytes_[ 4], norm_bytes_[ 4]);
    arg.H1.restore(&H1_h, &H1norm_h, bytes_[ 5], norm_bytes_[ 5]);
    arg.Z1.restore(&Z1_h, &Z1norm_h, bytes_[ 6], norm_bytes_[ 6]);
    arg.P1.restore(&P1_h, &P1norm_h, bytes_[ 7], norm_bytes_[ 7]);
    arg.U1.restore(&U1_h, &U1norm_h, bytes_[ 8], norm_bytes_[ 8]);
    arg.G1.restore(&G1_h, &G1norm_h, bytes_[ 9], norm_bytes_[ 9]);

    arg.X2.restore(&X2_h, &X2norm_h, bytes_[10], norm_bytes_[10]);
    arg.R2.restore(&R2_h, &R2norm_h, bytes_[11], norm_bytes_[11]);
    arg.W2.restore(&W2_h, &W2norm_h, bytes_[12], norm_bytes_[12]);
    arg.Q2.restore(&Q2_h, &Q2norm_h, bytes_[13], norm_bytes_[13]);
    arg.D2.restore(&D2_h, &D2norm_h, bytes_[14], norm_bytes_[14]);
    arg.H2.restore(&H2_h, &H2norm_h, bytes_[15], norm_bytes_[15]);
    arg.Z2.restore(&Z2_h, &Z2norm_h, bytes_[16], norm_bytes_[16]);
    arg.P2.restore(&P2_h, &P2norm_h, bytes_[17], norm_bytes_[17]);
    arg.U2.restore(&U2_h, &U2norm_h, bytes_[18], norm_bytes_[18]);
    arg.G2.restore(&G2_h, &G2norm_h, bytes_[19], norm_bytes_[19]);
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
    size_t base_bytes = arg.X1.Precision()*vec_length<FloatN>::value*M_;
    if (arg.X1.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.R1.Precision()*vec_length<FloatN>::value*M_;
    if (arg.R1.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
    return ((arg.reduce_functor.streams()-2)*base_bytes + 2*extra_bytes)*arg.length*nParity;
  }

  int tuningIter() const { return 3; }
};


template <int Nreduce, typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename zType,
	  int M_, template <int Nreduce_, typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeX,int writeR,int writeW,int writeQ, int writeD, int writeH,int writeZ,int writeP,int writeU, int writeG>
void reduceComponentwiseCudaExp(doubleN reduce_buffer[Nreduce], const double2 &a, const double2 &b, const double2 &c, const double2 &a2, const double2 &b2, const double2& c2, 
                   ColorSpinorField &x1, ColorSpinorField &r1,
		   ColorSpinorField &w1, ColorSpinorField &q1, ColorSpinorField &d1,
		   ColorSpinorField &h1, ColorSpinorField &z1, ColorSpinorField &p1, 
                   ColorSpinorField &u1, ColorSpinorField &g1,
                   ColorSpinorField &x2, ColorSpinorField &r2,
		   ColorSpinorField &w2, ColorSpinorField &q2, ColorSpinorField &d2,
		   ColorSpinorField &h2, ColorSpinorField &z2, ColorSpinorField &p2, 
                   ColorSpinorField &u2, ColorSpinorField &g2, int length) {

  checkLength(x1, r1); checkLength(x1, w1); checkLength(x1, q1); checkLength(x1, d1);
  checkLength(x1, h1); checkLength(x1, z1); checkLength(x1, p1); checkLength(x1, u1);
  checkLength(x1, g1);

  checkLength(x2, r2); checkLength(x2, w2); checkLength(x2, q2); checkLength(x2, d2);
  checkLength(x2, h2); checkLength(x2, z2); checkLength(x2, p2); checkLength(x2, u2);
  checkLength(x2, g2); checkLength(x1, x2);

//Warning m and n are single component

  if (!x1.isNative() or !x2.isNative()) {
    warningQuda("Device reductions on non-native fields is not supported\n");
    for(int j = 0; j < Nreduce; j++) ::quda::zero(reduce_buffer[j]);
    return;
  }

  blasStrings.vol_str = x1.VolString();
  strcpy(blasStrings.aux_tmp, x1.AuxString());
  if (typeid(StoreType) != typeid(zType)) {//?
    strcat(blasStrings.aux_tmp, ",");
    strcat(blasStrings.aux_tmp, z1.AuxString());
  }

  size_t bytes[] = {x1.Bytes(), r1.Bytes(), w1.Bytes(), q1.Bytes(), d1.Bytes(), h1.Bytes(), z1.Bytes(), p1.Bytes(), u1.Bytes(), g1.Bytes(), x2.Bytes(), r2.Bytes(), w2.Bytes(), q2.Bytes(), d2.Bytes(), h2.Bytes(), z2.Bytes(), p2.Bytes(), u2.Bytes(), g2.Bytes()};
  size_t norm_bytes[] = {x1.NormBytes(), r1.NormBytes(), w1.NormBytes(), q1.NormBytes(), d1.NormBytes(), h1.NormBytes(), z1.NormBytes(), p1.NormBytes(), u1.NormBytes(), g1.NormBytes(),
x2.NormBytes(), r2.NormBytes(), w2.NormBytes(), q2.NormBytes(), d2.NormBytes(), h2.NormBytes(), z2.NormBytes(), p2.NormBytes(), u2.NormBytes(), g2.NormBytes()};

  Spinor<RegType,StoreType,M_,writeX, 0> X1(x1);
  Spinor<RegType,StoreType,M_,writeP, 1> R1(r1);
  Spinor<RegType,StoreType,M_,writeW, 2> W1(w1);
  Spinor<RegType,StoreType,M_,writeQ, 3> Q1(q1);
  Spinor<RegType,StoreType,M_,writeD, 4> D1(d1);
  Spinor<RegType,    zType,M_,writeH, 5> H1(h1);
  Spinor<RegType,StoreType,M_,writeZ, 6> Z1(z1);
  Spinor<RegType,StoreType,M_,writeP, 7> P1(p1);
  Spinor<RegType,StoreType,M_,writeU, 8> U1(u1);
  Spinor<RegType,    zType,M_,writeG, 9> G1(g1);

  Spinor<RegType,StoreType,M_,writeX,10> X2(x2);
  Spinor<RegType,StoreType,M_,writeP,11> R2(r2);
  Spinor<RegType,StoreType,M_,writeW,12> W2(w2);
  Spinor<RegType,StoreType,M_,writeQ,13> Q2(q2);
  Spinor<RegType,StoreType,M_,writeD,14> D2(d2);
  Spinor<RegType,    zType,M_,writeH,15> H2(h2);
  Spinor<RegType,StoreType,M_,writeZ,16> Z2(z2);
  Spinor<RegType,StoreType,M_,writeP,17> P2(p2);
  Spinor<RegType,StoreType,M_,writeU,18> U2(u2);
  Spinor<RegType,    zType,M_,writeG,19> G2(g2);

  typedef typename scalar<RegType>::type Float;
  typedef typename vector<Float,2>::type Float2;
  typedef vector<Float,2> vec2;
  Reducer<Nreduce, ReduceType, Float2, RegType> reducer_((Float2)vec2(a), (Float2)vec2(b),(Float2)vec2(c), (Float2)vec2(a2),(Float2)vec2(b),(Float2)vec2(c) );

  if(x1.IsComposite()) errorQuda("Composite fields are not supported.\n");

  int partitions = x1.SiteSubset();//(x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset());

  ReduceComponentwiseCudaExp<Nreduce,doubleN,ReduceType,RegType,M_,
  Spinor<RegType,StoreType,M_,writeX, 0>,
  Spinor<RegType,StoreType,M_,writeR, 1>,
  Spinor<RegType,StoreType,M_,writeW, 2>,
  Spinor<RegType,StoreType,M_,writeQ, 3>,
  Spinor<RegType,StoreType,M_,writeD, 4>,
  Spinor<RegType,    zType,M_,writeH, 5>,
  Spinor<RegType,StoreType,M_,writeZ, 6>,
  Spinor<RegType,StoreType,M_,writeP, 7>,
  Spinor<RegType,StoreType,M_,writeU, 8>,
  Spinor<RegType,    zType,M_,writeG, 9>,
  Spinor<RegType,StoreType,M_,writeX,10>,
  Spinor<RegType,StoreType,M_,writeR,11>,
  Spinor<RegType,StoreType,M_,writeW,12>,
  Spinor<RegType,StoreType,M_,writeQ,13>,
  Spinor<RegType,StoreType,M_,writeD,14>,
  Spinor<RegType,    zType,M_,writeH,15>,
  Spinor<RegType,StoreType,M_,writeZ,16>,
  Spinor<RegType,StoreType,M_,writeP,17>,
  Spinor<RegType,StoreType,M_,writeU,18>,
  Spinor<RegType,    zType,M_,writeG,19>,
    Reducer<Nreduce, ReduceType, Float2, RegType> >
    reduce(reduce_buffer, X1, R1, W1, Q1, D1, H1, Z1, P1, U1, G1, X2, R2, W2, Q2, D2, H2, Z2, P2, U2, G2, reducer_, length, partitions, bytes, norm_bytes);
  reduce.apply(*(blas::getStream()));

  blas::bytes += reduce.bytes();
  blas::flops += reduce.flops();

  checkCudaError();

  return;
}


