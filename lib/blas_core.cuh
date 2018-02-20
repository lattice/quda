/**
  Parameter struct for generic blas kernel
*/
template <typename SpinorX, typename SpinorY, typename SpinorZ,
  typename SpinorW, typename Functor>
struct BlasArg {
  SpinorX X;
  SpinorY Y;
  SpinorZ Z;
  SpinorW W;
  Functor f;
  const int length;
  BlasArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, Functor f, int length)
  : X(X), Y(Y), Z(Z), W(W), f(f), length(length) { ; }
};

/**
   Generic blas kernel with four loads and up to four stores.
 */
template <typename FloatN, int M, typename SpinorX, typename SpinorY,
  typename SpinorZ, typename SpinorW, typename Functor>
  __global__ void blasKernel(BlasArg<SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int parity = blockIdx.y;
  unsigned int gridSize = gridDim.x*blockDim.x;

  arg.f.init();

  while (i < arg.length) {
    FloatN x[M], y[M], z[M], w[M];
    arg.X.load(x, i, parity);
    arg.Y.load(y, i, parity);
    arg.Z.load(z, i, parity);
    arg.W.load(w, i, parity);

#pragma unroll
    for (int j=0; j<M; j++) arg.f(x[j], y[j], z[j], w[j]);

    arg.X.save(x, i, parity);
    arg.Y.save(y, i, parity);
    arg.Z.save(z, i, parity);
    arg.W.save(w, i, parity);
    i += gridSize;
  }
}

template <typename FloatN, int M, typename SpinorX, typename SpinorY,
  typename SpinorZ, typename SpinorW, typename Functor>
class BlasCuda : public Tunable {

private:
  mutable BlasArg<SpinorX,SpinorY,SpinorZ,SpinorW,Functor> arg;

  const int nParity;

  // host pointers used for backing up fields when tuning
  // dont't these curry these in to minimize Arg size
  char *X_h, *Y_h, *Z_h, *W_h;
  char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h;
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
  BlasCuda(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor &f,
	   int length, int nParity, const size_t *bytes, const size_t *norm_bytes) :
    arg(X, Y, Z, W, f, length/nParity), nParity(nParity), X_h(0), Y_h(0), Z_h(0), W_h(0),
    Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), bytes_(bytes), norm_bytes_(norm_bytes) { }

  virtual ~BlasCuda() { }

  inline TuneKey tuneKey() const {
    return TuneKey(blasStrings.vol_str, typeid(arg.f).name(), blasStrings.aux_tmp);
  }

  inline void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    blasKernel<FloatN,M> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
  }

  void preTune() {
    arg.X.backup(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.backup(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.backup(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.backup(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
  }

  void postTune() {
    arg.X.restore(&X_h, &Xnorm_h, bytes_[0], norm_bytes_[0]);
    arg.Y.restore(&Y_h, &Ynorm_h, bytes_[1], norm_bytes_[1]);
    arg.Z.restore(&Z_h, &Znorm_h, bytes_[2], norm_bytes_[2]);
    arg.W.restore(&W_h, &Wnorm_h, bytes_[3], norm_bytes_[3]);
  }

  void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.grid.y = nParity;
  }

  void defaultTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.grid.y = nParity;
  }

  long long flops() const { return arg.f.flops()*vec_length<FloatN>::value*arg.length*nParity*M; }
  long long bytes() const
  {
    // bytes for low-precision vector
    size_t base_bytes = arg.X.Precision()*vec_length<FloatN>::value*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) base_bytes += sizeof(float);

    // bytes for high precision vector
    size_t extra_bytes = arg.Y.Precision()*vec_length<FloatN>::value*M;
    if (arg.Y.Precision() == QUDA_HALF_PRECISION) extra_bytes += sizeof(float);

    // the factor two here assumes we are reading and writing to the high precision vector
    return ((arg.f.streams()-2)*base_bytes + 2*extra_bytes)*arg.length*nParity;
  }
  int tuningIter() const { return 3; }
};

template <typename RegType, typename StoreType, typename yType, int M,
	  template <typename,typename> class Functor,
	  int writeX, int writeY, int writeZ, int writeW>
void blasCuda(const double2 &a, const double2 &b, const double2 &c,
	      ColorSpinorField &x, ColorSpinorField &y,
	      ColorSpinorField &z, ColorSpinorField &w, int length) {

  checkLength(x, y); checkLength(x, z); checkLength(x, w);

  if (!x.isNative() && !(x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && x.Precision() == QUDA_SINGLE_PRECISION) ) {
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
  Functor<Float2, RegType> f( (Float2)vec2(a), (Float2)vec2(b), (Float2)vec2(c));

  int partitions = (x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset());
  BlasCuda<RegType,M,
    decltype(X), decltype(Y), decltype(Z), decltype(W),
    Functor<Float2, RegType> >
    blas(X, Y, Z, W, f, length, partitions, bytes, norm_bytes);
  blas.apply(*blasStream);

  blas::bytes += blas.bytes();
  blas::flops += blas.flops();

  checkCudaError();
}


/**
   Generic blas kernel with four loads and up to four stores.
  */
template <typename Float, int writeX, int writeY, int writeZ, int writeW,
  typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW,
  typename Functor>
void genericBlas(SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, Functor f) {

  for (int parity=0; parity<X.Nparity(); parity++) {
    for (int x=0; x<X.VolumeCB(); x++) {
      for (int s=0; s<X.Nspin(); s++) {
	for (int c=0; c<X.Ncolor(); c++) {
	  complex<Float> X_(X(parity, x, s, c));
	  complex<Float> Y_ = Y(parity, x, s, c);
	  complex<Float> Z_ = Z(parity, x, s, c);
	  complex<Float> W_ = W(parity, x, s, c);
	  f(X_, Y_, Z_, W_);
	  if (writeX) X(parity, x, s, c) = X_;
	  if (writeY) Y(parity, x, s, c) = Y_;
	  if (writeZ) Z(parity, x, s, c) = Z_;
	  if (writeW) W(parity, x, s, c) = W_;
	}
      }
    }
  }
}

template <typename Float, typename yFloat, int nSpin, int nColor, QudaFieldOrder order,
  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z,
		   ColorSpinorField &w, Functor f) {
  colorspinor::FieldOrderCB<Float,nSpin,nColor,1,order> X(x), Z(z), W(w);
  colorspinor::FieldOrderCB<yFloat,nSpin,nColor,1,order> Y(y);
  genericBlas<yFloat,writeX,writeY,writeZ,writeW>(X, Y, Z, W, f);
}

template <typename Float, typename yFloat, int nSpin, QudaFieldOrder order,
	  int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Ncolor() == 3) {
    genericBlas<Float,yFloat,nSpin,3,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 4) {
    genericBlas<Float,yFloat,nSpin,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 6) { // free field Wilson
    genericBlas<Float,yFloat,nSpin,6,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 8) {
    genericBlas<Float,yFloat,nSpin,8,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 12) {
    genericBlas<Float,yFloat,nSpin,12,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 16) {
    genericBlas<Float,yFloat,nSpin,16,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 20) {
    genericBlas<Float,yFloat,nSpin,20,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 24) {
    genericBlas<Float,yFloat,nSpin,24,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Ncolor() == 32) {
    genericBlas<Float,yFloat,nSpin,32,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else {
    errorQuda("nColor = %d not implemeneted",x.Ncolor());
  }
}

template <typename Float, typename yFloat, QudaFieldOrder order, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.Nspin() == 4) {
    genericBlas<Float,yFloat,4,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
  } else if (x.Nspin() == 2) {
    genericBlas<Float,yFloat,2,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#ifdef GPU_STAGGERED_DIRAC
  } else if (x.Nspin() == 1) {
    genericBlas<Float,yFloat,1,order,writeX,writeY,writeZ,writeW,Functor>(x, y, z, w, f);
#endif
  } else {
    errorQuda("nSpin = %d not implemeneted",x.Nspin());
  }
}

template <typename Float, typename yFloat, int writeX, int writeY, int writeZ, int writeW, typename Functor>
  void genericBlas(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, Functor f) {
  if (x.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
    genericBlas<Float,yFloat,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,writeX,writeY,writeZ,writeW,Functor>
      (x, y, z, w, f);
  } else {
    errorQuda("Not implemeneted");
  }
}
