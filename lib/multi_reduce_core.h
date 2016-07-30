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
    : r(r), length(length){
    
    for(int i=0; i<N; ++i){
      this->X[i] = X[i];
      this->Y[i] = Y[i];
      this->Z[i] = Z[i];
      this->W[i] = W[i];
      this->V[i] = V[i];
    }
  }
};


template<int block_size, int N, typename ReduceType, typename ReduceSimpleType,
  typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  __global__ void multiReduceKernel(MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg){

    unsigned int gridSize = gridDim.x*blockDim.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int src_idx = blockIdx.y*blockDim.y + threadIdx.y;

    ReduceType sum;
    ::quda::zero(sum);

    while (i < arg.length) {
      FloatN x[M], y[M], z[M], w[M], v[M];
      arg.X[src_idx].load(x, i);
      arg.Y[src_idx].load(y, i);
      arg.Z[src_idx].load(z, i);
      arg.W[src_idx].load(w, i);
      arg.V[src_idx].load(v, i);

      arg.r.pre();

#pragma unroll
      for (int j=0; j<M; j++) arg.r(sum, x[j], y[j], z[j], w[j], v[j]);

      arg.r.post(sum);

      arg.X[src_idx].save(x, i);
      arg.Y[src_idx].save(y, i);
      arg.Z[src_idx].save(z, i);
      arg.W[src_idx].save(w, i);
      arg.V[src_idx].save(v, i);

      i += gridSize;
    }

    ::quda::reduce<block_size, ReduceType>(arg, sum, src_idx);

  } // multiReduceKernel

template<int N, typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN,
int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
void multiReduceLaunch(doubleN result[], 
                       MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> &arg,
                       const TuneParam &tp, const cudaStream_t &stream){

  if(tp.grid.x > REDUCE_MAX_BLOCKS)
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, REDUCE_MAX_BLOCKS);

  LAUNCH_KERNEL(multiReduceKernel, tp, stream, arg, N, ReduceType, ReduceSimpleType, FloatN, M);

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

template<int N, typename doubleN, typename ReduceType, typename ReduceSimpleType, 
  typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, 
  typename SpinorW, typename SpinorV, typename Reducer>
  class MultiReduceCuda : public Tunable {

    private: 
      mutable MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg;

      doubleN *result;

      // host pointer used for backing up fields when tuning
      char *X_h[N], *Y_h[N], *Z_h[N], *W_h[N], *V_h[N];
      char *Xnorm_h[N], *Ynorm_h[N], *Znorm_h[N], *Wnorm_h[N], *Vnorm_h[N];

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
		      Reducer &r, int length) :
      arg(X, Y, Z, W, V, r, length), result(result),
	X_h(), Y_h(), Z_h(), W_h(), V_h(), Xnorm_h(),
	Ynorm_h(), Znorm_h(), Wnorm_h(), Vnorm_h() { }

      virtual ~MultiReduceCuda(){}

      inline TuneKey tuneKey() const {
	char name[TuneKey::name_n];
	strcpy(name, num_to_string<N>::value);
	strcat(name, typeid(arg.r).name());
        return TuneKey(blasStrings.vol_str, name, blasStrings.aux_str);
      }

      unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock; }

      void apply(const cudaStream_t &stream){
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        multiReduceLaunch<N,doubleN,ReduceType,ReduceSimpleType,FloatN,M>(result,arg,tp,stream);
      }

      bool advanceGridDim(TuneParam &param) const {
	bool rtn = Tunable::advanceGridDim(param);
	if (N > deviceProp.maxGridSize[1]) errorQuda("N=%d is greater than the maximum support grid size", N);
	param.grid.y = N;
	return rtn;
      }


#define BYTES(X) ( arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.X.Stride() )
#define NORM_BYTES(X) ( (arg.X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0 )

      void preTune() {
        for(int i=0; i<N; ++i){
          arg.X[i].save(&X_h[i], &Xnorm_h[i], BYTES(X[i]), NORM_BYTES(X[i]));
          arg.Y[i].save(&Y_h[i], &Ynorm_h[i], BYTES(Y[i]), NORM_BYTES(Y[i]));
          arg.Z[i].save(&Z_h[i], &Znorm_h[i], BYTES(Z[i]), NORM_BYTES(Z[i]));
          arg.W[i].save(&W_h[i], &Wnorm_h[i], BYTES(W[i]), NORM_BYTES(W[i]));
          arg.V[i].save(&V_h[i], &Vnorm_h[i], BYTES(V[i]), NORM_BYTES(V[i]));
        }
      }  

      void postTune() {
        for(int i=0; i<N; ++i){
          arg.X[i].load(&X_h[i], &Xnorm_h[i], BYTES(X[i]), NORM_BYTES(X[i]));
          arg.Y[i].load(&Y_h[i], &Ynorm_h[i], BYTES(Y[i]), NORM_BYTES(Y[i]));
          arg.Z[i].load(&Z_h[i], &Znorm_h[i], BYTES(Z[i]), NORM_BYTES(Z[i]));
          arg.W[i].load(&W_h[i], &Wnorm_h[i], BYTES(W[i]), NORM_BYTES(W[i]));
          arg.V[i].load(&V_h[i], &Vnorm_h[i], BYTES(V[i]), NORM_BYTES(V[i]));
        }
      } 
#undef BYTES
#undef NORM_BYTES

      // Need to check this!
      long long flops() const { return N*arg.r.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }
      long long bytes() const {
        size_t bytes = N*arg.X[0].Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
        if (arg.X[0].Precision() == QUDA_HALF_PRECISION) bytes += N*sizeof(float);
        return arg.r.streams()*bytes*arg.length; }
      int tuningIter() const { return 3; }
  };
// NB - need to change d_reduce and hd_reduce




template<int N, typename doubleN, typename ReduceType, typename ReduceSimpleType, 
  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll> 
  void multiReduceCuda(doubleN result[], const double2& a, const double2& b,
      std::vector<cudaColorSpinorField*>& x, std::vector<cudaColorSpinorField*>& y, 
      std::vector<cudaColorSpinorField*>& z, std::vector<cudaColorSpinorField*>& w,
      std::vector<cudaColorSpinorField*>& v){

    if(x[0]->SiteSubset() == QUDA_FULL_SITE_SUBSET){
      doubleN evenResult[N];
      doubleN oddResult[N];
      std::vector<cudaColorSpinorField> xp; xp.reserve(N);
      std::vector<cudaColorSpinorField> yp; yp.reserve(N);
      std::vector<cudaColorSpinorField> zp; zp.reserve(N);
      std::vector<cudaColorSpinorField> wp; wp.reserve(N);
      std::vector<cudaColorSpinorField> vp; vp.reserve(N);

      std::vector<cudaColorSpinorField*> xpp; xpp.reserve(N);
      std::vector<cudaColorSpinorField*> ypp; ypp.reserve(N);
      std::vector<cudaColorSpinorField*> zpp; zpp.reserve(N);
      std::vector<cudaColorSpinorField*> wpp; wpp.reserve(N);
      std::vector<cudaColorSpinorField*> vpp; vpp.reserve(N);

      for(int i=0; i<N; ++i){
        xp.push_back(x[i]->Even());
        yp.push_back(y[i]->Even());
        zp.push_back(z[i]->Even());
        wp.push_back(w[i]->Even());
        vp.push_back(v[i]->Even());

        xpp.push_back(&xp[i]);
        ypp.push_back(&yp[i]);
        zpp.push_back(&zp[i]);
        wpp.push_back(&wp[i]);
        vpp.push_back(&vp[i]);
      }

      multiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, Reducer, writeX,
        writeY, writeZ, writeW, writeV, siteUnroll>
          (evenResult, a, b, xpp, ypp, zpp, wpp, vpp);

      for(int i=0; i<N; ++i){
        xp.push_back(x[i]->Odd());
        yp.push_back(y[i]->Odd());
        zp.push_back(z[i]->Odd());
        wp.push_back(w[i]->Odd());
        vp.push_back(v[i]->Odd());

        xpp.push_back(&xp[i]);
        ypp.push_back(&yp[i]);
        zpp.push_back(&zp[i]);
        wpp.push_back(&wp[i]);
        vpp.push_back(&vp[i]);
      }

      multiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, Reducer, writeX,
        writeY, writeZ, writeW, writeV, siteUnroll> 
          (oddResult, a, b, xpp, ypp, zpp, wpp, vpp);

      for(int i=0; i<N; ++i) result[i] = evenResult[i] + oddResult[i];
      return;
    }
    for(int i=0; i<N; ++i){
      checkSpinor( (*(x[i])), (*(y[i])) );
      checkSpinor( (*(x[i])), (*(z[i])) );
      checkSpinor( (*(x[i])), (*(w[i])) );
      checkSpinor( (*(x[i])), (*(v[i])) );

      if(!x[i]->isNative()){
        warningQuda("Reductions on non-native fields are not supported\n");
        memset(result, 0, N*sizeof(doubleN));
        return; 
      }
    }

    memset(result, 0, N*sizeof(doubleN));

    blasStrings.vol_str = x[0]->VolString();
    blasStrings.aux_str = x[0]->AuxString(); // FIXME for mixed precision

    int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

    if (x[0]->Precision() == QUDA_DOUBLE_PRECISION) {
      if (x[0]->Nspin() == 4 || x[0]->Nspin() == 2) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
	if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");

        Spinor<double2, double2, M, writeX> X[N];
        Spinor<double2, double2, M, writeY> Y[N];
        Spinor<double2, double2, M, writeZ> Z[N];
        Spinor<double2, double2, M, writeW> W[N];
        Spinor<double2, double2, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }
        Reducer<ReduceType, double2, double2> r(a,b);
        MultiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, double2, M,
          Spinor<double2, double2, M, writeX>, Spinor<double2, double2, M, writeY>,
          Spinor<double2, double2, M, writeZ>, Spinor<double2, double2, M, writeW>,
          Spinor<double2, double2, M, writeV>, Reducer<ReduceType, double2, double2> >
	  reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));
        reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
        const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do

        Spinor<double2, double2, M, writeX> X[N];
        Spinor<double2, double2, M, writeY> Y[N];
        Spinor<double2, double2, M, writeZ> Z[N];
        Spinor<double2, double2, M, writeW> W[N];
        Spinor<double2, double2, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }

        Reducer<ReduceType, double2, double2> r(a,b);
        MultiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, double2, M,
          Spinor<double2, double2, M, writeX>, Spinor<double2, double2, M, writeY>,
          Spinor<double2, double2, M, writeZ>, Spinor<double2, double2, M, writeW>,
          Spinor<double2, double2, M, writeV>, Reducer<ReduceType, double2, double2> >
            reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));
        
        reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x[0]->Nspin()); }

    } else if (x[0]->Precision() == QUDA_SINGLE_PRECISION) {
      if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
        const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do

        Spinor<float4, float4, M, writeX> X[N];
        Spinor<float4, float4, M, writeY> Y[N];
        Spinor<float4, float4, M, writeZ> Z[N];
        Spinor<float4, float4, M, writeW> W[N];
        Spinor<float4, float4, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }

	Reducer<ReduceType, float2, float4> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float4,M,
	  Spinor<float4,float4,M,writeX>, Spinor<float4,float4,M,writeY>,
	  Spinor<float4,float4,M,writeZ>, Spinor<float4,float4,M,writeW>,
	  Spinor<float4,float4,M,writeV>, Reducer<ReduceType, float2, float4> >
	reduce(result, X, Y, Z, W, V, r, reduce_length/(4*M));
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if(x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
	const int M = siteUnroll ? 3 : 1; 
	if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");

        Spinor<float2,float2,M,writeY> X[N];
        Spinor<float2,float2,M,writeY> Y[N];
        Spinor<float2,float2,M,writeZ> Z[N];
        Spinor<float2,float2,M,writeW> W[N];
        Spinor<float2,float2,M,writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }

	Reducer<ReduceType, float2, float2> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float2,M,
	  Spinor<float2,float2,M,writeX>, Spinor<float2,float2,M,writeY>,
	  Spinor<float2,float2,M,writeZ>, Spinor<float2,float2,M,writeW>,
	  Spinor<float2,float2,M,writeV>, Reducer<ReduceType, float2, float2> >
	reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x[0]->Nspin()); }
    } else { // half precision
      if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
        Spinor<float4,short4,6,writeX> X[N];
	Spinor<float4,short4,6,writeY> Y[N];
	Spinor<float4,short4,6,writeZ> Z[N];
	Spinor<float4,short4,6,writeV> V[N];
	Spinor<float4,short4,6,writeV> W[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          V[i].set(*v[i]);
          W[i].set(*w[i]);
        }

	Reducer<ReduceType, float2, float4> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float4,6,
	  Spinor<float4,short4,6,writeX>, Spinor<float4,short4,6,writeY>,
	  Spinor<float4,short4,6,writeZ>, Spinor<float4,short4,6,writeW>,
	  Spinor<float4,short4,6,writeV>, Reducer<ReduceType, float2, float4> >
	reduce(result, X, Y, Z, W, V, r, y[0]->Volume());
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else if(x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
        Spinor<float2,short2,3,writeX> X[N];
	Spinor<float2,short2,3,writeY> Y[N];
	Spinor<float2,short2,3,writeZ> Z[N];
	Spinor<float2,short2,3,writeV> V[N];
	Spinor<float2,short2,3,writeV> W[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          V[i].set(*v[i]);
          W[i].set(*w[i]);
        }

	Reducer<ReduceType, float2, float2> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float2,3,
	  Spinor<float2,short2,3,writeX>, Spinor<float2,short2,3,writeY>,
	  Spinor<float2,short2,3,writeZ>, Spinor<float2,short2,3,writeW>,
	  Spinor<float2,short2,3,writeV>, Reducer<ReduceType, float2, float2> >
	  reduce(result, X, Y, Z, W, V, r, y[0]->Volume());
	reduce.apply(*blas::getStream());
#else
	errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x[0]->Nspin()); }
    }

    for(int i=0; i<N; ++i){
      blas::bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x[i]->RealLength()*x[i]->Precision();
      blas::flops += Reducer<ReduceType,double2,double2>::flops()*(unsigned long long)x[i]->RealLength();
    }
    checkCudaError();

    // now do multi-node reduction
    const int Nreduce = N*(sizeof(doubleN)/sizeof(double));
    reduceDoubleArray((double*)result, Nreduce);

    return;
  }
