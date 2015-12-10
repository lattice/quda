#ifdef SSTEP

template <int N, typename ReduceType, typename SpinorX, typename SpinorY, 
         typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
struct MultiReduceArg {
  
  SpinorX X[N];
  SpinorY Y[N];
  SpinorZ Z[N];
  SpinorW W[N];
  SpinorV V[N];
  Reducer r;
  ReduceType *partial;
  ReduceType *complete;
  const int length;
  MultiReduceArg(SpinorX X[N], SpinorY Y[N], SpinorZ Z[N], SpinorW W[N], SpinorV V[N],
                 Reducer r, ReduceType *partial, ReduceType *complete, int length)
    : r(r), length(length){
    
    for(int i=0; i<N; ++i){
      this->X[i]         = X[i];
      this->Y[i]         = Y[i];
      this->Z[i]         = Z[i];
      this->W[i]         = W[i];
      this->V[i]         = V[i];
      this->partial      = partial;
      this->complete     = complete;
    }
  }
};


template<int block_size, int N, typename ReduceType, typename ReduceSimpleType,
  typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  __global__ void multiReduceKernel(MultiReduceArg<N,ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg){
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    ReduceType sum[N];

    for(int i=0; i<N; ++i){
      zero(sum[i]);
      unsigned int id = blockIdx.x*(blockDim.x) + threadIdx.x;
      FloatN x[M], y[M], z[M], w[M], v[M];
      while(id < arg.length){

        arg.X[i].load(x, id);
        arg.Y[i].load(y, id);
        arg.Z[i].load(z, id);
        arg.W[i].load(w, id);
        arg.V[i].load(v, id);
        arg.r.pre();

#pragma unroll
        for (int j=0; j<M; j++) arg.r(sum[i], x[j], y[j], z[j], w[j], v[j]);

        arg.r.post(sum[i]);

        arg.X[i].save(x, id);
        arg.Y[i].save(y, id);
        arg.Z[i].save(z, id);
        arg.W[i].save(w, id);
        arg.V[i].save(v, id);

        id += gridSize;
      } // loop over id
    } // loop over i



    extern __shared__ ReduceSimpleType sdata[];
    ReduceSimpleType *s = sdata + tid;

    // Copy data into shared memory
    for(int i=0; i<N; ++i){
      if (i>0) __syncthreads();
      if(tid >= warpSize) copytoshared(s, 0, sum[i], block_size);
      __syncthreads();

      // now reduce using the first warp only
      if(tid < warpSize){
        for(int j=warpSize; j<block_size; j+=warpSize) add<ReduceType>(sum[i], s, j, block_size);
        warpReduce<block_size>(s, sum[i]);

        // write result for this block to global memory
        if(tid == 0){
          ReduceType tmp;
          copyfromshared(tmp, s, 0, block_size);
          arg.partial[i*gridDim.x + blockIdx.x] = tmp;
        }
      } 
    } // loop over i

    if(tid==0){ 
       __threadfence(); // flush result
      unsigned int value = atomicInc(&count, gridDim.x);

      isLastBlockDone = (value == (gridDim.x-1));
    }

    __syncthreads();

    // Finish the reduction if last block
    if(isLastBlockDone){
      for(int i=0; i<N; ++i){
       unsigned int id = threadIdx.x;

        zero(sum[i]); 
        while(id < gridDim.x){
          sum[i] += arg.partial[i*gridDim.x + id]; 
          id += block_size;
        }
      } // loop over i

      extern __shared__ ReduceSimpleType sdata[];
      ReduceSimpleType *s = sdata + tid;

      for(int i=0; i<N; ++i){
	if (i>0) __syncthreads();
        if(tid >= warpSize) copytoshared(s, 0, sum[i], block_size);
        __syncthreads();

        if(tid < warpSize){
          for(int j=warpSize; j<block_size; j+=warpSize){ add<ReduceType>(sum[i], s, j, block_size); }
          warpReduce<block_size>(s, sum[i]);

          if(tid == 0){
            ReduceType tmp;
            copyfromshared(tmp, s, 0, block_size);
            arg.complete[i] = tmp;
          }
        }
      } // loop over i
      if(threadIdx.x == 0) count = 0;
    } // isLastBlockDone
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

  memset(result, 0, N*sizeof(doubleN));
  for(int i=0; i<N; ++i) result[i] += ((ReduceType*)h_reduce)[i]; // Need to check this
  
  const int Nreduce = N*(sizeof(doubleN)/sizeof(double));

  reduceDoubleArray((double*)result, Nreduce);
}




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

      unsigned int sharedBytesPerThread() const { return sizeof(ReduceType); }

      // When there is only one warp per block, we need to allocate two warps
      // worth of shared memory so that we don't index shared memory out of bounds  
      unsigned int sharedBytesPerBlock(const TuneParam &param) const {
        int warpSize = 32;
        return 2*warpSize*sizeof(ReduceType);
      }

      virtual bool advanceSharedBytes(TuneParam &param) const
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread()*nthreads > sharedBytesPerBlock(param) ? sharedBytesPerThread()*nthreads : sharedBytesPerBlock(param);
        return false; 
      }

    public:
      MultiReduceCuda(doubleN result[], SpinorX X[], SpinorY Y[], 
          SpinorZ Z[], SpinorW W[], SpinorV V[], Reducer &r, int length) :
        arg(X, Y, Z, W, V, r, (ReduceType*)d_reduce, (ReduceType*)hd_reduce, length), result(result) {
            for(int i=0; i<N; ++i){
              X_h[i] = 0;
              Y_h[i] = 0;
              Z_h[i] = 0;
              W_h[i] = 0;
              V_h[i] = 0;

              Xnorm_h[i] = 0;
              Ynorm_h[i] = 0;
              Znorm_h[i] = 0;
              Wnorm_h[i] = 0;
              Vnorm_h[i] = 0;
            }
          }

      virtual ~MultiReduceCuda(){}

      inline TuneKey tuneKey() const {
        return TuneKey(blasStrings.vol_str, typeid(*this).name(), blasStrings.aux_str);
      }

      void apply(const cudaStream_t &stream){
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        multiReduceLaunch<N,doubleN,ReduceType,ReduceSimpleType,FloatN,M>(result,arg,tp,stream);
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
      long long flops() const { return arg.r.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }
      long long bytes() const {
        size_t bytes = N*arg.X[0].Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
        if (arg.X[0].Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
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
    blasStrings.aux_str = x[0]->AuxString();

    int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

    if(x[0]->Precision() == QUDA_DOUBLE_PRECISION){
      if(x[0]->Nspin() == 4){ // wilson
        const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do

        Spinor<double2, double2, double2, M, writeX> X[N];
        Spinor<double2, double2, double2, M, writeY> Y[N];
        Spinor<double2, double2, double2, M, writeZ> Z[N];
        Spinor<double2, double2, double2, M, writeW> W[N];
        Spinor<double2, double2, double2, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }
        Reducer<ReduceType, double2, double2> r(a,b);
        MultiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, double2, M,
          Spinor<double2, double2, double2, M, writeX>, Spinor<double2, double2, double2, M, writeY>,
          Spinor<double2, double2, double2, M, writeZ>, Spinor<double2, double2, double2, M, writeW>,
          Spinor<double2, double2, double2, M, writeV>, Reducer<ReduceType, double2, double2> >
            reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));
        reduce.apply(*getBlasStream());

        

      }else if(x[0]->Nspin() == 1){

        const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do

        Spinor<double2, double2, double2, M, writeX> X[N];
        Spinor<double2, double2, double2, M, writeY> Y[N];
        Spinor<double2, double2, double2, M, writeZ> Z[N];
        Spinor<double2, double2, double2, M, writeW> W[N];
        Spinor<double2, double2, double2, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }

        Reducer<ReduceType, double2, double2> r(a,b);
        MultiReduceCuda<N, doubleN, ReduceType, ReduceSimpleType, double2, M,
          Spinor<double2, double2, double2, M, writeX>, Spinor<double2, double2, double2, M, writeY>,
          Spinor<double2, double2, double2, M, writeZ>, Spinor<double2, double2, double2, M, writeW>,
          Spinor<double2, double2, double2, M, writeV>, Reducer<ReduceType, double2, double2> >
            reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));

        
        reduce.apply(*getBlasStream());
      } else { errorQuda("ERROR: nSpin=%d is not supported\n", x[0]->Nspin()); }

    }else if (x[0]->Precision() == QUDA_SINGLE_PRECISION) {
      if(x[0]->Nspin() == 4){ // wilson
        const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do

        Spinor<float4, float4, float4, M, writeX> X[N];
        Spinor<float4, float4, float4, M, writeY> Y[N];
        Spinor<float4, float4, float4, M, writeZ> Z[N];
        Spinor<float4, float4, float4, M, writeW> W[N];
        Spinor<float4, float4, float4, M, writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }


	Reducer<ReduceType, float2, float4> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float4,M,
	  Spinor<float4,float4,float4,M,writeX>, Spinor<float4,float4,float4,M,writeY>,
	  Spinor<float4,float4,float4,M,writeZ>, Spinor<float4,float4,float4,M,writeW>,
	  Spinor<float4,float4,float4,M,writeV>, Reducer<ReduceType, float2, float4> >
	reduce(result, X, Y, Z, W, V, r, reduce_length/(4*M));
	reduce.apply(*getBlasStream());
      }else if(x[0]->Nspin() == 1){ // staggered

	const int M = siteUnroll ? 3 : 1; 

      	Spinor<float2,float2,float2,M,writeX> X[N];
        Spinor<float2,float2,float2,M,writeY> Y[N];
        Spinor<float2,float2,float2,M,writeZ> Z[N];
        Spinor<float2,float2,float2,M,writeW> W[N];
        Spinor<float2,float2,float2,M,writeV> V[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          W[i].set(*w[i]);
          V[i].set(*v[i]);
        }

	Reducer<ReduceType, float2, float2> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float2,M,
	  Spinor<float2,float2,float2,M,writeX>, Spinor<float2,float2,float2,M,writeY>,
	  Spinor<float2,float2,float2,M,writeZ>, Spinor<float2,float2,float2,M,writeW>,
	  Spinor<float2,float2,float2,M,writeV>, Reducer<ReduceType, float2, float2> >
	reduce(result, X, Y, Z, W, V, r, reduce_length/(2*M));
	reduce.apply(*getBlasStream());
      }
    }else{ // half precision
      if(x[0]->Nspin() == 4){ // wilson
        Spinor<float4,float4,short4,6,writeX> X[N];
	Spinor<float4,float4,short4,6,writeY> Y[N];
	Spinor<float4,float4,short4,6,writeZ> Z[N];
	Spinor<float4,float4,short4,6,writeV> V[N];
	Spinor<float4,float4,short4,6,writeV> W[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          V[i].set(*v[i]);
          W[i].set(*w[i]);
        }

	Reducer<ReduceType, float2, float4> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float4,6,
	  Spinor<float4,float4,short4,6,writeX>, Spinor<float4,float4,short4,6,writeY>,
	  Spinor<float4,float4,short4,6,writeZ>, Spinor<float4,float4,short4,6,writeW>,
	  Spinor<float4,float4,short4,6,writeV>, Reducer<ReduceType, float2, float4> >
	reduce(result, X, Y, Z, W, V, r, y[0]->Volume());
	reduce.apply(*getBlasStream());	  
      }else if(x[0]->Nspin() == 1){ // staggered
        Spinor<float2,float2,short2,3,writeX> X[N];
	Spinor<float2,float2,short2,3,writeY> Y[N];
	Spinor<float2,float2,short2,3,writeZ> Z[N];
	Spinor<float2,float2,short2,3,writeV> V[N];
	Spinor<float2,float2,short2,3,writeV> W[N];

        for(int i=0; i<N; ++i){
          X[i].set(*x[i]);
          Y[i].set(*y[i]);
          Z[i].set(*z[i]);
          V[i].set(*v[i]);
          W[i].set(*w[i]);
        }

	Reducer<ReduceType, float2, float2> r(make_float2(a.x,a.y), make_float2(b.x,b.y));
	MultiReduceCuda<N,doubleN,ReduceType,ReduceSimpleType,float2,3,
	  Spinor<float2,float2,short2,3,writeX>, Spinor<float2,float2,short2,3,writeY>,
	  Spinor<float2,float2,short2,3,writeZ>, Spinor<float2,float2,short2,3,writeW>,
	  Spinor<float2,float2,short2,3,writeV>, Reducer<ReduceType, float2, float2> >
	  reduce(result, X, Y, Z, W, V, r, y[0]->Volume());
	reduce.apply(*getBlasStream());

      }else{ errorQuda("ERROR: nSpin=%d is not supported\n", x[0]->Nspin()); }
    }

    for(int i=0; i<N; ++i){
      blas_bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x[i]->RealLength()*x[i]->Precision();
      blas_flops += Reducer<ReduceType,double2,double2>::flops()*(unsigned long long)x[i]->RealLength();
    }
    checkCudaError();
    return;
  }

#endif // SSTEP
