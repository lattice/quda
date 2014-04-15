__host__ __device__ void zero(double &x) { x = 0.0; }
__host__ __device__ void zero(double2 &x) { x.x = 0.0; x.y = 0.0; }
__host__ __device__ void zero(double3 &x) { x.x = 0.0; x.y = 0.0; x.z = 0.0; }
__device__ void copytoshared(double *s, const int i, const double x, const int block) { s[i] = x; }
__device__ void copytoshared(double *s, const int i, const double2 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }
__device__ void copytoshared(double *s, const int i, const double3 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; s[i+2*block] = x.z; }
__device__ void copytoshared(volatile double *s, const int i, const double x, const int block) { s[i] = x; }
__device__ void copytoshared(volatile double *s, const int i, const double2 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }
__device__ void copytoshared(volatile double *s, const int i, const double3 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; s[i+2*block] = x.z; }
__device__ void copyfromshared(double &x, const double *s, const int i, const int block) { x = s[i]; }
__device__ void copyfromshared(double2 &x, const double *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }
__device__ void copyfromshared(double3 &x, const double *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; x.z = s[i+2*block]; }

template<typename ReduceType, typename ReduceSimpleType> 
__device__ void add(ReduceType &sum, ReduceSimpleType *s, const int i, const int block) { }
template<> __device__ void add<double,double>(double &sum, double *s, const int i, const int block) 
{ sum += s[i]; }
template<> __device__ void add<double2,double>(double2 &sum, double *s, const int i, const int block) 
{ sum.x += s[i]; sum.y += s[i+block]; }
template<> __device__ void add<double3,double>(double3 &sum, double *s, const int i, const int block) 
{ sum.x += s[i]; sum.y += s[i+block]; sum.z += s[i+2*block]; }

template<typename ReduceType, typename ReduceSimpleType> 
__device__ void add(ReduceSimpleType *s, const int i, const int j, const int block) { }
template<typename ReduceType, typename ReduceSimpleType> 
__device__ void add(volatile ReduceSimpleType *s, const int i, const int j, const int block) { }

template<> __device__ void add<double,double>(double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; }
template<> __device__ void add<double,double>(volatile double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; }

template<> __device__ void add<double2,double>(double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block];}
template<> __device__ void add<double2,double>(volatile double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block];}

template<> __device__ void add<double3,double>(double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block]; s[i+2*block] += s[j+2*block];}
template<> __device__ void add<double3,double>(volatile double *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block]; s[i+2*block] += s[j+2*block];}

#if (__COMPUTE_CAPABILITY__ < 130)
__host__ __device__ void zero(doublesingle &x) { x = 0.0; }
__host__ __device__ void zero(doublesingle2 &x) { x.x = 0.0; x.y = 0.0; }
__host__ __device__ void zero(doublesingle3 &x) { x.x = 0.0; x.y = 0.0; x.z = 0.0; }
__device__ void copytoshared(doublesingle *s, const int i, const doublesingle x, const int block) { s[i] = x; }
__device__ void copytoshared(doublesingle *s, const int i, const doublesingle2 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }
__device__ void copytoshared(doublesingle *s, const int i, const doublesingle3 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; s[i+2*block] = x.z; }
__device__ void copytoshared(volatile doublesingle *s, const int i, const doublesingle x, const int block) { s[i].a.x = x.a.x; s[i].a.y = x.a.y; }
__device__ void copytoshared(volatile doublesingle *s, const int i, const doublesingle2 x, const int block) 
{ s[i].a.x = x.x.a.x; s[i].a.y = x.x.a.y; s[i+block].a.x = x.y.a.x; s[i+block].a.y = x.y.a.y; }
__device__ void copytoshared(volatile doublesingle *s, const int i, const doublesingle3 x, const int block) 
{ s[i].a.x = x.x.a.x; s[i].a.y = x.x.a.y; s[i+block].a.x = x.y.a.x; s[i+block].a.y = x.y.a.y; 
  s[i+2*block].a.x = x.z.a.x; s[i+2*block].a.y = x.z.a.y; }
__device__ void copyfromshared(doublesingle &x, const doublesingle *s, const int i, const int block) { x = s[i]; }
__device__ void copyfromshared(doublesingle2 &x, const doublesingle *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }
__device__ void copyfromshared(doublesingle3 &x, const doublesingle *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; x.z = s[i+2*block]; }

template<> __device__ void add<doublesingle,doublesingle>(doublesingle &sum, doublesingle *s, const int i, const int block) 
{ sum += s[i]; }
template<> __device__ void add<doublesingle2,doublesingle>(doublesingle2 &sum, doublesingle *s, const int i, const int block) 
{ sum.x += s[i]; sum.y += s[i+block]; }
template<> __device__ void add<doublesingle3,doublesingle>(doublesingle3 &sum, doublesingle *s, const int i, const int block) 
{ sum.x += s[i]; sum.y += s[i+block]; sum.z += s[i+2*block]; }

template<> __device__ void add<doublesingle,doublesingle>(doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; }
template<> __device__ void add<doublesingle,doublesingle>(volatile doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; }

template<> __device__ void add<doublesingle2,doublesingle>(doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block];}
template<> __device__ void add<doublesingle2,doublesingle>(volatile doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block];}

template<> __device__ void add<doublesingle3,doublesingle>(doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block]; s[i+2*block] += s[j+2*block];}
template<> __device__ void add<doublesingle3,doublesingle>(volatile doublesingle *s, const int i, const int j, const int block) 
{ s[i] += s[j]; s[i+block] += s[j+block]; s[i+2*block] += s[j+2*block];}
#endif

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

template <typename ReduceType, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
struct ReduceArg {
  SpinorX X;
  SpinorY Y;
  SpinorZ Z;
  SpinorW W;
  SpinorV V;
  Reducer r;
  ReduceType *partial;
  ReduceType *complete;
  const int length;
  ReduceArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, SpinorV V, Reducer r, 
	    ReduceType *partial, ReduceType *complete, int length) 
  : X(X), Y(Y), Z(Z), W(W), V(V), r(r), partial(partial), complete(complete), length(length) { ; }
};

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename ReduceSimpleType, 
  typename FloatN, int M, typename SpinorX, typename SpinorY, 
  typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
  __global__ void reduceKernel(ReduceArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  zero(sum); 
  while (i < arg.length) {
    FloatN x[M], y[M], z[M], w[M], v[M];
    arg.X.load(x, i);
    arg.Y.load(y, i);
    arg.Z.load(z, i);
    arg.W.load(w, i);
    arg.V.load(v, i);

#if (__COMPUTE_CAPABILITY__ >= 200)
    arg.r.pre();
#endif

#pragma unroll
    for (int j=0; j<M; j++) arg.r(sum, x[j], y[j], z[j], w[j], v[j]);

#if (__COMPUTE_CAPABILITY__ >= 200)
    arg.r.post(sum);
#endif

    arg.X.save(x, i);
    arg.Y.save(y, i);
    arg.Z.save(z, i);
    arg.W.save(w, i);
    arg.V.save(v, i);

    i += gridSize;
  }

  extern __shared__ ReduceSimpleType sdata[];
  ReduceSimpleType *s = sdata + tid;  
  if (tid >= warpSize) copytoshared(s, 0, sum, block_size);
  __syncthreads();
  
  // now reduce using the first warp only
  if (tid<warpSize) {
    // Warp raking
#pragma unroll
    for (int i=warpSize; i<block_size; i+=warpSize) { add<ReduceType>(sum, s, i, block_size); }

    // Intra-warp reduction
    volatile ReduceSimpleType *sv = s;
    copytoshared(sv, 0, sum, block_size);

    if (block_size >= 32) { add<ReduceType>(sv, 0, 16, block_size); } 
    if (block_size >= 16) { add<ReduceType>(sv, 0, 8, block_size); } 
    if (block_size >= 8) { add<ReduceType>(sv, 0, 4, block_size); } 
    if (block_size >= 4) { add<ReduceType>(sv, 0, 2, block_size); } 
    if (block_size >= 2) { add<ReduceType>(sv, 0, 1, block_size); } 

    // warpSize generic warp reduction - open64 can't handle it, only nvvm
    //#pragma unroll
    //for (int i=warpSize/2; i>0; i/=2) { add<ReduceType>(sv, 0, i, block_size); }
  }

  // write result for this block to global mem 
  if (tid == 0) {    
    ReduceType tmp;
    copyfromshared(tmp, s, 0, block_size);
    arg.partial[blockIdx.x] = tmp;
    
    __threadfence(); // flush result
    
    // increment global block counter
    unsigned int value = atomicInc(&count, gridDim.x);
    
    // Determine if this block is the last block to be done
    isLastBlockDone = (value == (gridDim.x-1));
  }

  __syncthreads();

  // Finish the reduction if last block
  if (isLastBlockDone) {
    unsigned int i = threadIdx.x;

    ReduceType sum;
    zero(sum); 
    while (i < gridDim.x) {
      sum += arg.partial[i];
      i += block_size;
    }
    
    extern __shared__ ReduceSimpleType sdata[];
    ReduceSimpleType *s = sdata + tid;  
    if (tid >= warpSize) copytoshared(s, 0, sum, block_size);
    __syncthreads();
   
    // now reduce using the first warp only
    if (tid<warpSize) {
      // Warp raking
#pragma unroll
      for (int i=warpSize; i<block_size; i+=warpSize) { add<ReduceType>(sum, s, i, block_size); }
      
      // Intra-warp reduction
      volatile ReduceSimpleType *sv = s;
      copytoshared(sv, 0, sum, block_size);

      if (block_size >= 32) { add<ReduceType>(sv, 0, 16, block_size); } 
      if (block_size >= 16) { add<ReduceType>(sv, 0, 8, block_size); } 
      if (block_size >= 8) { add<ReduceType>(sv, 0, 4, block_size); } 
      if (block_size >= 4) { add<ReduceType>(sv, 0, 2, block_size); } 
      if (block_size >= 2) { add<ReduceType>(sv, 0, 1, block_size); } 

      //#pragma unroll
      //for (int i=warpSize/2; i>0; i/=2) { add<ReduceType>(sv, 0, i, block_size); } 
    }
 
    // write out the final reduced value
    if (threadIdx.x == 0) {
      ReduceType tmp;
      copyfromshared(tmp, s, 0, block_size);
      arg.complete[0] = tmp;
      count = 0;
    }
  }

}
/**
   Generic reduction kernel launcher
*/
template <typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN, 
  int M, typename SpinorX, typename SpinorY, typename SpinorZ,  
  typename SpinorW, typename SpinorV, typename Reducer>
doubleN reduceLaunch(ReduceArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> &arg, 
		     const TuneParam &tp, const cudaStream_t &stream) {
  if (tp.grid.x > REDUCE_MAX_BLOCKS) 
    errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, REDUCE_MAX_BLOCKS);

  switch (tp.block.x) {
  case 32:
    reduceKernel<32,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 64:
    reduceKernel<64,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 96:
    reduceKernel<96,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 128:
    reduceKernel<128,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 160:
    reduceKernel<160,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 192:
    reduceKernel<192,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 224:
    reduceKernel<224,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 256:
    reduceKernel<256,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 288:
    reduceKernel<288,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 320:
    reduceKernel<320,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 352:
    reduceKernel<352,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 384:
    reduceKernel<384,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 416:
    reduceKernel<416,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 448:
    reduceKernel<448,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 480:
    reduceKernel<480,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 512:
    reduceKernel<512,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 544:
    reduceKernel<544,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 576:
    reduceKernel<576,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 608:
    reduceKernel<608,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 640:
    reduceKernel<640,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 672:
    reduceKernel<672,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 704:
    reduceKernel<704,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 736:
    reduceKernel<736,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 768:
    reduceKernel<768,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 800:
    reduceKernel<800,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 832:
    reduceKernel<832,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 864:
    reduceKernel<864,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 896:
    reduceKernel<896,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 928:
    reduceKernel<928,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 960:
    reduceKernel<960,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 992:
    reduceKernel<992,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  case 1024:
    reduceKernel<1024,ReduceType,ReduceSimpleType,FloatN,M>
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);
    break;
  default:
    errorQuda("Reduction not implemented for %d threads", tp.block.x);
  }

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
  if(deviceProp.canMapHostMemory) {
    cudaEventRecord(reduceEnd, stream);
    while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
  } else 
#endif
    { cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType), cudaMemcpyDeviceToHost); }

  doubleN cpu_sum;
  zero(cpu_sum);
  cpu_sum += ((ReduceType*)h_reduce)[0];

  const int Nreduce = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray((double*)&cpu_sum, Nreduce);

  return cpu_sum;
}


template <typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN, 
  int M, typename SpinorX, typename SpinorY, typename SpinorZ,  
  typename SpinorW, typename SpinorV, typename Reducer>
class ReduceCuda : public Tunable {

private:
  mutable ReduceArg<ReduceType,SpinorX,SpinorY,SpinorZ,SpinorW,SpinorV,Reducer> arg;
  doubleN &result;

  // host pointers used for backing up fields when tuning
  // these can't be curried into the Spinors because of Tesla argument length restriction
  char *X_h, *Y_h, *Z_h, *W_h, *V_h;
  char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h, *Vnorm_h;

  unsigned int sharedBytesPerThread() const { return sizeof(ReduceType); }

  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { 
    int warpSize = 32; // FIXME - use device property query
    return 2*warpSize*sizeof(ReduceType); 
  }

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
	     SpinorW &W, SpinorV &V, Reducer &r, int length) :
  arg(X, Y, Z, W, V, r, (ReduceType*)d_reduce, (ReduceType*)hd_reduce, length),
    result(result), X_h(0), Y_h(0), Z_h(0), W_h(0), V_h(0), 
    Xnorm_h(0), Ynorm_h(0), Znorm_h(0), Wnorm_h(0), Vnorm_h(0)
    { ; }
  virtual ~ReduceCuda() { }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << blasConstants.x[0] << "x";
    vol << blasConstants.x[1] << "x";
    vol << blasConstants.x[2] << "x";
    vol << blasConstants.x[3];    
    aux << "stride=" << blasConstants.stride << ",prec=" << arg.X.Precision();
    return TuneKey(vol.str(), typeid(arg.r).name(), aux.str());
  }  

  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    result = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,FloatN,M>(arg, tp, stream);
  }

  void preTune() { 
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.X.Stride();
    size_t norm_bytes = (arg.X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    arg.X.save(&X_h, &Xnorm_h, bytes, norm_bytes);
    arg.Y.save(&Y_h, &Ynorm_h, bytes, norm_bytes);
    arg.Z.save(&Z_h, &Znorm_h, bytes, norm_bytes);
    arg.W.save(&W_h, &Wnorm_h, bytes, norm_bytes);
    arg.V.save(&V_h, &Vnorm_h, bytes, norm_bytes);
  }

  void postTune() {
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M*arg.X.Stride();
    size_t norm_bytes = (arg.X.Precision() == QUDA_HALF_PRECISION) ? sizeof(float)*arg.length : 0;
    arg.X.load(&X_h, &Xnorm_h, bytes, norm_bytes);
    arg.Y.load(&Y_h, &Ynorm_h, bytes, norm_bytes);
    arg.Z.load(&Z_h, &Znorm_h, bytes, norm_bytes);
    arg.W.load(&W_h, &Wnorm_h, bytes, norm_bytes);
    arg.V.load(&V_h, &Vnorm_h, bytes, norm_bytes);
  }

  long long flops() const { return arg.r.flops()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*arg.length*M; }
  long long bytes() const { 
    size_t bytes = arg.X.Precision()*(sizeof(FloatN)/sizeof(((FloatN*)0)->x))*M;
    if (arg.X.Precision() == QUDA_HALF_PRECISION) bytes += sizeof(float);
    return arg.r.streams()*bytes*arg.length; }
};


/*
  Wilson
  double double2 M = 1/12
  single float4  M = 1/6
  half   short4  M = 6/6

  Staggered 
  double double2 M = 1/3
  single float2  M = 1/3
  half   short2  M = 3/3
 */

/**
   Driver for generic reduction routine with two loads.
   @param ReduceType 
   @param siteUnroll - if this is true, then one site corresponds to exactly one thread
 */
template <typename doubleN, typename ReduceType, typename ReduceSimpleType,
	  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
  int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
doubleN reduceCuda(const double2 &a, const double2 &b, cudaColorSpinorField &x, 
		   cudaColorSpinorField &y, cudaColorSpinorField &z, cudaColorSpinorField &w,
		   cudaColorSpinorField &v) {
  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    doubleN even =
      reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,
      writeY,writeZ,writeW,writeV,siteUnroll>
      (a, b, x.Even(), y.Even(), z.Even(), w.Even(), v.Even());
    doubleN odd = 
      reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,
      writeY,writeZ,writeW,writeV,siteUnroll>
      (a, b, x.Odd(), y.Odd(), z.Odd(), w.Odd(), v.Odd());
    return even + odd;
  }

  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);
  checkSpinor(x, v);

  if (!x.isNative()) {
    warningQuda("Reductions on non-native fields is not supported\n");
    doubleN value;
    zero(value);
    return value;
  }

  for (int d=0; d<QUDA_MAX_DIM; d++) blasConstants.x[d] = x.X()[d];
  blasConstants.stride = x.Stride();

  int reduce_length = siteUnroll ? x.RealLength() : x.Length();
  doubleN value;

  // FIXME: use traits to encapsulate register type for shorts -
  // will reduce template type parameters from 3 to 2

  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    if (x.Nspin() == 4){ //wilson
      const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
      Spinor<double2,double2,double2,M,writeX> X(x);
      Spinor<double2,double2,double2,M,writeY> Y(y);
      Spinor<double2,double2,double2,M,writeZ> Z(z);
      Spinor<double2,double2,double2,M,writeW> W(w);
      Spinor<double2,double2,double2,M,writeV> V(v);
      Reducer<ReduceType, double2, double2> r(a,b);
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,double2,M,
	Spinor<double2,double2,double2,M,writeX>, Spinor<double2,double2,double2,M,writeY>,
	Spinor<double2,double2,double2,M,writeZ>, Spinor<double2,double2,double2,M,writeW>,
	Spinor<double2,double2,double2,M,writeV>, Reducer<ReduceType, double2, double2> >
	reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M));
      reduce.apply(*getBlasStream());
    } else if (x.Nspin() == 1){ //staggered
      const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
      Spinor<double2,double2,double2,M,writeX> X(x);
      Spinor<double2,double2,double2,M,writeY> Y(y);
      Spinor<double2,double2,double2,M,writeZ> Z(z);
      Spinor<double2,double2,double2,M,writeW> W(w);
      Spinor<double2,double2,double2,M,writeV> V(v);
      Reducer<ReduceType, double2, double2> r(a,b);
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,double2,M,
	Spinor<double2,double2,double2,M,writeX>, Spinor<double2,double2,double2,M,writeY>,
	Spinor<double2,double2,double2,M,writeZ>, Spinor<double2,double2,double2,M,writeW>,
	Spinor<double2,double2,double2,M,writeV>, Reducer<ReduceType, double2, double2> >
	reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M));
      reduce.apply(*getBlasStream());
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
      Spinor<float4,float4,float4,M,writeX,0> X(x);
      Spinor<float4,float4,float4,M,writeY,1> Y(y);
      Spinor<float4,float4,float4,M,writeZ,2> Z(z);
      Spinor<float4,float4,float4,M,writeW,3> W(w);
      Spinor<float4,float4,float4,M,writeV,4> V(v);
      Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float4,M,
	Spinor<float4,float4,float4,M,writeX,0>,  Spinor<float4,float4,float4,M,writeY,1>,
	Spinor<float4,float4,float4,M,writeZ,2>,  Spinor<float4,float4,float4,M,writeW,3>,
	Spinor<float4,float4,float4,M,writeV,4>, Reducer<ReduceType, float2, float4> >
	reduce(value, X, Y, Z, W, V, r, reduce_length/(4*M));
      reduce.apply(*getBlasStream());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else if (x.Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
      const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
      Spinor<float2,float2,float2,M,writeX,0> X(x);
      Spinor<float2,float2,float2,M,writeY,1> Y(y);
      Spinor<float2,float2,float2,M,writeZ,2> Z(z);
      Spinor<float2,float2,float2,M,writeW,3> W(w);
      Spinor<float2,float2,float2,M,writeV,4> V(v);
      Reducer<ReduceType, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float2,M,
	Spinor<float2,float2,float2,M,writeX,0>,  Spinor<float2,float2,float2,M,writeY,1>,
	Spinor<float2,float2,float2,M,writeZ,2>,  Spinor<float2,float2,float2,M,writeW,3>,
	Spinor<float2,float2,float2,M,writeV,4>, Reducer<ReduceType, float2, float2> >
	reduce(value, X, Y, Z, W, V, r, reduce_length/(2*M));
      reduce.apply(*getBlasStream());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
  } else {
    if (x.Nspin() == 4){ //wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      Spinor<float4,float4,short4,6,writeX,0> X(x);
      Spinor<float4,float4,short4,6,writeY,1> Y(y);
      Spinor<float4,float4,short4,6,writeZ,2> Z(z);
      Spinor<float4,float4,short4,6,writeW,3> W(w);
      Spinor<float4,float4,short4,6,writeV,4> V(v);
      Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float4,6,
	Spinor<float4,float4,short4,6,writeX,0>, Spinor<float4,float4,short4,6,writeY,1>,
	Spinor<float4,float4,short4,6,writeZ,2>, Spinor<float4,float4,short4,6,writeW,3>,
	Spinor<float4,float4,short4,6,writeV,4>, Reducer<ReduceType, float2, float4> >
	reduce(value, X, Y, Z, W, V, r, y.Volume());
      reduce.apply(*getBlasStream());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else if (x.Nspin() == 1) {//staggered
#ifdef GPU_STAGGERED_DIRAC
      Spinor<float2,float2,short2,3,writeX,0> X(x);
      Spinor<float2,float2,short2,3,writeY,1> Y(y);
      Spinor<float2,float2,short2,3,writeZ,2> Z(z);
      Spinor<float2,float2,short2,3,writeW,3> W(w);
      Spinor<float2,float2,short2,3,writeV,4> V(v);
      Reducer<ReduceType, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      ReduceCuda<doubleN,ReduceType,ReduceSimpleType,float2,3,
	Spinor<float2,float2,short2,3,writeX,0>, Spinor<float2,float2,short2,3,writeY,1>,
	Spinor<float2,float2,short2,3,writeZ,2>, Spinor<float2,float2,short2,3,writeW,3>,
	Spinor<float2,float2,short2,3,writeV,4>, Reducer<ReduceType, float2, float2> >
	reduce(value, X, Y, Z, W, V, r, y.Volume());
      reduce.apply(*getBlasStream());
#else
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    blas_bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x.Volume()*sizeof(float);
  }
  blas_bytes += Reducer<ReduceType,double2,double2>::streams()*(unsigned long long)x.RealLength()*x.Precision();
  blas_flops += Reducer<ReduceType,double2,double2>::flops()*(unsigned long long)x.RealLength();

  checkCudaError();

  return value;
}

