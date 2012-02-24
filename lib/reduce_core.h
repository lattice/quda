__host__ __device__ void zero(double &x) { x = 0.0; }
__host__ __device__ void zero(double2 &x) { x.x = 0.0; x.y = 0.0; }
__host__ __device__ void zero(double3 &x) { x.x = 0.0; x.y = 0.0; x.z = 0.0; }
__device__ void copytoshared(double *s, const int i, const double x, const int block) { s[i] = x; }
__device__ void copytoshared(double *s, const int i, const double2 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; }
__device__ void copytoshared(double *s, const int i, const double3 x, const int block) 
{ s[i] = x.x; s[i+block] = x.y; s[i+2*block] = x.z; }
__device__ void copyfromshared(double &x, const double *s, const int i, const int block) { x = s[i]; }
__device__ void copyfromshared(double2 &x, const double *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }
__device__ void copyfromshared(double3 &x, const double *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; x.z = s[i+2*block]; }

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
__device__ void copyfromshared(doublesingle &x, const doublesingle *s, const int i, const int block) { x = s[i]; }
__device__ void copyfromshared(doublesingle2 &x, const doublesingle *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; }
__device__ void copyfromshared(doublesingle3 &x, const doublesingle *s, const int i, const int block) 
{ x.x = s[i]; x.y = s[i+block]; x.z = s[i+2*block]; }

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

/**
   Generic reduction kernel with up to four loads and three saves.
 */
template <int block_size, typename ReduceType, typename ReduceSimpleType, 
	  typename FloatN, int M, int writeX, int writeY, int writeZ,
	  typename InputX, typename InputY, typename InputZ, typename InputW, typename InputV,
	  typename OutputX, typename OutputY, typename OutputZ, typename Reducer>
__global__ void reduceKernel(InputX X, InputY Y, InputZ Z, InputW W, InputV V, Reducer r, 
			     ReduceType *partial, ReduceType *complete,
			     OutputX XX, OutputY YY, OutputZ ZZ, int length) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;

  ReduceType sum;
  zero(sum); 
  while (i < length) {
    FloatN x[M], y[M], z[M], w[M], v[M];
    X.load(x, i);
    Y.load(y, i);
    Z.load(z, i);
    W.load(w, i);
    V.load(v, i);
#pragma unroll
    for (int j=0; j<M; j++) r(sum, x[j], y[j], z[j], w[j], v[j]);

    if (writeX) XX.save(x, i);
    if (writeY) YY.save(y, i);
    if (writeZ) ZZ.save(z, i);

    i += gridSize;
  }

  extern __shared__ ReduceSimpleType sdata[];
  ReduceSimpleType *s = sdata + tid;  
  copytoshared(s, 0, sum, block_size);
  __syncthreads();
  
  // do reduction in shared mem
  if (block_size >= 1024){ if (tid < 512) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 992) { if (tid < 480) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 960) { if (tid < 448) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 928) { if (tid < 416) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 896) { if (tid < 384) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 864) { if (tid < 352) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 832) { if (tid < 320) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 800) { if (tid < 288) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 768) { if (tid < 256) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 736) { if (tid < 224) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 704) { if (tid < 192) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 672) { if (tid < 160) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 640) { if (tid < 128) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 608) { if (tid <  96) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 576) { if (tid <  64) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
  if (block_size == 544) { if (tid <  32) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }

  if (block_size >= 512) { if (tid < 256) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 480) { if (tid < 224) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 448) { if (tid < 192) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 416) { if (tid < 160) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 384) { if (tid < 128) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 352) { if (tid <  96) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 320) { if (tid <  64) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
  if (block_size == 288) { if (tid <  32) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }

  if (block_size >= 256) { if (tid < 128) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
  if (block_size == 224) { if (tid <  96) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
  if (block_size == 192) { if (tid <  64) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
  if (block_size == 160) { if (tid <  32) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }

  if (block_size >= 128) { if (tid <  64) { add<ReduceType>(s, 0, 64, block_size); } __syncthreads(); }
  
  if (tid < 32) {
    volatile ReduceSimpleType *sv = s;
    if (block_size ==  96)  { add<ReduceType>(sv, 0, 64, block_size); }
    if (block_size >=  64) { add<ReduceType>(sv, 0, 32, block_size); }
    if (block_size >=  32) { add<ReduceType>(sv, 0, 16, block_size); }
    if (block_size >=  16) { add<ReduceType>(sv, 0, 8, block_size); }
    if (block_size >=  8)  { add<ReduceType>(sv, 0, 4, block_size); }
    if (block_size >=  4)  { add<ReduceType>(sv, 0, 2, block_size); }
    if (block_size >=  2)  { add<ReduceType>(sv, 0, 1, block_size); }
  }
  
  // write result for this block to global mem 
  if (tid == 0) {    
    ReduceType tmp;
    copyfromshared(tmp, s, 0, block_size);
    partial[blockIdx.x] = tmp;

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
      sum += partial[i];
      i += block_size;
    }
    
    extern __shared__ ReduceSimpleType sdata[];
    ReduceSimpleType *s = sdata + tid;  
    copytoshared(s, 0, sum, block_size);
    __syncthreads();
    
    // do reduction in shared mem
    if (block_size >= 1024){ if (tid < 512) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 992) { if (tid < 480) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 960) { if (tid < 448) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 928) { if (tid < 416) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 896) { if (tid < 384) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 864) { if (tid < 352) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 832) { if (tid < 320) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 800) { if (tid < 288) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 768) { if (tid < 256) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 736) { if (tid < 224) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 704) { if (tid < 192) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 672) { if (tid < 160) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 640) { if (tid < 128) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 608) { if (tid <  96) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 576) { if (tid <  64) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }
    if (block_size == 544) { if (tid <  32) { add<ReduceType>(s, 0, 512, block_size); } __syncthreads(); }

    if (block_size >= 512) { if (tid < 256) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size >= 512) { if (tid < 256) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 480) { if (tid < 224) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 448) { if (tid < 192) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 416) { if (tid < 160) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 384) { if (tid < 128) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 352) { if (tid <  96) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 320) { if (tid <  64) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }
    if (block_size == 288) { if (tid <  32) { add<ReduceType>(s, 0, 256, block_size); } __syncthreads(); }

    if (block_size >= 256) { if (tid < 128) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
    if (block_size == 224) { if (tid <  96) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
    if (block_size == 192) { if (tid <  64) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }
    if (block_size == 160) { if (tid <  32) { add<ReduceType>(s, 0, 128, block_size); } __syncthreads(); }

    if (block_size >= 128) { if (tid <  64) { add<ReduceType>(s, 0, 64, block_size); } __syncthreads(); }
    
    if (tid < 32) {
      volatile ReduceSimpleType *sv = s;
      if (block_size ==  96)  { add<ReduceType>(sv, 0, 64, block_size); }
      if (block_size >=  64) { add<ReduceType>(sv, 0, 32, block_size); }
      if (block_size >=  32) { add<ReduceType>(sv, 0, 16, block_size); }
      if (block_size >=  16) { add<ReduceType>(sv, 0, 8, block_size); }
      if (block_size >=  8)  { add<ReduceType>(sv, 0, 4, block_size); }
      if (block_size >=  4)  { add<ReduceType>(sv, 0, 2, block_size); }
      if (block_size >=  2)  { add<ReduceType>(sv, 0, 1, block_size); }
    }
   
    // write out the final reduced value
    if (threadIdx.x == 0) {
      ReduceType tmp;
      copyfromshared(tmp, s, 0, block_size);
      complete[0] = tmp;
      count = 0;
    }
 
  }

}

/**
   Generic reduction Kernel launcher
*/
template <typename doubleN, typename ReduceType, typename ReduceSimpleType, typename FloatN, 
	  int M, int writeX, int writeY, int writeZ, 
	  typename InputX, typename InputY, typename InputZ, typename InputW, typename InputV,
	  typename Reducer, typename OutputX, typename OutputY, typename OutputZ>
doubleN reduceLaunch(InputX X, InputY Y, InputZ Z, InputW W, InputV V, Reducer r, 
		     OutputX XX, OutputY YY, OutputZ ZZ, int length) {
  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(ReduceType) : 
    blasBlock.x * sizeof(ReduceType);

  ReduceType *partial = (ReduceType*)d_reduce;
  ReduceType *complete = (ReduceType*)hd_reduce;

  if (blasBlock.x == 32) {
    reduceKernel<32,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 64) {
    reduceKernel<64,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 96) {
    reduceKernel<96,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 128) {
    reduceKernel<128,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 160) {
    reduceKernel<160,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 192) {
    reduceKernel<192,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 224) {
    reduceKernel<224,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 256) {
    reduceKernel<256,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 288) {
    reduceKernel<288,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 320) {
    reduceKernel<320,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 352) {
    reduceKernel<352,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 384) {
    reduceKernel<384,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 416) {
    reduceKernel<416,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 448) {
    reduceKernel<448,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 480) {
    reduceKernel<480,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 512) {
    reduceKernel<512,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } /*else if (blasBlock.x == 544) {
    reduceKernel<544,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 576) {
    reduceKernel<576,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 608) {
    reduceKernel<608,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 640) {
    reduceKernel<640,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 672) {
    reduceKernel<672,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 704) {
    reduceKernel<704,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 736) {
    reduceKernel<736,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 768) {
    reduceKernel<768,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 800) {
    reduceKernel<800,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 832) {
    reduceKernel<832,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 864) {
    reduceKernel<864,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 896) {
    reduceKernel<896,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 928) {
    reduceKernel<928,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 960) {
    reduceKernel<960,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 992) {
    reduceKernel<992,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
  } else if (blasBlock.x == 1024) {
    reduceKernel<1024,ReduceType,ReduceSimpleType,FloatN,M,writeX,writeY,writeZ>
      <<< blasGrid, blasBlock, smemSize, *blasStream >>>(X, Y, Z, W, V, r, partial, complete, XX, YY, ZZ, length);
      } */ else {
    errorQuda("Reduction not implemented for %d threads", blasBlock.x);
  }

  if(deviceProp.canMapHostMemory) {
    cudaEventRecord(reduceEnd, *blasStream);
    while (cudaSuccess != cudaEventQuery(reduceEnd)) { ; }
  } else {
    cudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType), cudaMemcpyDeviceToHost);
  }

  doubleN cpu_sum;
  zero(cpu_sum);
  cpu_sum += ((ReduceType*)h_reduce)[0];

  const int Nreduce = sizeof(doubleN) / sizeof(double);
  reduceDoubleArray((double*)&cpu_sum, Nreduce);

  return cpu_sum;
}

/**
   Driver for generic reduction routine with two loads.
   @param ReduceType 
 */
template <typename doubleN, typename ReduceType, typename ReduceSimpleType,
	  template <typename ReducerType, typename Float, typename FloatN> class Reducer,
	  int writeX, int writeY, int writeZ>
doubleN reduceCuda(const int kernel, const double2 &a, const double2 &b, cudaColorSpinorField &x, 
		   cudaColorSpinorField &y, cudaColorSpinorField &z, cudaColorSpinorField &w,
		   cudaColorSpinorField &v) {
  if (x.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
    doubleN even =
      reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,writeY,writeZ>
      (kernel, a, b, x.Even(), y.Even(), z.Even(), w.Even(), v.Even());
    doubleN odd = 
      reduceCuda<doubleN,ReduceType,ReduceSimpleType,Reducer,writeX,writeY,writeZ>
      (kernel, a, b, x.Odd(), y.Odd(), z.Odd(), w.Odd(), v.Odd());
    return even + odd;
  }

  int block_length = (x.Precision() == QUDA_HALF_PRECISION) ? x.Stride() : x.Length();
  setBlock(kernel, block_length, x.Precision());
  checkSpinor(x, y);
  checkSpinor(x, z);
  checkSpinor(x, w);
  checkSpinor(x, v);

  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }

  doubleN value;
  if (x.Precision() == QUDA_DOUBLE_PRECISION) {
    const int M = 1; // determines how much work per thread to do
    SpinorTexture<double2,double2,double2,M,0> xTex(x);
    SpinorTexture<double2,double2,double2,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<double2,double2,double2,M,1>(y);    
    SpinorTexture<double2,double2,double2,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<double2,double2,double2,M,2>(z);    
    Spinor<double2,double2,double2,M> X(x);
    Spinor<double2,double2,double2,M> Y(y);
    Spinor<double2,double2,double2,M> Z(z);
    Spinor<double2,double2,double2,M> W(w);
    Spinor<double2,double2,double2,M> V(v);
    Reducer<ReduceType, double2, double2> r(a,b);
    value = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,double2,M,writeX,writeY,writeZ>
      (xTex, yTex, zTex, W, V, r, X, Y, Z, x.Length()/(2*M));
  } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
    const int M = 1;
    SpinorTexture<float4,float4,float4,M,0> xTex(x);
    SpinorTexture<float4,float4,float4,M,1> yTex;
    if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,float4,M,1>(y);
    SpinorTexture<float4,float4,float4,M,2> zTex;
    if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,float4,M,2>(z);
    SpinorTexture<float4,float4,float4,M,3> wTex;
    if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,float4,M,3>(w);
    SpinorTexture<float4,float4,float4,M,4> vTex;
    if (x.V() != v.V()) vTex = SpinorTexture<float4,float4,float4,M,4>(v);
    Spinor<float4,float4,float4,M> X(x);
    Spinor<float4,float4,float4,M> Y(y);
    Spinor<float4,float4,float4,M> Z(z);
    Spinor<float4,float4,float4,M> W(w);
    Spinor<float4,float4,float4,M> V(v);
    Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
    value = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,float4,M,writeX,writeY,writeZ>
      (xTex, yTex, zTex, wTex, vTex, r, X, Y, Z, x.Length()/(4*M));
  } else {
    if (x.Nspin() == 4){ //wilson
      SpinorTexture<float4,float4,short4,6,0> xTex(x);
      SpinorTexture<float4,float4,short4,6,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float4,float4,short4,6,1>(y);
      SpinorTexture<float4,float4,short4,6,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float4,float4,short4,6,2>(z);
      SpinorTexture<float4,float4,short4,6,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float4,float4,short4,6,3>(w);
      SpinorTexture<float4,float4,short4,6,4> vTex;
      if (x.V() != v.V()) vTex = SpinorTexture<float4,float4,short4,6,4>(v);
      Spinor<float4,float4,short4,6> xOut(x);
      Spinor<float4,float4,short4,6> yOut(y);
      Spinor<float4,float4,short4,6> zOut(z);
      Reducer<ReduceType, float2, float4> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      value = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,float4,6,writeX,writeY,writeZ>
	(xTex,yTex,zTex,wTex,vTex,r,xOut,yOut,zOut,y.Volume());
    } else if (x.Nspin() == 1) {//staggered
      SpinorTexture<float2,float2,short2,3,0> xTex(x);
      SpinorTexture<float2,float2,short2,3,1> yTex;
      if (x.V() != y.V()) yTex = SpinorTexture<float2,float2,short2,3,1>(y);
      SpinorTexture<float2,float2,short2,3,2> zTex;
      if (x.V() != z.V()) zTex = SpinorTexture<float2,float2,short2,3,2>(z);
      SpinorTexture<float2,float2,short2,3,3> wTex;
      if (x.V() != w.V()) wTex = SpinorTexture<float2,float2,short2,3,3>(w);
      SpinorTexture<float2,float2,short2,3,4> vTex;
      if (x.V() != v.V()) vTex = SpinorTexture<float2,float2,short2,3,4>(v);
      Spinor<float2,float2,short2,3> xOut(x);
      Spinor<float2,float2,short2,3> yOut(y);
      Spinor<float2,float2,short2,3> zOut(z);
      Reducer<ReduceType, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
      value = reduceLaunch<doubleN,ReduceType,ReduceSimpleType,float2,3,writeX,writeY,writeZ>
	(xTex,yTex,zTex,wTex,vTex,r,xOut,yOut,zOut,y.Volume());
    } else { errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin()); }
    quda::blas_bytes += Reducer<ReduceType,double2,double2>::streams()*x.Volume()*sizeof(float);
  }
  quda::blas_bytes += Reducer<ReduceType,double2,double2>::streams()*x.RealLength()*x.Precision();
  quda::blas_flops += Reducer<ReduceType,double2,double2>::flops()*x.RealLength();

  if (!blasTuning) checkCudaError();

  return value;
}

