#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>

#include <comm_quda.h>

#include <hisq_links_quda.h> //reunit gauge links!!!!!

#include <pgauge_monte.h> 


#include <random.h>


#define BORDER_RADIUS 2

#ifndef PI
#define PI    3.1415926535897932384626433832795    // pi
#endif
#ifndef PII
#define PII   6.2831853071795864769252867665590    // 2 * pi
#endif

namespace quda {







__device__ __host__ inline int linkIndex2(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}



static __device__ __host__ inline int linkIndex3(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndex(int x[], const int X[4]) {
  int idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndexM1(int x[], const int X[4], const int mu) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = (y[mu] -1 + X[mu]) % X[mu];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}



static __device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) {
  /*x[3] = cb_index/(X[2]*X[1]*X[0]/2);
  x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  x[1] = (cb_index/(X[0]/2)) % X[1];
  x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);*/
  int za = (cb_index / (X[0]/2));
  int zb =  (za / X[1]);
  x[1] = za - zb * X[1];
  x[3] = (zb / X[2]);
  x[2] = zb - x[3] * X[2];
  int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = (2 * cb_index + x1odd)  - za * X[0];
  return;
}





template <typename Gauge>
struct InitGaugeColdArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
  Gauge dataOr;
  InitGaugeColdArg(const Gauge &dataOr, const cudaGaugeField &data)
    : dataOr(dataOr) {
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
    threads = X[0]*X[1]*X[2]*X[3];
  }
};




template<typename Float, typename Gauge, int NCOLORS>
__global__ void compute_InitGauge_ColdStart(InitGaugeColdArg<Gauge> arg){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= arg.threads) return;
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    int parity = 0;
    if(idx >= arg.threads/2) {
      parity = 1;
      idx -= arg.threads/2;
    }
    Matrix<Cmplx,NCOLORS> U;
    setIdentity(&U); 
    for(int d = 0; d < 4; d++)
      arg.dataOr.save((Float*)(U.data),idx, d, parity);
}


template<typename Float, typename Gauge, int NCOLORS>
class InitGaugeCold : Tunable {
  InitGaugeColdArg<Gauge> arg;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  InitGaugeCold(InitGaugeColdArg<Gauge> &arg)
    : arg(arg) {}
  ~InitGaugeCold () { }

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      compute_InitGauge_ColdStart<Float, Gauge, NCOLORS><<<tp.grid,tp.block>>>(arg);
      //cudaDeviceSynchronize();
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
    
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune(){}
  void postTune(){}
  long long flops() const { return 0; }// Only correct if there is no link reconstruction, no cub reduction accounted also
  long long bytes() const { return 0; }//no accounting the reduction!!!!

}; 

template<typename Float, int NCOLORS, typename Gauge>
void InitGaugeField( Gauge dataOr,  cudaGaugeField& data) {
  InitGaugeColdArg<Gauge> initarg(dataOr, data);
  InitGaugeCold<Float, Gauge, NCOLORS> init(initarg);
  init.apply(0);
  checkCudaError();
}



template<typename Float>
void InitGaugeField( cudaGaugeField& data) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2, 12>(data), data);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2,  8>(data), data);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4, 18>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4, 12>(data), data);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4,  8>(data), data);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}

/** @brief Perform a cold start to the gauge field, identity SU(3) matrix, also fills the ghost links in multi-GPU case (no need to exchange data)
* 
* @param[in,out] data Gauge field
*/
void InitGaugeField( cudaGaugeField& data) {

/*#ifdef MULTI_GPU
  errorQuda("Gauge Fixing with multi-GPU support NOT implemented yet!\n");
#else*/
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    InitGaugeField<float> (data);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    InitGaugeField<double>(data);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
//#endif
}








template <typename Gauge>
struct InitGaugeHotArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
  cuRNGState *rngstate;
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  InitGaugeHotArg(const Gauge &dataOr, const cudaGaugeField &data, cuRNGState *rngstate)
    : dataOr(dataOr), rngstate(rngstate) {
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    //the optimal number of RNG states in rngstate array must be equal to half the lattice volume
    //this number is the same used in heatbath...
    threads = X[0]*X[1]*X[2]*X[3] >> 1;
  }
};



template<typename Cmplx>
__device__ __host__ static inline typename RealTypeId<Cmplx>::Type Abs2(const Cmplx & a){
  return a.x * a.x + a.y * a.y;
}



template <typename Float> 
__host__ __device__ static inline void reunit_link( Matrix<typename ComplexTypeId<Float>::Type,3> &U ){

  typedef typename ComplexTypeId<Float>::Type Cmplx;

  Cmplx t2 = makeComplex((Float)0.0, (Float)0.0);
  Float t1 = 0.0;
  //first normalize first row
  //sum of squares of row
#pragma unroll
  for(int c = 0; c < 3; c++)    t1 += Abs2(U(0, c));
  t1 = (Float)1.0 / sqrt(t1);
  //14
  //used to normalize row
#pragma unroll
  for(int c = 0; c < 3; c++)    U(0,c) *= t1;
  //6      
#pragma unroll
  for(int c = 0; c < 3; c++)    t2 += Conj(U(0,c)) * U(1,c);
  //24
#pragma unroll
  for(int c = 0; c < 3; c++)    U(1,c) -= t2 * U(0,c);
  //24
  //normalize second row
  //sum of squares of row
  t1 = 0.0;
#pragma unroll
  for(int c = 0; c < 3; c++)    t1 += Abs2(U(1,c));
  t1 = (Float)1.0 / sqrt(t1);
  //14
  //used to normalize row
#pragma unroll
  for(int c = 0; c < 3; c++)    U(1, c) *= t1;
  //6      
  //Reconstruct lat row
  U(2,0) = Conj(U(0,1) * U(1,2) - U(0,2) * U(1,1));
  U(2,1) = Conj(U(0,2) * U(1,0) - U(0,0) * U(1,2));
  U(2,2) = Conj(U(0,0) * U(1,1) - U(0,1) * U(1,0));
  //42
  //T=130
}






/**
    @brief Generate the four random real elements of the SU(2) matrix
    @param localstate CURAND rng state
    @return four real numbers of the SU(2) matrix
*/
template <class T>
__device__ static inline Matrix<T,2> randomSU2(cuRNGState& localState){
    Matrix<T,2> a;
    T aabs, ctheta, stheta, phi;
    a(0,0) = Random<T>(localState, (T)-1.0, (T)1.0);
    aabs = sqrt( 1.0 - a(0,0) * a(0,0));
    ctheta = Random<T>(localState, (T)-1.0, (T)1.0);
    phi = PII * Random<T>(localState);
    stheta = ( curand(&localState) & 1 ? 1 : - 1 ) * sqrt( (T)1.0 - ctheta * ctheta );
    a(0,1) = aabs * stheta * cos( phi );
    a(1,0) = aabs * stheta * sin( phi );
    a(1,1) = aabs * ctheta;
    return a;
}


/**
    @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
    @param u SU(2) matrix represented by four real numbers
    @param link SU(Nc) matrix
    @param id indices
*/
template <class T, int NCOLORS>
__host__ __device__ static inline void mul_block_sun( Matrix<T,2> u, Matrix<typename ComplexTypeId<T>::Type,NCOLORS> &link, int2 id ){
    typename ComplexTypeId<T>::Type tmp;
    for(int j = 0; j < NCOLORS; j++){
        tmp = makeComplex( u(0,0), u(1,1) ) * link(id.x, j) + makeComplex( u(1,0), u(0,1) ) * link(id.y, j);
        link(id.y, j) = makeComplex(-u(1,0), u(0,1) ) * link(id.x, j) + makeComplex( u(0,0),-u(1,1) ) * link(id.y, j);
        link(id.x, j) = tmp;
    }
}


/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate the index's, the total number of blocks is NCOLORS * ( NCOLORS - 1) / 2.
    @return Returns two index's in int2 type, accessed by .x and .y.
*/
template<int NCOLORS>
__host__ __device__ static inline   int2 IndexBlock(int block){
    int2 id;
    int i1;
    int found = 0;
    int del_i = 0;
    int index = -1;
    while ( del_i < (NCOLORS-1) && found == 0 ){
        del_i++;
        for ( i1 = 0; i1 < (NCOLORS-del_i); i1++ ){
            index++;
            if ( index == block ){
                found = 1;
                break;
            }
        }
    }
    id.y = i1 + del_i;
    id.x = i1;
    return id;
}

/**
    @brief Generate a SU(Nc) random matrix
    @param localstate CURAND rng state
    @return SU(Nc) matrix
*/
template <class Float, int NCOLORS>
__device__ inline Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> randomize( cuRNGState& localState ){
    Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> U;
     
    for(int i=0; i<NCOLORS; i++)
        for(int j=0; j<NCOLORS; j++)
            U(i,j) = makeComplex( (Float)(Random<Float>(localState) - 0.5), (Float)(Random<Float>(localState) - 0.5));
    reunit_link<Float>(U);
    return U;

    /*setIdentity(&U);
    for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
      Matrix<Float,2> rr = randomSU2<Float>(localState);
      int2 id = IndexBlock<NCOLORS>( block );
      mul_block_sun<Float, NCOLORS>(rr, U, id);
      //U = block_su2_to_su3<Float>( U, a00, a01, a10, a11, block );
      }
    return U;*/
}

template<typename Float, typename Gauge, int NCOLORS>
__global__ void compute_InitGauge_HotStart(InitGaugeHotArg<Gauge> arg){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= arg.threads) return;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  #ifdef MULTI_GPU
  int X[4], x[4]; 
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
  for(int dr=0; dr<4; ++dr) X[dr] += 2*arg.border[dr]; 
  int id = idx;
  cuRNGState localState = arg.rngstate[ id ];
  #else
  cuRNGState localState = arg.rngstate[ idx ];
  #endif
  for(int parity = 0; parity < 2; parity++){
    #ifdef MULTI_GPU
    getCoords3(x, id, arg.X, parity);
    for(int dr=0; dr<4; ++dr) x[dr] += arg.border[dr];
    idx = linkIndex(x,X);
    #endif
    for(int d = 0; d < 4; d++){
      Matrix<Cmplx,NCOLORS> U;
      U = randomize<Float, NCOLORS>(localState);
      arg.dataOr.save((Float*)(U.data),idx, d, parity);
    }
  }
  #ifdef MULTI_GPU
  arg.rngstate[ id ] = localState;
  #else
  arg.rngstate[ idx ] = localState;
  #endif
}




template<typename Float, typename Gauge, int NCOLORS>
class InitGaugeHot : Tunable {
  InitGaugeHotArg<Gauge> arg;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  InitGaugeHot(InitGaugeHotArg<Gauge> &arg)
    : arg(arg) {}
  ~InitGaugeHot () { }

  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      compute_InitGauge_HotStart<Float, Gauge, NCOLORS><<<tp.grid,tp.block>>>(arg);
      //cudaDeviceSynchronize();
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
    
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune(){}
  void postTune(){}
  long long flops() const { return 0; }// Only correct if there is no link reconstruction, no cub reduction accounted also
  long long bytes() const { return 0; }//no accounting the reduction!!!!

}; 





template<typename Float, int NCOLORS, typename Gauge>
void InitGaugeField( Gauge dataOr,  cudaGaugeField& data, cuRNGState *rngstate) {
  InitGaugeHotArg<Gauge> initarg(dataOr, data, rngstate);
  InitGaugeHot<Float, Gauge, NCOLORS> init(initarg);
  init.apply(0);
  checkCudaError();
  cudaDeviceSynchronize();
  int R[4] = {0,0,0,0};
  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = BORDER_RADIUS;

  data.exchangeExtendedGhost(R,false);
  /*cudaDeviceSynchronize();
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
  const int reunit_allow_svd = 1;
  const int reunit_svd_only  = 0;
  const double svd_rel_error = 1e-6;
  const double svd_abs_error = 1e-6;
  setUnitarizeLinksConstants(unitarize_eps, max_error,
      reunit_allow_svd, reunit_svd_only,
      svd_rel_error, svd_abs_error);
  int num_failures=0;
  int* num_failures_dev;
  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));
  if(num_failures_dev == NULL) errorQuda("cudaMalloc failed for dev_pointer\n");

  unitarizeLinksQuda(data, num_failures_dev);
  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
  if(num_failures>0){
    cudaFree(num_failures_dev); 
    errorQuda("Error in the unitarization\n"); 
    exit(1);
  }
  cudaFree(num_failures_dev);*/
}



template<typename Float>
void InitGaugeField( cudaGaugeField& data, cuRNGState *rngstate) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2, 18>(data), data, rngstate);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2, 12>(data), data, rngstate);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 2,  8>(data), data, rngstate);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4, 18>(data), data, rngstate);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4, 12>(data), data, rngstate);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      InitGaugeField<Float, 3>(FloatNOrder<Float, 18, 4,  8>(data), data, rngstate);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}

/** @brief Perform a hot start to the gauge field, random SU(3) matrix, followed by reunitarization, also exchange borders links in multi-GPU case.
* 
* @param[in,out] data Gauge field
* @param[in,out] rngstate state of the CURAND random number generator
*/
void InitGaugeField( cudaGaugeField& data, cuRNGState *rngstate) {
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    InitGaugeField<float> (data, rngstate);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    InitGaugeField<double>(data, rngstate);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
}
}
