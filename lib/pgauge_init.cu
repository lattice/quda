#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <comm_quda.h>
#include <unitarization_links.h>
#include <pgauge_monte.h>
#include <random_quda.h>
#include <cub_helper.cuh>
#include <index_helper.cuh>

#ifndef PI
#define PI    3.1415926535897932384626433832795    // pi
#endif
#ifndef PII
#define PII   6.2831853071795864769252867665590    // 2 * pi
#endif

namespace quda {

#ifdef GPU_GAUGE_ALG

  template <typename Gauge>
  struct InitGaugeColdArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    Gauge dataOr;
    InitGaugeColdArg(const Gauge &dataOr, const cudaGaugeField &data)
      : dataOr(dataOr) {
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      threads = X[0] * X[1] * X[2] * X[3];
    }
  };

  template<typename Float, typename Gauge, int NCOLORS>
  __global__ void compute_InitGauge_ColdStart(InitGaugeColdArg<Gauge> arg){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= arg.threads ) return;
    int parity = 0;
    if ( idx >= arg.threads / 2 ) {
      parity = 1;
      idx -= arg.threads / 2;
    }
    Matrix<complex<Float>,NCOLORS> U;
    setIdentity(&U);
    for ( int d = 0; d < 4; d++ ) arg.dataOr(d, idx, parity) = U;
  }


  template<typename Float, typename Gauge, int NCOLORS>
  class InitGaugeCold : Tunable {
    InitGaugeColdArg<Gauge> arg;
    mutable char aux_string[128]; // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    //bool tuneSharedBytes() const { return false; }  // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                        // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    InitGaugeCold(InitGaugeColdArg<Gauge> &arg)
      : arg(arg) {
    }
    ~InitGaugeCold () {
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      compute_InitGauge_ColdStart<Float, Gauge, NCOLORS> <<< tp.grid,tp.block >>> (arg);
      //cudaDeviceSynchronize();
    }

    TuneKey tuneKey() const {
      std::stringstream vol;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3];
      sprintf(aux_string,"threads=%d,prec=%lu", arg.threads, sizeof(Float));
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

    }

    long long flops() const {
      return 0;
    }                                  // Only correct if there is no link reconstruction, no cub reduction accounted also
    long long bytes() const {
      return 0;
    }                                  //no accounting the reduction!!!!

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

    if ( data.isNative() ) {
      if ( data.Reconstruct() == QUDA_RECONSTRUCT_NO ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_12 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_8 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data);
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

    if ( data.Precision() == QUDA_SINGLE_PRECISION ) {
      InitGaugeField<float> (data);
    } else if ( data.Precision() == QUDA_DOUBLE_PRECISION ) {
      InitGaugeField<double>(data);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
    
  }








  template <typename Gauge>
  struct InitGaugeHotArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    RNG rngstate;
#ifdef MULTI_GPU
    int border[4];
#endif
    Gauge dataOr;
    InitGaugeHotArg(const Gauge &dataOr, const cudaGaugeField &data, RNG &rngstate)
      : dataOr(dataOr), rngstate(rngstate) {
#ifdef MULTI_GPU
      for ( int dir = 0; dir < 4; ++dir ) {
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir] * 2;
      } 
#else
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
#endif
      //the optimal number of RNG states in rngstate array must be equal to half the lattice volume
      //this number is the same used in heatbath...
      threads = X[0] * X[1] * X[2] * X[3] >> 1;
    }
  };


  template <typename Float>
  __host__ __device__ static inline void reunit_link( Matrix<complex<Float>,3> &U ){

    complex<Float> t2((Float)0.0, (Float)0.0);
    Float t1 = 0.0;
    //first normalize first row
    //sum of squares of row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(0,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(0,c) *= t1;
    //6
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t2 += conj(U(0,c)) * U(1,c);
    //24
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1,c) -= t2 * U(0,c);
    //24
    //normalize second row
    //sum of squares of row
    t1 = 0.0;
#pragma unroll
    for ( int c = 0; c < 3; c++ ) t1 += norm(U(1,c));
    t1 = (Float)1.0 / sqrt(t1);
    //14
    //used to normalize row
#pragma unroll
    for ( int c = 0; c < 3; c++ ) U(1, c) *= t1;
    //6
    //Reconstruct lat row
    U(2,0) = conj(U(0,1) * U(1,2) - U(0,2) * U(1,1));
    U(2,1) = conj(U(0,2) * U(1,0) - U(0,0) * U(1,2));
    U(2,2) = conj(U(0,0) * U(1,1) - U(0,1) * U(1,0));
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
    stheta = ( curand(&localState) & 1 ? 1 : -1 ) * sqrt( (T)1.0 - ctheta * ctheta );
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
  __host__ __device__ static inline void mul_block_sun( Matrix<T,2> u, Matrix<complex<T>,NCOLORS> &link, int2 id ){
    for ( int j = 0; j < NCOLORS; j++ ) {
      complex<T> tmp = complex<T>( u(0,0), u(1,1) ) * link(id.x, j) + complex<T>( u(1,0), u(0,1) ) * link(id.y, j);
      link(id.y, j) = complex<T>(-u(1,0), u(0,1) ) * link(id.x, j) + complex<T>( u(0,0),-u(1,1) ) * link(id.y, j);
      link(id.x, j) = tmp;
    }
  }


/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate the index's, the total number of blocks is NCOLORS * ( NCOLORS - 1) / 2.
    @return Returns two index's in int2 type, accessed by .x and .y.
 */
  template<int NCOLORS>
  __host__ __device__ static inline int2 IndexBlock(int block){
    int2 id;
    int i1;
    int found = 0;
    int del_i = 0;
    int index = -1;
    while ( del_i < (NCOLORS - 1) && found == 0 ) {
      del_i++;
      for ( i1 = 0; i1 < (NCOLORS - del_i); i1++ ) {
        index++;
        if ( index == block ) {
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
  __device__ inline Matrix<complex<Float>,NCOLORS> randomize( cuRNGState& localState ){
    Matrix<complex<Float>,NCOLORS> U;

    for ( int i = 0; i < NCOLORS; i++ )
      for ( int j = 0; j < NCOLORS; j++ )
        U(i,j) = complex<Float>( (Float)(Random<Float>(localState) - 0.5), (Float)(Random<Float>(localState) - 0.5) );
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
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= arg.threads ) return;
  #ifdef MULTI_GPU
    int X[4], x[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] += 2 * arg.border[dr];
    int id = idx;
    cuRNGState localState = arg.rngstate.State()[ id ];
  #else
    cuRNGState localState = arg.rngstate.State()[ idx ];
  #endif
    for ( int parity = 0; parity < 2; parity++ ) {
    #ifdef MULTI_GPU
      getCoords(x, id, arg.X, parity);
      for ( int dr = 0; dr < 4; ++dr ) x[dr] += arg.border[dr];
      idx = linkIndex(x,X);
    #endif
      for ( int d = 0; d < 4; d++ ) {
        Matrix<complex<Float>,NCOLORS> U;
        U = randomize<Float, NCOLORS>(localState);
        arg.dataOr(d, idx, parity) = U;
      }
    }
  #ifdef MULTI_GPU
    arg.rngstate.State()[ id ] = localState;
  #else
    arg.rngstate.State()[ idx ] = localState;
  #endif
  }




  template<typename Float, typename Gauge, int NCOLORS>
  class InitGaugeHot : Tunable {
    InitGaugeHotArg<Gauge> arg;
    mutable char aux_string[128]; // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    bool tuneSharedBytes() const {
      return false;
    }                                            // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                        // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    InitGaugeHot(InitGaugeHotArg<Gauge> &arg)
      : arg(arg) {
    }
    ~InitGaugeHot () {
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      compute_InitGauge_HotStart<Float, Gauge, NCOLORS> <<< tp.grid,tp.block >>> (arg);
      //cudaDeviceSynchronize();
    }

    TuneKey tuneKey() const {
      std::stringstream vol;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3];
      sprintf(aux_string,"threads=%d,prec=%lud", arg.threads, sizeof(Float));
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

    }

    void preTune(){ arg.rngstate.backup(); }
    void postTune(){ arg.rngstate.restore(); }
    long long flops() const {
      return 0;
    }                                  // Only correct if there is no link reconstruction, no cub reduction accounted also
    long long bytes() const {
      return 0;
    }                                  //no accounting the reduction!!!!

  };


  template<typename Float, int NCOLORS, typename Gauge>
  void InitGaugeField( Gauge dataOr,  cudaGaugeField& data, RNG &rngstate) {
    InitGaugeHotArg<Gauge> initarg(dataOr, data, rngstate);
    InitGaugeHot<Float, Gauge, NCOLORS> init(initarg);
    init.apply(0);
    checkCudaError();
    qudaDeviceSynchronize();

    data.exchangeExtendedGhost(data.R(),false);
  }

  template<typename Float>
  void InitGaugeField( cudaGaugeField& data, RNG &rngstate) {

    if ( data.isNative() ) {
      if ( data.Reconstruct() == QUDA_RECONSTRUCT_NO ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data, rngstate);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_12 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data, rngstate);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_8 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
        InitGaugeField<Float, 3>(Gauge(data), data, rngstate);
      } else {
        errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
      }
    } else {
      errorQuda("Invalid Gauge Order\n");
    }
  }
#endif // GPU_GAUGE_ALG

/** @brief Perform a hot start to the gauge field, random SU(3) matrix, followed by reunitarization, also exchange borders links in multi-GPU case.
 *
 * @param[in,out] data Gauge field
 * @param[in,out] rngstate state of the CURAND random number generator
 */
  void InitGaugeField( cudaGaugeField& data, RNG &rngstate) {
#ifdef GPU_GAUGE_ALG
    if ( data.Precision() == QUDA_SINGLE_PRECISION ) {
      InitGaugeField<float> (data, rngstate);
    } else if ( data.Precision() == QUDA_DOUBLE_PRECISION ) {
      InitGaugeField<double>(data, rngstate);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
#else
    errorQuda("Pure gauge code has not been built");
#endif
  }
}
