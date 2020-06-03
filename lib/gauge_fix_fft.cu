#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <unitarization_links.h>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <index_helper.cuh>

#include <cufft.h>

#ifdef GPU_GAUGE_ALG
#include <CUFFT_Plans.h>
#endif

namespace quda {

#ifdef GPU_GAUGE_ALG

//UNCOMMENT THIS IF YOU WAN'T TO USE LESS MEMORY
#define GAUGEFIXING_DONT_USE_GX
//Without using the precalculation of g(x),
//we loose some performance, because Delta(x) is written in normal lattice coordinates need for the FFTs
//and the gauge array in even/odd format

#ifdef HOST_DEBUG
#ifdef GAUGEFIXING_DONT_USE_GX
#warning Not using precalculated g(x)
#else
#warning Using precalculated g(x)
#endif
#endif


#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif

  template <typename Float>
  struct GaugeFixFFTRotateArg {
    int threads;     // number of active threads required
    int X[4];     // grid dimensions
    complex<Float> *tmp0;
    complex<Float> *tmp1;
    GaugeFixFFTRotateArg(const cudaGaugeField &data){
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      threads = X[0] * X[1] * X[2] * X[3];
      tmp0 = 0;
      tmp1 = 0;
    }
  };

  template <int direction, typename Float>
  __global__ void fft_rotate_kernel_2D2D(GaugeFixFFTRotateArg<Float> arg){ //Cmplx *data_in, Cmplx *data_out){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= arg.threads ) return;
    if ( direction == 0 ) {
      int x3 = id / (arg.X[0] * arg.X[1] * arg.X[2]);
      int x2 = (id / (arg.X[0] * arg.X[1])) % arg.X[2];
      int x1 = (id / arg.X[0]) % arg.X[1];
      int x0 = id % arg.X[0];

      int id  =  x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];
      int id_out =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
      arg.tmp1[id_out] = arg.tmp0[id];
      //data_out[id_out] = data_in[id];
    }
    if ( direction == 1 ) {

      int x1 = id / (arg.X[2] * arg.X[3] * arg.X[0]);
      int x0 = (id / (arg.X[2] * arg.X[3])) % arg.X[0];
      int x3 = (id / arg.X[2]) % arg.X[3];
      int x2 = id % arg.X[2];

      int id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
      int id_out =  x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0];
      arg.tmp1[id_out] = arg.tmp0[id];
      //data_out[id_out] = data_in[id];
    }
  }






  template<typename Float>
  class GaugeFixFFTRotate : Tunable {
    GaugeFixFFTRotateArg<Float> arg;
    int direction;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    //bool tuneSharedBytes() const { return false; } // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                              // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    GaugeFixFFTRotate(GaugeFixFFTRotateArg<Float> &arg) : arg(arg) {
      direction = 0;
    }
    ~GaugeFixFFTRotate () {
    }
    void setDirection(int dir, complex<Float> *data_in, complex<Float> *data_out){
      direction = dir;
      arg.tmp0 = data_in;
      arg.tmp1 = data_out;
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if ( direction == 0 )
        fft_rotate_kernel_2D2D<0, Float > <<< tp.grid, tp.block, 0, stream >>> (arg);
      else if ( direction == 1 )
        fft_rotate_kernel_2D2D<1, Float > <<< tp.grid, tp.block, 0, stream >>> (arg);
      else
        errorQuda("Error in GaugeFixFFTRotate option.\n");
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
    }
    long long bytes() const {
      return 4LL * sizeof(Float) * arg.threads;
    }

  };


  template <typename Float, typename Gauge>
  struct GaugeFixQualityArg : public ReduceArg<double2> {
    int threads;     // number of active threads required
    int X[4];     // grid dimensions
    Gauge dataOr;
    complex<Float> *delta;

    GaugeFixQualityArg(const Gauge &dataOr, const cudaGaugeField &data, complex<Float> * delta)
      : ReduceArg<double2>(), dataOr(dataOr), delta(delta) {
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      threads = data.VolumeCB();
    }
    double getAction(){ return result_h[0].x; }
    double getTheta(){ return result_h[0].y; }
  };

  template<int blockSize, int Elems, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFix_quality(GaugeFixQualityArg<Float, Gauge> argQ){
    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double2 data = make_double2(0.0,0.0);
    while (idx_cb < argQ.threads) {
      typedef complex<Float> Cmplx;

      int x[4];
      getCoords(x, idx_cb, argQ.X, parity);
      Matrix<Cmplx,3> delta;
      setZero(&delta);
      //idx = linkIndex(x,X);
      for ( int mu = 0; mu < gauge_dir; mu++ ) {
        Matrix<Cmplx,3> U = argQ.dataOr(mu, idx_cb, parity);
        delta -= U;
      }
      //18*gauge_dir
      data.x += -delta(0, 0).x - delta(1, 1).x - delta(2, 2).x;
      //2
      for ( int mu = 0; mu < gauge_dir; mu++ ) {
        Matrix<Cmplx,3> U = argQ.dataOr(mu, linkIndexM1(x,argQ.X,mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      //SAVE DELTA!!!!!
      SubTraceUnit(delta);
      int idx = getIndexFull(idx_cb, argQ.X, parity);
      //Saving Delta
      argQ.delta[idx] = delta(0,0);
      argQ.delta[idx + 2 * argQ.threads] = delta(0,1);
      argQ.delta[idx + 4 * argQ.threads] = delta(0,2);
      argQ.delta[idx + 6 * argQ.threads] = delta(1,1);
      argQ.delta[idx + 8 * argQ.threads] = delta(1,2);
      argQ.delta[idx + 10 * argQ.threads] = delta(2,2);
      //12
      data.y += getRealTraceUVdagger(delta, delta);
      //35
      //T=36*gauge_dir+65

      idx_cb += blockDim.x * gridDim.x;
    }

    reduce2d<blockSize,2>(argQ, data);
  }



  template<int Elems, typename Float, typename Gauge, int gauge_dir>
  class GaugeFixQuality : TunableLocalParity {
    GaugeFixQualityArg<Float, Gauge> argQ;
    mutable char aux_string[128];     // used as a label in the autotuner

  private:
    bool tuneGridDim() const { return true; }

  public:
    GaugeFixQuality(GaugeFixQualityArg<Float, Gauge> &argQ)
      : argQ(argQ) {
    }
    ~GaugeFixQuality () { }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      argQ.result_h[0] = make_double2(0.0,0.0);
      LAUNCH_KERNEL_LOCAL_PARITY(computeFix_quality, (*this), tp, stream, argQ, Elems, Float, Gauge, gauge_dir);
      qudaDeviceSynchronize();
      argQ.result_h[0].x  /= (double)(3 * gauge_dir * 2 * argQ.threads);
      argQ.result_h[0].y  /= (double)(3 * 2 * argQ.threads);
    }

    TuneKey tuneKey() const {
      std::stringstream vol;
      vol << argQ.X[0] << "x" << argQ.X[1] << "x" << argQ.X[2] << "x" << argQ.X[3];
      sprintf(aux_string,"threads=%d,prec=%lu,gaugedir=%d", argQ.threads, sizeof(Float), gauge_dir);
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
    }

    long long flops() const {
      return (36LL * gauge_dir + 65LL) * 2 * argQ.threads;
    }                                                                         // Only correct if there is no link reconstruction, no cub reduction accounted also
    long long bytes() const {
      return (2LL * gauge_dir + 2LL) * Elems * 2 * argQ.threads * sizeof(Float);
    }                                                                                                    //Not accounting the reduction!!!

  };



  template <typename Float>
  struct GaugeFixArg {
    int threads;     // number of active threads required
    int X[4];     // grid dimensions
    cudaGaugeField &data;
    Float *invpsq;
    complex<Float> *delta;
    complex<Float> *gx;

    GaugeFixArg( cudaGaugeField & data, const int Elems) : data(data){
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      threads = X[0] * X[1] * X[2] * X[3];
      invpsq = (Float*)device_malloc(sizeof(Float) * threads);
      delta = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads * 6);
#ifdef GAUGEFIXING_DONT_USE_GX
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads);
#else
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads * Elems);
#endif
    }
    void free(){
      device_free(invpsq);
      device_free(delta);
      device_free(gx);
    }
  };




  template <typename Float>
  __global__ void kernel_gauge_set_invpsq(GaugeFixArg<Float> arg){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= arg.threads ) return;
    int x1 = id / (arg.X[2] * arg.X[3] * arg.X[0]);
    int x0 = (id / (arg.X[2] * arg.X[3])) % arg.X[0];
    int x3 = (id / arg.X[2]) % arg.X[3];
    int x2 = id % arg.X[2];
    //id  =  x2 + (x3 +  (x0 + x1 * arg.X[0]) * arg.X[3]) * arg.X[2];
    Float sx = sin( (Float)x0 * FL_UNITARIZE_PI / (Float)arg.X[0]);
    Float sy = sin( (Float)x1 * FL_UNITARIZE_PI / (Float)arg.X[1]);
    Float sz = sin( (Float)x2 * FL_UNITARIZE_PI / (Float)arg.X[2]);
    Float st = sin( (Float)x3 * FL_UNITARIZE_PI / (Float)arg.X[3]);
    Float sinsq = sx * sx + sy * sy + sz * sz + st * st;
    Float prcfact = 0.0;
    //The FFT normalization is done here
    if ( sinsq > 0.00001 ) prcfact = 4.0 / (sinsq * (Float)arg.threads);
    arg.invpsq[id] = prcfact;
  }


  template<typename Float>
  class GaugeFixSETINVPSP : Tunable {
    GaugeFixArg<Float> arg;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    bool tuneSharedBytes() const {
      return false;
    }                                                  // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                              // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    GaugeFixSETINVPSP(GaugeFixArg<Float> &arg) : arg(arg) { }
    ~GaugeFixSETINVPSP () { }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_gauge_set_invpsq<Float> <<< tp.grid, tp.block, 0, stream >>> (arg);
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
      return 21 * arg.threads;
    }
    long long bytes() const {
      return sizeof(Float) * arg.threads;
    }

  };

  template<typename Float>
  __global__ void kernel_gauge_mult_norm_2D(GaugeFixArg<Float> arg){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id < arg.threads ) arg.gx[id] = arg.gx[id] * arg.invpsq[id];
  }


  template<typename Float>
  class GaugeFixINVPSP : Tunable {
    GaugeFixArg<Float> arg;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    //bool tuneSharedBytes() const { return false; } // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                              // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    GaugeFixINVPSP(GaugeFixArg<Float> &arg)
      : arg(arg){
      cudaFuncSetCacheConfig( kernel_gauge_mult_norm_2D<Float>,   cudaFuncCachePreferL1);
    }
    ~GaugeFixINVPSP () {
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_gauge_mult_norm_2D<Float> <<< tp.grid, tp.block, 0, stream >>> (arg);
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

    void preTune(){
      //since delta contents are irrelevant at this point, we can swap gx with delta
      complex<Float> *tmp = arg.gx;
      arg.gx = arg.delta;
      arg.delta = tmp;
    }
    void postTune(){
      arg.gx = arg.delta;
    }
    long long flops() const {
      return 2LL * arg.threads;
    }
    long long bytes() const {
      return 5LL * sizeof(Float) * arg.threads;
    }

  };



  template <typename Float>
  __host__ __device__ inline void reunit_link( Matrix<complex<Float>,3> &U ){

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

#ifdef GAUGEFIXING_DONT_USE_GX

  template <typename Float, typename Gauge>
  __global__ void kernel_gauge_fix_U_EO_NEW( GaugeFixArg<Float> arg, Gauge dataOr, Float half_alpha){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    if ( id >= arg.threads/2 ) return;

    typedef complex<Float> Cmplx;

    int x[4];
    getCoords(x, id, arg.X, parity);
    int idx = ((x[3] * arg.X[2] + x[2]) * arg.X[1] + x[1]) * arg.X[0] + x[0];
    Matrix<Cmplx,3> de;
    //Read Delta
    de(0,0) = arg.delta[idx + 0 * arg.threads];
    de(0,1) = arg.delta[idx + 1 * arg.threads];
    de(0,2) = arg.delta[idx + 2 * arg.threads];
    de(1,1) = arg.delta[idx + 3 * arg.threads];
    de(1,2) = arg.delta[idx + 4 * arg.threads];
    de(2,2) = arg.delta[idx + 5 * arg.threads];

    de(1,0) = Cmplx(-de(0,1).x, de(0,1).y);
    de(2,0) = Cmplx(-de(0,2).x, de(0,2).y);
    de(2,1) = Cmplx(-de(1,2).x, de(1,2).y);
    Matrix<Cmplx,3> g;
    setIdentity(&g);
    g += de * half_alpha;
    //36
    reunit_link<Float>( g );
    //130


    for ( int mu = 0; mu < 4; mu++ ) {
      Matrix<Cmplx,3> U = dataOr(mu, id, parity);
      Matrix<Cmplx,3> g0;
      U = g * U;
      //198
      idx = linkNormalIndexP1(x,arg.X,mu);
      //Read Delta
      de(0,0) = arg.delta[idx + 0 * arg.threads];
      de(0,1) = arg.delta[idx + 1 * arg.threads];
      de(0,2) = arg.delta[idx + 2 * arg.threads];
      de(1,1) = arg.delta[idx + 3 * arg.threads];
      de(1,2) = arg.delta[idx + 4 * arg.threads];
      de(2,2) = arg.delta[idx + 5 * arg.threads];

      de(1,0) = Cmplx(-de(0,1).x, de(0,1).y);
      de(2,0) = Cmplx(-de(0,2).x, de(0,2).y);
      de(2,1) = Cmplx(-de(1,2).x, de(1,2).y);

      setIdentity(&g0);
      g0 += de * half_alpha;
      //36
      reunit_link<Float>( g0 );
      //130

      U = U * conj(g0);
      //198
      dataOr(mu, id, parity) = U;
    }
  }


  template<typename Float, typename Gauge>
  class GaugeFixNEW : TunableLocalParity {
    GaugeFixArg<Float> arg;
    Float half_alpha;
    Gauge dataOr;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:

    // since GaugeFixArg is used by other kernels that don't use
    // tunableLocalParity, arg.threads stores Volume and not VolumeCB
    // so we need to divide by two
    unsigned int minThreads() const { return arg.threads/2; }

    public:
    GaugeFixNEW(Gauge & dataOr, GaugeFixArg<Float> &arg, Float alpha)
      : dataOr(dataOr), arg(arg) {
      half_alpha = alpha * 0.5;
      cudaFuncSetCacheConfig( kernel_gauge_fix_U_EO_NEW<Float, Gauge>,   cudaFuncCachePreferL1);
    }
    ~GaugeFixNEW () { }

    void setAlpha(Float alpha){ half_alpha = alpha * 0.5; }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_gauge_fix_U_EO_NEW<Float, Gauge> <<< tp.grid, tp.block, 0, stream >>> (arg, dataOr, half_alpha);
    }

    TuneKey tuneKey() const {
      std::stringstream vol;
      vol << arg.X[0] << "x" << arg.X[1] << "x" << arg.X[2] << "x" << arg.X[3];
      sprintf(aux_string,"threads=%d,prec=%lu", arg.threads, sizeof(Float));
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);

    }

    //need this
    void preTune() {
      arg.data.backup();
    }
    void postTune() {
      arg.data.restore();
    }
    long long flops() const {
      return 2414LL * arg.threads;
      //Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
    }
    long long bytes() const {
      return ( dataOr.Bytes() * 4LL + 5 * 12LL * sizeof(Float)) * arg.threads;
    }

  };



#else
  template <int Elems, typename Float>
  __global__ void kernel_gauge_GX(GaugeFixArg<Float> arg, Float half_alpha){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( id >= arg.threads ) return;

    typedef complex<Float> Cmplx;

    Matrix<Cmplx,3> de;
    //Read Delta
    de(0,0) = arg.delta[id];
    de(0,1) = arg.delta[id + arg.threads];
    de(0,2) = arg.delta[id + 2 * arg.threads];
    de(1,1) = arg.delta[id + 3 * arg.threads];
    de(1,2) = arg.delta[id + 4 * arg.threads];
    de(2,2) = arg.delta[id + 5 * arg.threads];

    de(1,0) = makeComplex(-de(0,1).x, de(0,1).y);
    de(2,0) = makeComplex(-de(0,2).x, de(0,2).y);
    de(2,1) = makeComplex(-de(1,2).x, de(1,2).y);


    Matrix<Cmplx,3> g;
    setIdentity(&g);
    g += de * half_alpha;
    //36
    reunit_link<Float>( g );
    //130
    //gx is represented in even/odd order
    //normal lattice index to even/odd index
    int x3 = id / (arg.X[0] * arg.X[1] * arg.X[2]);
    int x2 = (id / (arg.X[0] * arg.X[1])) % arg.X[2];
    int x1 = (id / arg.X[0]) % arg.X[1];
    int x0 = id % arg.X[0];
    id  =  (x0 + (x1 + (x2 + x3 * arg.X[2]) * arg.X[1]) * arg.X[0]) >> 1;
    id += ((x0 + x1 + x2 + x3) & 1 ) * arg.threads / 2;

    for ( int i = 0; i < Elems; i++ ) arg.gx[id + i * arg.threads] = g.data[i];
    //T=166 for Elems 9
    //T=208 for Elems 6
  }




  template<int Elems, typename Float>
  class GaugeFix_GX : Tunable {
    GaugeFixArg<Float> arg;
    Float half_alpha;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    //bool tuneSharedBytes() const { return false; } // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                              // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    GaugeFix_GX(GaugeFixArg<Float> &arg, Float alpha)
      : arg(arg) {
      half_alpha = alpha * 0.5;
      cudaFuncSetCacheConfig( kernel_gauge_GX<Elems, Float>,   cudaFuncCachePreferL1);
    }
    ~GaugeFix_GX () {
    }

    void setAlpha(Float alpha){
      half_alpha = alpha * 0.5;
    }


    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_gauge_GX<Elems, Float> <<< tp.grid, tp.block, 0, stream >>> (arg, half_alpha);
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
      if ( Elems == 6 ) return 208LL * arg.threads;
      else return 166LL * arg.threads;
    }
    long long bytes() const {
      return 4LL * Elems * sizeof(Float) * arg.threads;
    }

  };


  template <int Elems, typename Float, typename Gauge>
  __global__ void kernel_gauge_fix_U_EO( GaugeFixArg<Float> arg, Gauge dataOr){
    int idd = threadIdx.x + blockIdx.x * blockDim.x;

    if ( idd >= arg.threads ) return;

    int parity = 0;
    int id = idd;
    if ( idd >= arg.threads / 2 ) {
      parity = 1;
      id -= arg.threads / 2;
    }
    typedef complex<Float> Cmplx;

    Matrix<Cmplx,3> g;
    //for(int i = 0; i < Elems; i++) g.data[i] = arg.gx[idd + i * arg.threads];
    for ( int i = 0; i < Elems; i++ ) {
      g.data[i] = arg.gx[idd + i * arg.threads];
    }
    if ( Elems == 6 ) {
      g(2,0) = conj(g(0,1) * g(1,2) - g(0,2) * g(1,1));
      g(2,1) = conj(g(0,2) * g(1,0) - g(0,0) * g(1,2));
      g(2,2) = conj(g(0,0) * g(1,1) - g(0,1) * g(1,0));
      //42
    }
    int x[4];
    getCoords(x, id, arg.X, parity);
    for ( int mu = 0; mu < 4; mu++ ) {
      Matrix<Cmplx,3> U = dataOr(mu, id, parity);
      Matrix<Cmplx,3> g0;
      U = g * U;
      //198
      int idm1 = linkIndexP1(x,arg.X,mu);
      idm1 += (1 - parity) * arg.threads / 2;
      //for(int i = 0; i < Elems; i++) g0.data[i] = arg.gx[idm1 + i * arg.threads];
      for ( int i = 0; i < Elems; i++ ) {
        g0.data[i] = arg.gx[idm1 + i * arg.threads];
      }
      if ( Elems == 6 ) {
        g0(2,0) = conj(g0(0,1) * g0(1,2) - g0(0,2) * g0(1,1));
        g0(2,1) = conj(g0(0,2) * g0(1,0) - g0(0,0) * g0(1,2));
        g0(2,2) = conj(g0(0,0) * g0(1,1) - g0(0,1) * g0(1,0));
        //42
      }
      U = U * conj(g0);
      //198
      dataOr.save(mu, id, parity) = U;
    }
    //T=42+4*(198*2+42) Elems=6
    //T=4*(198*2) Elems=9
    //Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
  }


  template<int Elems, typename Float, typename Gauge>
  class GaugeFix : Tunable {
    GaugeFixArg<Float> arg;
    Gauge dataOr;
    mutable char aux_string[128];     // used as a label in the autotuner
    private:
    unsigned int sharedBytesPerThread() const {
      return 0;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      return 0;
    }
    //bool tuneSharedBytes() const { return false; } // Don't tune shared memory
    bool tuneGridDim() const {
      return false;
    }                                              // Don't tune the grid dimensions.
    unsigned int minThreads() const {
      return arg.threads;
    }

    public:
    GaugeFix(Gauge & dataOr, GaugeFixArg<Float> &arg)
      : dataOr(dataOr), arg(arg) {
      cudaFuncSetCacheConfig( kernel_gauge_fix_U_EO<Elems, Float, Gauge>,   cudaFuncCachePreferL1);
    }
    ~GaugeFix () { }


    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      kernel_gauge_fix_U_EO<Elems, Float, Gauge> <<< tp.grid, tp.block, 0, stream >>> (arg, dataOr);
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

    //need this
    void preTune() {
      arg.data.backup();
    }
    void postTune() {
      arg.data.restore();
    }
    long long flops() const {
      if ( Elems == 6 ) return 1794LL * arg.threads;
      else return 1536LL * arg.threads;
      //Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
    }
    long long bytes() const {
      return 26LL * Elems * sizeof(Float) * arg.threads;
    }

  };
#endif
//GAUGEFIXING_DONT_USE_GX


  template<int Elems, typename Float, typename Gauge, int gauge_dir>
  void gaugefixingFFT( Gauge dataOr,  cudaGaugeField& data, \
                       const int Nsteps, const int verbose_interval, \
                       const Float alpha0, const int autotune, const double tolerance, \
                       const int stopWtheta) {

    TimeProfile profileInternalGaugeFixFFT("InternalGaugeFixQudaFFT", false);

    profileInternalGaugeFixFFT.TPSTART(QUDA_PROFILE_COMPUTE);

    Float alpha = alpha0;
    std::cout << "\tAlpha parameter of the Steepest Descent Method: " << alpha << std::endl;
    if ( autotune ) std::cout << "\tAuto tune active: yes" << std::endl;
    else std::cout << "\tAuto tune active: no" << std::endl;
    std::cout << "\tStop criterium: " << tolerance << std::endl;
    if ( stopWtheta ) std::cout << "\tStop criterium method: theta" << std::endl;
    else std::cout << "\tStop criterium method: Delta" << std::endl;
    std::cout << "\tMaximum number of iterations: " << Nsteps << std::endl;
    std::cout << "\tPrint convergence results at every " << verbose_interval << " steps" << std::endl;


    unsigned int delta_pad = data.X()[0] * data.X()[1] * data.X()[2] * data.X()[3];
    int4 size = make_int4( data.X()[0], data.X()[1], data.X()[2], data.X()[3] );
    cufftHandle plan_xy;
    cufftHandle plan_zt;

    GaugeFixArg<Float> arg(data, Elems);
    SetPlanFFT2DMany( plan_zt, size, 0, arg.delta);     //for space and time ZT
    SetPlanFFT2DMany( plan_xy, size, 1, arg.delta);    //with space only XY


    GaugeFixFFTRotateArg<Float> arg_rotate(data);
    GaugeFixFFTRotate<Float> GFRotate(arg_rotate);

    GaugeFixSETINVPSP<Float> setinvpsp(arg);
    setinvpsp.apply(0);
    GaugeFixINVPSP<Float> invpsp(arg);


#ifdef GAUGEFIXING_DONT_USE_GX
    //without using GX, gx will be created only for plane rotation but with less size
    GaugeFixNEW<Float, Gauge> gfixNew(dataOr, arg, alpha);
#else
    //using GX
    GaugeFix_GX<Elems, Float> calcGX(arg, alpha);
    GaugeFix<Elems, Float, Gauge> gfix(dataOr, arg);
#endif

    GaugeFixQualityArg<Float, Gauge> argQ(dataOr, data, arg.delta);
    GaugeFixQuality<Elems, Float, Gauge, gauge_dir> gfixquality(argQ);

    gfixquality.apply(0);
    double action0 = argQ.getAction();
    printf("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    double diff = 0.0;
    int iter = 0;
    for ( iter = 0; iter < Nsteps; iter++ ) {
      for ( int k = 0; k < 6; k++ ) {
        //------------------------------------------------------------------------
        // Set a pointer do the element k in lattice volume
        // each element is stored with stride lattice volume
        // it uses gx as temporary array!!!!!!
        //------------------------------------------------------------------------
        complex<Float> *_array = arg.delta + k * delta_pad;
        //////  2D FFT + 2D FFT
        //------------------------------------------------------------------------
        // Perform FFT on xy plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_xy, _array, arg.gx, CUFFT_FORWARD);
        //------------------------------------------------------------------------
        // Rotate hypercube, xyzt -> ztxy
        //------------------------------------------------------------------------
        GFRotate.setDirection(0, arg.gx, _array);
        GFRotate.apply(0);
        //------------------------------------------------------------------------
        // Perform FFT on zt plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_zt, _array, arg.gx, CUFFT_FORWARD);
        //------------------------------------------------------------------------
        // Normalize FFT and apply pmax^2/p^2
        //------------------------------------------------------------------------
        invpsp.apply(0);
        //------------------------------------------------------------------------
        // Perform IFFT on zt plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_zt, arg.gx, _array, CUFFT_INVERSE);
        //------------------------------------------------------------------------
        // Rotate hypercube, ztxy -> xyzt
        //------------------------------------------------------------------------
        GFRotate.setDirection(1, _array, arg.gx);
        GFRotate.apply(0);
        //------------------------------------------------------------------------
        // Perform IFFT on xy plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_xy, arg.gx, _array, CUFFT_INVERSE);
      }
                #ifdef GAUGEFIXING_DONT_USE_GX
      //------------------------------------------------------------------------
      // Apply gauge fix to current gauge field
      //------------------------------------------------------------------------
      gfixNew.apply(0);
                #else
      //------------------------------------------------------------------------
      // Calculate g(x)
      //------------------------------------------------------------------------
      calcGX.apply(0);
      //------------------------------------------------------------------------
      // Apply gauge fix to current gauge field
      //------------------------------------------------------------------------
      gfix.apply(0);
                #endif
      //------------------------------------------------------------------------
      // Measure gauge quality and recalculate new Delta(x)
      //------------------------------------------------------------------------
      gfixquality.apply(0);
      double action = argQ.getAction();
      diff = abs(action0 - action);
      if ((iter % verbose_interval) == (verbose_interval - 1))
        printf("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
      if ( autotune && ((action - action0) < -1e-14) ) {
        if ( alpha > 0.01 ) {
          alpha = 0.95 * alpha;
                                #ifdef GAUGEFIXING_DONT_USE_GX
          gfixNew.setAlpha(alpha);
                                #else
          calcGX.setAlpha(alpha);
                                #endif
          printf(">>>>>>>>>>>>>> Warning: changing alpha down -> %.4e\n", alpha );
        }
      }
      //------------------------------------------------------------------------
      // Check gauge fix quality criterium
      //------------------------------------------------------------------------
      if ( stopWtheta ) {   if ( argQ.getTheta() < tolerance ) break; }
      else { if ( diff < tolerance ) break; }

      action0 = action;
    }
    if ((iter % verbose_interval) != 0 )
      printf("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter, argQ.getAction(), argQ.getTheta(), diff);

    // Reunitarize at end
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error,
                               reunit_allow_svd, reunit_svd_only,
                               svd_rel_error, svd_abs_error);
    int num_failures = 0;
    int* num_failures_dev = static_cast<int*>(pool_device_malloc(sizeof(int)));
    cudaMemset(num_failures_dev, 0, sizeof(int));
    unitarizeLinks(data, data, num_failures_dev);
    qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);

    pool_device_free(num_failures_dev);
    if ( num_failures > 0 ) {
      errorQuda("Error in the unitarization\n");
      exit(1);
    }
    // end reunitarize


    arg.free();
    CUFFT_SAFE_CALL(cufftDestroy(plan_zt));
    CUFFT_SAFE_CALL(cufftDestroy(plan_xy));
    checkCudaError();
    qudaDeviceSynchronize();
    profileInternalGaugeFixFFT.TPSTOP(QUDA_PROFILE_COMPUTE);

    if (getVerbosity() > QUDA_SUMMARIZE){
      double secs = profileInternalGaugeFixFFT.Last(QUDA_PROFILE_COMPUTE);
      double fftflop = 5.0 * (log2((double)( data.X()[0] * data.X()[1]) ) + log2( (double)(data.X()[2] * data.X()[3] )));
      fftflop *= (double)( data.X()[0] * data.X()[1] * data.X()[2] * data.X()[3] );
      double gflops = setinvpsp.flops() + gfixquality.flops();
      double gbytes = setinvpsp.bytes() + gfixquality.bytes();
      double flop = invpsp.flops() * Elems;
      double byte = invpsp.bytes() * Elems;
      flop += (GFRotate.flops() + fftflop) * Elems * 2;
      byte += GFRotate.bytes() * Elems * 4;     //includes FFT reads, assuming 1 read and 1 write per site
      #ifdef GAUGEFIXING_DONT_USE_GX
      flop += gfixNew.flops();
      byte += gfixNew.bytes();
      #else
      flop += calcGX.flops();
      byte += calcGX.bytes();
      flop += gfix.flops();
      byte += gfix.bytes();
      #endif
      flop += gfixquality.flops();
      byte += gfixquality.bytes();
      gflops += flop * iter;
      gbytes += byte * iter;
      gflops += 4588.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3]; //Reunitarize at end
      gbytes += 8.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3] * dataOr.Bytes() ; //Reunitarize at end

      gflops = (gflops * 1e-9) / (secs);
      gbytes = gbytes / (secs * 1e9);
      printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    }
  }

  template<int Elems, typename Float, typename Gauge>
  void gaugefixingFFT( Gauge dataOr,  cudaGaugeField& data, const int gauge_dir, \
                       const int Nsteps, const int verbose_interval, const Float alpha, const int autotune, \
                       const double tolerance, const int stopWtheta) {
    if ( gauge_dir != 3 ) {
      printf("Starting Landau gauge fixing with FFTs...\n");
      gaugefixingFFT<Elems, Float, Gauge, 4>(dataOr, data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
    }
    else {
      printf("Starting Coulomb gauge fixing with FFTs...\n");
      gaugefixingFFT<Elems, Float, Gauge, 3>(dataOr, data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
    }
  }



  template<typename Float>
  void gaugefixingFFT( cudaGaugeField& data, const int gauge_dir, \
                       const int Nsteps, const int verbose_interval, const Float alpha, const int autotune, \
                       const double tolerance, const int stopWtheta) {

    // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
    // Need to fix this!!
    //9 and 6 means the number of complex elements used to store g(x) and Delta(x)
    if ( data.isNative() ) {
      if ( data.Reconstruct() == QUDA_RECONSTRUCT_NO ) {
        //printfQuda("QUDA_RECONSTRUCT_NO\n");
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
        gaugefixingFFT<9, Float>(Gauge(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_12 ) {
        //printfQuda("QUDA_RECONSTRUCT_12\n");
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
        gaugefixingFFT<6, Float>(Gauge(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_8 ) {
        //printfQuda("QUDA_RECONSTRUCT_8\n");
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
        gaugefixingFFT<6, Float>(Gauge(data), data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);

      } else {
        errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
      }
    } else {
      errorQuda("Invalid Gauge Order\n");
    }
  }

#endif // GPU_GAUGE_ALG


  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] data, quda gauge field
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value 
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   */
  void gaugefixingFFT( cudaGaugeField& data, const int gauge_dir, \
                       const int Nsteps, const int verbose_interval, const double alpha, const int autotune, \
                       const double tolerance, const int stopWtheta) {

#ifdef GPU_GAUGE_ALG
#ifdef MULTI_GPU
    if(comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3))
      errorQuda("Gauge Fixing with FFTs in multi-GPU support NOT implemented yet!\n");
#endif
    if ( data.Precision() == QUDA_HALF_PRECISION ) {
      errorQuda("Half precision not supported\n");
    }
    if ( data.Precision() == QUDA_SINGLE_PRECISION ) {
      gaugefixingFFT<float> (data, gauge_dir, Nsteps, verbose_interval, (float)alpha, autotune, tolerance, stopWtheta);
    } else if ( data.Precision() == QUDA_DOUBLE_PRECISION ) {
      gaugefixingFFT<double>(data, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
#else
    errorQuda("Gauge fixing has bot been built");
#endif
  }



}
