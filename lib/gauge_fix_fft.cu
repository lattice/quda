#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <unitarization_links.h>
#include <atomic.cuh>
#include <index_helper.cuh>

#include <cufft.h>
#include <CUFFT_Plans.h>
#include <instantiate.h>

#include <tunable_reduction.h>
#include <kernels/gauge_fix_fft.cuh>

namespace quda {

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
    GaugeFixFFTRotateArg(const GaugeField &data){
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      threads = X[0] * X[1] * X[2] * X[3];
      tmp0 = 0;
      tmp1 = 0;
    }
  };

  template <int direction, typename Arg>
  __global__ void fft_rotate_kernel_2D2D(Arg arg){ //Cmplx *data_in, Cmplx *data_out){
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

  template <typename Float, typename Arg>
  class GaugeFixFFTRotate : Tunable {
    Arg &arg;
    const GaugeField &meta;
    int direction;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

    public:
    GaugeFixFFTRotate(Arg &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    {
      direction = 0;
    }

    void setDirection(int dir, complex<Float> *data_in, complex<Float> *data_out){
      direction = dir;
      arg.tmp0 = data_in;
      arg.tmp1 = data_out;
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if ( direction == 0 )      qudaLaunchKernel(fft_rotate_kernel_2D2D<0, Arg>, tp, stream, arg);
      else if ( direction == 1 ) qudaLaunchKernel(fft_rotate_kernel_2D2D<1, Arg>, tp, stream, arg);
      else                       errorQuda("Error in GaugeFixFFTRotate option.\n");
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    long long flops() const { return 0; }
    long long bytes() const { return 4LL * sizeof(Float) * arg.threads; }
  };

  template <typename Arg>
  class GaugeFixQuality : TunableReduction2D<> {
    Arg &arg;
    const GaugeField &meta;

  public:
    GaugeFixQuality(Arg &arg, const GaugeField &meta) :
      TunableReduction2D(meta),
      arg(arg),
      meta(meta) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<FixQualityFFT>(tp, stream, arg);
      auto reset = true; // apply called multiple times with the same arg so need to reset
      arg.complete(arg.result, stream, reset);
      if (!activeTuning()) {
        arg.result.x /= (double)(3 * Arg::gauge_dir * 2 * arg.threads.x);
        arg.result.y /= (double)(3 * 2 * arg.threads.x);
      }
    }

    long long flops() const { return (36LL * Arg::gauge_dir + 65LL) * 2 * arg.threads.x; }
    long long bytes() const
    { return (Arg::gauge_dir * meta.Bytes() / 4) + 12 * meta.Volume() * meta.Precision(); }
  };

  template <typename store_t, QudaReconstructType recon>
  struct GaugeFixArg {
    using Float = typename mapper<store_t>::type;
    using Gauge = typename gauge_mapper<store_t, recon>::type;
    Gauge data;
    int threads;     // number of active threads required
    int X[4];     // grid dimensions
    Float *invpsq;
    complex<Float> *delta;
    complex<Float> *gx;

    GaugeFixArg(GaugeField &data, const int Elems) :
      data(data),
      threads(data.Volume())
    {
      for (int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
      invpsq = (Float*)device_malloc(sizeof(Float) * threads);
      delta = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads * 6);
#ifdef GAUGEFIXING_DONT_USE_GX
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads);
#else
      gx = (complex<Float>*)device_malloc(sizeof(complex<Float>) * threads * Elems);
#endif
    }
    void free() {
      device_free(invpsq);
      device_free(delta);
      device_free(gx);
    }
  };

  template <typename Arg> __global__ void kernel_gauge_set_invpsq(Arg arg)
  {
    using Float = typename Arg::Float;
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

  template <typename Arg>
  class GaugeFixSETINVPSP : Tunable {
    Arg arg;
    const GaugeField &meta;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugeFixSETINVPSP(Arg &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta) { }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(kernel_gauge_set_invpsq<Arg>, tp, stream, arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    long long flops() const { return 21 * arg.threads; }
    long long bytes() const { return sizeof(typename Arg::Float) * arg.threads; }
  };

  template <typename Arg> __global__ void kernel_gauge_mult_norm_2D(Arg arg)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < arg.threads) arg.gx[id] = arg.gx[id] * arg.invpsq[id];
  }

  template <typename Arg>
  class GaugeFixINVPSP : Tunable {
    Arg &arg;
    const GaugeField &meta;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugeFixINVPSP(Arg &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(kernel_gauge_mult_norm_2D<Arg>, tp, stream, arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    //since delta contents are irrelevant at this point, we can swap gx with delta
    void preTune() { std::swap(arg.gx, arg.delta); }
    void postTune() { std::swap(arg.gx, arg.delta); }
    long long flops() const { return 2LL * arg.threads; }
    long long bytes() const { return 5LL * sizeof(typename Arg::Float) * arg.threads; }
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

  template <typename Arg> __global__ void kernel_gauge_fix_U_EO_NEW(Arg arg, typename Arg::Float half_alpha)
  {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    if (id >= arg.threads/2) return;

    using Float = typename Arg::Float;
    using complex = complex<Float>;
    using matrix = Matrix<complex, 3>;

    int x[4];
    getCoords(x, id, arg.X, parity);
    int idx = ((x[3] * arg.X[2] + x[2]) * arg.X[1] + x[1]) * arg.X[0] + x[0];
    matrix de;
    //Read Delta
    de(0,0) = arg.delta[idx + 0 * arg.threads];
    de(0,1) = arg.delta[idx + 1 * arg.threads];
    de(0,2) = arg.delta[idx + 2 * arg.threads];
    de(1,1) = arg.delta[idx + 3 * arg.threads];
    de(1,2) = arg.delta[idx + 4 * arg.threads];
    de(2,2) = arg.delta[idx + 5 * arg.threads];

    de(1,0) = complex(-de(0,1).real(), de(0,1).imag());
    de(2,0) = complex(-de(0,2).real(), de(0,2).imag());
    de(2,1) = complex(-de(1,2).real(), de(1,2).imag());

    matrix g;
    setIdentity(&g);
    g += de * half_alpha;
    //36
    reunit_link<Float>( g );
    //130

    for ( int mu = 0; mu < 4; mu++ ) {
      matrix U = arg.data(mu, id, parity);

      matrix g0;
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

      de(1,0) = complex(-de(0,1).real(), de(0,1).imag());
      de(2,0) = complex(-de(0,2).real(), de(0,2).imag());
      de(2,1) = complex(-de(1,2).real(), de(1,2).imag());

      setIdentity(&g0);
      g0 += de * half_alpha;
      //36
      reunit_link<Float>( g0 );
      //130

      U = U * conj(g0);
      //198
      arg.data(mu, id, parity) = U;
    }
  }

  template <typename Arg>
  class GaugeFixNEW : TunableVectorY {
    Arg &arg;
    const GaugeField &meta;
    double half_alpha;

    bool tuneGridDim() const { return false; }
    // since GaugeFixArg is used by other kernels that don't keep
    // parity separate, arg.threads stores Volume and not VolumeCB so
    // we need to divide by two
    unsigned int minThreads() const { return arg.threads/2; }

  public:
    GaugeFixNEW(Arg &arg, double alpha, const GaugeField &meta) :
      TunableVectorY(2),
      arg(arg),
      meta(meta)
    {
      half_alpha = alpha * 0.5;
    }

    void setAlpha(double alpha) { half_alpha = alpha * 0.5; }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(kernel_gauge_fix_U_EO_NEW<Arg>, tp, stream, arg, (typename Arg::Float)half_alpha);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    void preTune() { meta.backup(); }
    void postTune() { meta.restore(); }
    long long flops() const { return 2414LL * arg.threads; }
    long long bytes() const { return meta.Bytes() + (5 * 12LL * sizeof(typename Arg::Float)) * arg.threads; }
  };

#else

  template <int Elems, typename Arg>
  __global__ void kernel_gauge_GX(Arg arg, typename Arg::Float half_alpha)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= arg.threads) return;

    using Float = typename Arg::Float;
    using complex = complex<Float>;

    Matrix<complex,3> de;
    //Read Delta
    de(0,0) = arg.delta[id];
    de(0,1) = arg.delta[id + arg.threads];
    de(0,2) = arg.delta[id + 2 * arg.threads];
    de(1,1) = arg.delta[id + 3 * arg.threads];
    de(1,2) = arg.delta[id + 4 * arg.threads];
    de(2,2) = arg.delta[id + 5 * arg.threads];

    de(1,0) = complex(-de(0,1).x, de(0,1).y);
    de(2,0) = complex(-de(0,2).x, de(0,2).y);
    de(2,1) = complex(-de(1,2).x, de(1,2).y);

    Matrix<complex, 3> g;
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

  template<int Elems, typename Arg>
  class GaugeFix_GX : Tunable {
    Arg arg;
    const GaugeField &meta;
    double half_alpha;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

    public:
    GaugeFix_GX(Arg &arg, double alpha, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    {
      half_alpha = alpha * 0.5;
    }

    void setAlpha(double alpha) { half_alpha = alpha * 0.5; }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(kernel_gauge_GX<Elems, Arg>, tp, stream, arg, half_alpha);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    long long flops() const {
      if ( Elems == 6 ) return 208LL * arg.threads;
      else return 166LL * arg.threads;
    }
    long long bytes() const { return 4LL * Elems * meta.Precision() * arg.threads; }
  };

  template <int Elems, typename Arg>
  __global__ void kernel_gauge_fix_U_EO(Arg arg)
  {
    int idd = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idd >= arg.threads ) return;

    int parity = 0;
    int id = idd;
    if ( idd >= arg.threads / 2 ) {
      parity = 1;
      id -= arg.threads / 2;
    }
    typedef complex<typename Arg::Float> Cmplx;

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
      Matrix<Cmplx,3> U = arg.data(mu, id, parity);
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
      arg.data(mu, id, parity) = U;
    }
    //T=42+4*(198*2+42) Elems=6
    //T=4*(198*2) Elems=9
    //Not accounting here the reconstruction of the gauge if 12 or 8!!!!!!
  }

  template<int Elems, typename Arg>
  class GaugeFix : Tunable {
    Arg arg;
    const GaugeField &meta;
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugeFix(Arg &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(kernel_gauge_fix_U_EO<Elems, Arg>, tp, stream, arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    void preTune() { meta.backup(); }
    void postTune() { meta.restore(); }
    long long flops() const {
      if ( Elems == 6 ) return 1794LL * arg.threads;
      else return 1536LL * arg.threads;
    }
    long long bytes() const { return 26LL * Elems * meta.Precision() * arg.threads; }
  };
#endif
//GAUGEFIXING_DONT_USE_GX

  template <typename Float, QudaReconstructType recon, int gauge_dir>
  void gaugefixingFFT(GaugeField& data, const int Nsteps, const int verbose_interval,
                      const Float alpha0, const int autotune, const double tolerance, const int stopWtheta)
  {
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

    constexpr int Elems = recon / 2; // number of complex elements used to store g(x) and Delta(x)
    GaugeFixArg<Float, recon> arg(data, Elems);
    SetPlanFFT2DMany( plan_zt, size, 0, arg.delta);     //for space and time ZT
    SetPlanFFT2DMany( plan_xy, size, 1, arg.delta);    //with space only XY

    GaugeFixFFTRotateArg<Float> arg_rotate(data);
    GaugeFixFFTRotate<Float, decltype(arg_rotate)> GFRotate(arg_rotate, data);

    GaugeFixSETINVPSP<decltype(arg)> setinvpsp(arg, data);
    setinvpsp.apply(0);
    GaugeFixINVPSP<decltype(arg)> invpsp(arg, data);

#ifdef GAUGEFIXING_DONT_USE_GX
    //without using GX, gx will be created only for plane rotation but with less size
    GaugeFixNEW<decltype(arg)> gfixNew(arg, alpha, data);
#else
    //using GX
    GaugeFix_GX<Elems, decltype(arg)> calcGX(arg, alpha, data);
    GaugeFix<Elems, decltype(arg)> gfix(arg, data);
#endif

    GaugeFixQualityFFTArg<Float, recon, gauge_dir> argQ(data, arg.delta);
    GaugeFixQuality<decltype(argQ)> gfixquality(argQ, data);
    gfixquality.apply(0);
    double action0 = argQ.getAction();
    printf("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    double diff = 0.0;
    int iter = 0;
    for (iter = 0; iter < Nsteps; iter++) {
      for (int k = 0; k < 6; k++) {
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

      gfixquality.apply(0);
      printfQuda("Debug %e\n", argQ.getAction());
#ifdef GAUGEFIXING_DONT_USE_GX
      //------------------------------------------------------------------------
      // Apply gauge fix to current gauge field
      //------------------------------------------------------------------------
      gfixNew.apply(0); // something wrong here, perhaps quality is messing up delta?
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
      gfixquality.apply(0);
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
    qudaMemset(num_failures_dev, 0, sizeof(int));
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
      gflops += 4588.0 * data.Volume(); //Reunitarize at end
      gbytes += 2 * data.Bytes(); //Reunitarize at end

      gflops = (gflops * 1e-9) / (secs);
      gbytes = gbytes / (secs * 1e9);
      printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    }
  }

  template<typename Float, int nColors, QudaReconstructType recon> struct GaugeFixingFFT {
    GaugeFixingFFT(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval,
                   const Float alpha, const int autotune, const double tolerance, const int stopWtheta)
    {
      if (gauge_dir != 3) {
        printfQuda("Starting Landau gauge fixing with FFTs...\n");
        gaugefixingFFT<Float, recon, 4>(data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
      } else {
        printfQuda("Starting Coulomb gauge fixing with FFTs...\n");
        gaugefixingFFT<Float, recon, 3>(data, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta);
      }
    }
  };

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
  void gaugeFixingFFT(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval, const double alpha,
                      const int autotune, const double tolerance, const int stopWtheta)
  {
#ifdef GPU_GAUGE_ALG
#ifdef MULTI_GPU
    if (comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3))
      errorQuda("Gauge Fixing with FFTs in multi-GPU support NOT implemented yet!\n");
#endif
    instantiate<GaugeFixingFFT>(data, gauge_dir, Nsteps, verbose_interval, (float)alpha, autotune, tolerance, stopWtheta);
#else
    errorQuda("Gauge fixing has bot been built");
#endif
  }

}
