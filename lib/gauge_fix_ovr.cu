#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <gauge_fix_ovr_extra.h>
#include <gauge_fix_ovr_hit_devf.cuh>
#include <reduce_helper.h>
#include <index_helper.cuh>
#include <instantiate.h>

namespace quda {

#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, arg, parity, ...)                                                   \
  if (tp.aux.x == 0) {                                                                                                 \
    switch (tp.block.x) {                                                                                              \
    case 256: qudaLaunchKernel(kernel<0, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<0, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 768: qudaLaunchKernel(kernel<0, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 1024: qudaLaunchKernel(kernel<0, 128, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else if (tp.aux.x == 1) {                                                                                          \
    switch (tp.block.x) {                                                                                              \
    case 256: qudaLaunchKernel(kernel<1, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<1, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 768: qudaLaunchKernel(kernel<1, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 1024: qudaLaunchKernel(kernel<1, 128, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else if (tp.aux.x == 2) {                                                                                          \
    switch (tp.block.x) {                                                                                              \
    case 256: qudaLaunchKernel(kernel<2, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<2, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 768: qudaLaunchKernel(kernel<2, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 1024: qudaLaunchKernel(kernel<2, 128, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else if (tp.aux.x == 3) {                                                                                          \
    switch (tp.block.x) {                                                                                              \
    case 128: qudaLaunchKernel(kernel<3, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 256: qudaLaunchKernel(kernel<3, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 384: qudaLaunchKernel(kernel<3, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<3, 128, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 640: qudaLaunchKernel(kernel<3, 160, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 768: qudaLaunchKernel(kernel<3, 192, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 896: qudaLaunchKernel(kernel<3, 224, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 1024: qudaLaunchKernel(kernel<3, 256, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else if (tp.aux.x == 4) {                                                                                          \
    switch (tp.block.x) {                                                                                              \
    case 128: qudaLaunchKernel(kernel<4, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 256: qudaLaunchKernel(kernel<4, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 384: qudaLaunchKernel(kernel<4, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<4, 128, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 640: qudaLaunchKernel(kernel<4, 160, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 768: qudaLaunchKernel(kernel<4, 192, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 896: qudaLaunchKernel(kernel<4, 224, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 1024: qudaLaunchKernel(kernel<4, 256, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else if (tp.aux.x == 5) {                                                                                          \
    switch (tp.block.x) {                                                                                              \
    case 128: qudaLaunchKernel(kernel<5, 32, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 256: qudaLaunchKernel(kernel<5, 64, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 384: qudaLaunchKernel(kernel<5, 96, __VA_ARGS__>, tp, stream, arg, parity); break;      \
    case 512: qudaLaunchKernel(kernel<5, 128, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 640: qudaLaunchKernel(kernel<5, 160, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 768: qudaLaunchKernel(kernel<5, 192, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 896: qudaLaunchKernel(kernel<5, 224, __VA_ARGS__>, tp, stream, arg, parity); break;     \
    case 1024: qudaLaunchKernel(kernel<5, 256, __VA_ARGS__>, tp, stream, arg, parity); break;    \
    default: errorQuda("%s not implemented for %d threads", #kernel, tp.block.x);                                      \
    }                                                                                                                  \
  } else {                                                                                                             \
    errorQuda("Not implemented for %d", tp.aux.x);                                                                     \
  }

  /**
   * @brief container to pass parameters for the gauge fixing quality kernel
   */
  template <typename Gauge>
  struct GaugeFixQualityArg : public ReduceArg<double2> {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4];
#endif
    Gauge dataOr;
    double2 result;
    GaugeFixQualityArg(const Gauge &dataOr, const GaugeField &data)
      : ReduceArg<double2>(), dataOr(dataOr) {

      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
      #ifdef MULTI_GPU
        border[dir] = data.R()[dir];
      #endif
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
    double getAction(){ return result.x; }
    double getTheta(){ return result.y; }
  };

  /**
   * @brief Measure gauge fixing quality
   */
  template<int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFix_quality(GaugeFixQualityArg<Gauge> argQ){
    typedef complex<Float> Cmplx;

    int idx_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double2 data = make_double2(0.0,0.0);
    while (idx_cb < argQ.threads) {
      int X[4];
#pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = argQ.X[dr];

      int x[4];
      getCoords(x, idx_cb, X, parity);
#ifdef MULTI_GPU
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) {
        x[dr] += argQ.border[dr];
        X[dr] += 2 * argQ.border[dr];
      }
#endif
      Matrix<Cmplx,3> delta;
      setZero(&delta);
      //load upward links
      for ( int mu = 0; mu < gauge_dir; mu++ ) {
        Matrix<Cmplx,3> U = argQ.dataOr(mu, linkIndex(x, X), parity);
        delta -= U;
      }
      //18*gauge_dir
      data.x += -delta(0, 0).x - delta(1, 1).x - delta(2, 2).x;
      //2
      //load downward links
      for ( int mu = 0; mu < gauge_dir; mu++ ) {
        Matrix<Cmplx,3> U = argQ.dataOr(mu, linkIndexM1(x,X,mu), 1 - parity);
        delta += U;
      }
      //18*gauge_dir
      delta -= conj(delta);
      //18
      SubTraceUnit(delta);
      //12
      data.y += getRealTraceUVdagger(delta, delta);
      //35
      //T=36*gauge_dir+65

      idx_cb += blockDim.x * gridDim.x;
    }
    argQ.template reduce2d<blockSize,2>(data);
  }

  /**
   * @brief Tunable object for the gauge fixing quality kernel
   */
  template<typename Float, typename Gauge, int gauge_dir>
  class GaugeFixQuality : TunableLocalParityReduction {
    GaugeFixQualityArg<Gauge> &arg;
    const GaugeField &meta;

  public:
    GaugeFixQuality(GaugeFixQualityArg<Gauge> &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_LOCAL_PARITY(computeFix_quality, (*this), tp, stream, arg, Float, Gauge, gauge_dir);
      auto reset = true; // apply is called multiple times with the same arg instance so we need to reset
      arg.complete(arg.result, stream, reset);
      if (!activeTuning()) {
        comm_allreduce_array((double*)&arg.result, 2);
        arg.result.x /= (double)(3 * gauge_dir * 2 * arg.threads * comm_size());
        arg.result.y /= (double)(3 * 2 * arg.threads * comm_size());
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    long long flops() const { return (36LL * gauge_dir + 65LL) * meta.Volume(); }
    //long long bytes() const { return (1)*2*gauge_dir*arg.Bytes(); }
    long long bytes() const { return 2LL * gauge_dir * meta.Volume() * meta.Reconstruct() * sizeof(Float); }
  };

  /**
   * @brief container to pass parameters for the gauge fixing kernel
   */
  template <typename Float, typename Gauge>
  struct GaugeFixArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4];
#endif
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;

    GaugeFixArg(Gauge & dataOr, GaugeField & data, const Float relax_boost)
      : dataOr(dataOr), data(data), relax_boost(relax_boost) {

      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
      #ifdef MULTI_GPU
        border[dir] = data.R()[dir];
      #endif
      }
      threads = X[0] * X[1] * X[2] * X[3] >> 1;
    }
  };


  /**
   * @brief Kernel to perform gauge fixing with overrelaxation for single-GPU
   */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFix(GaugeFixArg<Float, Gauge> arg, int parity)
  {
    typedef complex<Float> Cmplx;
    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;

    if ( idx >= arg.threads ) return;

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      int X[4];
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
  #ifdef MULTI_GPU
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
  #endif
      int mu = (threadIdx.x / blockSize);
      int oddbit = parity;
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        oddbit = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, oddbit);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, oddbit) = link;
    }
    // 4 threads per lattice site
    else{
      int X[4];
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, idx, X, parity);
  #ifdef MULTI_GPU
    #pragma unroll
      for ( int dr = 0; dr < 4; ++dr ) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
  #endif
      int mu = (threadIdx.x / blockSize);
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      //load upward link
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);

      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      //load downward link
      Matrix<Cmplx,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  /**
   * @brief Tunable object for the gauge fixing kernel
   */
  template<typename Float, typename Gauge, int gauge_dir>
  class GaugeFix : Tunable {
    GaugeFixArg<Float, Gauge> &arg;
    const GaugeField &meta;
    int parity;

    dim3 createGrid(const TuneParam &param) const
    {
      unsigned int blockx = param.block.x / 8;
      if (param.aux.x > 2) blockx = param.block.x / 4;
      unsigned int gx  = std::max((arg.threads + blockx - 1) / blockx, 1u);
      return dim3(gx, 1, 1);
    }

    bool advanceBlockDim (TuneParam &param) const
    {
      // Use param.aux.x to tune and save state for best kernel option
      // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
      const unsigned int min_threads0 = 32 * 8;
      const unsigned int min_threads1 = 32 * 4;
      const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= max_threads)) {
        param.shared_bytes = sharedBytesPerBlock(param);
        return true;
      } else if (param.aux.x == 0) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 1; // USE FOR ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 1) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 2; // USE FOR NO ATOMIC ADD and LESS SHARED MEM
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 2) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 3; // USE FOR NO ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float);
        return true;
      } else if (param.aux.x == 3) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 4;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else if (param.aux.x == 4) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 5;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else {
        return false;
      }
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      switch (param.aux.x) {
      case 0: return param.block.x * 4 * sizeof(Float);
      case 1: return param.block.x * 4 * sizeof(Float) / 8;
      case 2: return param.block.x * 4 * sizeof(Float) / 8;
      case 3: return param.block.x * 4 * sizeof(Float);
      default: return param.block.x * sizeof(Float);
      }
    }

    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

public:
    GaugeFix(GaugeFixArg<Float, Gauge> &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta),
      parity(0) { }

    void setParity(const int par) { parity = par; }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_GAUGEFIX(computeFix, tp, stream, arg, parity, Float, Gauge, gauge_dir);
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      param.block = dim3(256, 1, 1);
      param.aux.x = 0;
      param.grid = createGrid(param);
      param.shared_bytes = sharedBytesPerBlock(param);
    }

    virtual void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    std::string paramString(const TuneParam &param) const {
      std::stringstream ps(Tunable::paramString(param));
      ps << ", atomicadd=" << param.aux.x;
      return ps.str();
    }

    void preTune() { arg.data.backup(); }
    void postTune() { arg.data.restore(); }
    long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3) * arg.threads; }
    long long bytes() const { return 8LL * 2 * arg.threads * meta.Reconstruct() * sizeof(Float);  }
  };

#ifdef MULTI_GPU
  template <typename Float, typename Gauge>
  struct GaugeFixInteriorPointsArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4];
#endif
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;
    GaugeFixInteriorPointsArg(Gauge & dataOr, GaugeField & data, const Float relax_boost)
      : dataOr(dataOr), data(data), relax_boost(relax_boost) {

#ifdef MULTI_GPU
      for ( int dir = 0; dir < 4; ++dir ) {
        if ( comm_dim_partitioned(dir)) border[dir] = data.R()[dir] + 1;  //skip BORDER_RADIUS + face border point
        else border[dir] = 0;
      }
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir] - border[dir] * 2;
#else
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
#endif
      threads = X[0] * X[1] * X[2] * X[3] >> 1;
      if (this->threads == 0) errorQuda("Local volume is too small");
    }
  };


  /**
   * @brief Kernel to perform gauge fixing with overrelaxation in the interior points for multi-GPU implementation
   */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFixInteriorPoints(GaugeFixInteriorPointsArg<Float, Gauge> arg, int parity){
    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;
    if ( idx >= arg.threads ) return;
    typedef complex<Float> Complex;
    int X[4];
#pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
#ifdef MULTI_GPU
    int za = (idx / (X[0] / 2));
    int zb =  (za / X[1]);
    x[1] = za - zb * X[1];
    x[3] = (zb / X[2]);
    x[2] = zb - x[3] * X[2];
    int p = 0; for ( int dr = 0; dr < 4; ++dr ) p += arg.border[dr];
    p = p & 1;
    int x1odd = (x[1] + x[2] + x[3] + parity + p) & 1;
    //int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * idx + x1odd)  - za * X[0];
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }
#else
    getCoords(x, idx, X, parity);
#endif
    int mu = (threadIdx.x / blockSize);

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        parity = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link = arg.dataOr(mu, idx, parity);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, parity) = link;
    }
    // 4 threads per lattice site
    else{
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link = arg.dataOr(mu, idx, parity);


      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Complex,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  /**
   * @brief Tunable object for the interior points of the gauge fixing
   * kernel in multi-GPU implementation
   */
  template<typename Float, typename Gauge, int gauge_dir>
  class GaugeFixInteriorPoints : Tunable {
    GaugeFixInteriorPointsArg<Float, Gauge> &arg;
    const GaugeField &meta;
    int parity;

    dim3 createGrid(const TuneParam &param) const
    {
      unsigned int blockx = param.block.x / 8;
      if (param.aux.x > 2) blockx = param.block.x / 4;
      unsigned int gx  = (arg.threads + blockx - 1) / blockx;
      return dim3(gx, 1, 1);
    }

    bool advanceBlockDim(TuneParam &param) const
    {
      // Use param.aux.x to tune and save state for best kernel option
      // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
      const unsigned int min_threads0 = 32 * 8;
      const unsigned int min_threads1 = 32 * 4;
      const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= max_threads)) {
        param.shared_bytes = sharedBytesPerBlock(param);
        return true;
      } else if (param.aux.x == 0) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 1; // USE FOR ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 1) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 2; // USE FOR NO ATOMIC ADD and LESS SHARED MEM
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 2) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 3; // USE FOR NO ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float);
        return true;
      } else if (param.aux.x == 3) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 4;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else if (param.aux.x == 4) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 5;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else {
        return false;
      }
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      switch (param.aux.x) {
      case 0: return param.block.x * 4 * sizeof(Float);
      case 1: return param.block.x * 4 * sizeof(Float) / 8;
      case 2: return param.block.x * 4 * sizeof(Float) / 8;
      case 3: return param.block.x * 4 * sizeof(Float);
      default: return param.block.x * sizeof(Float);
      }
    }

    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

public:
    GaugeFixInteriorPoints(GaugeFixInteriorPointsArg<Float, Gauge> &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta),
      parity(0) {}

    void setParity(const int par) { parity = par; }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_GAUGEFIX(computeFixInteriorPoints, tp, stream, arg, parity, Float, Gauge, gauge_dir);
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      param.block = dim3(256, 1, 1);
      param.aux.x = 0;
      param.grid = createGrid(param);
      param.shared_bytes = sharedBytesPerBlock(param);
    }

    virtual void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    std::string paramString(const TuneParam &param) const {
      std::stringstream ps(Tunable::paramString(param));
      ps << ", atomicadd=" << param.aux.x;
      return ps.str();
    }

    void preTune() { arg.data.backup(); }
    void postTune() { arg.data.restore(); }
    long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3) * arg.threads; }
    long long bytes() const { return 8LL * 2 * arg.threads * meta.Reconstruct() * sizeof(Float); }
  };

  template <typename Float, typename Gauge>
  struct GaugeFixBorderPointsArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
    int border[4];
    int *borderpoints[2];
    int *faceindicessize[2];
    size_t faceVolume[4];
    size_t faceVolumeCB[4];
    Gauge dataOr;
    GaugeField &data;
    const Float relax_boost;

    GaugeFixBorderPointsArg(Gauge & dataOr, GaugeField & data, const Float relax_boost, size_t faceVolume_[4], size_t faceVolumeCB_[4])
      : dataOr(dataOr), data(data), relax_boost(relax_boost)
    {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        border[dir] = data.R()[dir];
      }

      /*for(int dir=0; dir<4; ++dir){
         if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
         else border[dir] = 0;
         }
         for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;*/
      for ( int dir = 0; dir < 4; ++dir ) {
        faceVolume[dir] = faceVolume_[dir];
        faceVolumeCB[dir] = faceVolumeCB_[dir];
      }
      if ( comm_partitioned() ) PreCalculateLatticeIndices(faceVolume, faceVolumeCB, X, border, threads, borderpoints);
    }
  };

  /**
   * @brief Kernel to perform gauge fixing with overrelaxation in the border points for multi-GPU implementation
  */
  template<int ImplementationType, int blockSize, typename Float, typename Gauge, int gauge_dir>
  __global__ void computeFixBorderPoints(GaugeFixBorderPointsArg<Float, Gauge> arg, int parity){
    typedef complex<Float> Cmplx;

    int tid = (threadIdx.x + blockSize) % blockSize;
    int idx = blockIdx.x * blockSize + tid;
    if ( idx >= arg.threads ) return;
    int mu = (threadIdx.x / blockSize);
    idx = arg.borderpoints[parity][idx];
    int X[4], x[4];
    x[3] = idx / (arg.X[0] * arg.X[1]  * arg.X[2]);
    x[2] = (idx / (arg.X[0] * arg.X[1])) % arg.X[2];
    x[1] = (idx / arg.X[0]) % arg.X[1];
    x[0] = idx % arg.X[0];
  #pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) x[dr] += arg.border[dr];
  #pragma unroll
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr] + 2 * arg.border[dr];

    // 8 threads per lattice site
    if ( ImplementationType < 3 ) {
      if ( threadIdx.x >= blockSize * 4 ) {
        mu -= 4;
        x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
        parity = 1 - parity;
      }
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 8x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 0 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 1 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      // 8 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 2 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, arg.relax_boost, tid);
      arg.dataOr(mu, idx, parity) = link;
    }
    // 4 threads per lattice site
    else{
      idx = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link = arg.dataOr(mu, idx, parity);


      x[mu] = (x[mu] - 1 + X[mu]) % X[mu];
      int idx1 = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
      Matrix<Cmplx,3> link1 = arg.dataOr(mu, idx1, 1 - parity);

      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // this implementation needs 4x more shared memory than the implementation using atomicadd
      if ( ImplementationType == 3 ) GaugeFixHit_NoAtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory using atomicadd
      if ( ImplementationType == 4 ) GaugeFixHit_AtomicAdd<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);
      // 4 treads per lattice site, the reduction is performed by shared memory without using atomicadd.
      // uses the same amount of shared memory as the atomicadd implementation with more thread block synchronization
      if ( ImplementationType == 5 ) GaugeFixHit_NoAtomicAdd_LessSM<blockSize, Float, gauge_dir, 3>(link, link1, arg.relax_boost, tid);

      arg.dataOr(mu, idx, parity) = link;
      arg.dataOr(mu, idx1, 1 - parity) = link1;
    }
  }

  /**
   * @brief Tunable object for the border points of the gauge fixing kernel in multi-GPU implementation
   */
  template<typename Float, typename Gauge, int gauge_dir>
  class GaugeFixBorderPoints : Tunable {
    GaugeFixBorderPointsArg<Float, Gauge> &arg;
    const GaugeField &meta;
    int parity;

    dim3 createGrid(const TuneParam &param) const
    {
      unsigned int blockx = param.block.x / 8;
      if (param.aux.x > 2) blockx = param.block.x / 4;
      unsigned int gx = (arg.threads + blockx - 1) / blockx;
      return dim3(gx, 1, 1);
    }

    bool advanceBlockDim(TuneParam &param) const
    {
      // Use param.aux.x to tune and save state for best kernel option
      // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
      const unsigned int min_threads0 = 32 * 8;
      const unsigned int min_threads1 = 32 * 4;
      const unsigned int max_threads = 1024; // FIXME: use deviceProp.maxThreadsDim[0];
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= max_threads)) {
        param.shared_bytes = sharedBytesPerBlock(param);
        return true;
      } else if (param.aux.x == 0) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 1; // USE FOR ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 1) {
        param.block.x = min_threads0;
        param.block.y = 1;
        param.aux.x = 2; // USE FOR NO ATOMIC ADD and LESS SHARED MEM
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float) / 8;
        return true;
      } else if (param.aux.x == 2) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 3; // USE FOR NO ATOMIC ADD
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * 4 * sizeof(Float);
        return true;
      } else if (param.aux.x == 3) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 4;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else if (param.aux.x == 4) {
        param.block.x = min_threads1;
        param.block.y = 1;
        param.aux.x = 5;
        param.grid = createGrid(param);
        param.shared_bytes = param.block.x * sizeof(Float);
        return true;
      } else {
        return false;
      }
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      switch (param.aux.x) {
      case 0: return param.block.x * 4 * sizeof(Float);
      case 1: return param.block.x * 4 * sizeof(Float) / 8;
      case 2: return param.block.x * 4 * sizeof(Float) / 8;
      case 3: return param.block.x * 4 * sizeof(Float);
      default: return param.block.x * sizeof(Float);
      }
    }

    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

public:
    GaugeFixBorderPoints(GaugeFixBorderPointsArg<Float, Gauge> &arg, const GaugeField &meta) :
      arg(arg),
      meta(meta),
      parity(0) { }

    ~GaugeFixBorderPoints () {
      if ( comm_partitioned() ) for ( int i = 0; i < 2; i++ ) pool_device_free(arg.borderpoints[i]);
    }

    void setParity(const int par) { parity = par; }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_GAUGEFIX(computeFixBorderPoints, tp, stream, arg, parity, Float, Gauge, gauge_dir);
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      param.block = dim3(256, 1, 1);
      param.aux.x = 0;
      param.grid = createGrid(param);
      param.shared_bytes = sharedBytesPerBlock(param);
    }

    virtual void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    std::string paramString(const TuneParam &param) const {
      std::stringstream ps(Tunable::paramString(param));
      ps << ", atomicadd=" << param.aux.x;
      return ps.str();
    }

    void preTune() { arg.data.backup(); }
    void postTune() { arg.data.restore(); }
    long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3) * arg.threads; }
    //long long bytes() const { return (1)*8*2*arg.dataOr.Bytes(); } // Only correct if there is no link reconstruction load+save
    long long bytes() const { return 8LL * 2 * arg.threads * meta.Reconstruct() * sizeof(Float); }
  };

  template <int NElems_, typename Gauge>
  struct GaugeFixUnPackArg {
    static constexpr int NElems = NElems_;
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4];
#endif
    Gauge dataOr;
    GaugeFixUnPackArg(Gauge & dataOr, GaugeField & data)
      : dataOr(dataOr) {
      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
      #ifdef MULTI_GPU
        border[dir] = data.R()[dir];
      #endif
      }
    }
  };

  template <typename Float, bool pack, typename Arg>
  __global__ void Kernel_UnPackGhost(int size, Arg arg, complex<Float> *array, int parity, int face, int dir)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    int borderid = 0;
    parity = 1 - parity;
    switch ( face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }
    x[face] -= 1;
    parity = 1 - parity;
    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Cmplx;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[Arg::NElems];
    Cmplx data[9];
    if ( pack ) {
      arg.dataOr.load(data, id, dir, parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        array[idx + size * i] = Cmplx(tmp[2*i+0], tmp[2*i+1]);
      }
    } else {
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        tmp[2*i+0] = array[idx + size * i].real();
        tmp[2*i+1] = array[idx + size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, dir, parity);
    }
  }

  template <typename Float, bool pack, typename Arg>
  __global__ void Kernel_UnPackTop(int size, Arg arg, complex<Float> *array, int parity, int face, int dir)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    int borderid = arg.X[face] - 1;
    switch ( face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }
    for ( int dr = 0; dr < 4; ++dr ) {
      x[dr] += arg.border[dr];
      X[dr] += 2 * arg.border[dr];
    }
    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Cmplx;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[Arg::NElems];
    Cmplx data[9];
    if ( pack ) {
      arg.dataOr.load(data, id, dir, parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < Arg::NElems / 2; ++i ) array[idx + size * i] = Cmplx(tmp[2*i+0], tmp[2*i+1]);
    }
    else{
      for ( int i = 0; i < Arg::NElems / 2; ++i ) {
        tmp[2*i+0] = array[idx + size * i].real();
        tmp[2*i+1] = array[idx + size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, dir, parity);
    }
  }
#endif


  template<typename Float, typename Gauge, int NElems, int gauge_dir>
  void gaugefixingOVR( Gauge dataOr, GaugeField& data,
		       const int Nsteps, const int verbose_interval,
		       const Float relax_boost, const double tolerance,
		       const int reunit_interval, const int stopWtheta)
  {
    TimeProfile profileInternalGaugeFixOVR("InternalGaugeFixQudaOVR", false);

    profileInternalGaugeFixOVR.TPSTART(QUDA_PROFILE_COMPUTE);
    double flop = 0;
    double byte = 0;

    printfQuda("\tOverrelaxation boost parameter: %lf\n", (double)relax_boost);
    printfQuda("\tStop criterium: %lf\n", tolerance);
    if ( stopWtheta ) printfQuda("\tStop criterium method: theta\n");
    else printfQuda("\tStop criterium method: Delta\n");
    printfQuda("\tMaximum number of iterations: %d\n", Nsteps);
    printfQuda("\tReunitarize at every %d steps\n", reunit_interval);
    printfQuda("\tPrint convergence results at every %d steps\n", verbose_interval);

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

    GaugeFixQualityArg<Gauge> argQ(dataOr, data);
    GaugeFixQuality<Float,Gauge, gauge_dir> GaugeFixQuality(argQ, data);

    GaugeFixArg<Float, Gauge> arg(dataOr, data, relax_boost);
    GaugeFix<Float,Gauge, gauge_dir> gaugeFix(arg, data);

#ifdef MULTI_GPU
    void *send[4];
    void *recv[4];
    void *sendg[4];
    void *recvg[4];
    void *send_d[4];
    void *recv_d[4];
    void *sendg_d[4];
    void *recvg_d[4];
    void *hostbuffer_h[4];
    qudaStream_t GFStream[9];
    size_t offset[4];
    size_t bytes[4];
    size_t faceVolume[4];
    size_t faceVolumeCB[4];
    // do the exchange
    MsgHandle *mh_recv_back[4];
    MsgHandle *mh_recv_fwd[4];
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_send_back[4];
    int X[4];
    TuneParam tp[4];

    if ( comm_partitioned() ) {

      for ( int dir = 0; dir < 4; ++dir ) {
        X[dir] = data.X()[dir] - data.R()[dir] * 2;
        if ( !commDimPartitioned(dir) && data.R()[dir] != 0 ) errorQuda("Not supported!");
      }
      for ( int i = 0; i < 4; i++ ) {
        faceVolume[i] = 1;
        for ( int j = 0; j < 4; j++ ) {
          if ( i == j ) continue;
          faceVolume[i] *= X[j];
        }
        faceVolumeCB[i] = faceVolume[i] / 2;
      }

      for ( int d = 0; d < 4; d++ ) {
        if ( !commDimPartitioned(d)) continue;
        offset[d] = faceVolumeCB[d] * NElems;
        bytes[d] =  sizeof(Float) * offset[d];
        send_d[d] = device_malloc(bytes[d]);
        recv_d[d] = device_malloc(bytes[d]);
        sendg_d[d] = device_malloc(bytes[d]);
        recvg_d[d] = device_malloc(bytes[d]);
        cudaStreamCreate(&GFStream[d]);
        cudaStreamCreate(&GFStream[4 + d]);
      #ifndef GPU_COMMS
        hostbuffer_h[d] = (void*)pinned_malloc(4 * bytes[d]);
      #endif
        tp[d].block = make_uint3(128, 1, 1);
        tp[d].grid = make_uint3((faceVolumeCB[d] + tp[d].block.x - 1) / tp[d].block.x, 1, 1);
      }
      cudaStreamCreate(&GFStream[8]);
      for ( int d = 0; d < 4; d++ ) {
        if ( !commDimPartitioned(d)) continue;
      #ifdef GPU_COMMS
        recv[d] = recv_d[d];
        send[d] = send_d[d];
        recvg[d] = recvg_d[d];
        sendg[d] = sendg_d[d];
      #else
        recv[d] = hostbuffer_h[d];
        send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
        recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
        sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];
      #endif
        mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
        mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
        mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
        mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
      }
    }
    GaugeFixUnPackArg<NElems,Gauge> dataexarg(dataOr, data);
    GaugeFixBorderPointsArg<Float, Gauge> argBorder(dataOr, data, relax_boost, faceVolume, faceVolumeCB);
    GaugeFixBorderPoints<Float,Gauge, gauge_dir> gfixBorderPoints(argBorder, data);
    GaugeFixInteriorPointsArg<Float, Gauge> argInt(dataOr, data, relax_boost);
    GaugeFixInteriorPoints<Float,Gauge, gauge_dir> gfixIntPoints(argInt, data);
  #endif

    GaugeFixQuality.apply(0);
    flop += (double)GaugeFixQuality.flops();
    byte += (double)GaugeFixQuality.bytes();
    double action0 = argQ.getAction();
    printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    unitarizeLinks(data, data, num_failures_dev);
    qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if ( num_failures > 0 ) {
      pool_device_free(num_failures_dev);
      errorQuda("Error in the unitarization\n");
      exit(1);
    }
    qudaMemset(num_failures_dev, 0, sizeof(int));

    int iter = 0;
    for ( iter = 0; iter < Nsteps; iter++ ) {
      for ( int p = 0; p < 2; p++ ) {
      #ifndef MULTI_GPU
        gaugeFix.setParity(p);
        gaugeFix.apply(0);
        flop += (double)gaugeFix.flops();
        byte += (double)gaugeFix.bytes();
      #else
        if ( !comm_partitioned() ) {
          gaugeFix.setParity(p);
          gaugeFix.apply(0);
          flop += (double)gaugeFix.flops();
          byte += (double)gaugeFix.bytes();
        }
        else{
          gfixIntPoints.setParity(p);
          gfixBorderPoints.setParity(p); //compute border points
          gfixBorderPoints.apply(0);
          flop += (double)gfixBorderPoints.flops();
          byte += (double)gfixBorderPoints.bytes();
          flop += (double)gfixIntPoints.flops();
          byte += (double)gfixIntPoints.bytes();
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_start(mh_recv_back[d]);
            comm_start(mh_recv_fwd[d]);
          }
          //wait for the update to the halo points before start packing...
          qudaDeviceSynchronize();
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            //extract top face
            qudaLaunchKernel(Kernel_UnPackTop<Float, true, decltype(dataexarg)>, tp[d], GFStream[d],
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(send_d[d]), p, d, d);
            //extract bottom ghost
            qudaLaunchKernel(Kernel_UnPackGhost<Float, true, decltype(dataexarg)>, tp[d], GFStream[4 + d],
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(sendg_d[d]), 1 - p, d, d);
          }
        #ifdef GPU_COMMS
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaStreamSynchronize(GFStream[d]);
            comm_start(mh_send_fwd[d]);
            qudaStreamSynchronize(GFStream[4 + d]);
            comm_start(mh_send_back[d]);
          }
        #else
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaMemcpyAsync(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[d]);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[4 + d]);
          }
        #endif
          //compute interior points
          gfixIntPoints.apply(GFStream[8]);

        #ifndef GPU_COMMS
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaStreamSynchronize(GFStream[d]);
            comm_start(mh_send_fwd[d]);
            qudaStreamSynchronize(GFStream[4 + d]);
            comm_start(mh_send_back[d]);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_recv_back[d]);
            qudaMemcpyAsync(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice, GFStream[d]);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_recv_fwd[d]);
            qudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice, GFStream[4 + d]);
          }
        #endif
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
          #ifdef GPU_COMMS
            comm_wait(mh_recv_back[d]);
          #endif
            qudaLaunchKernel(Kernel_UnPackGhost<Float, false, decltype(dataexarg)>, tp[d], GFStream[d],
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recv_d[d]), p, d, d);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
          #ifdef GPU_COMMS
            comm_wait(mh_recv_fwd[d]);
          #endif
            qudaLaunchKernel(Kernel_UnPackTop<Float, false, decltype(dataexarg)>, tp[d], GFStream[4 + d],
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recvg_d[d]), 1 - p, d, d);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_send_back[d]);
            comm_wait(mh_send_fwd[d]);
            qudaStreamSynchronize(GFStream[d]);
            qudaStreamSynchronize(GFStream[4 + d]);
          }
          qudaStreamSynchronize(GFStream[8]);
        }
      #endif
      }
      if ((iter % reunit_interval) == (reunit_interval - 1)) {
        unitarizeLinks(data, data, num_failures_dev);
        qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
        if ( num_failures > 0 ) errorQuda("Error in the unitarization\n");
        qudaMemset(num_failures_dev, 0, sizeof(int));
        flop += 4588.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3];
        byte += 8.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3] * dataOr.Bytes();
      }
      GaugeFixQuality.apply(0);
      flop += (double)GaugeFixQuality.flops();
      byte += (double)GaugeFixQuality.bytes();
      double action = argQ.getAction();
      double diff = abs(action0 - action);
      if ((iter % verbose_interval) == (verbose_interval - 1))
        printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
      if ( stopWtheta ) {
        if ( argQ.getTheta() < tolerance ) break;
      }
      else{
        if ( diff < tolerance ) break;
      }
      action0 = action;
    }
    if ((iter % reunit_interval) != 0 )  {
      unitarizeLinks(data, data, num_failures_dev);
      qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
      if ( num_failures > 0 ) errorQuda("Error in the unitarization\n");
      qudaMemset(num_failures_dev, 0, sizeof(int));
      flop += 4588.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3];
      byte += 8.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3] * dataOr.Bytes();
    }
    if ((iter % verbose_interval) != 0 ) {
      GaugeFixQuality.apply(0);
      flop += (double)GaugeFixQuality.flops();
      byte += (double)GaugeFixQuality.bytes();
      double action = argQ.getAction();
      double diff = abs(action0 - action);
      printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
    }
    pool_device_free(num_failures_dev);
  #ifdef MULTI_GPU
    if ( comm_partitioned() ) {
      data.exchangeExtendedGhost(data.R(),false);
      for ( int d = 0; d < 4; d++ ) {
        if ( commDimPartitioned(d)) {
          comm_free(mh_send_fwd[d]);
          comm_free(mh_send_back[d]);
          comm_free(mh_recv_back[d]);
          comm_free(mh_recv_fwd[d]);
          device_free(send_d[d]);
          device_free(recv_d[d]);
          device_free(sendg_d[d]);
          device_free(recvg_d[d]);
          cudaStreamDestroy(GFStream[d]);
          cudaStreamDestroy(GFStream[4 + d]);
        #ifndef GPU_COMMS
          host_free(hostbuffer_h[d]);
        #endif
        }
      }
      cudaStreamDestroy(GFStream[8]);
    }
  #endif
    qudaDeviceSynchronize();
    profileInternalGaugeFixOVR.TPSTOP(QUDA_PROFILE_COMPUTE);
    if (getVerbosity() > QUDA_SUMMARIZE){
      double secs = profileInternalGaugeFixOVR.Last(QUDA_PROFILE_COMPUTE);
      double gflops = (flop * 1e-9) / (secs);
      double gbytes = byte / (secs * 1e9);
      printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops * comm_size(), gbytes * comm_size());
    }
  }

  template <typename Float, int nColor, QudaReconstructType recon> struct GaugeFixingOVR {
    GaugeFixingOVR(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval,
                   const Float relax_boost, const double tolerance, const int reunit_interval, const int stopWtheta)
    {
      using Gauge = typename gauge_mapper<Float, recon>::type;
      if (gauge_dir == 4) {
        printfQuda("Starting Landau gauge fixing...\n");
        gaugefixingOVR<Float, Gauge, recon, 4>(Gauge(data), data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
      } else if (gauge_dir == 3) {
        printfQuda("Starting Coulomb gauge fixing...\n");
        gaugefixingOVR<Float, Gauge, recon, 3>(Gauge(data), data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
      } else {
        errorQuda("Unexpected gauge_dir = %d", gauge_dir);
      }
    }
  };

  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] data, quda gauge field
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   */
  void gaugeFixingOVR(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval, const double relax_boost,
                      const double tolerance, const int reunit_interval, const int stopWtheta) {
#ifdef GPU_GAUGE_ALG
    instantiate<GaugeFixingOVR>(data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
#else
    errorQuda("Gauge fixing has not been built");
#endif // GPU_GAUGE_ALG
  }

}   //namespace quda
