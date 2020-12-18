#include <quda_internal.h>
#include <gauge_field.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <gauge_fix_ovr_hit_devf.cuh>
#include <reduce_helper.h>
#ifndef QUDA_BACKEND_OMPTARGET
#include <thrust_helper.cuh>
#endif
#include <instantiate.h>
#include <tunable_reduction.h>
#include <tunable_nd.h>
#include <kernels/gauge_fix_ovr.cuh>

namespace quda {

#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, arg, parity, ...)    \
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
      unsigned int gx  = (arg.threads + blockx - 1) / blockx;
      return dim3(gx, 1, 1);
    }

    bool advanceBlockDim (TuneParam &param) const
    {
      // Use param.aux.x to tune and save state for best kernel option
      // to make use or not of atomicAdd operations and 4 or 8 threads per lattice site!!!
      const unsigned int min_threads0 = 32 * 8;
      const unsigned int min_threads1 = 32 * 4;
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= device::max_threads_per_block_dim(0))) {
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
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= device::max_threads_per_block_dim(0))) {
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
      const unsigned int atmadd = 0;
      unsigned int min_threads = min_threads0;
      param.aux.x += atmadd; // USE TO SELECT BEST KERNEL OPTION WITH/WITHOUT USING ATOMICADD
      if (param.aux.x > 2) min_threads = 32 * 4;
      param.block.x += min_threads;
      param.block.y = 1;
      param.grid = createGrid(param);

      if ((param.block.x >= min_threads) && (param.block.x <= device::max_threads_per_block_dim(0))) {
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

    virtual ~GaugeFixBorderPoints () {
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

  /**
   * @brief Pre-calculate lattice border points used by the gauge
   * fixing with overrelaxation in multi-GPU implementation
   */
  void PreCalculateLatticeIndices(size_t faceVolume[4], int X[4], int border[4], int &threads, int *borderpoints[2])
  {
#ifdef QUDA_BACKEND_OMPTARGET
    ompwip("unimplemented");
#else
    BorderIdArg arg(X, border);
    int nlinksfaces = 0;
    for (int dir = 0; dir < 4; dir++)
      if (comm_dim_partitioned(dir)) nlinksfaces += faceVolume[dir];

    thrust::device_ptr<int> array_faceT[2];
    thrust::device_ptr<int> array_interiorT[2];

    for (int i = 0; i < 2; i++) { //even and odd ids
      borderpoints[i] = static_cast<int*>(pool_device_malloc(nlinksfaces * sizeof(int)));
      qudaMemset(borderpoints[i], 0, nlinksfaces * sizeof(int) );
      array_faceT[i] = thrust::device_pointer_cast(borderpoints[i]);
    }

    TuneParam tp;
    tp.block = dim3(128, 1, 1);
    int start = 0;
    for (int dir = 0; dir < 4; ++dir) {
      if (comm_dim_partitioned(dir)) {
        tp.grid = dim3((faceVolume[dir] + tp.block.x - 1) / tp.block.x,1,1);
        for (int oddbit = 0; oddbit < 2; oddbit++) {
          auto faceindices = borderpoints[oddbit] + start;
          qudaLaunchKernel(ComputeBorderPointsActiveFaceIndex, tp, device::get_default_stream(), arg, faceindices, faceVolume[dir], dir, oddbit);
        }
        start += faceVolume[dir];
      }
    }
    int size[2];
    for (int i = 0; i < 2; i++) {
      //sort and remove duplicated lattice indices
      thrust_allocator alloc;
      thrust::sort(thrust::cuda::par(alloc), array_faceT[i], array_faceT[i] + nlinksfaces);
      thrust::device_ptr<int> new_end = thrust::unique(array_faceT[i], array_faceT[i] + nlinksfaces);
      size[i] = thrust::raw_pointer_cast(new_end) - thrust::raw_pointer_cast(array_faceT[i]);
    }
    if (size[0] == size[1]) threads = size[0];
    else errorQuda("BORDER: Even and Odd sizes does not match, not supported!!!!, %d:%d", size[0], size[1]);
#endif
  }

  /**
   * @brief Tunable object for the gauge fixing quality kernel
   */
  template <typename Arg>
  class GaugeFixQuality : TunableReduction2D<> {
    Arg &arg;
    const GaugeField &meta;

  public:
    GaugeFixQuality(Arg &arg, const GaugeField &meta) :
      TunableReduction2D(meta),
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<FixQualityOVR>(arg.result, tp, stream, arg);
      if (!activeTuning()) {
        arg.result.x /= (double)(3 * Arg::gauge_dir * 2 * arg.threads.x * comm_size());
        arg.result.y /= (double)(3 * 2 * arg.threads.x * comm_size());
      }
    }

    long long flops() const { return (36LL * Arg::gauge_dir + 65LL) * meta.Volume(); }
    //long long bytes() const { return (1)*2*gauge_dir*arg.Bytes(); }
    long long bytes() const { return 2LL * Arg::gauge_dir * meta.Volume() * meta.Reconstruct() * meta.Precision(); }
  };

  template <typename Float, QudaReconstructType recon, int gauge_dir>
  void gaugeFixingOVR(GaugeField &data,const int Nsteps, const int verbose_interval,
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

    GaugeFixQualityOVRArg<Float, recon, gauge_dir> argQ(data);
    GaugeFixQuality<decltype(argQ)> GaugeFixQuality(argQ, data);

    using Gauge = typename gauge_mapper<Float, recon>::type;
    Gauge dataOr(data);
    GaugeFixArg<Float, Gauge> arg(dataOr, data, relax_boost);
    GaugeFix<Float,Gauge, gauge_dir> gaugeFix(arg, data);

    void *send[4];
    void *recv[4];
    void *sendg[4];
    void *recvg[4];
    void *send_d[4];
    void *recv_d[4];
    void *sendg_d[4];
    void *recvg_d[4];
    void *hostbuffer_h[4];
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
        offset[d] = faceVolumeCB[d] * recon;
        bytes[d] =  sizeof(Float) * offset[d];
        send_d[d] = device_malloc(bytes[d]);
        recv_d[d] = device_malloc(bytes[d]);
        sendg_d[d] = device_malloc(bytes[d]);
        recvg_d[d] = device_malloc(bytes[d]);
        hostbuffer_h[d] = (void*)pinned_malloc(4 * bytes[d]);
        tp[d].block = dim3(128, 1, 1);
        tp[d].grid = dim3((faceVolumeCB[d] + tp[d].block.x - 1) / tp[d].block.x, 1, 1);
      }
      for ( int d = 0; d < 4; d++ ) {
        if ( !commDimPartitioned(d)) continue;
        recv[d] = hostbuffer_h[d];
        send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
        recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
        sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];
        mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
        mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
        mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
        mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
      }
    }
    GaugeFixUnPackArg<recon,Gauge> dataexarg(dataOr, data);
    GaugeFixBorderPointsArg<Float, Gauge> argBorder(dataOr, data, relax_boost, faceVolume, faceVolumeCB);
    GaugeFixBorderPoints<Float,Gauge, gauge_dir> gfixBorderPoints(argBorder, data);
    GaugeFixInteriorPointsArg<Float, Gauge> argInt(dataOr, data, relax_boost);
    GaugeFixInteriorPoints<Float,Gauge, gauge_dir> gfixIntPoints(argInt, data);

    GaugeFixQuality.apply(device::get_default_stream());
    flop += (double)GaugeFixQuality.flops();
    byte += (double)GaugeFixQuality.bytes();
    double action0 = argQ.getAction();
    printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    unitarizeLinks(data, data, num_failures_dev);
    qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), qudaMemcpyDeviceToHost);
    if ( num_failures > 0 ) {
      pool_device_free(num_failures_dev);
      errorQuda("Error in the unitarization\n");
      exit(1);
    }
    qudaMemset(num_failures_dev, 0, sizeof(int));

    int iter = 0;
    for ( iter = 0; iter < Nsteps; iter++ ) {
      for ( int p = 0; p < 2; p++ ) {
        if ( !comm_partitioned() ) {
          gaugeFix.setParity(p);
          gaugeFix.apply(device::get_default_stream());
          flop += (double)gaugeFix.flops();
          byte += (double)gaugeFix.bytes();
        } else {
          gfixIntPoints.setParity(p);
          gfixBorderPoints.setParity(p); //compute border points
          gfixBorderPoints.apply(device::get_default_stream());
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
            qudaLaunchKernel(Kernel_UnPackTop<Float, true, decltype(dataexarg)>, tp[d], device::get_stream(d),
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(send_d[d]), p, d, d);
            //extract bottom ghost
            qudaLaunchKernel(Kernel_UnPackGhost<Float, true, decltype(dataexarg)>, tp[d], device::get_stream(4 + d),
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(sendg_d[d]), 1 - p, d, d);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaMemcpyAsync(send[d], send_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(d));
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(4 + d));
          }
          //compute interior points
          gfixIntPoints.apply(device::get_stream(8));

          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaStreamSynchronize(device::get_stream(d));
            comm_start(mh_send_fwd[d]);
            qudaStreamSynchronize(device::get_stream(4 + d));
            comm_start(mh_send_back[d]);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_recv_back[d]);
            qudaMemcpyAsync(recv_d[d], recv[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(d));
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_recv_fwd[d]);
            qudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(4 + d));
          }

          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaLaunchKernel(Kernel_UnPackGhost<Float, false, decltype(dataexarg)>, tp[d], device::get_stream(d),
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recv_d[d]), p, d, d);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            qudaLaunchKernel(Kernel_UnPackTop<Float, false, decltype(dataexarg)>, tp[d], device::get_stream(4 + d),
                             faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recvg_d[d]), 1 - p, d, d);
          }
          for ( int d = 0; d < 4; d++ ) {
            if ( !commDimPartitioned(d)) continue;
            comm_wait(mh_send_back[d]);
            comm_wait(mh_send_fwd[d]);
            qudaStreamSynchronize(device::get_stream(d));
            qudaStreamSynchronize(device::get_stream(4 + d));
          }
          qudaStreamSynchronize(device::get_stream(8));
        }
      }
      if ((iter % reunit_interval) == (reunit_interval - 1)) {
        unitarizeLinks(data, data, num_failures_dev);
        qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), qudaMemcpyDeviceToHost);
        if ( num_failures > 0 ) errorQuda("Error in the unitarization\n");
        qudaMemset(num_failures_dev, 0, sizeof(int));
        flop += 4588.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3];
        byte += 8.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3] * dataOr.Bytes();
      }
      GaugeFixQuality.apply(device::get_default_stream());
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
      qudaMemcpy(&num_failures, num_failures_dev, sizeof(int), qudaMemcpyDeviceToHost);
      if ( num_failures > 0 ) errorQuda("Error in the unitarization\n");
      qudaMemset(num_failures_dev, 0, sizeof(int));
      flop += 4588.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3];
      byte += 8.0 * data.X()[0]*data.X()[1]*data.X()[2]*data.X()[3] * dataOr.Bytes();
    }
    if ((iter % verbose_interval) != 0 ) {
      GaugeFixQuality.apply(device::get_default_stream());
      flop += (double)GaugeFixQuality.flops();
      byte += (double)GaugeFixQuality.bytes();
      double action = argQ.getAction();
      double diff = abs(action0 - action);
      printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
    }
    pool_device_free(num_failures_dev);

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
          host_free(hostbuffer_h[d]);
        }
      }
    }

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
      if (gauge_dir == 4) {
        printfQuda("Starting Landau gauge fixing...\n");
        gaugeFixingOVR<Float, recon, 4>(data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
      } else if (gauge_dir == 3) {
        printfQuda("Starting Coulomb gauge fixing...\n");
        gaugeFixingOVR<Float, recon, 3>(data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
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
#ifdef GPU_GAUGE_ALG
  void gaugeFixingOVR(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval, const double relax_boost,
                      const double tolerance, const int reunit_interval, const int stopWtheta)
  {
    instantiate<GaugeFixingOVR>(data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
  }
#else
  void gaugeFixingOVR(GaugeField&, const int, const int, const int, const double, const double, const int, const int)
  {
    errorQuda("Gauge fixing has not been built");
  }
#endif

}   //namespace quda
