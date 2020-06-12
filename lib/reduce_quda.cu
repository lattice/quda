#include <atomic>
#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/reduce_core.cuh>

// These are used for reduction kernels
static device_reduce_t *d_reduce = nullptr;
static device_reduce_t *h_reduce = nullptr;
static device_reduce_t *hd_reduce = nullptr;
static cudaEvent_t reduceEnd;
static bool fast_reduce_enabled = false;

namespace quda {

  namespace blas {

#include <generic_reduce.cuh>

    qudaStream_t* getStream();

    void* getDeviceReduceBuffer() { return d_reduce; }
    void* getMappedHostReduceBuffer() { return hd_reduce; }
    void* getHostReduceBuffer() { return h_reduce; }
    cudaEvent_t* getReduceEvent() { return &reduceEnd; }
    bool getFastReduce() { return fast_reduce_enabled; }

    void initFastReduce(int32_t words)
    {
      // initialize the reduction values in 32-bit increments to INT_MIN
      for (int32_t i = 0; i < words; i++) {
        reinterpret_cast<int32_t *>(h_reduce)[i] = std::numeric_limits<int32_t>::min();
      }

      // ensure that the host memory write is complete before we launch the kernel
      atomic_thread_fence(std::memory_order_release);
    }

    void completeFastReduce(int32_t words)
    {
      volatile int32_t *check = reinterpret_cast<int32_t *>(h_reduce);
      int count = 0;
      int complete = 0;
      while (complete < words) {
        // ensure visiblity to any changes in memory
        atomic_thread_fence(std::memory_order_acquire);

        complete = 0;
        for (int32_t i = 0; i < words; i++) {
          // spin-wait until all values have been updated
          if (check[i] != std::numeric_limits<int32_t>::min()) complete++;
        }
        if (count++ % 10000 == 0) { // check error every 10000 iterations
          // if there is an error in the kernel then we need to exit the spin-wait
          if (cudaSuccess != cudaPeekAtLastError()) break;
        }
      }
    }

    void initReduce()
    {
      /* we have these different reductions to cater for:

         - regular reductions (reduce_quda.cu) where are reducing to a
           single vector type (max length 4 presently), with possibly
           parity dimension, and a grid-stride loop with max number of
           blocks = 2 x SM count

         - multi-reductions where we are reducing to a matrix of size
           of size QUDA_MAX_MULTI_REDUCE of vectors (max length 4), with
           possible parity dimension, and a grid-stride loop with
           maximum number of blocks = 2 x SM count
      */

      const int reduce_size = 4 * sizeof(device_reduce_t);
      const int max_reduce_blocks = 2*deviceProp.multiProcessorCount;

      const int max_reduce = 2 * max_reduce_blocks * reduce_size;
      const int max_multi_reduce = 2 * QUDA_MAX_MULTI_REDUCE * max_reduce_blocks * reduce_size;

      // reduction buffer size
      size_t bytes = max_reduce > max_multi_reduce ? max_reduce : max_multi_reduce;

      if (!d_reduce) d_reduce = (device_reduce_t *) device_malloc(bytes);

      // these arrays are actually oversized currently (only needs to be device_reduce_t x 3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
	// only use zero copy reductions when using 64-bit
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
	if(deviceProp.canMapHostMemory) {
	  h_reduce = (device_reduce_t *) mapped_malloc(bytes);
	  cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0); // set the matching device pointer
	} else
#endif
	  {
	    h_reduce = (device_reduce_t *) pinned_malloc(bytes);
	    hd_reduce = d_reduce;
	  }
	memset(h_reduce, 0, bytes); // added to ensure that valgrind doesn't report h_reduce is unitialised
      }

      cudaEventCreateWithFlags(&reduceEnd, cudaEventDisableTiming);

      // enable fast reductions with CPU spin waiting as opposed to using CUDA events
      char *fast_reduce_env = getenv("QUDA_ENABLE_FAST_REDUCE");
      if (fast_reduce_env && strcmp(fast_reduce_env,"1") == 0) {
        warningQuda("Experimental fast reductions enabled");
        fast_reduce_enabled = true;
      }

      checkCudaError();
    }

    void endReduce(void)
    {
      if (d_reduce) {
	device_free(d_reduce);
	d_reduce = 0;
      }
      if (h_reduce) {
	host_free(h_reduce);
	h_reduce = 0;
      }
      hd_reduce = 0;

      cudaEventDestroy(reduceEnd);
    }

    /**
       Generic reduction kernel launcher
    */
    template <typename host_reduce_t, typename real, int len, typename Arg>
    auto reduceLaunch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream, Tunable &tunable)
    {
      using device_reduce_t = typename Arg::Reducer::reduce_t;
      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

      const int32_t words = tp.grid.y * sizeof(device_reduce_t) / sizeof(int32_t);
      if (getFastReduce() && !commAsyncReduction()) initFastReduce(words);

#ifdef JITIFY
      using namespace jitify::reflection;
      tunable.jitifyError() = program->kernel("quda::blas::reduceKernel")
                                  .instantiate((int)tp.block.x, Type<real>(), len, Type<Arg>())
                                  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                                  .launch(arg);
#else
      LAUNCH_KERNEL(reduceKernel, tunable, tp, stream, arg, real, len);
#endif

      if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
        if (deviceProp.canMapHostMemory) {
          if (getFastReduce()) {
            completeFastReduce(words);
          } else {
            qudaEventRecord(reduceEnd, stream);
            while (cudaSuccess != qudaEventQuery(reduceEnd)) { ; }
          }
        } else
#endif
        {
          qudaMemcpy(h_reduce, hd_reduce, sizeof(device_reduce_t), cudaMemcpyDeviceToHost);
        }
      }

      host_reduce_t cpu_sum = set(((device_reduce_t *)h_reduce)[0]);
      if (tp.grid.y == 2) sum(cpu_sum, ((device_reduce_t *)h_reduce)[1]); // add other parity if needed
      return cpu_sum;
    }

    template <typename host_reduce_t, typename real, int len, typename SpinorX, typename SpinorY,
              typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
    class Reduce : public Tunable
    {
      const int nParity; // for composite fields this includes the number of composites
      mutable ReductionArg<SpinorX, SpinorY, SpinorZ, SpinorW, SpinorV, Reducer> arg;
      host_reduce_t &result;

      const ColorSpinorField &x, &y, &z, &w, &v;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      virtual bool advanceSharedBytes(TuneParam &param) const
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(param) ?
            sharedBytesPerThread() * nthreads :
            sharedBytesPerBlock(param);
        return false;
      }

  public:
      Reduce(host_reduce_t &result, SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Reducer &r,
             ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v,
          int length) :
          nParity((x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset())),
          arg(X, Y, Z, W, V, r, length / nParity),
          x(x),
          y(y),
          z(z),
          w(w),
          v(v),
          result(result)
      {
        strcpy(aux, x.AuxString());
        if (x.Precision() != z.Precision()) {
          strcat(aux, ",");
          strcat(aux, z.AuxString());
        }
        if (getFastReduce()) strcat(aux, ",fast_reduce");

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/reduce_core.cuh");
#endif
      }
      virtual ~Reduce() {}

      inline TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(arg.r).name(), aux); }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        result = reduceLaunch<host_reduce_t, real, len>(arg, tp, stream, *this);
      }

      void preTune()
      {
        if (arg.r.write.X) x.backup();
        if (arg.r.write.Y) y.backup();
        if (arg.r.write.Z) z.backup();
        if (arg.r.write.W) w.backup();
        if (arg.r.write.V) v.backup();
      }

      void postTune()
      {
        if (arg.r.write.X) x.restore();
        if (arg.r.write.Y) y.restore();
        if (arg.r.write.Z) z.restore();
        if (arg.r.write.W) w.restore();
        if (arg.r.write.V) v.restore();
      }

      void initTuneParam(TuneParam &param) const
      {
        Tunable::initTuneParam(param);
        param.grid.y = nParity;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        Tunable::defaultTuneParam(param);
        param.grid.y = nParity;
      }

      long long flops() const { return arg.r.flops() * x.Length(); }

      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
        return (arg.r.streams() - 2) * x.Bytes() + 2 * z.Bytes();
      }

      int tuningIter() const { return 3; }
    };

    template <template <typename ReducerType, typename real> class Reducer, typename real,
              typename store_t, int len, int N, typename z_store_t = store_t, int Nz = N, typename coeff_t>
    auto nativeReduce(const coeff_t &a, const coeff_t &b, ColorSpinorField &x, ColorSpinorField &y,
                      ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, int length)
    {
      checkLength(x, y);
      checkLength(x, z);
      checkLength(x, w);
      checkLength(x, v);

      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      Reducer<device_reduce_t, real> r(a, b);

      Spinor<store_t, N> X(x);
      Spinor<store_t, N> Y(y);
      Spinor<z_store_t, Nz> Z(z);
      Spinor<store_t, N> W(w);
      Spinor<store_t, N> V(v);

      host_reduce_t value;
      Reduce<host_reduce_t, real, len, decltype(X), decltype(Y), decltype(Z), decltype(W), decltype(V), decltype(r)>
        reduce(value, X, Y, Z, W, V, r, x, y, z, w, v, length);
      reduce.apply(*(blas::getStream()));

      blas::bytes += reduce.bytes();
      blas::flops += reduce.flops();
      checkCudaError();
      return value;
    }

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
       Driver for generic reduction routine with five loads.
       @param ReduceType
    */
    template <template <typename reduce_t, typename real> class Reducer, typename coeff_t>
    auto uni_reduce(const coeff_t &a, const coeff_t &b, ColorSpinorField &x, ColorSpinorField &y,
                    ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      checkPrecision(x, y, z, w, v);

      constexpr bool siteUnroll = Reducer<double, double>::site_unroll;
      using host_reduce_t = typename Reducer<double, double>::reduce_t;
      host_reduce_t value;

      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative() && x.FieldOrder() != QUDA_FLOAT8_FIELD_ORDER) {
          warningQuda("Device reductions on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          host_reduce_t value;
          ::quda::zero(value);
          return value;
        }

        // cannot do site unrolling for arbitrary color (needs JIT)
        if (siteUnroll && x.Ncolor() != 3) errorQuda("Not supported");

        int reduce_length = siteUnroll ? x.RealLength() : x.Length();

        if (x.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
          if (x.Nspin() == 4 || x.Nspin() == 2) { // wilson
#if defined(NSPIN4) || defined(NSPIN2)
            const int M = siteUnroll ? 24 : 2; // determines how much work per thread to do
            if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
            value = nativeReduce<Reducer, double, double, M, 2>(a, b, x, y, z, w, v, reduce_length / M);
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = siteUnroll ? 6 : 2; // determines how much work per thread to do
            value = nativeReduce<Reducer, double, double, M, 2>(a, b, x, y, z, w, v, reduce_length / M);
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) { // wilson
#if defined(NSPIN4)
            const int M = siteUnroll ? 24 : 4; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, float, M, 4>(a, b, x, y, z, w, v, reduce_length / M);
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1 || x.Nspin() == 2 || (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER)) {
#if defined(NSPIN1) || defined(NSPIN2) || defined(GPU_MULTIGRID)
            const int M = siteUnroll ? 6 : 2; // determines how much work per thread to do
            if ((x.Nspin() == 2 || x.Nspin() == 4) && siteUnroll) errorQuda("siteUnroll not supported here for nSpin=%d", x.Nspin());
            value = nativeReduce<Reducer, float, float, M, 2>(a, b, x, y, z, w, v, reduce_length / M);
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_HALF_PRECISION) { // half precision

#if QUDA_PRECISION & 2
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) { // wilson
#if defined(NSPIN4)
            const int M = 24; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, short, M, 4>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 24; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, short, M, 8>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 6; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, short, M, 2>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else if (x.Precision() == QUDA_QUARTER_PRECISION) { // quarter precision

#if QUDA_PRECISION & 1
          if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) { // wilson
#if defined(NSPIN4)
            const int M = 24; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, char, M, 4>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 24;
            value = nativeReduce<Reducer, float, char, M, 8>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#ifdef NSPIN1
            const int M = 3; // determines how much work per thread to do
            value = nativeReduce<Reducer, float, char, M, 2>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x.Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else {
          errorQuda("precision=%d is not supported\n", x.Precision());
        }
      } else { // fields are on the CPU
        if (x.Precision() == QUDA_DOUBLE_PRECISION) {
          Reducer<double, double> r(a, b);
          value = genericReduce<host_reduce_t, double, double, decltype(r)>(x, y, z, w, v, r);
        } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
          Reducer<double, float> r(a, b);
          value = genericReduce<host_reduce_t, float, float, decltype(r)>(x, y, z, w, v, r);
        } else {
          errorQuda("Precision %d not implemented", x.Precision());
        }
      }

      const int Nreduce = sizeof(host_reduce_t) / sizeof(double);
      reduceDoubleArray((double *)&value, Nreduce);

      return value;
    }

    template <template <typename ReducerType, typename real> class Reducer, typename coeff_t>
    auto mixed_reduce(const coeff_t &a, const coeff_t &b, ColorSpinorField &x, ColorSpinorField &y,
                      ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      checkPrecision(x, y, w, v);

      using host_reduce_t = typename Reducer<double, double>::reduce_t;
      host_reduce_t value;

      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative()) {
          warningQuda("Device reductions on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          host_reduce_t value;
          ::quda::zero(value);
          return value;
        }

        // cannot do site unrolling for arbitrary color (needs JIT)
        if (x.Ncolor() != 3) errorQuda("Not supported");

        if (z.Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
          if (x.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 24; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, float, M, 4, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 6; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, float, M, 2, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

          } else if (x.Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 24; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, short, M, 4, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 6; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, short, M, 2, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

          } else if (x.Precision() == QUDA_QUARTER_PRECISION) {

#if QUDA_PRECISION & 1
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 24; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, char, M, 4, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 6; // determines how much work per thread to do
              value = nativeReduce<Reducer, double, char, M, 2, double, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

          } else {
            errorQuda("Not implemented for this precision combination %d %d", x.Precision(), z.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, z.Precision());
#endif

        } else if (z.Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x.Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 24;
              value = nativeReduce<Reducer, float, short, M, 4, float, 4>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 6;
              value = nativeReduce<Reducer, float, short, M, 2, float, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

          } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 24;
              value = nativeReduce<Reducer, float, char, M, 4, float, 4>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 6;
              value = nativeReduce<Reducer, float, char, M, 2, float, 2>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif
          } else {
            errorQuda("Not implemented for this precision combination %d %d", x.Precision(), z.Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

        } else {
          errorQuda("Not implemented for this precision combination %d %d", x.Precision(), z.Precision());
        }

      } else {
        if (x.Precision() == QUDA_SINGLE_PRECISION && z.Precision() == QUDA_DOUBLE_PRECISION) {
          Reducer<double, double> r(a, b);
          value = genericReduce<host_reduce_t, float, double, decltype(r)>(x, y, z, w, v, r);
        } else {
          errorQuda("Precision %d not implemented", x.Precision());
        }
      }

      const int Nreduce = sizeof(host_reduce_t) / sizeof(double);
      reduceDoubleArray((double *)&value, Nreduce);

      return value;
    }

    double norm1(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x); // FIXME
      return uni_reduce<Norm1>(0.0, 0.0, y, y, y, y, y);
    }

    double norm2(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x);
      return uni_reduce<Norm2>(0.0, 0.0, y, y, y, y, y);
    }

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<Dot>(0.0, 0.0, x, y, x, x, x);
    }

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<axpbyzNorm2>(a, b, x, y, z, x, x);
    }

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<AxpyReDot>(a, 0.0, x, y, x, x, x);
    }

    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<caxpyNorm2>(a, Complex(0.0), x, y, x, x, x);
    }

    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<caxpyxmaznormx>(a, Complex(0.0), x, y, z, x, x);
    }

    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<cabxpyzaxnorm>(Complex(a), b, x, y, z, x, x);
    }

    Complex cDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      auto cdot = uni_reduce<Cdot>(0.0, 0.0, x, y, x, x, x);
      return Complex(cdot.x, cdot.y);
    }

    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      double2 cdot = uni_reduce<caxpydotzy>(a, Complex(0.0), x, y, z, x, x);
      return Complex(cdot.x, cdot.y);
    }

    double3 cDotProductNormA(ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<CdotNormA>(0.0, 0.0, x, y, x, x, x);
    }

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y,
                                           ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &u)
    {
      if (x.Precision() != z.Precision()) {
        return mixed_reduce<caxpbypzYmbwcDotProductUYNormY_>(a, b, x, y, z, w, u);
      } else {
        return uni_reduce<caxpbypzYmbwcDotProductUYNormY_>(a, b, x, y, z, w, u);
      }
    }

    Complex axpyCGNorm(double a, ColorSpinorField &x, ColorSpinorField &y)
    {
      // swizzle since mixed is on z
      double2 cg_norm ;
      if (x.Precision() != y.Precision()) {
        cg_norm = mixed_reduce<axpyCGNorm2>(a, 0.0, x, x, y, x, x);
      } else {
        cg_norm = uni_reduce<axpyCGNorm2>(a, 0.0, x, x, y, x, x);
      }
      return Complex(cg_norm.x, cg_norm.y);
    }

    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = uni_reduce<HeavyQuarkResidualNorm_>(0.0, 0.0, x, r, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = uni_reduce<xpyHeavyQuarkResidualNorm_>(0.0, 0.0, x, y, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<tripleCGReduction_>(0.0, 0.0, x, y, z, x, x);
    }

    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<quadrupleCGReduction_>(0.0, 0.0, x, y, z, x, x);
    }

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v) {
      return uni_reduce<quadrupleCG3InitNorm_>(a, 0.0, x, y, z, w, v);
    }

    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y,
                                  ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      return uni_reduce<quadrupleCG3UpdateNorm_>(a, b, x, y, z, w, v);
    }

    double doubleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<doubleCG3InitNorm_>(a, 0.0, x, y, z, z, z);
    }

    double doubleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<doubleCG3UpdateNorm_>(a, b, x, y, z, z, z);
    }

  } // namespace blas

} // namespace quda
