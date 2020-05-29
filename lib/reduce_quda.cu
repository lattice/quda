#include <atomic>
#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/reduce_core.cuh>

// These are used for reduction kernels
static QudaSumFloat *d_reduce=0;
static QudaSumFloat *h_reduce=0;
static QudaSumFloat *hd_reduce=0;
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

      const int reduce_size = 4 * sizeof(QudaSumFloat);
      const int max_reduce_blocks = 2*deviceProp.multiProcessorCount;

      const int max_reduce = 2 * max_reduce_blocks * reduce_size;
      const int max_multi_reduce = 2 * QUDA_MAX_MULTI_REDUCE * max_reduce_blocks * reduce_size;

      // reduction buffer size
      size_t bytes = max_reduce > max_multi_reduce ? max_reduce : max_multi_reduce;

      if (!d_reduce) d_reduce = (QudaSumFloat *) device_malloc(bytes);

      // these arrays are actually oversized currently (only needs to be QudaSumFloat3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
	// only use zero copy reductions when using 64-bit
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
	if(deviceProp.canMapHostMemory) {
	  h_reduce = (QudaSumFloat *) mapped_malloc(bytes);
	  cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0); // set the matching device pointer
	} else
#endif
	  {
	    h_reduce = (QudaSumFloat *) pinned_malloc(bytes);
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
    template <typename doubleN, typename ReduceType, typename FloatN, int M, typename Arg>
    doubleN reduceLaunch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream, Tunable &tunable)
    {
      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

      const int32_t words = tp.grid.y * sizeof(ReduceType) / sizeof(int32_t);
      if (getFastReduce() && !commAsyncReduction()) initFastReduce(words);

#ifdef JITIFY
      using namespace jitify::reflection;
      tunable.jitifyError() = program->kernel("quda::blas::reduceKernel")
                                  .instantiate((int)tp.block.x, Type<ReduceType>(), Type<FloatN>(), M, Type<Arg>())
                                  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                                  .launch(arg);
#else
      LAUNCH_KERNEL(reduceKernel, tunable, tp, stream, arg, ReduceType, FloatN, M);
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
          qudaMemcpy(h_reduce, hd_reduce, sizeof(ReduceType), cudaMemcpyDeviceToHost);
        }
      }
      doubleN cpu_sum = set(((ReduceType *)h_reduce)[0]);
      if (tp.grid.y == 2) sum(cpu_sum, ((ReduceType *)h_reduce)[1]); // add other parity if needed
      return cpu_sum;
    }

    template <typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX, typename SpinorY,
        typename SpinorZ, typename SpinorW, typename SpinorV, typename Reducer>
    class ReduceCuda : public Tunable
    {

  private:
      const int nParity; // for composite fields this includes the number of composites
      mutable ReductionArg<ReduceType, SpinorX, SpinorY, SpinorZ, SpinorW, SpinorV, Reducer> arg;
      doubleN &result;

      const ColorSpinorField &x, &y, &z, &w, &v;

      // host pointers used for backing up fields when tuning
      // these can't be curried into the Spinors because of Tesla argument length restriction
      char *X_h, *Y_h, *Z_h, *W_h, *V_h;
      char *Xnorm_h, *Ynorm_h, *Znorm_h, *Wnorm_h, *Vnorm_h;

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
      ReduceCuda(doubleN &result, SpinorX &X, SpinorY &Y, SpinorZ &Z, SpinorW &W, SpinorV &V, Reducer &r,
          ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v,
          int length) :
          nParity((x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset())),
          arg(X, Y, Z, W, V, r, length / nParity),
          x(x),
          y(y),
          z(z),
          w(w),
          v(v),
          result(result),
          X_h(0),
          Y_h(0),
          Z_h(0),
          W_h(0),
          V_h(0),
          Xnorm_h(0),
          Ynorm_h(0),
          Znorm_h(0),
          Wnorm_h(0),
          Vnorm_h(0)
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
      virtual ~ReduceCuda() {}

      inline TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(arg.r).name(), aux); }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        result = reduceLaunch<doubleN, ReduceType, FloatN, M>(arg, tp, stream, *this);
      }

      void preTune()
      {
        arg.X.backup(&X_h, &Xnorm_h, x.Bytes(), x.NormBytes());
        arg.Y.backup(&Y_h, &Ynorm_h, y.Bytes(), y.NormBytes());
        arg.Z.backup(&Z_h, &Znorm_h, z.Bytes(), z.NormBytes());
        arg.W.backup(&W_h, &Wnorm_h, w.Bytes(), w.NormBytes());
        arg.V.backup(&V_h, &Vnorm_h, v.Bytes(), v.NormBytes());
      }

      void postTune()
      {
        arg.X.restore(&X_h, &Xnorm_h, x.Bytes(), x.NormBytes());
        arg.Y.restore(&Y_h, &Ynorm_h, y.Bytes(), y.NormBytes());
        arg.Z.restore(&Z_h, &Znorm_h, z.Bytes(), z.NormBytes());
        arg.W.restore(&W_h, &Wnorm_h, w.Bytes(), w.NormBytes());
        arg.V.restore(&V_h, &Vnorm_h, v.Bytes(), v.NormBytes());
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

      long long flops() const { return arg.r.flops() * vec_length<FloatN>::value * arg.length * nParity * M; }

      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
        return (arg.r.streams() - 2) * x.Bytes() + 2 * z.Bytes();
      }

      int tuningIter() const { return 3; }
    };

    template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename zType, int M,
        template <typename ReducerType, typename Float, typename FloatN> class Reducer, int writeX, int writeY,
        int writeZ, int writeW, int writeV>
    doubleN nativeReduce(const double2 &a, const double2 &b, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, int length)
    {

      checkLength(x, y);
      checkLength(x, z);
      checkLength(x, w);
      checkLength(x, v);

      Spinor<RegType, StoreType, M, writeX> X(x);
      Spinor<RegType, StoreType, M, writeY> Y(y);
      Spinor<RegType, zType, M, writeZ> Z(z);
      Spinor<RegType, StoreType, M, writeW> W(w);
      Spinor<RegType, StoreType, M, writeV> V(v);

      doubleN value;
      typedef typename scalar<RegType>::type Float;
      typedef typename vector<Float, 2>::type Float2;
      typedef vector<Float, 2> vec2;

      Reducer<ReduceType, Float2, RegType> r((Float2)vec2(a), (Float2)vec2(b));
      ReduceCuda<doubleN, ReduceType, RegType, M, decltype(X), decltype(Y), decltype(Z), decltype(W), decltype(V),
          Reducer<ReduceType, Float2, RegType>>
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
       @param siteUnroll - if this is true, then one site corresponds to exactly one thread
    */
    template <typename doubleN, typename ReduceType, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
        int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
    doubleN uni_reduce(const double2 &a, const double2 &b, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {

      checkPrecision(x, y, z, w, v);

      doubleN value;
      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative() && x.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER && x.FieldOrder() != QUDA_FLOAT8_FIELD_ORDER) {
          warningQuda("Device reductions on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          doubleN value;
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
            const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
            if (x.Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
            value = nativeReduce<doubleN, ReduceType, double2, double2, double2, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, reduce_length / (2 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
            value = nativeReduce<doubleN, ReduceType, double2, double2, double2, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, reduce_length / (2 * M));
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
            const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
            value = nativeReduce<doubleN, ReduceType, float4, float4, float4, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, reduce_length / (4 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1 || x.Nspin() == 2 || (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER)) {
#if defined(NSPIN1) || defined(NSPIN2) || defined(GPU_MULTIGRID)
            const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
            if ((x.Nspin() == 2 || x.Nspin() == 4) && siteUnroll) errorQuda("siteUnroll not supported here for nSpin=%d", x.Nspin());
            value = nativeReduce<doubleN, ReduceType, float2, float2, float2, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, reduce_length / (2 * M));
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
            const int M = 6; // determines how much work per thread to do
            value = nativeReduce<doubleN, ReduceType, float4, short4, short4, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) { // wilson
#if defined(GPU_MULTIGRID)  // FIXME eventually we should get rid of this and use float4 ordering
            const int M = 12; // determines how much work per thread to do
            value
                = nativeReduce<doubleN, ReduceType, float2, short2, short2, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                    a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 3; // determines how much work per thread to do
            value
                = nativeReduce<doubleN, ReduceType, float8, short8, short8, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                    a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 3; // determines how much work per thread to do
            value = nativeReduce<doubleN, ReduceType, float2, short2, short2, M, Reducer, writeX, writeY, writeZ,
                writeW, writeV>(a, b, x, y, z, w, v, y.Volume());
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
            const int M = 6; // determines how much work per thread to do
            value
                = nativeReduce<doubleN, ReduceType, float4, char4, char4, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                    a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) { // wilson
#if defined(GPU_MULTIGRID)  // FIXME eventually we should get rid of this and use float4 ordering
            const int M = 12; // determines how much work per thread to do
            value
              = nativeReduce<doubleN, ReduceType, float2, char2, char2, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT8_FIELD_ORDER) { // wilson
#if defined(NSPIN4) && defined(FLOAT8)
            const int M = 3;
            value
              = nativeReduce<doubleN, ReduceType, float8, char8, char8, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                a, b, x, y, z, w, v, y.Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
          } else if (x.Nspin() == 1) { // staggered
#ifdef NSPIN1
            const int M = 3; // determines how much work per thread to do
            value
                = nativeReduce<doubleN, ReduceType, float2, char2, char2, M, Reducer, writeX, writeY, writeZ, writeW, writeV>(
                    a, b, x, y, z, w, v, y.Volume());
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
        // we don't have quad precision support on the GPU so use doubleN instead of ReduceType
        if (x.Precision() == QUDA_DOUBLE_PRECISION) {
          Reducer<doubleN, double2, double2> r(a, b);
          value = genericReduce<doubleN, doubleN, double, double, writeX, writeY, writeZ, writeW, writeV,
              Reducer<doubleN, double2, double2>>(x, y, z, w, v, r);
        } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
          Reducer<doubleN, float2, float2> r(make_float2(a.x, a.y), make_float2(b.x, b.y));
          value = genericReduce<doubleN, doubleN, float, float, writeX, writeY, writeZ, writeW, writeV,
              Reducer<doubleN, float2, float2>>(x, y, z, w, v, r);
        } else {
          errorQuda("Precision %d not implemented", x.Precision());
        }
      }

      const int Nreduce = sizeof(doubleN) / sizeof(double);
      reduceDoubleArray((double *)&value, Nreduce);

      return value;
    }

    /**
       Driver for generic reduction routine with two loads.
       @param ReduceType
       @param siteUnroll - if this is true, then one site corresponds to exactly one thread
    */
    template <typename doubleN, typename ReduceType, template <typename ReducerType, typename Float, typename FloatN> class Reducer,
        int writeX, int writeY, int writeZ, int writeW, int writeV, bool siteUnroll>
    doubleN mixed_reduce(const double2 &a, const double2 &b, ColorSpinorField &x, ColorSpinorField &y,
        ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      checkPrecision(x, y, w, v);

      doubleN value;
      if (checkLocation(x, y, z, w, v) == QUDA_CUDA_FIELD_LOCATION) {

        if (!x.isNative() && !(x.Nspin() == 4 && x.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && x.Precision() == QUDA_SINGLE_PRECISION)) {
          warningQuda("Device reductions on non-native fields is not supported (prec = %d, order = %d)", x.Precision(),
                      x.FieldOrder());
          doubleN value;
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
              const int M = 12; // determines how much work per thread to do
              value = nativeReduce<doubleN, ReduceType, double2, float4, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
              const int reduce_length = siteUnroll ? x.RealLength() : x.Length();
              value = nativeReduce<doubleN, ReduceType, double2, float2, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, reduce_length / (2 * M));
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
              const int M = 12; // determines how much work per thread to do
              value = nativeReduce<doubleN, ReduceType, double2, short4, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 3; // determines how much work per thread to do
              value = nativeReduce<doubleN, ReduceType, double2, short2, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
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
              const int M = 12; // determines how much work per thread to do
              value = nativeReduce<doubleN, ReduceType, double2, char4, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 3; // determines how much work per thread to do
              value = nativeReduce<doubleN, ReduceType, double2, char2, double2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
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
              const int M = 6;
              value = nativeReduce<doubleN, ReduceType, float4, short4, float4, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 3;
              value = nativeReduce<doubleN, ReduceType, float2, short2, float2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
            blas::bytes
                += Reducer<ReduceType, double2, double2>::streams() * (unsigned long long)x.Volume() * sizeof(float);
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x.Precision());
#endif

          } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
            if (x.Nspin() == 4) { // wilson
#if defined(NSPIN4)
              const int M = 6;
              value = nativeReduce<doubleN, ReduceType, float4, char4, float4, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else if (x.Nspin() == 1) { // staggered
#if defined(NSPIN1)
              const int M = 3;
              value = nativeReduce<doubleN, ReduceType, float2, char2, float2, M, Reducer, writeX, writeY, writeZ,
                  writeW, writeV>(a, b, x, y, z, w, v, x.Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x.Nspin(), x.FieldOrder());
#endif
            } else {
              errorQuda("ERROR: nSpin=%d is not supported\n", x.Nspin());
            }
            blas::bytes
                += Reducer<ReduceType, double2, double2>::streams() * (unsigned long long)x.Volume() * sizeof(float);
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
        // we don't have quad precision support on the GPU so use doubleN instead of ReduceType
        if (x.Precision() == QUDA_SINGLE_PRECISION && z.Precision() == QUDA_DOUBLE_PRECISION) {
          Reducer<doubleN, double2, double2> r(a, b);
          value = genericReduce<doubleN, doubleN, float, double, writeX, writeY, writeZ, writeW, writeV,
              Reducer<doubleN, double2, double2>>(x, y, z, w, v, r);
        } else {
          errorQuda("Precision %d not implemented", x.Precision());
        }
      }

      const int Nreduce = sizeof(doubleN) / sizeof(double);
      reduceDoubleArray((double *)&value, Nreduce);

      return value;
    }

    double norm1(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x); // FIXME
      return uni_reduce<double, QudaSumFloat, Norm1, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), y, y, y, y, y);
    }

    double norm2(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x);
      return uni_reduce<double, QudaSumFloat, Norm2, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), y, y, y, y, y);
    }

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<double, QudaSumFloat, Dot, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<double, QudaSumFloat, axpbyzNorm2, 0, 0, 1, 0, 0, false>(
          make_double2(a, 0.0), make_double2(b, 0.0), x, y, z, x, x);
    }

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<double, QudaSumFloat, AxpyReDot, 0, 1, 0, 0, 0, false>(
          make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }

    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return uni_reduce<double, QudaSumFloat, caxpyNorm2, 0, 1, 0, 0, 0, false>(
          make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, x, x, x);
    }

    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<double, QudaSumFloat, caxpyxmaznormx, 1, 1, 0, 0, 0, false>(
          make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, z, x, x);
    }

    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return uni_reduce<double, QudaSumFloat, cabxpyzaxnorm, 1, 0, 1, 0, 0, false>(
          make_double2(a, 0.0), make_double2(REAL(b), IMAG(b)), x, y, z, x, x);
    }

    Complex cDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      double2 cdot = uni_reduce<double2, QudaSumFloat2, Cdot, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
      return Complex(cdot.x, cdot.y);
    }

    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      double2 cdot = uni_reduce<double2, QudaSumFloat2, caxpydotzy, 0, 1, 0, 0, 0, false>(
          make_double2(REAL(a), IMAG(a)), make_double2(0.0, 0.0), x, y, z, x, x);
      return Complex(cdot.x, cdot.y);
    }

    double3 cDotProductNormA(ColorSpinorField &x, ColorSpinorField &y) {
      return uni_reduce<double3, QudaSumFloat3, CdotNormA, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, x, x);
    }

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x,
					   const Complex &b, ColorSpinorField &y,
					   ColorSpinorField &z, ColorSpinorField &w,
					   ColorSpinorField &u) {
      if (x.Precision() != z.Precision()) {
        return mixed_reduce<double3, QudaSumFloat3, caxpbypzYmbwcDotProductUYNormY_, 0, 1, 1, 0, 0, false>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), x, y, z, w, u);
      } else {
        return uni_reduce<double3, QudaSumFloat3, caxpbypzYmbwcDotProductUYNormY_, 0, 1, 1, 0, 0, false>(
            make_double2(REAL(a), IMAG(a)), make_double2(REAL(b), IMAG(b)), x, y, z, w, u);
      }
    }

    Complex axpyCGNorm(double a, ColorSpinorField &x, ColorSpinorField &y) {
      // swizzle since mixed is on z
      double2 cg_norm ;
      if (x.Precision() != y.Precision()) {
        cg_norm = mixed_reduce<double2, QudaSumFloat2, axpyCGNorm2, 0, 0, 1, 0, 0, false>(
            make_double2(a, 0.0), make_double2(0.0, 0.0), x, x, y, x, x);
      } else {
        cg_norm = uni_reduce<double2, QudaSumFloat2, axpyCGNorm2, 0, 0, 1, 0, 0, false>(
            make_double2(a, 0.0), make_double2(0.0, 0.0), x, x, y, x, x);
      }
      return Complex(cg_norm.x, cg_norm.y);
    }

    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r) {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = uni_reduce<double3, QudaSumFloat3, HeavyQuarkResidualNorm_, 0, 0, 0, 0, 0, true>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, r, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y,
				      ColorSpinorField &r) {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = uni_reduce<double3, QudaSumFloat3, xpyHeavyQuarkResidualNorm_, 0, 0, 0, 0, 0, true>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return uni_reduce<double3, QudaSumFloat3, tripleCGReduction_, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, x);
    }

    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return uni_reduce<double4, QudaSumFloat4, quadrupleCGReduction_, 0, 0, 0, 0, 0, false>(
          make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, z, x, x);
    }

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v) {
      return uni_reduce<double, QudaSumFloat, quadrupleCG3InitNorm_, 1, 1, 1, 1, 0, false>(
          make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, z, w, v);
    }

    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v) {
      return uni_reduce<double, QudaSumFloat, quadrupleCG3UpdateNorm_, 1, 1, 1, 1, 0, false>(
          make_double2(a, 0.0), make_double2(b, 1. - b), x, y, z, w, v);
    }

    double doubleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return uni_reduce<double, QudaSumFloat, doubleCG3InitNorm_, 1, 1, 0, 0, 0, false>(
          make_double2(a, 0.0), make_double2(0.0, 0.0), x, y, z, z, z);
    }

    double doubleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      return uni_reduce<double, QudaSumFloat, doubleCG3UpdateNorm_, 1, 1, 0, 0, 0, false>(
          make_double2(a, 0.0), make_double2(b, 1.0 - b), x, y, z, z, z);
    }

   } // namespace blas

} // namespace quda
