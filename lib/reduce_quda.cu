#include <atomic>

#include <blas_quda.h>
#include <tune_quda.h>
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

    template <template <typename ReducerType, typename real> class Reducer,
              typename store_t, typename y_store_t, int nSpin, typename coeff_t>
    class Reduce : public Tunable
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      Reducer<device_reduce_t, real> r;
      const int nParity; // for composite fields this includes the number of composites
      host_reduce_t &result;

      const coeff_t &a, &b;
      ColorSpinorField &x, &y, &z, &w, &v;
      QudaFieldLocation location;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool advanceSharedBytes(TuneParam &param) const
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
      Reduce(const coeff_t &a, const coeff_t &b, const coeff_t &c, ColorSpinorField &x, ColorSpinorField &y,
             ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v, host_reduce_t &result) :
        r(a, b),
        nParity((x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset())),
        a(a),
        b(b),
        x(x),
        y(y),
        z(z),
        w(w),
        v(v),
        result(result),
        location(checkLocation(x, y, z, w, v))
      {
        checkLength(x, y, z, w, v);
        auto x_prec = checkPrecision(x, z, w);
        auto y_prec = checkPrecision(y, v);
        auto x_order = checkOrder(x, z, w);
        auto y_order = checkOrder(y, v);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        strcpy(aux, x.AuxString());
        if (x.Precision() != z.Precision()) {
          strcat(aux, ",");
          strcat(aux, z.AuxString());
        }
        if (location == QUDA_CPU_FIELD_LOCATION) strcat(aux, ",CPU");
        else if (getFastReduce()) strcat(aux, ",fast_reduce");

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/reduce_core.cuh");
#endif

        apply(*(blas::getStream()));
        checkCudaError();

        blas::bytes += bytes();
        blas::flops += flops();

        const int Nreduce = sizeof(host_reduce_t) / sizeof(double);
        reduceDoubleArray((double *)&result, Nreduce);
      }

      TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(r).name(), aux); }

      void apply(const qudaStream_t &stream)
      {
        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value || decltype(r)::site_unroll;
        if (site_unroll_check && (x.Ncolor() != 3 || x.Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x.Nspin(), x.Ncolor());

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(x, y, z, w, v); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Reducer<device_reduce_t, device_real_t> r_(a, b);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value || decltype(r)::site_unroll;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x.Length() / (nParity * M);

          ReductionArg<device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity);
          result = reduceLaunch<host_reduce_t, device_real_t, M>(arg, tp, stream, *this);
        } else {
          if (checkOrder(x, y, z, w, v) != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
            warningQuda("CPU Blas functions expect AoS field order");
            return;
          }

          using host_store_t = typename host_type_mapper<store_t>::type;
          using host_y_store_t = typename host_type_mapper<y_store_t>::type;
          using host_real_t = typename mapper<host_y_store_t>::type;
          Reducer<double, host_real_t> r_(a, b);

          // redefine site_unroll with host_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<host_store_t, host_y_store_t>::value || isFixed<host_store_t>::value || decltype(r)::site_unroll;
          constexpr int N = n_vector<host_store_t, false, nSpin, site_unroll>();
          constexpr int Ny = n_vector<host_y_store_t, false, nSpin, site_unroll>();
          constexpr int M = N; // if site unrolling then M=N will be 24/6, e.g., full AoS
          const int length = x.Length() / (nParity * M);

          ReductionArg<host_store_t, N, host_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity);
          result = reduceCPU<host_real_t, M>(arg);
        }
      }

      void preTune()
      {
        if (r.write.X) x.backup();
        if (r.write.Y) y.backup();
        if (r.write.Z) z.backup();
        if (r.write.W) w.backup();
        if (r.write.V) v.backup();
      }

      void postTune()
      {
        if (r.write.X) x.restore();
        if (r.write.Y) y.restore();
        if (r.write.Z) z.restore();
        if (r.write.W) w.restore();
        if (r.write.V) v.restore();
      }

      bool advanceTuneParam(TuneParam &param) const
      {
        return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
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

      long long flops() const { return r.flops() * x.Length(); }

      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        // this will evaluate correctly for non-mixed kernels since the +2/-2 will cancel out
        return (r.streams() - 2) * x.Bytes() + 2 * z.Bytes();
      }

      int tuningIter() const { return 3; }
    };

    template <template <typename reduce_t, typename real> class Functor, bool mixed, typename... Args>
    auto instantiateReduce(Args &&... args)
    {
      using host_reduce_t = typename Functor<double, double>::reduce_t;
      host_reduce_t value;
      ::quda::zero(value); // no default constructor so we need to explicitly zero
      instantiate<Functor, Reduce, mixed>(args..., value);
      return value;
    }

    double norm1(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x); // FIXME
      return instantiateReduce<Norm1, false>(0.0, 0.0, 0.0, y, y, y, y, y);
    }

    double norm2(const ColorSpinorField &x)
    {
      ColorSpinorField &y = const_cast<ColorSpinorField &>(x);
      return instantiateReduce<Norm2, false>(0.0, 0.0, 0.0, y, y, y, y, y);
    }

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<Dot, false>(0.0, 0.0, 0.0, x, y, x, x, x);
    }

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<axpbyzNorm2, false>(a, b, 0.0, x, y, z, x, x);
    }

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<AxpyReDot, false>(a, 0.0, 0.0, x, y, x, x, x);
    }

    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<caxpyNorm2, false>(a, Complex(0.0), Complex(0.0), x, y, x, x, x);
    }

    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<caxpyxmaznormx, false>(a, Complex(0.0), Complex(0.0), x, y, z, x, x);
    }

    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<cabxpyzaxnorm, false>(Complex(a), b, Complex(0.0), x, y, z, x, x);
    }

    Complex cDotProduct(ColorSpinorField &x, ColorSpinorField &y)
    {
      auto cdot = instantiateReduce<Cdot, false>(0.0, 0.0, 0.0, x, y, x, x, x);
      return Complex(cdot.x, cdot.y);
    }

    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      auto cdot = instantiateReduce<caxpydotzy, false>(a, Complex(0.0), Complex(0.0), x, y, z, x, x);
      return Complex(cdot.x, cdot.y);
    }

    double3 cDotProductNormA(ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<CdotNormA, false>(0.0, 0.0, 0.0, x, y, x, x, x);
    }

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y,
                                           ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &u)
    {
      return instantiateReduce<caxpbypzYmbwcDotProductUYNormY_, true>(a, b, Complex(0.0), x, z, y, w, u);
    }

    Complex axpyCGNorm(double a, ColorSpinorField &x, ColorSpinorField &y)
    {
      double2 cg_norm = instantiateReduce<axpyCGNorm2, true>(a, 0.0, 0.0, x, y, x, x, x);
      return Complex(cg_norm.x, cg_norm.y);
    }

    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor() != 3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = instantiateReduce<HeavyQuarkResidualNorm_, false>(0.0, 0.0, 0.0, x, r, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      double3 rtn = instantiateReduce<xpyHeavyQuarkResidualNorm_, false>(0.0, 0.0, 0.0, x, y, r, r, r);
      rtn.z /= (x.Volume()*comm_size());
      return rtn;
    }

    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<tripleCGReduction_, false>(0.0, 0.0, 0.0, x, y, z, x, x);
    }

    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<quadrupleCGReduction_, false>(0.0, 0.0, 0.0, x, y, z, x, x);
    }

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y,
                                ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      return instantiateReduce<quadrupleCG3InitNorm_, false>(a, 0.0, 0.0, x, y, z, w, v);
    }

    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y,
                                  ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v)
    {
      return instantiateReduce<quadrupleCG3UpdateNorm_, false>(a, b, 0.0, x, y, z, w, v);
    }

  } // namespace blas

} // namespace quda
