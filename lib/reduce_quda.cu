#include <blas_quda.h>
#include <tune_quda.h>
#include <color_spinor_field_order.h>
#include <jitify_helper.cuh>
#include <kernels/reduce_core.cuh>

namespace quda {

  namespace blas {

    qudaStream_t* getStream();

    template <int block_size, typename real, int len, typename Arg>
    typename std::enable_if<block_size!=32, qudaError_t>::type launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (tp.block.x == block_size)
        return qudaLaunchKernel(reduceKernel<block_size, real, len, Arg>, tp, stream, arg);
      else
        return launch<block_size - 32, real, len>(arg, tp, stream);
    }

    template <int block_size, typename real, int len, typename Arg>
    typename std::enable_if<block_size==32, qudaError_t>::type launch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream)
    {
      if (block_size != tp.block.x) errorQuda("Unexpected block size %d\n", tp.block.x);
      return qudaLaunchKernel(reduceKernel<block_size, real, len, Arg>, tp, stream, arg);
    }

#ifdef QUDA_FAST_COMPILE_REDUCE
    constexpr static unsigned int max_block_size() { return 32; }
#else
    constexpr static unsigned int max_block_size() { return 1024; }
#endif

   /**
       Generic reduction kernel launcher
    */
    template <typename host_reduce_t, typename real, int len, typename Arg>
    auto reduceLaunch(Arg &arg, const TuneParam &tp, const qudaStream_t &stream, Tunable &tunable)
    {
      using device_reduce_t = typename Arg::Reducer::reduce_t;
      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

#ifdef JITIFY
      using namespace jitify::reflection;
      tunable.jitifyError() = program->kernel("quda::blas::reduceKernel")
                                  .instantiate((int)tp.block.x, Type<real>(), len, Type<Arg>())
                                  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                                  .launch(arg);
      arg.launch_error = tunable.jitifyError() == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
#else
      arg.launch_error = launch<max_block_size(), real, len>(arg, tp, stream);
#endif

      host_reduce_t result;
      ::quda::zero(result);
      if (!commAsyncReduction()) arg.complete(result, stream);
      return result;
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

      unsigned int maxBlockSize(const TuneParam &param) const { return max_block_size(); }

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
        auto x_prec = checkPrecision(x, z, w, v);
        auto y_prec = y.Precision();
        auto x_order = checkOrder(x, z, w, v);
        auto y_order = y.FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        strcpy(aux, x.AuxString());
        if (x_prec != y_prec) {
          strcat(aux, ",");
          strcat(aux, y.AuxString());
        }
        strcat(aux, nParity == 2 ? ",nParity=2" : ",nParity=1");
        if (location == QUDA_CPU_FIELD_LOCATION) strcat(aux, ",CPU");
        if (commAsyncReduction()) strcat(aux, ",async");

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/reduce_core.cuh");
#endif

        apply(*(blas::getStream()));

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

          ReductionArg<device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity, tp);
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

          ReductionArg<host_store_t, N, host_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity, tp);
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
      }

      void defaultTuneParam(TuneParam &param) const
      {
        Tunable::defaultTuneParam(param);
      }

      long long flops() const { return r.flops() * x.Length(); }

      long long bytes() const
      {
        return (r.read.X + r.write.X) * x.Bytes() + (r.read.Y + r.write.Y) * y.Bytes() +
          (r.read.Z + r.write.Z) * z.Bytes() + (r.read.W + r.write.W) * w.Bytes() + (r.read.V + r.write.V) * v.Bytes();
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
