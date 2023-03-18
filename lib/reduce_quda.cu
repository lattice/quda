#include <blas_quda.h>
#include <color_spinor_field_order.h>
#include <tunable_reduction.h>
#include <kernels/reduce_core.cuh>

namespace quda {

  namespace blas {

    template <template <typename ReducerType, typename real> class Reducer,
              typename store_t, typename y_store_t, int nSpin, typename coeff_t>
    class Reduce : public TunableReduction2D
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      Reducer<device_reduce_t, real> r;
      const int nParity; // for composite fields this includes the number of composites

      const coeff_t &a, &b;
      ColorSpinorField &x, &y, &z, &w, &v;
      host_reduce_t &result;

      bool advanceSharedBytes(TuneParam &param) const override
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
      template <typename Vx, typename Vy, typename Vz, typename Vw, typename Vv>
      Reduce(const coeff_t &a, const coeff_t &b, const coeff_t &, Vx &x, Vy &y, Vz &z, Vw &w, Vv &v, host_reduce_t &result) :
        TunableReduction2D(x, 1u),
        r(a, b),
        nParity((x.IsComposite() ? x.CompositeDim() : 1) * (x.SiteSubset())),
        a(a),
        b(b),
        x(const_cast<ColorSpinorField&>(x)),
        y(const_cast<ColorSpinorField&>(y)),
        z(const_cast<ColorSpinorField&>(z)),
        w(const_cast<ColorSpinorField&>(w)),
        v(const_cast<ColorSpinorField&>(v)),
        result(result)
      {
        checkLocation(x, y, z, w, v);
        checkLength(x, y, z, w, v);
        auto x_prec = checkPrecision(x, z, w, v);
        auto y_prec = y.Precision();
        auto x_order = checkOrder(x, z, w, v);
        auto y_order = y.FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        if (x_prec != y_prec) {
          strcat(aux, ",");
          strcat(aux, y.AuxString().c_str());
        }

        apply(device::get_default_stream());

        blas::bytes += bytes();
        blas::flops += flops();
      }

      TuneKey tuneKey() const override { return TuneKey(vol, typeid(r).name(), aux); }

      void apply(const qudaStream_t &stream) override
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
          const int length = x.Length() / M;

          ReductionArg<device_real_t, M, device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity);
          launch<Reduce_>(result, tp, stream, arg);
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
          const int length = x.Length() / M;

          ReductionArg<host_real_t, M, host_store_t, N, host_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length, nParity);
          launch_host<Reduce_>(result, tp, stream, arg);
        }
      }

      void preTune() override
      {
        if (r.write.X) x.backup();
        if (r.write.Y) y.backup();
        if (r.write.Z) z.backup();
        if (r.write.W) w.backup();
        if (r.write.V) v.backup();
      }

      void postTune() override
      {
        if (r.write.X) x.restore();
        if (r.write.Y) y.restore();
        if (r.write.Z) z.restore();
        if (r.write.W) w.restore();
        if (r.write.V) v.restore();
      }

      long long flops() const override { return r.flops() * x.Length(); }

      long long bytes() const override
      {
        return (r.read.X + r.write.X) * x.Bytes() + (r.read.Y + r.write.Y) * y.Bytes() +
          (r.read.Z + r.write.Z) * z.Bytes() + (r.read.W + r.write.W) * w.Bytes() + (r.read.V + r.write.V) * v.Bytes();
      }
    };

    template <template <typename reduce_t, typename real> class Functor, bool mixed, typename... Args>
    auto instantiateReduce(Args &&... args)
    {
      using host_reduce_t = typename Functor<double, double>::reduce_t;
      host_reduce_t value = ::quda::zero<host_reduce_t>();
      instantiate<Functor, Reduce, mixed>(args..., value);
      return value;
    }

    double max(const ColorSpinorField &x)
    {
      return instantiateReduce<Max, false>(0.0, 0.0, 0.0, x, x, x, x, x);
    }

    array<double, 2> max_deviation(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto deviation = instantiateReduce<MaxDeviation, false>(0.0, 0.0, 0.0, x, y, y, y, y);
      // ensure that if the absolute deviation is zero, so is the relative deviation
      return {deviation.diff, deviation.diff > 0.0 ? deviation.diff / deviation.ref : 0.0};
    }

    double norm1(const ColorSpinorField &x)
    {
      return instantiateReduce<Norm1, false>(0.0, 0.0, 0.0, x, x, x, x, x);
    }

    double norm2(const ColorSpinorField &x)
    {
      return instantiateReduce<Norm2, false>(0.0, 0.0, 0.0, x, x, x, x, x);
    }

    double reDotProduct(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      return instantiateReduce<Dot, false>(0.0, 0.0, 0.0, x, y, x, x, x);
    }

    double axpbyzNorm(double a, const ColorSpinorField &x, double b, const ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<axpbyzNorm2, false>(a, b, 0.0, x, y, z, x, x);
    }

    double axpyReDot(double a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<AxpyReDot, false>(a, 0.0, 0.0, x, y, x, x, x);
    }

    double caxpyNorm(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      return instantiateReduce<caxpyNorm2, false>(a, Complex(0.0), Complex(0.0), x, y, x, x, x);
    }

    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &z)
    {
      return instantiateReduce<cabxpyzaxnorm, false>(Complex(a), b, Complex(0.0), x, y, z, x, x);
    }

    Complex cDotProduct(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto cdot = instantiateReduce<Cdot, false>(0.0, 0.0, 0.0, x, y, x, x, x);
      return Complex(cdot[0], cdot[1]);
    }

    Complex caxpyDotzy(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z)
    {
      auto cdot = instantiateReduce<caxpydotzy, false>(a, Complex(0.0), Complex(0.0), x, y, z, x, x);
      return Complex(cdot[0], cdot[1]);
    }

    double4 cDotProductNormAB(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto ab = instantiateReduce<CdotNormAB, false>(0.0, 0.0, 0.0, x, y, x, x, x);
      return make_double4(ab[0], ab[1], ab[2], ab[3]);
    }

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, const ColorSpinorField &x, const Complex &b,
                                           ColorSpinorField &y, ColorSpinorField &z,
                                           const ColorSpinorField &w, const ColorSpinorField &v)
    {
      auto rtn = instantiateReduce<caxpbypzYmbwcDotProductUYNormY_, true>(a, b, Complex(0.0), x, z, y, w, v);
      return make_double3(rtn[0], rtn[1], rtn[2]);
    }

    double2 axpyCGNorm(double a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      auto cg_norm = instantiateReduce<axpyCGNorm2, true>(a, 0.0, 0.0, x, y, x, x, x);
      return make_double2(cg_norm[0], cg_norm[1]);
    }

    double3 HeavyQuarkResidualNorm(const ColorSpinorField &x, const ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor() != 3) return make_double3(0.0, 0.0, 0.0);
      auto rtn = instantiateReduce<HeavyQuarkResidualNorm_, false>(0.0, 0.0, 0.0, x, r, r, r, r);
      rtn[2] /= (x.Volume()*comm_size());
      return make_double3(rtn[0], rtn[1], rtn[2]);
    }

    double3 xpyHeavyQuarkResidualNorm(const ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &r)
    {
      // in case of x.Ncolor()!=3 (MG mainly) reduce_core do not support this function.
      if (x.Ncolor()!=3) return make_double3(0.0, 0.0, 0.0);
      auto rtn = instantiateReduce<xpyHeavyQuarkResidualNorm_, false>(0.0, 0.0, 0.0, x, y, r, r, r);
      rtn[2] /= (x.Volume()*comm_size());
      return make_double3(rtn[0], rtn[1], rtn[2]);
    }

    double3 tripleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z)
    {
      auto rtn = instantiateReduce<tripleCGReduction_, false>(0.0, 0.0, 0.0, x, y, z, x, x);
      return make_double3(rtn[0], rtn[1], rtn[2]);
    }

    double4 quadrupleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z)
    {
      auto red = instantiateReduce<quadrupleCGReduction_, false>(0.0, 0.0, 0.0, x, y, z, x, x);
      return make_double4(red[0], red[1], red[2], red[3]);
    }

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y,
                                ColorSpinorField &z, ColorSpinorField &w, const ColorSpinorField &v)
    {
      return instantiateReduce<quadrupleCG3InitNorm_, false>(a, 0.0, 0.0, x, y, z, w, v);
    }

    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y,
                                  ColorSpinorField &z, ColorSpinorField &w, const ColorSpinorField &v)
    {
      return instantiateReduce<quadrupleCG3UpdateNorm_, false>(a, b, 0.0, x, y, z, w, v);
    }

  } // namespace blas

} // namespace quda
