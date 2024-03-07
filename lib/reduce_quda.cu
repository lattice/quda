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

    cvector<double> max(cvector_ref<const ColorSpinorField> &x)
    {
      vector<double> max(x.size());
      for (auto i = 0u; i < x.size(); i++)
        max[i] = instantiateReduce<Max, false>(0.0, 0.0, 0.0, x[i], x[i], x[i], x[i], x[i]);
      return max;
    }

    cvector<array<double, 2>> max_deviation(cvector_ref<const ColorSpinorField> &x,
                                            cvector_ref<const ColorSpinorField> &y)
    {
      check_size(x, y);
      vector<array<double, 2>> deviation(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto dev = instantiateReduce<MaxDeviation, true>(0.0, 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
        // ensure that if the absolute deviation is zero, so is the relative deviation
        deviation[i] = {dev.diff, dev.diff > 0.0 ? dev.diff / dev.ref : 0.0};
      }
      return deviation;
    }

    cvector<double> norm1(cvector_ref<const ColorSpinorField> &x)
    {
      vector<double> norm1(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm1[i] = instantiateReduce<Norm1, false>(0.0, 0.0, 0.0, x[i], x[i], x[i], x[i], x[i]);
      return norm1;
    }

    cvector<double> norm2(cvector_ref<const ColorSpinorField> &x)
    {
      vector<double> norm2(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm2[i] = instantiateReduce<Norm2, false>(0.0, 0.0, 0.0, x[i], x[i], x[i], x[i], x[i]);
      return norm2;
    }

    cvector<double> reDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      check_size(x, y);
      vector<double> dot(x.size());
      for (auto i = 0u; i < x.size(); i++)
        dot[i] = instantiateReduce<Dot, false>(0.0, 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
      return dot;
    }

    cvector<double> axpbyzNorm(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                               cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      check_size(a, x, b, y, z);
      vector<double> norm(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm[i] = instantiateReduce<axpbyzNorm2, false>(a[i], b[i], 0.0, x[i], y[i], z[i], x[i], x[i]);
      return norm;
    }

    cvector<double> axpyReDot(cvector<double> &a, cvector_ref<const ColorSpinorField> &x,
                              cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, y);
      vector<double> dot(x.size());
      for (auto i = 0u; i < x.size(); i++)
        dot[i] = instantiateReduce<AxpyReDot, false>(a[i], 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
      return dot;
    }

    cvector<double> caxpbyNorm(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector<Complex> &b,
                               cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, b, y);
      vector<double> norm(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm[i] = instantiateReduce<caxpyNorm2, false>(a[i], b[i], Complex(0.0), x[i], y[i], x[i], x[i], x[i]);
      return norm;
    }

    cvector<double> cabxpyzAxNorm(cvector<double> &a, cvector<Complex> &b, cvector_ref<ColorSpinorField> &x,
                                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      check_size(a, b, x, y, z);
      vector<double> norm(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm[i]
          = instantiateReduce<cabxpyzaxnorm, false>(Complex(a[i]), b[i], Complex(0.0), x[i], y[i], z[i], x[i], x[i]);
      return norm;
    }

    cvector<Complex> cDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      check_size(x, y);
      vector<Complex> cdots(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto cdot = instantiateReduce<Cdot, false>(0.0, 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
        cdots[i] = {cdot[0], cdot[1]};
      }
      return cdots;
    }

    cvector<Complex> caxpyDotzy(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      check_size(a, x, y, z);
      vector<Complex> cdot(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto c = instantiateReduce<caxpydotzy, false>(a[i], Complex(0.0), Complex(0.0), x[i], y[i], z[i], x[i], x[i]);
        cdot[i] = {c[0], c[1]};
      }
      return cdot;
    }

    cvector<double4> cDotProductNormAB(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      check_size(x, y);
      vector<double4> abs(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto ab = instantiateReduce<CdotNormAB, false>(0.0, 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
        abs[i] = {ab[0], ab[1], ab[2], ab[3]};
      }
      return abs;
    }

    cvector<double3> caxpbypzYmbwcDotProductUYNormY(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                                                    cvector<Complex> &b, cvector_ref<ColorSpinorField> &y,
                                                    cvector_ref<ColorSpinorField> &z,
                                                    cvector_ref<const ColorSpinorField> &w,
                                                    cvector_ref<const ColorSpinorField> &v)
    {
      check_size(a, x, b, y, z, w, v);
      vector<double3> abs(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto ab = instantiateReduce<caxpbypzYmbwcDotProductUYNormY_, true>(a[i], b[i], Complex(0.0), x[i], z[i], y[i],
                                                                           w[i], v[i]);
        abs[i] = {ab[0], ab[1], ab[2]};
      }
      return abs;
    }

    cvector<double2> axpyCGNorm(cvector<double> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, y);
      vector<double2> norm(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto cg_norm = instantiateReduce<axpyCGNorm2, true>(a[i], 0.0, 0.0, x[i], y[i], x[i], x[i], x[i]);
        norm[i] = {cg_norm[0], cg_norm[1]};
      }
      return norm;
    }

    cvector<double3> HeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                            cvector_ref<const ColorSpinorField> &r)
    {
      check_size(x, r);
      vector<double3> norm(x.size(), {});
      if (x[0].Ncolor() == 3) {// Nc != 3 (MG mainly) not suppored
        for (auto i = 0u; i < x.size(); i++) {
          auto n = instantiateReduce<HeavyQuarkResidualNorm_, false>(0.0, 0.0, 0.0, x[i], r[i], x[i], x[i], x[i]);
          norm[i] = {n[0], n[1], n[2] / (x[0].Volume() * comm_size())};
        }
      }
      return norm;
    }

    cvector<double3> xpyHeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                               cvector_ref<const ColorSpinorField> &y,
                                               cvector_ref<const ColorSpinorField> &r)
    {
      check_size(x, y, r);
      vector<double3> norm(x.size(), {});
      if (x[0].Ncolor() == 3) // Nc != 3 (MG mainly) not suppored
        for (auto i = 0u; i < x.size(); i++) {
          auto n = instantiateReduce<xpyHeavyQuarkResidualNorm_, true>(0.0, 0.0, 0.0, x[i], y[i], r[i], r[i], r[i]);
          norm[i] = {n[0], n[1], n[2] / (x[0].Volume() * comm_size())};
        }
      return norm;
    }

    cvector<double3> tripleCGReduction(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                                       cvector_ref<const ColorSpinorField> &z)
    {
      check_size(x, y, z);
      vector<double3> norm(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto cg = instantiateReduce<tripleCGReduction_, false>(0.0, 0.0, 0.0, x[i], y[i], z[i], x[i], x[i]);
        norm[i] = {cg[0], cg[1], cg[2]};
      }
      return norm;
    }

    cvector<double4> quadrupleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                          cvector_ref<const ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      check_size(x, y, z);
      vector<double4> norm(x.size());
      for (auto i = 0u; i < x.size(); i++) {
        auto cg = instantiateReduce<quadrupleCGReduction_, false>(0.0, 0.0, 0.0, x[i], y[i], z[i], x[i], x[i]);
        norm[i] = {cg[0], cg[1], cg[2], cg[3]};
      }
      return norm;
    }

    cvector<double> quadrupleCG3InitNorm(cvector<double> &a, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                         cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      check_size(a, x, y, z, w, v);
      vector<double> norm(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm[i] = instantiateReduce<quadrupleCG3InitNorm_, false>(a[i], 0.0, 0.0, x[i], y[i], z[i], w[i], v[i]);
      return norm;
    }

    cvector<double> quadrupleCG3UpdateNorm(cvector<double> &a, cvector<double> &b, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      check_size(a, b, x, y, z, w, v);
      vector<double> norm(x.size());
      for (auto i = 0u; i < x.size(); i++)
        norm[i] = instantiateReduce<quadrupleCG3UpdateNorm_, false>(a[i], b[i], 0.0, x[i], y[i], z[i], w[i], v[i]);
      return norm;
    }

  } // namespace blas

} // namespace quda
