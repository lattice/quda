#include <blas_quda.h>
#include <color_spinor_field_order.h>
#include <tunable_reduction.h>
#include <kernels/reduce_core.cuh>

namespace quda {

  namespace blas {

    template <template <typename ReducerType, typename real> class Reducer, typename store_t, typename y_store_t,
              int nSpin, typename coeff_t>
    class Reduce : public TunableMultiReduction
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      Reducer<device_reduce_t, real> r;
      const int nParity; // for composite fields this includes the number of composites

      coeff_t a, b;
      cvector_ref<ColorSpinorField> &x, &y, &z, &w, &v;
      vector<host_reduce_t> &result;

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
      Reduce(const coeff_t &a, const coeff_t &b, const coeff_t &, Vx &x, Vy &y, Vz &z, Vw &w, Vv &v,
             vector<host_reduce_t> &result) :
        TunableMultiReduction(x[0], 1u, x.size()),
        r(a, b),
        nParity((x[0].IsComposite() ? x[0].CompositeDim() : 1) * (x.SiteSubset())),
        a(a),
        b(b),
        x(reinterpret_cast<cvector_ref<ColorSpinorField> &>(x)),
        y(reinterpret_cast<cvector_ref<ColorSpinorField> &>(y)),
        z(reinterpret_cast<cvector_ref<ColorSpinorField> &>(z)),
        w(reinterpret_cast<cvector_ref<ColorSpinorField> &>(w)),
        v(reinterpret_cast<cvector_ref<ColorSpinorField> &>(v)),
        result(result)
      {
        if (a.size() != x.size()) this->a.resize(x.size(), a.size() == 1 ? a[0] : 0.0);
        if (b.size() != x.size()) this->b.resize(x.size(), b.size() == 1 ? b[0] : 0.0);
        check_size(this->a, this->b, x, y, z, w, v);
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
        setRHSstring(aux, x.size());

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

    // split the fields and recurse if needed
    template <template <typename reduce_t, typename real> class Functor, bool mixed, typename coeff_t, typename X,
              typename Y, typename Z, typename W, typename V>
    auto instantiateReduce(const coeff_t &a, const coeff_t &b, const coeff_t &c, X &x, Y &y, Z &z, W &w, V &v)
      -> vector<typename Functor<double, double>::reduce_t>
    {
      if (x.size() > get_max_multi_rhs()) {
        auto value0 = instantiateReduce<Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin(), x.begin() + x.size() / 2}, {y.begin(), y.begin() + y.size() / 2},
          {z.begin(), z.begin() + z.size() / 2}, {w.begin(), w.begin() + w.size() / 2},
          {v.begin(), v.begin() + v.size() / 2});
        auto value1 = instantiateReduce<Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin() + x.size() / 2, x.end()}, {y.begin() + y.size() / 2, y.end()},
          {z.begin() + z.size() / 2, z.end()}, {w.begin() + w.size() / 2, w.end()}, {v.begin() + v.size() / 2, v.end()});
        value0.reserve(value0.size() + value1.size());
        value0.insert(value0.end(), value1.begin(), value1.end());
        return value0;
      }

      using host_reduce_t = typename Functor<double, double>::reduce_t;
      vector<host_reduce_t> value(x.size());
      instantiate<Functor, Reduce, mixed>(a, b, c, x, y, z, w, v, value);
      return value;
    }

    cvector<double> max(cvector_ref<const ColorSpinorField> &x)
    {
      return instantiateReduce<Max, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0), x, x, x, x,
                                           x);
    }

    cvector<array<double, 2>> max_deviation(cvector_ref<const ColorSpinorField> &x,
                                            cvector_ref<const ColorSpinorField> &y)
    {
      vector<array<double, 2>> deviation(x.size());
      auto dev = instantiateReduce<MaxDeviation, true>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0),
                                                       x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) {
        // ensure that if the absolute deviation is zero, so is the relative deviation
        deviation[i] = {dev[i].diff, dev[i].diff > 0.0 ? dev[i].diff / dev[i].ref : 0.0};
      }
      return deviation;
    }

    cvector<double> norm1(cvector_ref<const ColorSpinorField> &x)
    {
      return instantiateReduce<Norm1, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0), x, x, x,
                                             x, x);
    }

    cvector<double> norm2(cvector_ref<const ColorSpinorField> &x)
    {
      return instantiateReduce<Norm2, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0), x, x, x,
                                             x, x);
    }

    cvector<double> reDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return instantiateReduce<Dot, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0), x, y, x, x,
                                           x);
    }

    cvector<double> axpbyzNorm(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                               cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return instantiateReduce<axpbyzNorm2, false>(a, b, cvector<double>(0.0), x, y, z, x, x);
    }

    cvector<double> axpyReDot(cvector<double> &a, cvector_ref<const ColorSpinorField> &x,
                              cvector_ref<ColorSpinorField> &y)
    {
      return instantiateReduce<AxpyReDot, false>(a, cvector<double>(0.0), cvector<double>(0.0), x, y, x, x, x);
    }

    cvector<double> caxpbyNorm(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector<Complex> &b,
                               cvector_ref<ColorSpinorField> &y)
    {
      return instantiateReduce<caxpyNorm2, true>(a, b, cvector<Complex>(0.0), x, y, x, x, x);
    }

    cvector<double> cabxpyzAxNorm(cvector<double> &a, cvector<Complex> &b, cvector_ref<ColorSpinorField> &x,
                                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return instantiateReduce<cabxpyzaxnorm, false>(cvector<Complex>(a), b, cvector<Complex>(0.0), x, y, z, x, x);
    }

    cvector<Complex> cDotProduct(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      vector<Complex> cdots(x.size());
      auto cdot = instantiateReduce<Cdot, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0), x, y,
                                                 x, x, x);
      for (auto i = 0u; i < x.size(); i++) cdots[i] = {cdot[i][0], cdot[i][1]};
      return cdots;
    }

    cvector<Complex> caxpyDotzy(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      vector<Complex> cdot(x.size());
      auto c = instantiateReduce<caxpydotzy, false>(a, cvector<Complex>(0.0), cvector<Complex>(0.0), x, y, z, x, x);
      for (auto i = 0u; i < x.size(); i++) cdot[i] = {c[i][0], c[i][1]};
      return cdot;
    }

    cvector<double4> cDotProductNormAB(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      vector<double4> abs(x.size());
      auto ab = instantiateReduce<CdotNormAB, false>(cvector<double>(0.0), cvector<double>(0.0), cvector<double>(0.0),
                                                     x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) abs[i] = {ab[i][0], ab[i][1], ab[i][2], ab[i][3]};
      return abs;
    }

    cvector<double3> caxpbypzYmbwcDotProductUYNormY(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                                                    cvector<Complex> &b, cvector_ref<ColorSpinorField> &y,
                                                    cvector_ref<ColorSpinorField> &z,
                                                    cvector_ref<const ColorSpinorField> &w,
                                                    cvector_ref<const ColorSpinorField> &v)
    {
      vector<double3> abs(x.size());
      auto ab = instantiateReduce<caxpbypzYmbwcDotProductUYNormY_, true>(a, b, cvector<Complex>(), x, z, y, w, v);
      for (auto i = 0u; i < x.size(); i++) abs[i] = {ab[i][0], ab[i][1], ab[i][2]};
      return abs;
    }

    cvector<double2> axpyCGNorm(cvector<double> &a, cvector_ref<const ColorSpinorField> &x,
                                cvector_ref<ColorSpinorField> &y)
    {
      vector<double2> norm(x.size());
      auto cg_norm = instantiateReduce<axpyCGNorm2, true>(a, cvector<double>(0.0), cvector<double>(0.0), x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = {cg_norm[i][0], cg_norm[i][1]};
      return norm;
    }

    cvector<double3> HeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                            cvector_ref<const ColorSpinorField> &r)
    {
      vector<double3> norm(x.size(), {});
      if (x.Ncolor() == 3) { // Nc != 3 (MG mainly) not suppored
        auto n = instantiateReduce<HeavyQuarkResidualNorm_, false>(cvector<double>(0.0), cvector<double>(0.0),
                                                                   cvector<double>(0.0), x, r, x, x, x);
        for (auto i = 0u; i < x.size(); i++) norm[i] = {n[i][0], n[i][1], n[i][2] / (x.Volume() * comm_size())};
      }
      return norm;
    }

    cvector<double3> xpyHeavyQuarkResidualNorm(cvector_ref<const ColorSpinorField> &x,
                                               cvector_ref<const ColorSpinorField> &y,
                                               cvector_ref<const ColorSpinorField> &r)
    {
      vector<double3> norm(x.size(), {});
      if (x.Ncolor() == 3) { // Nc != 3 (MG mainly) not suppored
        auto n = instantiateReduce<xpyHeavyQuarkResidualNorm_, true>(cvector<double>(0.0), cvector<double>(0.0),
                                                                     cvector<double>(0.0), x, y, r, r, r);
        for (auto i = 0u; i < x.size(); i++) norm[i] = {n[i][0], n[i][1], n[i][2] / (x.Volume() * comm_size())};
      }
      return norm;
    }

    cvector<double3> tripleCGReduction(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                                       cvector_ref<const ColorSpinorField> &z)
    {
      vector<double3> norm(x.size());
      auto cg = instantiateReduce<tripleCGReduction_, false>(cvector<double>(0.0), cvector<double>(0.0),
                                                             cvector<double>(0.0), x, y, z, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = {cg[i][0], cg[i][1], cg[i][2]};
      return norm;
    }

    cvector<double4> quadrupleCGReduction(cvector_ref<const ColorSpinorField> &x,
                                          cvector_ref<const ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      vector<double4> norm(x.size());
      auto cg = instantiateReduce<quadrupleCGReduction_, false>(cvector<double>(0.0), cvector<double>(0.0),
                                                                cvector<double>(0.0), x, y, z, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = {cg[i][0], cg[i][1], cg[i][2], cg[i][3]};
      return norm;
    }

    cvector<double> quadrupleCG3InitNorm(cvector<double> &a, cvector_ref<ColorSpinorField> &x,
                                         cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                         cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return instantiateReduce<quadrupleCG3InitNorm_, false>(a, cvector<double>(0.0), cvector<double>(0.0), x, y, z, w,
                                                             v);
    }

    cvector<double> quadrupleCG3UpdateNorm(cvector<double> &a, cvector<double> &b, cvector_ref<ColorSpinorField> &x,
                                           cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                           cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return instantiateReduce<quadrupleCG3UpdateNorm_, false>(a, b, cvector<double>(0.0), x, y, z, w, v);
    }
  } // namespace blas

} // namespace quda
