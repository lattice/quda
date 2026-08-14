#pragma once

#include <blas_quda.h>
#include <comm_quda.h>
#include <color_spinor_field_order.h>
#include <tunable_reduction.h>
#include <kernels/reduce_core.cuh>

namespace quda
{

  namespace blas
  {

    /** Promote a device/host reduction value to host scalar real_t (e.g. doubledouble -> __float128). */
    template <typename T> real_t reduction_to_real(const T &x)
    {
      if constexpr (is_rfa<T>::value) {
        return reduction_to_real(x.conv());
      } else {
        return static_cast<real_t>(x);
      }
    }

    template <int N, typename T> array<real_t, N> reduction_to_array(const T &v)
    {
      array<real_t, N> out {};
      for (int i = 0; i < N; i++) out[i] = reduction_to_real(v[i]);
      return out;
    }

    template <typename T> vector<real_t> to_real_vector(const vector<T> &v)
    {
      vector<real_t> out(v.size());
      for (size_t i = 0; i < v.size(); i++) out[i] = reduction_to_real(v[i]);
      return out;
    }

    template <template <typename ReducerType, typename real> class Reducer, typename store_t, typename y_store_t,
              int nSpin, typename coeff_t>
    class Reduce : public TunableMultiReduction
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<device_reduce_t, real>::reduce_t;
      Reducer<device_reduce_t, real> r;
      const int nParity; // for composite fields this includes the number of composites

      coeff_t a, b;
      cvector_ref<ColorSpinorField> &x, &y, &z, &w, &v;
      vector<host_reduce_t> &result;
      bool tuneSharedBytes() const override { return false; }

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
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          blas_tune_aux_prefetch(aux);
          blas_tune_aux_work_item_unroll(aux, reduce_unroll);
        }

        apply(device::get_default_stream());
      }

      TuneKey tuneKey() const override { return TuneKey(vol, typeid(r).name(), aux); }

      void apply(const qudaStream_t &stream) override
      {
        constexpr bool site_unroll_check
          = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value || decltype(r)::site_unroll;
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
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value
            || isFixed<device_store_t>::value || decltype(r)::site_unroll;
          constexpr int N = n_vector<device_store_t, true>(nSpin, site_unroll);
          constexpr int Ny = n_vector<device_y_store_t, true>(nSpin, site_unroll);
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x.Length() / M;

          ReductionArg<device_real_t, M, device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_,
                                                                                                   length, nParity);
          launch<Reduce_>(result, tp, stream, arg);
        } else {
          if (checkOrder(x, y, z, w, v) != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
            warningQuda("CPU Blas functions expect AoS field order");
            return;
          }

          using host_store_t = typename host_type_mapper<store_t>::type;
          using host_y_store_t = typename host_type_mapper<y_store_t>::type;
          using host_real_t = typename mapper<host_y_store_t>::type;
          Reducer<device_reduce_t, host_real_t> r_(a, b);

          // redefine site_unroll with host_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<host_store_t, host_y_store_t>::value || isFixed<host_store_t>::value
            || decltype(r)::site_unroll;
          constexpr int N = n_vector<host_store_t, false>(nSpin, site_unroll);
          constexpr int Ny = n_vector<host_y_store_t, false>(nSpin, site_unroll);
          constexpr int M = N; // if site unrolling then M=N will be 24/6, e.g., full AoS
          const int length = x.Length() / M;

          ReductionArg<host_real_t, M, host_store_t, N, host_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, v, r_, length,
                                                                                             nParity);
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

      long long flops() const override { return r.flops() * x.Length() * x.size(); }

      long long bytes() const override
      {
        return (r.read.X + r.write.X) * x.Bytes() + (r.read.Y + r.write.Y) * y.Bytes() + (r.read.Z + r.write.Z) * z.Bytes()
          + (r.read.W + r.write.W) * w.Bytes() + (r.read.V + r.write.V) * v.Bytes();
      }
    };

    /**
       Run the reduction kernel for a fixed x-field store type and return raw
       accumulators. Conversion to host real_t is done in the public wrappers.
    */
    template <typename store_t, template <typename reduce_t, typename real> class Functor, bool mixed, typename coeff_t,
              typename X, typename Y, typename Z, typename W, typename V>
    auto instantiateReduce(const coeff_t &a, const coeff_t &b, const coeff_t &c, X &x, Y &y, Z &z, W &w, V &v)
      -> vector<typename Functor<device_reduce_t, double>::reduce_t>
    {
      if (x.size() > get_max_multi_rhs()) {
        auto value0 = instantiateReduce<store_t, Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin(), x.begin() + x.size() / 2}, {y.begin(), y.begin() + y.size() / 2},
          {z.begin(), z.begin() + z.size() / 2}, {w.begin(), w.begin() + w.size() / 2},
          {v.begin(), v.begin() + v.size() / 2});
        auto value1 = instantiateReduce<store_t, Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin() + x.size() / 2, x.end()}, {y.begin() + y.size() / 2, y.end()},
          {z.begin() + z.size() / 2, z.end()}, {w.begin() + w.size() / 2, w.end()}, {v.begin() + v.size() / 2, v.end()});
        value0.reserve(value0.size() + value1.size());
        value0.insert(value0.end(), value1.begin(), value1.end());
        return value0;
      }

      using host_reduce_t = typename Functor<device_reduce_t, double>::reduce_t;
      vector<host_reduce_t> value(x.size());
      instantiate<Functor, Reduce, mixed, coeff_t, store_t>(a, b, c, x, y, z, w, v, value);
      return value;
    }

    template <typename store_t> cvector<real_t> max_impl(cvector_ref<const ColorSpinorField> &x)
    {
      return to_real_vector(instantiateReduce<store_t, Max, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                   cvector<real_t>(0.0), x, x, x, x, x));
    }

    template <typename store_t>
    cvector<array<real_t, 2>> max_deviation_impl(cvector_ref<const ColorSpinorField> &x,
                                                 cvector_ref<const ColorSpinorField> &y)
    {
      vector<array<real_t, 2>> deviation(x.size());
      auto dev = instantiateReduce<store_t, MaxDeviation, true>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                cvector<real_t>(0.0), x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) {
        deviation[i] = {reduction_to_real(dev[i].diff),
                        (dev[i].diff > reduction_t(0)) ? reduction_to_real(dev[i].diff / dev[i].ref) : real_t(0)};
      }
      return deviation;
    }

    template <typename store_t> cvector<real_t> norm1_impl(cvector_ref<const ColorSpinorField> &x)
    {
      return to_real_vector(instantiateReduce<store_t, Norm1, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                     cvector<real_t>(0.0), x, x, x, x, x));
    }

    template <typename store_t> cvector<real_t> norm2_impl(cvector_ref<const ColorSpinorField> &x)
    {
      return to_real_vector(instantiateReduce<store_t, Norm2, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                     cvector<real_t>(0.0), x, x, x, x, x));
    }

    template <typename store_t>
    cvector<real_t> reDotProduct_impl(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      return to_real_vector(instantiateReduce<store_t, Dot, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                   cvector<real_t>(0.0), x, y, x, x, x));
    }

    template <typename store_t>
    cvector<real_t> axpbyzNorm_impl(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return to_real_vector(instantiateReduce<store_t, axpbyzNorm2, false>(a, b, cvector<real_t>(0.0), x, y, z, x, x));
    }

    template <typename store_t>
    cvector<real_t> axpyReDot_impl(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                   cvector_ref<ColorSpinorField> &y)
    {
      return to_real_vector(
        instantiateReduce<store_t, AxpyReDot, false>(a, cvector<real_t>(0.0), cvector<real_t>(0.0), x, y, x, x, x));
    }

    template <typename store_t>
    cvector<real_t> caxpbyNorm_impl(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                                    cvector_ref<ColorSpinorField> &y)
    {
      return to_real_vector(instantiateReduce<store_t, caxpyNorm2, true>(a, b, cvector<complex_t>(0.0), x, y, x, x, x));
    }

    template <typename store_t>
    cvector<real_t> cabxpyzAxNorm_impl(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                                       cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      return to_real_vector(
        instantiateReduce<store_t, cabxpyzaxnorm, false>(cvector<complex_t>(a), b, cvector<complex_t>(0.0), x, y, z, x, x));
    }

    template <typename store_t>
    cvector<complex_t> cDotProduct_impl(cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y)
    {
      vector<complex_t> cdots(x.size());
      auto cdot = instantiateReduce<store_t, Cdot, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                          cvector<real_t>(0.0), x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++)
        cdots[i] = complex_t(reduction_to_real(cdot[i][0]), reduction_to_real(cdot[i][1]));
      return cdots;
    }

    template <typename store_t>
    cvector<complex_t> caxpyDotzy_impl(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                                       cvector_ref<ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z)
    {
      vector<complex_t> cdot(x.size());
      auto c = instantiateReduce<store_t, caxpydotzy, false>(a, cvector<complex_t>(0.0), cvector<complex_t>(0.0), x, y, z,
                                                             x, x);
      for (auto i = 0u; i < x.size(); i++) cdot[i] = complex_t(reduction_to_real(c[i][0]), reduction_to_real(c[i][1]));
      return cdot;
    }

    template <typename store_t>
    cvector<array<real_t, 4>> cDotProductNormAB_impl(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y)
    {
      vector<array<real_t, 4>> abs(x.size());
      auto ab = instantiateReduce<store_t, CdotNormAB, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                              cvector<real_t>(0.0), x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) abs[i] = reduction_to_array<4>(ab[i]);
      return abs;
    }

    template <typename store_t>
    cvector<array<real_t, 3>> caxpbypzYmbwcDotProductUYNormY_impl(
      cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z, cvector_ref<const ColorSpinorField> &w,
      cvector_ref<const ColorSpinorField> &v)
    {
      vector<array<real_t, 3>> abs(x.size());
      auto ab = instantiateReduce<store_t, caxpbypzYmbwcDotProductUYNormY_, true>(a, b, cvector<complex_t>(), x, z, y, w, v);
      for (auto i = 0u; i < x.size(); i++) abs[i] = reduction_to_array<3>(ab[i]);
      return abs;
    }

    template <typename store_t>
    cvector<array<real_t, 2>> axpyCGNorm_impl(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                                              cvector_ref<ColorSpinorField> &y)
    {
      vector<array<real_t, 2>> norm(x.size());
      auto cg_norm
        = instantiateReduce<store_t, axpyCGNorm2, true>(a, cvector<real_t>(0.0), cvector<real_t>(0.0), x, y, x, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = reduction_to_array<2>(cg_norm[i]);
      return norm;
    }

    template <typename store_t>
    cvector<array<real_t, 3>> HeavyQuarkResidualNorm_impl(cvector_ref<const ColorSpinorField> &x,
                                                          cvector_ref<const ColorSpinorField> &r)
    {
      vector<array<real_t, 3>> norm(x.size(), {});
      if (x.Ncolor() == 3) { // Nc != 3 (MG mainly) not suppored
        auto n = instantiateReduce<store_t, HeavyQuarkResidualNorm_, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                            cvector<real_t>(0.0), x, r, x, x, x);
        const auto scale = real_t(1.0) / real_t(x.Volume() * comm_size());
        for (auto i = 0u; i < x.size(); i++) {
          norm[i] = reduction_to_array<3>(n[i]);
          norm[i][2] = reduction_to_real(n[i][2]) * scale;
        }
      }
      return norm;
    }

    template <typename store_t>
    cvector<array<real_t, 3>> xpyHeavyQuarkResidualNorm_impl(cvector_ref<const ColorSpinorField> &x,
                                                             cvector_ref<const ColorSpinorField> &y,
                                                             cvector_ref<const ColorSpinorField> &r)
    {
      vector<array<real_t, 3>> norm(x.size(), {});
      if (x.Ncolor() == 3) { // Nc != 3 (MG mainly) not suppored
        auto n = instantiateReduce<store_t, xpyHeavyQuarkResidualNorm_, true>(
          cvector<real_t>(0.0), cvector<real_t>(0.0), cvector<real_t>(0.0), x, y, r, r, r);
        const auto scale = real_t(1.0) / real_t(x.Volume() * comm_size());
        for (auto i = 0u; i < x.size(); i++) {
          norm[i] = reduction_to_array<3>(n[i]);
          norm[i][2] = reduction_to_real(n[i][2]) * scale;
        }
      }
      return norm;
    }

    template <typename store_t>
    cvector<array<real_t, 3>> tripleCGReduction_impl(cvector_ref<const ColorSpinorField> &x,
                                                     cvector_ref<const ColorSpinorField> &y,
                                                     cvector_ref<const ColorSpinorField> &z)
    {
      vector<array<real_t, 3>> norm(x.size());
      auto cg = instantiateReduce<store_t, tripleCGReduction_, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                      cvector<real_t>(0.0), x, y, z, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = reduction_to_array<3>(cg[i]);
      return norm;
    }

    template <typename store_t>
    cvector<array<real_t, 4>> quadrupleCGReduction_impl(cvector_ref<const ColorSpinorField> &x,
                                                        cvector_ref<const ColorSpinorField> &y,
                                                        cvector_ref<const ColorSpinorField> &z)
    {
      vector<array<real_t, 4>> norm(x.size());
      auto cg = instantiateReduce<store_t, quadrupleCGReduction_, false>(cvector<real_t>(0.0), cvector<real_t>(0.0),
                                                                         cvector<real_t>(0.0), x, y, z, x, x);
      for (auto i = 0u; i < x.size(); i++) norm[i] = reduction_to_array<4>(cg[i]);
      return norm;
    }

    template <typename store_t>
    cvector<real_t> quadrupleCG3InitNorm_impl(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                                              cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                              cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return to_real_vector(instantiateReduce<store_t, quadrupleCG3InitNorm_, false>(
        a, cvector<real_t>(0.0), cvector<real_t>(0.0), x, y, z, w, v));
    }

    template <typename store_t>
    cvector<real_t> quadrupleCG3UpdateNorm_impl(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<ColorSpinorField> &x,
                                                cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                                                cvector_ref<ColorSpinorField> &w, cvector_ref<const ColorSpinorField> &v)
    {
      return to_real_vector(
        instantiateReduce<store_t, quadrupleCG3UpdateNorm_, false>(a, b, cvector<real_t>(0.0), x, y, z, w, v));
    }

  } // namespace blas

} // namespace quda
