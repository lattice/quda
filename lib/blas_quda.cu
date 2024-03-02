#include <blas_quda.h>
#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/blas_core.cuh>

namespace quda {

  namespace blas {

    template <template <typename real> class Functor, typename store_t, typename y_store_t,
              int nSpin, typename coeff_t>
    class Blas : public TunableGridStrideKernel2D
    {
      using real = typename mapper<y_store_t>::type;
      Functor<real> f;
      const int nParity; // for composite fields this includes the number of composites

      const coeff_t &a, &b, &c;
      ColorSpinorField &x, &y, &z, &w, &v;

      bool tuneSharedBytes() const override { return false; }
      // for these streaming kernels, there is no need to tune the grid size, just use max
      unsigned int minGridSize() const override { return maxGridSize(); }

    public:
      template <typename Vx, typename Vy, typename Vz, typename Vw, typename Vv>
      Blas(const coeff_t &a, const coeff_t &b, const coeff_t &c, Vx &x, Vy &y, Vz &z, Vw &w, Vv &v) :
        TunableGridStrideKernel2D(x, (x.IsComposite() ? x.CompositeDim() : 1) * x.SiteSubset()),
        f(a, b, c),
        nParity(vector_length_y),
        a(a),
        b(b),
        c(c),
        x(const_cast<ColorSpinorField&>(x)),
        y(const_cast<ColorSpinorField&>(y)),
        z(const_cast<ColorSpinorField&>(z)),
        w(const_cast<ColorSpinorField&>(w)),
        v(const_cast<ColorSpinorField&>(v))
      {
        checkLocation(x, y, z, w, v);
        checkLength(x, y, z, w, v);
        auto x_prec = checkPrecision(x, z, w);
        auto y_prec = checkPrecision(y, v);
        auto x_order = checkOrder(x, z, w);
        auto y_order = checkOrder(y, v);
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        if (x_prec != y_prec) {
          strcat(aux, ",");
          strcat(aux, y.AuxString().c_str());
        }

        apply(device::get_default_stream());
      }

      TuneKey tuneKey() const override { return TuneKey(vol, typeid(f).name(), aux); }

      void apply(const qudaStream_t &stream) override
      {
        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value;
        if (site_unroll_check && (x.Ncolor() != 3 || x.Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x.Nspin(), x.Ncolor());

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(x, y, z, w, v); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Functor<device_real_t> f_(a, b, c);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values 
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int threads = x.Length() / (nParity * M);

          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          BlasArg<device_real_t, M, device_store_t, N, device_y_store_t, Ny, decltype(f_)> arg(x, y, z, w, v, f_, threads, nParity);
          launch<Blas_>(tp, stream, arg);
        } else {
          if (checkOrder(x, y, z, w, v) != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER)
            errorQuda("CPU Blas functions expect AoS field order");

          using host_store_t = typename host_type_mapper<store_t>::type;
          using host_y_store_t = typename host_type_mapper<y_store_t>::type;
          using host_real_t = typename mapper<host_y_store_t>::type;
          Functor<host_real_t> f_(a, b, c);

          // redefine site_unroll with host_store types to ensure we have correct N/Ny/M values 
          constexpr bool site_unroll = !std::is_same<host_store_t, host_y_store_t>::value || isFixed<host_store_t>::value;
          constexpr int N = n_vector<host_store_t, false, nSpin, site_unroll>();
          constexpr int Ny = n_vector<host_y_store_t, false, nSpin, site_unroll>();
          constexpr int M = N; // if site unrolling then M=N will be 24/6, e.g., full AoS
          const int threads = x.Length() / (nParity * M);

          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
          BlasArg<host_real_t, M, host_store_t, N, host_y_store_t, Ny, decltype(f_)> arg(x, y, z, w, v, f_, threads, nParity);

          launch_host<Blas_>(tp, stream, arg);
        }
      }

      void preTune() override
      {
        if (f.write.X) x.backup();
        if (f.write.Y) y.backup();
        if (f.write.Z) z.backup();
        if (f.write.W) w.backup();
        if (f.write.V) v.backup();
      }

      void postTune() override
      {
        if (f.write.X) x.restore();
        if (f.write.Y) y.restore();
        if (f.write.Z) z.restore();
        if (f.write.W) w.restore();
        if (f.write.V) v.restore();
      }

      bool advanceTuneParam(TuneParam &param) const override
      {
        return location == QUDA_CPU_FIELD_LOCATION ? false : Tunable::advanceTuneParam(param);
      }

      long long flops() const override { return f.flops() * x.Length(); }
      long long bytes() const override
      {
        return (f.read.X + f.write.X) * x.Bytes() + (f.read.Y + f.write.Y) * y.Bytes() +
          (f.read.Z + f.write.Z) * z.Bytes() + (f.read.W + f.write.W) * w.Bytes() + (f.read.V + f.write.V) * v.Bytes();
      }
    };

    void axpbyz(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      check_size(a, x, b, y, z);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<axpbyz_, Blas, true>(a[i], b[i], 0.0, x[i], y[i], x[i], x[i], z[i]);
    }

    void axy(const cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, y);
      for (auto i = 0u; i < x.size(); i++) instantiate<axy_, Blas, false>(a[i], Complex(0.0), Complex(0.0), x[i], y[i], y[i], y[i], y[i]);
    }

    void caxpy(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, y);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpy_, Blas, true>(a[i], Complex(0.0), Complex(0.0), x[i], y[i], x[i], x[i], y[i]);
    }

    void caxpby(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector<Complex> &b,
                cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, x, b, y);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpby_, Blas, false>(a[i], b[i], Complex(0.0), x[i], y[i], x[i], x[i], y[i]);
    }

    void axpbypczw(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                   cvector_ref<const ColorSpinorField> &y, cvector<double> &c,
                   cvector_ref<const ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w)
    {
      check_size(a, x, b, y, c, z, w);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<axpbypczw_, Blas, false>(a[i], b[i], c[i], x[i], y[i], z[i], w[i], y[i]);
    }

    void cxpaypbz(cvector_ref<const ColorSpinorField> &x, cvector<Complex> &a,
                  cvector_ref<const ColorSpinorField> &y, cvector<Complex> &b, cvector_ref<ColorSpinorField> &z)
    {
      check_size(x, a, y, b, z);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<cxpaypbz_, Blas, false>(a[i], b[i], Complex(0.0), x[i], y[i], z[i], x[i], y[i]);
    }

    void axpyBzpcx(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<double> &b, cvector_ref<const ColorSpinorField> &z, cvector<double> &c)
    {
      check_size(a, x, y, b, z, c);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<axpyBzpcx_, Blas, true>(a[i], b[i], c[i], x[i], y[i], z[i], x[i], y[i]);
    }

    void axpyZpbx(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                  cvector_ref<const ColorSpinorField> &z, cvector<double> &b)
    {
      check_size(a, x, y, z, b);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<axpyZpbx_, Blas, true>(a[i], b[i], 0.0, x[i], y[i], z[i], x[i], y[i]);
    }

    void caxpyBzpx(cvector<Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<Complex> &b, cvector_ref<const ColorSpinorField> &z)
    {
      check_size(a, x, y, b, z);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpyBzpx_, Blas, true>(a[i], b[i], Complex(0.0), x[i], y[i], z[i], x[i], y[i]);
    }

    void caxpyBxpz(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y, cvector<Complex> &b, cvector_ref<ColorSpinorField> &z)
    {
      check_size(a, x, y, b, z);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpyBxpz_, Blas, true>(a[i], b[i], Complex(0.0), x[i], y[i], z[i], x[i], y[i]);
    }

    void caxpbypzYmbw(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector<Complex> &b, cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                      cvector_ref<const ColorSpinorField> &w)
    {
      check_size(a, x, b, y, z, w);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpbypzYmbw_, Blas, false>(a[i], b[i], Complex(0.0), x[i], y[i], z[i], w[i], y[i]);
    }

    void cabxpyAx(cvector<double> &a, cvector<Complex> &b, cvector_ref<ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y)
    {
      check_size(a, b, x, y);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<cabxpyAx_, Blas, false>(Complex(a[i]), b[i], Complex(0.0), x[i], y[i], x[i], x[i], y[i]);
    }

    void caxpyXmaz(cvector<Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const ColorSpinorField> &z)
    {
      check_size(a, x, y, z);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpyxmaz_, Blas, false>(a[i], Complex(0.0), Complex(0.0), x[i], y[i], z[i], x[i], y[i]);
    }

    void caxpyXmazMR(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z)
    {
      check_size(a, x, y, z);
      if (!commAsyncReduction())
	errorQuda("This kernel requires asynchronous reductions to be set");
      if (x[0].Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("This kernel cannot be run on CPU fields");
      for (auto i = 0u; i < x.size(); i++)
        instantiate<caxpyxmazMR_, Blas, false>(a[i], 0.0, 0.0, x[i], y[i], z[i], y[i], y[i]);
    }

    void tripleCGUpdate(cvector<double> &a, cvector<double> &b,
                        cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                        cvector_ref<ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w)
    {
      check_size(a, b, x, y, z, w);
      for (auto i = 0u; i < x.size(); i++)
        instantiate<tripleCGUpdate_, Blas, true>(a[i], b[i], 0.0, x[i], y[i], z[i], w[i], y[i]);
    }

  } // namespace blas

} // namespace quda
