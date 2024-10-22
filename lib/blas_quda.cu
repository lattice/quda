#include <blas_quda.h>
#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/blas_core.cuh>

namespace quda {

  namespace blas {

    template <template <typename real> class Functor, typename store_t, typename y_store_t, int nSpin, typename coeff_t>
    class Blas : public TunableGridStrideKernel3D
    {
      using real = typename mapper<y_store_t>::type;
      Functor<real> f;
      const int nParity; // for composite fields this includes the number of composites

      coeff_t a, b, c;
      cvector_ref<ColorSpinorField> &x, &y, &z, &w, &v;

      bool tuneSharedBytes() const override { return false; }
      // for these streaming kernels, there is no need to tune the grid size, just use max
      unsigned int minGridSize() const override { return maxGridSize(); }

    public:
      template <typename Vx, typename Vy, typename Vz, typename Vw, typename Vv>
      Blas(const coeff_t &a, const coeff_t &b, const coeff_t &c, Vx &x, Vy &y, Vz &z, Vw &w, Vv &v) :
        TunableGridStrideKernel3D(x[0], x.size(), (x[0].IsComposite() ? x[0].CompositeDim() : 1) * x.SiteSubset()),
        f(a, b, c),
        nParity(vector_length_z),
        a(a),
        b(b),
        c(c),
        x(reinterpret_cast<cvector_ref<ColorSpinorField> &>(x)),
        y(reinterpret_cast<cvector_ref<ColorSpinorField> &>(y)),
        z(reinterpret_cast<cvector_ref<ColorSpinorField> &>(z)),
        w(reinterpret_cast<cvector_ref<ColorSpinorField> &>(w)),
        v(reinterpret_cast<cvector_ref<ColorSpinorField> &>(v))
      {
        if (a.size() != x.size()) this->a.resize(x.size(), a.size() == 1 ? a[0] : 0.0);
        if (b.size() != x.size()) this->b.resize(x.size(), b.size() == 1 ? b[0] : 0.0);
        if (c.size() != x.size()) this->c.resize(x.size(), c.size() == 1 ? c[0] : 0.0);
        check_size(this->a, this->b, this->c, x, y, z, w, v);
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
        setRHSstring(aux, x.size());

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

      long long flops() const override { return f.flops() * x.Length() * x.size(); }
      long long bytes() const override
      {
        return (f.read.X + f.write.X) * x.Bytes() + (f.read.Y + f.write.Y) * y.Bytes() +
          (f.read.Z + f.write.Z) * z.Bytes() + (f.read.W + f.write.W) * w.Bytes() + (f.read.V + f.write.V) * v.Bytes();
      }
    };

    // split the fields and recurse if needed
    template <template <typename real> class Functor, bool mixed, typename coeff_t, typename X, typename Y, typename Z,
              typename W, typename V>
    void instantiateBlas(const coeff_t &a, const coeff_t &b, const coeff_t &c, X &x, Y &y, Z &z, W &w, V &v)
    {
      if (x.size() > get_max_multi_rhs()) {
        instantiateBlas<Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin(), x.begin() + x.size() / 2}, {y.begin(), y.begin() + y.size() / 2},
          {z.begin(), z.begin() + z.size() / 2}, {w.begin(), w.begin() + w.size() / 2},
          {v.begin(), v.begin() + v.size() / 2});
        instantiateBlas<Functor, mixed, coeff_t, X, Y, Z, W, V>(
          a, b, c, {x.begin() + x.size() / 2, x.end()}, {y.begin() + y.size() / 2, y.end()},
          {z.begin() + z.size() / 2, z.end()}, {w.begin() + w.size() / 2, w.end()}, {v.begin() + v.size() / 2, v.end()});
        return;
      }

      instantiate<Functor, Blas, mixed>(a, b, c, x, y, z, w, v);
    }

    void axpbyz(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      instantiateBlas<axpbyz_, true>(a, b, cvector<double>(), x, y, x, x, z);
    }

    void axy(const cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      instantiateBlas<axy_, false>(a, cvector<Complex>(), cvector<Complex>(), x, y, y, y, y);
    }

    void caxpy(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      instantiateBlas<caxpy_, true>(a, cvector<Complex>(), cvector<Complex>(), x, y, x, x, y);
    }

    void caxpby(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector<Complex> &b,
                cvector_ref<ColorSpinorField> &y)
    {
      instantiateBlas<caxpby_, false>(a, b, cvector<Complex>(), x, y, x, x, y);
    }

    void axpbypczw(cvector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector<double> &b,
                   cvector_ref<const ColorSpinorField> &y, cvector<double> &c, cvector_ref<const ColorSpinorField> &z,
                   cvector_ref<ColorSpinorField> &w)
    {
      instantiateBlas<axpbypczw_, false>(a, b, c, x, y, z, w, y);
    }

    void cxpaypbz(cvector_ref<const ColorSpinorField> &x, cvector<Complex> &a, cvector_ref<const ColorSpinorField> &y,
                  cvector<Complex> &b, cvector_ref<ColorSpinorField> &z)
    {
      instantiateBlas<cxpaypbz_, false>(a, b, cvector<Complex>(), x, y, z, x, y);
    }

    void axpyBzpcx(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<double> &b, cvector_ref<const ColorSpinorField> &z, cvector<double> &c)
    {
      instantiateBlas<axpyBzpcx_, true>(a, b, c, x, y, z, x, y);
    }

    void axpyZpbx(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                  cvector_ref<const ColorSpinorField> &z, cvector<double> &b)
    {
      instantiateBlas<axpyZpbx_, true>(a, b, cvector<double>(), x, y, z, x, y);
    }

    void caxpyBzpx(cvector<Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<Complex> &b, cvector_ref<const ColorSpinorField> &z)
    {
      instantiateBlas<caxpyBzpx_, true>(a, b, cvector<Complex>(), x, y, z, x, y);
    }

    void caxpyBxpz(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<Complex> &b, cvector_ref<ColorSpinorField> &z)
    {
      instantiateBlas<caxpyBxpz_, true>(a, b, cvector<Complex>(), x, y, z, x, y);
    }

    void caxpbypzYmbw(cvector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector<Complex> &b,
                      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                      cvector_ref<const ColorSpinorField> &w)
    {
      instantiateBlas<caxpbypzYmbw_, false>(a, b, cvector<Complex>(), x, y, z, w, y);
    }

    void cabxpyAx(cvector<double> &ar, cvector<Complex> &b, cvector_ref<ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y)
    {
      vector<Complex> a(ar.size());
      for (auto i = 0u; i < ar.size(); i++) a[i] = Complex(ar[i]);
      instantiateBlas<cabxpyAx_, false>(a, b, cvector<Complex>(), x, y, x, x, y);
    }

    void caxpyXmaz(cvector<Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const ColorSpinorField> &z)
    {
      instantiateBlas<caxpyxmaz_, false>(a, cvector<Complex>(), cvector<Complex>(), x, y, z, x, y);
    }

    void caxpyXmazMR(cvector<double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z)
    {
      if (!commAsyncReduction())
	errorQuda("This kernel requires asynchronous reductions to be set");
      if (x.Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("This kernel cannot be run on CPU fields");
      instantiateBlas<caxpyxmazMR_, false>(a, cvector<double>(), cvector<double>(), x, y, z, y, y);
    }

    void tripleCGUpdate(cvector<double> &a, cvector<double> &b, cvector_ref<const ColorSpinorField> &x,
                        cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                        cvector_ref<ColorSpinorField> &w)
    {
      instantiateBlas<tripleCGUpdate_, true>(a, b, cvector<double>(), x, y, z, w, y);
    }

  } // namespace blas

} // namespace quda
