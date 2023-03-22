#include <blas_quda.h>
#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/blas_core.cuh>

namespace quda {

  namespace blas {

    unsigned long long flops;
    unsigned long long bytes;

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

        blas::bytes += bytes();
        blas::flops += flops();
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

    void axpbyz(double a, const ColorSpinorField &x, double b, const ColorSpinorField &y, ColorSpinorField &z)
    {
      instantiate<axpbyz_, Blas, true>(a, b, 0.0, x, y, x, x, z);
    }

    void axy(double a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      instantiate<axy_, Blas, false>(a, 0.0, 0.0, x, y, y, y, y);
    }

    void caxpy(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      instantiate<caxpy_, Blas, true>(a, Complex(0.0), Complex(0.0), x, y, x, x, y);
    }

    void caxpby(const Complex &a, const ColorSpinorField &x, const Complex &b, ColorSpinorField &y)
    {
      instantiate<caxpby_, Blas, false>(a, b, Complex(0.0), x, y, x, x, y);
    }

    void axpbypczw(double a, const ColorSpinorField &x, double b, const ColorSpinorField &y,
                   double c, const ColorSpinorField &z, ColorSpinorField &w)
    {
      instantiate<axpbypczw_, Blas, false>(a, b, c, x, y, z, w, y);
    }

    void cxpaypbz(const ColorSpinorField &x, const Complex &a, const ColorSpinorField &y,
                  const Complex &b, ColorSpinorField &z)
    {
      instantiate<cxpaypbz_, Blas, false>(a, b, Complex(0.0), x, y, z, x, y);
    }

    void axpyBzpcx(double a, ColorSpinorField& x, ColorSpinorField& y, double b, const ColorSpinorField& z, double c)
    {
      instantiate<axpyBzpcx_, Blas, true>(a, b, c, x, y, z, x, y);
    }

    void axpyZpbx(double a, ColorSpinorField& x, ColorSpinorField& y, const ColorSpinorField& z, double b)
    {
      instantiate<axpyZpbx_, Blas, true>(a, b, 0.0, x, y, z, x, y);
    }

    void caxpyBzpx(const Complex &a, ColorSpinorField &x, ColorSpinorField &y,
                   const Complex &b, const ColorSpinorField &z)
    {
      instantiate<caxpyBzpx_, Blas, true>(a, b, Complex(0.0), x, y, z, x, y);
    }

    void caxpyBxpz(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y,
                   const Complex &b, ColorSpinorField &z)
    {
      instantiate<caxpyBxpz_, Blas, true>(a, b, Complex(0.0), x, y, z, x, y);
    }

    void caxpbypzYmbw(const Complex &a, const ColorSpinorField &x, const Complex &b,
                      ColorSpinorField &y, ColorSpinorField &z, const ColorSpinorField &w)
    {
      instantiate<caxpbypzYmbw_, Blas, false>(a, b, Complex(0.0), x, y, z, w, y);
    }

    void cabxpyAx(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y)
    {
      instantiate<cabxpyAx_, Blas, false>(Complex(a), b, Complex(0.0), x, y, x, x, y);
    }

    void caxpyXmaz(const Complex &a, ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z)
    {
      instantiate<caxpyxmaz_, Blas, false>(a, Complex(0.0), Complex(0.0), x, y, z, x, y);
    }

    void caxpyXmazMR(const double &a, ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z)
    {
      if (!commAsyncReduction())
	errorQuda("This kernel requires asynchronous reductions to be set");
      if (x.Location() == QUDA_CPU_FIELD_LOCATION)
	errorQuda("This kernel cannot be run on CPU fields");
      instantiate<caxpyxmazMR_, Blas, false>(a, 0.0, 0.0, x, y, z, y, y);
    }

    void tripleCGUpdate(double a, double b, const ColorSpinorField &x, ColorSpinorField &y,
                        ColorSpinorField &z, ColorSpinorField &w)
    {
      instantiate<tripleCGUpdate_, Blas, true>(a, b, 0.0, x, y, z, w, y);
    }

  } // namespace blas

} // namespace quda
