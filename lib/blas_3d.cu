#include <color_spinor_field.h>
#include <kernels/blas_3d.cuh>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <blas_3d.h>
#include <instantiate.h>

namespace quda
{

  namespace blas3d
  {

    template <typename Float, int nColor> class copy3D : TunableKernel2D
    {
      ColorSpinorField &y;
      ColorSpinorField &x;
      const int t_slice;
      const copyType type;
      unsigned int minThreads() const { return y.VolumeCB(); }

    public:
      copy3D(ColorSpinorField &y, ColorSpinorField &x, int t_slice, copyType type) :
        TunableKernel2D(y, y.SiteSubset()), y(y), x(x), t_slice(t_slice), type(type)
      {
        // Check slice value
        if (t_slice < 0 || t_slice >= y.X()[3]) errorQuda("Unexpected slice %d", t_slice);

        strcat(aux, type == copyType::SWAP_3D ? ",swap_3d" : type == copyType::COPY_TO_3D ? ",to_3d" : ",from_3d");
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        copy3dArg<Float, nColor> arg(y, x, t_slice);
        switch (type) {
        case copyType::COPY_TO_3D: launch<copyTo3d>(tp, stream, arg); break;
        case copyType::COPY_FROM_3D: launch<copyFrom3d>(tp, stream, arg); break;
        case copyType::SWAP_3D: launch<swap3d>(tp, stream, arg); break;
        default: errorQuda("Unknown 3D copy type");
        }
      }

      void preTune()
      {
        x.backup();
        y.backup();
      }
      void postTune()
      {
        x.restore();
        y.restore();
      }
      long long bytes() const
      {
        return (type == copyType::SWAP_3D ? 2 : 1) * (x.Bytes() / x.X(3) + y.Bytes() / y.X(3));
      }
    };

    void copy(const int slice, const copyType type, ColorSpinorField &x, ColorSpinorField &y)
    {
      checkPrecision(x, y);
      checkSpin(x, y);
      checkColor(x, y);
      // Check orth dim
      if (x.X()[3] != 1) errorQuda("Unexpected dimensions in x[3]=%d", x.X()[3]);
      // We must give a 4D Lattice field as the first argument
      instantiate<copy3D>(y, x, slice, type);
    }

    void swap(int slice, ColorSpinorField &x, ColorSpinorField &y)
    {
      checkPrecision(x, y);
      checkSpin(x, y);
      checkColor(x, y);
      instantiate<copy3D>(x, y, slice, copyType::SWAP_3D);
    }

    template <typename Float, int nColor> class axpby3D : TunableKernel2D
    {
      const ColorSpinorField &x;
      ColorSpinorField &y;
      const std::vector<double> &a;
      const std::vector<double> &b;
      unsigned int minThreads() const override { return x.VolumeCB(); }

    public:
      axpby3D(const ColorSpinorField &x, ColorSpinorField &y, const std::vector<double> &a, const std::vector<double> &b) :
        TunableKernel2D(x, x.SiteSubset()), x(x), y(y), a(a), b(b)
      {
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<axpby3d>(tp, stream, axpby3dArg<Float, nColor>(a, x, b, y));
      }

      void preTune() override { y.backup(); }
      void postTune() override { y.restore(); }
      long long flops() const override { return 6 * x.Volume() * x.Nspin() * x.Ncolor(); }
      long long bytes() const override { return x.Bytes() + 2 * y.Bytes(); }
    };

    void axpby(const std::vector<double> &a, const ColorSpinorField &x, const std::vector<double> &b, ColorSpinorField &y)
    {
      checkPrecision(x, y);
      checkSpin(x, y);
      checkColor(x, y);

      // Check coefficients
      if (a.size() != b.size() && a.size() != (unsigned int)x.X()[3])
        errorQuda("Unexpected coeff array sizes a=%lu b=%lu, x[3]=%d", a.size(), b.size(), x.X()[3]);

      // We must give a Lattice field as the first argument
      instantiate<axpby3D>(x, y, a, b);
    }

    void ax(const std::vector<double> &a, ColorSpinorField &x)
    {
      std::vector<double> zeros(a.size(), 0.0);
      axpby(a, x, zeros, x);
    }

    template <typename Float, int nColor> class caxpby3D : TunableKernel2D
    {
      const ColorSpinorField &x;
      ColorSpinorField &y;
      const std::vector<Complex> &a;
      const std::vector<Complex> &b;
      unsigned int minThreads() const override { return x.VolumeCB(); }

    public:
      caxpby3D(const ColorSpinorField &x, ColorSpinorField &y, const std::vector<Complex> &a,
               const std::vector<Complex> &b) :
        TunableKernel2D(x, x.SiteSubset()), x(x), y(y), a(a), b(b)
      {
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<caxpby3d>(tp, stream, caxpby3dArg<Float, nColor>(a, x, b, y));
      }

      void preTune() override { y.backup(); }
      void postTune() override { y.restore(); }
      long long flops() const override { return 14 * x.Volume() * x.Nspin() * x.Ncolor(); }
      long long bytes() const override { return x.Bytes() + 2 * y.Bytes(); }
    };

    void caxpby(const std::vector<Complex> &a, const ColorSpinorField &x, const std::vector<Complex> &b,
                ColorSpinorField &y)
    {
      checkPrecision(x, y);
      checkSpin(x, y);
      checkColor(x, y);

      // Check coefficients
      if (a.size() != b.size() && a.size() != (unsigned int)x.X()[3])
        errorQuda("Unexpected coeff array sizes a=%lu b=%lu, x[3]=%d", a.size(), b.size(), x.X()[3]);

      // We must give a Lattice field as the first argument
      instantiate<caxpby3D>(x, y, a, b);
    }

    template <typename Float, int nColor> class reDotProduct3D : TunableMultiReduction
    {
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      std::vector<double> &result;

    public:
      reDotProduct3D(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<double> &result) :
        TunableMultiReduction(x, x.SiteSubset(), x.X()[3]), x(x), y(y), result(result)
      {
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        reDotProduct3dArg<Float, nColor> arg(x, y);
        launch<reDotProduct3d>(result, tp, stream, arg);
      }

      long long flops() const { return x.Volume() * x.Nspin() * x.Ncolor() * 2; }
      long long bytes() const { return x.Bytes() + y.Bytes(); }
    };

    void reDotProduct(std::vector<double> &result, const ColorSpinorField &x, const ColorSpinorField &y)
    {
      checkSpin(x, y);
      checkColor(x, y);

      // Check coefficients
      if (result.size() != (unsigned int)x.X()[3])
        errorQuda("Unexpected coeff array size a=%lu, x[3]=%d", result.size(), x.X()[3]);

      // We must give a Lattice field as the first argument
      instantiate<reDotProduct3D>(x, y, result);
    }

    template <typename Float, int nColor> class cDotProduct3D : TunableMultiReduction
    {
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      std::vector<Complex> &result;

    public:
      cDotProduct3D(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result) :
        TunableMultiReduction(x, x.SiteSubset(), x.X()[3]), x(x), y(y), result(result)
      {
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        cDotProduct3dArg<Float, nColor> arg(x, y);
        launch<cDotProduct3d>(result, tp, stream, arg);
      }

      long long flops() const { return x.Volume() * x.Nspin() * x.Ncolor() * 8; }
      long long bytes() const { return x.Bytes() + y.Bytes(); }
    };

    void cDotProduct(std::vector<Complex> &result, const ColorSpinorField &x, const ColorSpinorField &y)
    {
      checkSpin(x, y);
      checkColor(x, y);

      // Check coefficients
      if (result.size() != (unsigned int)x.X()[3])
        errorQuda("Unexpected coeff array size a=%lu, x[3]=%d", result.size(), x.X()[3]);

      // We must give a Lattice field as the first argument
      instantiate<cDotProduct3D>(x, y, result);
    }

  } // namespace blas3d

} // namespace quda
