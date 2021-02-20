#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/color_contract.cuh>

namespace quda {

  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class ColorContract : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    complex<Float> *result;

    unsigned int minThreads() const { return x.VolumeCB(); }
    
  public:
    ColorContract(const ColorSpinorField &x, const ColorSpinorField &y, void *result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorContractArg<Float, nColor> arg(x, y, result);
      launch<ColorContraction>(tp, stream, arg);
    }

    long long flops() const
    {
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 1 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(Complex);
    }
  };

#ifdef GPU_CONTRACT
  void colorContractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result)
  {
    checkPrecision(x, y);
    if (x.Nspin() != 1 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    instantiate<ColorContract>(x, y, result);
  }
#else
  void colorContractQuda(const ColorSpinorField &, const ColorSpinorField &, void *)
  {
    errorQuda("Contraction code has not been built");
  }
#endif
  //----------------------------------------------------------------------------

  template <typename Float, int nColor> class ColorCross : TunableKernel2D
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    ColorSpinorField &result;

    unsigned int minThreads() const { return x.VolumeCB(); }

  public:
    ColorCross(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      result(result)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ColorCrossArg<Float, nColor> arg(x, y, result);      
      launch<ColorCrossCompute>(tp, stream, arg);
    }
    
    long long flops() const
    {
      // 1 spin, 3 color, 6 complex, lattice volume
      return 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

#ifdef GPU_CONTRACT  
  void colorCrossQuda(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result)
  {
    checkPrecision(x, y);
    checkPrecision(result, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3 || result.Ncolor() != 3) errorQuda("Unexpected number of colors x = %d y = %d result = %d", x.Ncolor(), y.Ncolor(), result.Ncolor());
    if (x.Nspin() != 1 || y.Nspin() != 1 || result.Nspin() != 1) errorQuda("Unexpected number of spins x = %d y = %d result = %d", x.Nspin(), y.Nspin(), result.Nspin());

    instantiate<ColorCross>(x, y, result);

  }
#else
  void colorCrossQuda(const ColorSpinorField &, const ColorSpinorField &, ColorSpinorField &)
  {
    errorQuda("Contraction code has not been built");
  }
#endif
}// namespace quda
