#include <color_spinor_field.h>
#include <contract_quda.h>

#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/contraction.cuh>

namespace quda {

  // Summed contraction type kernels.
  template <typename Float, int nColor> class ContractionSummed : TunableMultiReduction
  {
  protected:
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    std::vector<Complex> &result_global;
    const QudaContractType cType;
    const int *const source_position;
    const int *const mom_mode;
    const QudaFFTSymmType *const fft_type;
    const size_t s1;
    const size_t b1;

  public:
    ContractionSummed(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result_global,
                      const QudaContractType cType, const int *const source_position, const int *const mom_mode,
                      const QudaFFTSymmType *const fft_type, const size_t s1, const size_t b1) :
      TunableMultiReduction(
        x, 1u, x.X()[cType == QUDA_CONTRACT_TYPE_DR_FT_Z || cType == QUDA_CONTRACT_TYPE_OPEN_SUM_Z ? 2 : 3]),
      x(x),
      y(y),
      result_global(result_global),
      cType(cType),
      source_position(source_position),
      mom_mode(mom_mode),
      fft_type(fft_type),
      s1(s1),
      b1(b1)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN_SUM_T: strcat(aux, "open-sum-t,"); break;
      case QUDA_CONTRACT_TYPE_OPEN_SUM_Z: strcat(aux, "open-sum-z,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_T: strcat(aux, "degrand-rossi-ft-t,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_Z: strcat(aux, "degrand-rossi-ft-z,"); break;
      case QUDA_CONTRACT_TYPE_STAGGERED_FT_T: strcat(aux, "staggered-ft-t,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      int reduction_dim = 3;
      const int nSpinSq = x.Nspin() * x.Nspin();

      if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z) reduction_dim = 2;
      std::vector<double> result_local(2 * nSpinSq * x.X()[reduction_dim], 0.0);

      // Pass the integer value of the redection dim as a template arg
      switch (cType) {
      case QUDA_CONTRACT_TYPE_DR_FT_T: {
        constexpr int nSpin = 4;
        constexpr int ft_dir = 3;
        ContractionSummedArg<Float, nColor, nSpin, ft_dir> arg(x, y, source_position, mom_mode, fft_type, s1, b1);
        launch<DegrandRossiContractFT>(result_local, tp, stream, arg);
      } break;
      case QUDA_CONTRACT_TYPE_DR_FT_Z: {
        constexpr int nSpin = 4;
        constexpr int ft_dir = 2;
        ContractionSummedArg<Float, nColor, nSpin, ft_dir> arg(x, y, source_position, mom_mode, fft_type, s1, b1);
        launch<DegrandRossiContractFT>(result_local, tp, stream, arg);
      } break;
      case QUDA_CONTRACT_TYPE_STAGGERED_FT_T: {
        constexpr int nSpin = 1;
        constexpr int ft_dir = 3;
        ContractionSummedArg<Float, nColor, nSpin, ft_dir, staggered_spinor_array> arg(x, y, source_position, mom_mode,
                                                                                       fft_type, s1, b1);
        launch<StaggeredContractFT>(result_local, tp, stream, arg);
      } break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }

      // Copy results back to host array
      if (!activeTuning()) {
        for (int i = 0; i < nSpinSq * x.X()[reduction_dim]; i++) {
          result_global[nSpinSq * x.X()[reduction_dim] * comm_coord(reduction_dim) + i].real(result_local[2 * i]);
          result_global[nSpinSq * x.X()[reduction_dim] * comm_coord(reduction_dim) + i].imag(result_local[2 * i + 1]);
        }
      }
    }

    long long flops() const // DMH: Restore const qualifier for warning suppression
    {
      return ((x.Nspin() * x.Nspin() * x.Ncolor() * 6ll) + (x.Nspin() * x.Nspin() * (x.Nspin() + x.Nspin() * x.Ncolor())))
        * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

  void contractSummedQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result_global,
                          const QudaContractType cType, const int *const source_position, const int *const mom_mode,
                          const QudaFFTSymmType *const fft_type, const size_t s1, const size_t b1)
  {
    checkPrecision(x, y);
    if (x.Nspin() != y.Nspin())
      errorQuda("Contraction between unequal number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    if (x.Ncolor() != y.Ncolor())
      errorQuda("Contraction between unequal number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (cType != QUDA_CONTRACT_TYPE_STAGGERED_FT_T) {
      if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Expected four-spinors x=%d y=%d", x.Nspin(), y.Nspin());
      if (x.GammaBasis() != y.GammaBasis())
        errorQuda("Contracting spinors in different gamma bases x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    }
    if (cType == QUDA_CONTRACT_TYPE_DR_FT_T || cType == QUDA_CONTRACT_TYPE_DR_FT_Z) {
      if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
        errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    }
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());

    instantiate<ContractionSummed>(x, y, result_global, cType, source_position, mom_mode, fft_type, s1, b1);
  }

  template <typename Float, int nColor> class Contraction : TunableKernel2D
  {
    complex<Float> *result;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;
    unsigned int minThreads() const { return x.VolumeCB(); }

public:
    Contraction(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType) :
      TunableKernel2D(x, 2),
      result(static_cast<complex<Float>*>(result)),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: strcat(aux, "open,"); break;
      case QUDA_CONTRACT_TYPE_DR: strcat(aux, "degrand-rossi,"); break;
      case QUDA_CONTRACT_TYPE_STAGGERED: strcat(aux, "staggered,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN:
      case QUDA_CONTRACT_TYPE_DR: {
        constexpr int nSpin = 4;
        constexpr bool spin_project = true;
        ContractionArg<Float, nSpin, nColor, spin_project> arg(x, y, result);
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN: launch<ColorContract>(tp, stream, arg); break;
        case QUDA_CONTRACT_TYPE_DR: launch<DegrandRossiContract>(tp, stream, arg); break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
      }; break;
      case QUDA_CONTRACT_TYPE_STAGGERED: {
        constexpr int nSpin = 1;
        constexpr bool spin_project = false;
        ContractionArg<Float, nSpin, nColor, spin_project> arg(x, y, result);

        launch<StaggeredContract>(tp, stream, arg);
      }; break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
    }

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN)
        return x.Nspin() * x.Nspin() * x.Ncolor() * 6ll * x.Volume();
      else
        return ((x.Nspin() * x.Nspin() * x.Ncolor() * 6ll)
                + (x.Nspin() * x.Nspin() * (x.Nspin() + x.Nspin() * x.Ncolor())))
          * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    checkPrecision(x, y);
    if (x.Nspin() != y.Nspin())
      errorQuda("Contraction between unequal number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    if (x.Ncolor() != y.Ncolor())
      errorQuda("Contraction between unequal number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (cType == QUDA_CONTRACT_TYPE_OPEN || cType == QUDA_CONTRACT_TYPE_DR) {
      if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Expected four-spinors x=%d y=%d", x.Nspin(), y.Nspin());
      if (x.GammaBasis() != y.GammaBasis())
        errorQuda("Contracting spinors in different gamma bases x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    }
    if (cType == QUDA_CONTRACT_TYPE_DR) {
      if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
        errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    }

    instantiate<Contraction>(x, y, result, cType);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
