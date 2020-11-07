#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/contraction.cuh>

namespace quda {

  // Non-sliced contraction type kernels
  //-----------------------------------------------------------------------------
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
      x(x),
      y(y),
      result(static_cast<complex<Float>*>(result)),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: strcat(aux, "open,"); break;
      case QUDA_CONTRACT_TYPE_DR: strcat(aux, "degrand-rossi,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ContractionArg<Float, nColor> arg(x, y, result);
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: launch<ColorContract>(tp, stream, arg); break;
      case QUDA_CONTRACT_TYPE_DR:   launch<DegrandRossiContract>(tp, stream, arg); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
    }

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN)
        return 16 * 3 * 6ll * x.Volume();
      else
        return ((16 * 3 * 6ll) + (16 * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType)
  {
#ifdef GPU_CONTRACT
    checkPrecision(x, y);
    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    instantiate<Contraction>(x, y, result, cType);
#else
    errorQuda("Contraction code has not been built");
#endif
  }

  // Sliced contraction type kernels: Used to inherit from
  // TunableLocalParityReduction. Hacking this to inherit from TunableKernel2D
  // so it will compile
  //----------------------------------------------------------------------------
  template <typename Float, int nColor> class ContractionSlicedFT : TunableKernel2D
  {
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;
    const int *const source_position;
    const int *const mom_mode;
    const int s1;
    const int b1;
    unsigned int minThreads() const { return x.VolumeCB(); }
    
    // These need to be set accordingly in the new base class
    bool staticGridDim() const { return true; } // Maintain grid dims set in this class.
    bool tuneSharedBytes() const { return false; }
    void initTuneParam(TuneParam &param) const {
      TunableKernel2D::initTuneParam(param);
      param.block.y = 2;
      //param.grid.z = x.X(Arg::reduction_dim); // Reduction dimension is mapped to different blocks in the Z dimension
    }
    
    void defaultTuneParam(TuneParam &param) const {
      TunableKernel2D::defaultTuneParam(param);
      param.block.y = 2;
      //param.grid.z = x.X(Arg::reduction_dim); // Reduction dimension is mapped to different blocks in the Z dimension
    }    
    
  public:
    ContractionSlicedFT(const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType,
			const int *const source_position, const int *const mom_mode,
			const size_t s1, const size_t b1) :
      TunableKernel2D(x, 2),
      x(x),
      y(y),
      cType(cType),
      source_position(source_position),
      mom_mode(mom_mode),
      s1(s1),
      b1(b1)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN_SUM_T: strcat(aux, "open-sum-t,"); break;
      case QUDA_CONTRACT_TYPE_OPEN_SUM_Z: strcat(aux, "open-sum-z,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_T: strcat(aux, "degrand-rossi-ft-t,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_Z: strcat(aux, "degrand-rossi-ft-z,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ContractionSlicedFTArg<Float, nColor> arg_ft_t(x, y, source_position, mom_mode, s1, b1, 3);
      ContractionSlicedFTArg<Float, nColor> arg_ft_z(x, y, source_position, mom_mode, s1, b1, 2);      
      switch (cType) {
      case QUDA_CONTRACT_TYPE_DR_FT_T:
	launch<DegrandRossiContractFT>(tp, stream, arg_ft_t); break;
      case QUDA_CONTRACT_TYPE_DR_FT_Z:
	launch<DegrandRossiContractFT>(tp, stream, arg_ft_z); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
    }

    //void preTune() {}
    //void postTune() {}
    
    long long flops() const
    {
      return ((x.Nspin() * x.Nspin() * 3 * 6ll) + (x.Nspin() * x.Nspin() * (4 + 12))) * x.Volume();
    }
    
    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

  void contractSlicedFTQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result,
			    const QudaContractType cType, const int *const source_position,
			    const int *const mom_mode, const size_t s1, const size_t b1)
  {
#ifdef GPU_CONTRACT
    checkPrecision(x, y);
    
    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    instantiate<ContractionSlicedFT>(x, y, cType, source_position, mom_mode, s1, b1);
    // Reduction complete goes here?
    
#else
    errorQuda("Contraction code has not been built");
#endif
  }
} // namespace quda
//----------------------------------------------------------------------------







