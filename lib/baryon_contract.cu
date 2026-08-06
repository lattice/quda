#include <color_spinor_field.h>
#include <contract_quda.h>

#include <tunable_reduction.h>
#include <instantiate.h>
#include <kernels/baryon_contract.cuh>

namespace quda {

  template <typename Float, int nColor> class BaryonContractionSummed : TunableMultiReduction
  {
  protected:
    cvector_ref<const ColorSpinorField> &u;
    cvector_ref<const ColorSpinorField> &d;
    std::vector<Complex> &result_global;
    const int *const source_position;
    const int *const mom_mode;
    const QudaFFTSymmType *const fft_type;

  public:
    BaryonContractionSummed(cvector_ref<const ColorSpinorField> &u, cvector_ref<const ColorSpinorField> &d,
                            std::vector<Complex> &result_global, const int *const source_position,
                            const int *const mom_mode, const QudaFFTSymmType *const fft_type) :
      TunableMultiReduction(u[0], 1u, u.X(3)),
      u(u),
      d(d),
      result_global(result_global),
      source_position(source_position),
      mom_mode(mom_mode),
      fft_type(fft_type)
    {
      strcat(aux, "baryon-nucleon-ft-t,");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      constexpr int num_out_results = 16;
      std::vector<double> result_local(2 * num_out_results * u.X(3), 0.0);

      BaryonContractionSummedArg<Float, nColor> arg(u, d, source_position, mom_mode, fft_type);
      launch<NucleonContractFT>(result_local, tp, stream, arg);

      // Copy results back to host array
      if (!activeTuning()) {
        for (int i = 0; i < num_out_results * u.X(3); i++) {
          result_global[num_out_results * u.X(3) * comm_coord(3) + i].real(result_local[2 * i]);
          result_global[num_out_results * u.X(3) * comm_coord(3) + i].imag(result_local[2 * i + 1]);
        }
      }
    }

    long long flops() const
    {
      // 36 color permutation pairs, each dominated by ~144 complex multiply-adds
      return 36ll * 144 * 8 * u[0].Volume();
    }

    long long bytes() const { return u.size() * (u[0].Bytes() + d[0].Bytes()); }
  };

  void baryonContractSummedQuda(cvector_ref<const ColorSpinorField> &u, cvector_ref<const ColorSpinorField> &d,
                                std::vector<Complex> &result_global, QudaContractType cType,
                                const int *const source_position, const int *const mom_mode,
                                const QudaFFTSymmType *const fft_type)
  {
    if (cType != QUDA_CONTRACT_TYPE_BARYON_NUCLEON_FT_T) errorQuda("Unexpected contraction type %d", cType);
    if (u.size() != 12 || d.size() != 12)
      errorQuda("Baryon contraction requires 12 propagator components per flavor (u=%lu d=%lu)", u.size(), d.size());

    for (auto i = 0u; i < u.size(); i++) {
      checkPrecision(u[i], d[i], u[0]);
      if (u[i].Nspin() != 4 || d[i].Nspin() != 4)
        errorQuda("Expected four-spinors u=%d d=%d", u[i].Nspin(), d[i].Nspin());
      if (u[i].Ncolor() != 3 || d[i].Ncolor() != 3)
        errorQuda("Unexpected number of colors u=%d d=%d", u[i].Ncolor(), d[i].Ncolor());
      if (u[i].GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || d[i].GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
        errorQuda("Unexpected gamma basis u=%d d=%d", u[i].GammaBasis(), d[i].GammaBasis());
    }

    instantiate<BaryonContractionSummed>(u, d, result_global, source_position, mom_mode, fft_type);
  }

} // namespace quda
