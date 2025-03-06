#include "color_spinor_field.h"
#include "multigrid.h"
#include "kernels/dslash_gamma_helper_mg.cuh"
#include "tunable_nd.h"
#include "int_list.hpp"

namespace quda
{

  template <typename Float, int nColor, int nSpin> class CoarseChiralProjApply : TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const int proj;

    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    CoarseChiralProjApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int proj) :
      TunableKernel3D(in[0], in.size(), in.SiteSubset()), out(out), in(in), proj(proj)
    {
      strcat(aux, ",proj=");
      strcat(aux, proj == 1 ? "+1" : "-1");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CoarseChiralProj>(tp, stream, CoarseChiralProjArg<Float, nColor, nSpin>(out, in, proj));
    }

    long long bytes() const { return in.Bytes() / 2 + out.Bytes(); }
  };

  template <typename Float, int nSpin, int nColor, int... N>
  void ApplyCoarseChiralProj(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int proj, IntList<nColor, N...>)
  {
    if (out.Ncolor() == nColor) {
      CoarseChiralProjApply<Float, nColor, nSpin>(out, in, proj);
    } else {
      if constexpr (sizeof...(N) > 0)
        ApplyCoarseChiralProj<Float, nSpin>(out, in, proj, IntList<N...>());
      else
        errorQuda("nColor = %d not implemented", out.Ncolor());
    }
  }

  /**
     @brief Apply the coarse chiral projector to a coarse spinor.
     @param[out] out The result vector
     @param[in] in The input vector
     @param[in] proj +/-1 for the positive/negative projector
   */
  void ApplyCoarseChiralProj(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int proj)
  {
    checkColor(out, in);
    auto nSpin = checkSpin(out[0], in[0]);
    auto precision = checkPrecision(out, in);

    if (proj != 1 && proj != -1)
      errorQuda("Invalid projection direction %d, expected +/-1", proj);

    if (nSpin != 2)
      errorQuda("ApplyCoarseChiralProjector has not been enabled for nSpin = %d fields", nSpin);

    if constexpr (is_enabled_multigrid()) {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          // clang-format off
          ApplyCoarseChiralProj<double, 2>(out, in, proj, IntList<3, @QUDA_MULTIGRID_NVEC_LIST@>());
          // clang-format on
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        // clang-format off
        ApplyCoarseChiralProj<float, 2>(out, in, proj, IntList<3, @QUDA_MULTIGRID_NVEC_LIST@>());
        // clang-format on
      } else {
        errorQuda("Unsupported precision %d", out.Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
