#pragma once

#include <gauge_field.h>

namespace quda
{

  /**
     @brief Build the Kahler-Dirac inverse block for KD operators.
     @param[out] out Xinv resulting Kahler-Dirac inverse (assumed allocated)
     @param[in] in gauge original fine gauge field
     @param[in] in mass the mass of the original staggered operator w/out factor of 2 convention
  */
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass);

  /**
     @brief Allocate and build the Kahler-Dirac inverse block for KD operators
     @param[in] in gauge original fine gauge field
     @param[in] in mass the mass of the original staggered operator w/out factor of 2 convention
     @param[in] in precision of Xinv field
     @return constructed Xinv, which needs to be deleted manually
  */
  cudaGaugeField *AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass,
                                                              const QudaPrecision override_prec);

  // Note: see routine
  // void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv,
  // bool dagger); in dslash_quda.h as it is relevant for applying the above op.
} // namespace quda
