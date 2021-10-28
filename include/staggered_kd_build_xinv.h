#pragma once

#include <memory>

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
     @brief Perform the reordering of the Kahler-Dirac inverse block from a coarse scalar field to a KD geometry gauge field
     @param[out] out Kahler-Dirac inverse in KD geometry gauge field
     @param[in] in Kahler-Dirac inverse in coarse geometry MILC layout
  */
  void ReorderStaggeredKahlerDiracInverse(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout);

  /**
     @brief Allocate and build the Kahler-Dirac inverse block for KD operators
     @param[in] in gauge original fine gauge field
     @param[in] in mass the mass of the original staggered operator w/out factor of 2 convention
     @return constructed Xinv
  */
  std::unique_ptr<GaugeField> AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass);

} // namespace quda
