#pragma once

#include <memory>

#include <gauge_field.h>

namespace quda
{

  /**
     @brief Build the Kahler-Dirac inverse block for KD operators.
     @param Xinv[out] Resulting Kahler-Dirac inverse (assumed allocated)
     @param gauge[in] Original fine gauge field
     @param mass [in] Mass of the original staggered operator w/out factor of 2 convention
     @param dagger_approximation[in] Whether or not to use the dagger approximation, using the dagger of X instead of Xinv
     @param verify [in] Whether or not to verify the result as a numerical stability test
  */
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const GaugeField &gauge, double mass,
                                        bool dagger_approximation, bool verify = false);

  /**
     @brief Perform the reordering of the Kahler-Dirac inverse block from a coarse scalar field to a KD geometry gauge field
     @param xInvFineLayout[out] Kahler-Dirac inverse in KD geometry gauge field
     @param xInvCoarseLayout[in] Kahler-Dirac inverse in coarse geometry MILC layout
     @param dagger_approximation[in] Whether or not we're doing the dagger approximation, where you pass in X instead
     @param mass [in] Mass of the original staggered operator w/out factor of 2 convention, needed for dagger approx
  */
  void ReorderStaggeredKahlerDiracInverse(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout,
                                          bool dagger_approximation, double mass);

} // namespace quda
