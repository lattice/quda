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
  */
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass,
                                        const bool dagger_approximation);

  /**
     @brief Perform the reordering of the Kahler-Dirac inverse block from a coarse scalar field to a KD geometry gauge field
     @param xInvFineLayout[out] Kahler-Dirac inverse in KD geometry gauge field
     @param xInvCoarseLayout[in] Kahler-Dirac inverse in coarse geometry MILC layout
     @param dagger_approximation[in] Whether or not we're doing the dagger approximation, where you pass in X instead
     @param msas [in] Mass of the original staggered operator w/out factor of 2 convention, needed for dagger approx
  */
  void ReorderStaggeredKahlerDiracInverse(GaugeField &xInvFineLayout, const GaugeField &xInvCoarseLayout,
                                          const bool dagger_approximation, const double mass);

  /**
     @brief Allocate and build the Kahler-Dirac inverse block for KD operators
     @param gauge[in] Original fine gauge field
     @param mass[in] Mass of the original staggered operator w/out factor of 2 convention
     @param dagger_approximation[in] Whether or not to use the dagger approximation, using the dagger of X instead of Xinv
     @return constructed Xinv
  */
  std::shared_ptr<GaugeField> AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass,
                                                                          const bool dagger_approximation);

} // namespace quda
