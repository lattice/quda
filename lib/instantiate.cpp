#include <instantiate.h>

/**
   This file contains deinitions required when compiling with C++14.
   Without these, we can end up with undefined references at link
   time.  We can remove this file when we jump to C++17 and declare
   these are inline variables in instantiate.h.
 */

namespace quda
{

  // declared in instantiate.h
  constexpr std::array<QudaReconstructType, 6> ReconstructFull::recon;
#ifdef BUILD_OPENQCD_INTERFACE
  constexpr std::array<QudaReconstructType, 5> ReconstructWilson::recon;
#else
  constexpr std::array<QudaReconstructType, 3> ReconstructWilson::recon;
#endif
  constexpr std::array<QudaReconstructType, 3> ReconstructStaggered::recon;
  constexpr std::array<QudaReconstructType, 2> ReconstructNo12::recon;
  constexpr std::array<QudaReconstructType, 1> ReconstructNone::recon;
  constexpr std::array<QudaReconstructType, 2> ReconstructMom::recon;
  constexpr std::array<QudaReconstructType, 1> Reconstruct10::recon;

  // declared in dslash.h
#ifdef BUILD_OPENQCD_INTERFACE
  constexpr std::array<QudaReconstructType, 5> WilsonReconstruct::recon;
#else
  constexpr std::array<QudaReconstructType, 3> WilsonReconstruct::recon;
#endif
  constexpr std::array<QudaReconstructType, 3> StaggeredReconstruct::recon;

} // namespace quda
