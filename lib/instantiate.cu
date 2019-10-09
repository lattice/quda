#include <instantiate.h>

/**
   This file contains deinitions required when compiling with C++14.
   Without these, we can end up with undefined references at link
   time.  We can remove this file when we jump to C++17.

   For reasons I do not understand, this file fails to compile when
   compiled directly as a cpp file (it borks on the class member
   specialization in the header) but works fine as a cu file for both
   clang and g++.  No major deal since this is a temporary file.
 */

namespace quda {

  // declared in instantiate.h
  constexpr std::array<QudaReconstructType, 5> ReconstructFull::recon;
  constexpr std::array<QudaReconstructType, 3> ReconstructWilson::recon;
  constexpr std::array<QudaReconstructType, 3> ReconstructStaggered::recon;
  constexpr std::array<QudaReconstructType, 2> ReconstructNo12::recon;
  constexpr std::array<QudaReconstructType, 1> ReconstructNone::recon;
  constexpr std::array<QudaReconstructType, 2> ReconstructMom::recon;
  constexpr std::array<QudaReconstructType, 1> Reconstruct10::recon;

  // declared in dslash.h
  constexpr std::array<QudaReconstructType, 3> WilsonReconstruct::recon;
  constexpr std::array<QudaReconstructType, 3> StaggeredReconstruct::recon;

}
