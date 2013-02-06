#ifndef __TEST_UTILITIES_H_
#define __TEST_UTILITIES_H_

#include <quda.h>
#include <color_spinor_field.h>
#include <enum_quda.h>
// Here, we include declarations which are only needed for test routines

namespace quda {
  class Dirac;
  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve);

  void massRescale(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type,
                   QudaMassNormalization mass_normalization, cudaColorSpinorField &b);

}
#endif
