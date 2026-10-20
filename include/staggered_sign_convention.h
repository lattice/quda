#pragma once

#include <color_spinor_field.h>

namespace quda
{

  /** The sign convention of staggered fields saved by QUDA: QUDA's own, or MILC's */
  enum class StaggeredSignConvention { QUDA, MILC };

  /**
     @brief Return the name of a staggered sign convention ("QUDA" or "MILC"), as files record it
   */
  const char *get_staggered_sign_convention_str(StaggeredSignConvention convention);

  /**
     Negate the odd sites (odd global x + y + z + t) of full staggered fields, which converts between QUDA's and
     MILC's sign conventions: QUDA's staggered operator is eps M_MILC eps, with eps = (-1)^(x+y+z+t), so MILC's
     solution of M_MILC x = b is eps M_QUDA^{-1} (eps b).  Apply staggeredMILCToQUDASignConvention to b (and to an
     initial guess from MILC) before a QUDA solve, and staggeredQUDAToMILCSignConvention to the solution and to b
     after it.  The two are the same operation (it is its own inverse); the names say which way the data goes.

     eps is QUDA's staggered gamma5 (DiracMatrix::applyGamma5 for the unpreconditioned staggered and asqtad
     operators).  The fields are full 4-d fields of 1 spin, on the host or the device.
   */
  void staggeredQUDAToMILCSignConvention(cvector_ref<ColorSpinorField> &fields);
  void staggeredMILCToQUDASignConvention(cvector_ref<ColorSpinorField> &fields);

} // namespace quda
