#include <dslash_twisted_mass_preconditioned.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  typedef @QUDA_DSLASH_DDARG@ DDArg;
  typedef precision_type_mapper<precision>::type Float;

  template struct TwistedMassPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

} // namespace quda
