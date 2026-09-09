#include <dslash_twisted_clover_preconditioned.hpp>

namespace quda
{

  constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
  constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
  constexpr int reconI = @QUDA_DSLASH_RECONI@;

  using DDArg = @QUDA_DSLASH_DDARG@;
  using Float = precision_type_mapper<precision>::type;

  template struct TwistedCloverPreconditionedApply<Float, nColor, DDArg, ReconstructWilson::recon[reconI]>;

} // namespace quda
