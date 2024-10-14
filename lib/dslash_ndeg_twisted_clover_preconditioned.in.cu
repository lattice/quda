#include <dslash_ndeg_twisted_clover_preconditioned.hpp>

namespace quda
{

    constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
    constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
    constexpr int reconI = @QUDA_DSLASH_RECONI@;
    constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

    typedef @QUDA_DSLASH_DDARG@ DDArg;
    typedef precision_type_mapper<precision>::type Float;

    template struct NdegTwistedCloverPreconditionedApply<Float,nColor, DDArg, ReconstructWilson::recon[reconI]>;


    template NdegTwistedCloverPreconditionedApply<Float,nColor, DDArg, ReconstructWilson::recon[reconI]>::NdegTwistedCloverPreconditionedApply<distance_pc>(
		    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                    const CloverField &A, double a, double b, double c, bool xpay, int parity,
                    bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile);
    
} // namespace quda

