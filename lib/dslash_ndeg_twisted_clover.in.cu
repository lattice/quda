#include <dslash_ndeg_twisted_clover.hpp>

namespace quda
{

    constexpr QudaPrecision precision = QUDA_@QUDA_DSLASH_PREC@_PRECISION;
    constexpr int nColor = @QUDA_DSLASH_NCOLOR@;
    constexpr int reconI = @QUDA_DSLASH_RECONI@;
    constexpr bool distance_pc = @QUDA_DSLASH_DISTANCE@;

    typedef @QUDA_DSLASH_DDARG@ DDArg;
    typedef precision_type_mapper<precision>::type Float;

    template struct NdegTwistedCloverApply<Float,nColor, DDArg, ReconstructWilson::recon[reconI]>;

    template NdegTwistedCloverApply<Float,nColor, DDArg, ReconstructWilson::recon[reconI]>::NdegTwistedCloverApply<distance_pc>(
	    cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
            cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &A, double a,
	    double b, double c, int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile);
} // namespace quda
