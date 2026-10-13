#include <hisq_paths_force.hpp>

namespace quda
{

  namespace fermion_force
  {

    // clang-format off
    constexpr QudaPrecision precision = QUDA_@QUDA_HISQ_PATH_FORCE_PREC@_PRECISION;
    constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_@QUDA_HISQ_PATH_FORCE_RECON@;
    constexpr QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_@QUDA_HISQ_PATH_FORCE_PHASE@;
    // clang-format on
    using Float = precision_type_mapper<precision>::type;

    template void hisqStaplesForceImpl<Float, reconstruct, phase>(GaugeField &, const GaugeField &, const GaugeField &,
                                                                  const double[6]);
    template void hisqLongLinkForceImpl<Float, reconstruct, phase>(GaugeField &, const GaugeField &, const GaugeField &,
                                                                   double);
    template void hisqCompleteForceImpl<Float, reconstruct, phase>(GaugeField &, const GaugeField &);

  } // namespace fermion_force

} // namespace quda
