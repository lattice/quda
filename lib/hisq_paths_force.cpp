#include <gauge_field.h>
#include <instantiate.h>
#include <ks_improved_force.h>

namespace quda
{

  namespace fermion_force
  {

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase>
    void hisqStaplesForceImpl(GaugeField &, const GaugeField &, const GaugeField &, const double[6]);

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase>
    void hisqLongLinkForceImpl(GaugeField &, const GaugeField &, const GaugeField &, double);

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase>
    void hisqCompleteForceImpl(GaugeField &, const GaugeField &);

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase> struct StaplesForceApply {
      void operator()(GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link,
                      const double path_coeff_array[6])
      {
        hisqStaplesForceImpl<Float, recon, phase>(newOprod, oprod, link, path_coeff_array);
      }
    };

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase> struct LongLinkForceApply {
      void operator()(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        hisqLongLinkForceImpl<Float, recon, phase>(newOprod, oldOprod, link, coeff);
      }
    };

    template <typename Float, QudaReconstructType recon, QudaStaggeredPhase phase> struct CompleteForceApply {
      void operator()(GaugeField &force, const GaugeField &link)
      {
        hisqCompleteForceImpl<Float, recon, phase>(force, link);
      }
    };

    template <typename Float, template <typename, QudaReconstructType, QudaStaggeredPhase> class Apply, typename... Args>
    void instantiateHisqPathForce(const GaugeField &link, Args &&...args)
    {
      if (link.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_NO>()) {
          Apply<Float, QUDA_RECONSTRUCT_NO, QUDA_STAGGERED_PHASE_NO>()(args...);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable %d", QUDA_RECONSTRUCT, QUDA_RECONSTRUCT_NO);
        }
      } else if (link.Reconstruct() == QUDA_RECONSTRUCT_13) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
          if (link.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
            Apply<Float, QUDA_RECONSTRUCT_13, QUDA_STAGGERED_PHASE_NO>()(args...);
          } else if (link.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
            Apply<Float, QUDA_RECONSTRUCT_13, QUDA_STAGGERED_PHASE_MILC>()(args...);
          } else {
            errorQuda("Unsupported staggered phase type %d\n", link.StaggeredPhase());
          }
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable %d", QUDA_RECONSTRUCT, QUDA_RECONSTRUCT_13);
        }
      } else {
        errorQuda("Unsupported reconstruct type %d\n", link.Reconstruct());
      }
    }

    template <template <typename, QudaReconstructType, QudaStaggeredPhase> class Apply, typename... Args>
    void instantiateHisqPathForce(const GaugeField &link, Args &&...args)
    {
      if (link.Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled(QUDA_DOUBLE_PRECISION)) {
          instantiateHisqPathForce<double, Apply>(link, args...);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
        }
      } else if (link.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
          instantiateHisqPathForce<float, Apply>(link, args...);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
        }
      } else {
        errorQuda("Unsupported precision %d\n", link.Precision());
      }
    }

    void hisqStaplesForce(GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link,
                          const double path_coeff_array[6])
    {
      if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
        instantiateHisqPathForce<StaplesForceApply>(link, newOprod, oprod, link, path_coeff_array);
      } else {
        errorQuda("HISQ force requires staggered operator to be enabled");
      }
    }

    void hisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
    {
      if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
        instantiateHisqPathForce<LongLinkForceApply>(link, newOprod, oldOprod, link, coeff);
      } else {
        errorQuda("HISQ force requires staggered operator to be enabled");
      }
    }

    void hisqCompleteForce(GaugeField &force, const GaugeField &link)
    {
      if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
        instantiateHisqPathForce<CompleteForceApply>(link, force, link);
      } else {
        errorQuda("HISQ force requires staggered operator to be enabled");
      }
    }

  } // namespace fermion_force

} // namespace quda
