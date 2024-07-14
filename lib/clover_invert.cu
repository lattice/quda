#include <clover_field.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/clover_invert.cuh>

namespace quda {

  template <typename store_t>
  class CloverInvert : TunableReduction2D {
    CloverField &clover;
    bool compute_tr_log;

  public:
    CloverInvert(CloverField &clover, bool compute_tr_log) :
      TunableReduction2D(clover),
      clover(clover),
      compute_tr_log(compute_tr_log)
    {
      if (compute_tr_log) strcat(aux, ",trlog");
      strcat(aux,
             clover.TwistFlavor() == QUDA_TWIST_SINGLET || clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ?
               ",twist=true" :
               ",twist=false");
      apply(device::get_default_stream());

      if (std::isnan(clover.TrLog()[0]) || std::isnan(clover.TrLog()[1])) {
        printfQuda("Clover trlog = { %e, %e }\n", clover.TrLog()[0], clover.TrLog()[1]);
        errorQuda("Clover trlog has returned nan, clover matrix is likely non HPD (check coefficients)");
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (clover.TwistFlavor() == QUDA_TWIST_SINGLET ||
          clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
        if (compute_tr_log) {
          CloverInvertArg<store_t, true, true> arg(clover);
          launch<InvertClover>(clover.TrLog(), tp, stream, arg);
        } else {
          CloverInvertArg<store_t, true, false> arg(clover);
          launch<InvertClover>(clover.TrLog(), tp, stream, arg);
        }
      } else {
        if (compute_tr_log) {
          CloverInvertArg<store_t, false, true> arg(clover);
          launch<InvertClover>(clover.TrLog(), tp, stream, arg);
        } else {
          CloverInvertArg<store_t, false, false> arg(clover);
          launch<InvertClover>(clover.TrLog(), tp, stream, arg);
        }
      }
    }
    
    long long flops() const { return 0; }
    long long bytes() const { return (compute_tr_log ? 1 : 2) * clover.Bytes(); }
    void preTune() { if (clover::dynamic_inverse()) clover.backup(); }
    void postTune() { if (clover::dynamic_inverse()) clover.restore(); }
  };

  void cloverInvert(CloverField &clover, bool computeTraceLog)
  {
    if constexpr (is_enabled_clover()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      if (clover.Reconstruct() && !computeTraceLog) errorQuda("Cannot store the inverse with a reconstruct field");
      if (clover.Precision() < QUDA_SINGLE_PRECISION) errorQuda("Cannot use fixed-point precision here");
      instantiate<CloverInvert>(clover, computeTraceLog);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Clover has not been built");
    }
  }

} // namespace quda
