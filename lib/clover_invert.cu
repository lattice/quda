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
      strcat(aux, compute_tr_log ? ",trlog=true" : "trlog=false");
      strcat(aux, clover.TwistFlavor() == QUDA_TWIST_SINGLET || clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ?
             ",twist=true" : ",twist=false");
      apply(device::get_default_stream());

      if (compute_tr_log && (std::isnan(clover.TrLog()[0]) || std::isnan(clover.TrLog()[1]))) {
	printfQuda("clover.TrLog()[0]=%e, clover.TrLog()[1]=%e\n", clover.TrLog()[0], clover.TrLog()[1]);
	errorQuda("Clover trlog has returned -nan, likey due to the clover matrix being singular.");
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (clover.TwistFlavor() == QUDA_TWIST_SINGLET ||
          clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
        CloverInvertArg<store_t, true> arg(clover, compute_tr_log);
        launch<InvertClover>(clover.TrLog(), tp, stream, arg);
      } else {
        CloverInvertArg<store_t, false> arg(clover, compute_tr_log);
        launch<InvertClover>(clover.TrLog(), tp, stream, arg);
      }
    }
    
    long long flops() const { return 0; }
    long long bytes() const { return 2 * clover.Bytes(); }
    void preTune() { if (clover::dynamic_inverse()) clover.backup(); }
    void postTune() { if (clover::dynamic_inverse()) clover.restore(); }
  };

#ifdef GPU_CLOVER_DIRAC
  void cloverInvert(CloverField &clover, bool computeTraceLog)
  {
    if (clover.Reconstruct()) errorQuda("Cannot store the inverse with a reconstruct field");
    if (clover.Precision() < QUDA_SINGLE_PRECISION) errorQuda("Cannot use fixed-point precision here");
    instantiate<CloverInvert>(clover, computeTraceLog);
  }
#else
  void cloverInvert(CloverField &, bool)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda
