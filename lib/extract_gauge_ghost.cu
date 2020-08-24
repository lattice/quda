#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>
#include <instantiate.h>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtract {
    GhostExtract(const GaugeField &u, void **Ghost_, bool extract, int offset)
    {
      Float **Ghost = reinterpret_cast<Float**>(Ghost_);
      const int length = 18;

      if (u.isNative()) {
        if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type G;
          extractGhost<Float, length>(G(u, 0, Ghost), u, extract, offset);
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
          typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
          extractGhost<Float,length>(G(u, 0, Ghost), u, extract, offset);
#else
          errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_12", QUDA_RECONSTRUCT);
#endif
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
          typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
          extractGhost<Float,length>(G(u, 0, Ghost), u, extract, offset);
#else
          errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_8", QUDA_RECONSTRUCT);
#endif
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
          typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_13>::type G;
          extractGhost<Float,length>(G(u, 0, Ghost), u, extract, offset);
#else
          errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_13", QUDA_RECONSTRUCT);
#endif
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
          if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
            typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_9, 18, QUDA_STAGGERED_PHASE_MILC>::type G;
            extractGhost<Float, length>(G(u, 0, Ghost), u, extract, offset);
          } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
            typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type G;
            extractGhost<Float, length>(G(u, 0, Ghost), u, extract, offset);
          } else {
            errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
          }
#else
          errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_9", QUDA_RECONSTRUCT);
#endif
        }
      } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
        extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("QDP interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
        extractGhost<Float,length>(QDPJITOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("QDPJIT interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
        extractGhost<Float,length>(CPSOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("CPS interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
        extractGhost<Float,length>(MILCOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("MILC interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
        extractGhost<Float,length>(BQCDOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("BQCD interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
        extractGhost<Float,length>(TIFROrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("TIFR interface has not been built\n");
#endif

      } else if (u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
        extractGhost<Float,length>(TIFRPaddedOrder<Float,length>(u, 0, Ghost), u, extract, offset);
#else
        errorQuda("TIFR interface has not been built\n");
#endif

      } else {
        errorQuda("Gauge field %d order not supported", u.Order());
      }

    }
  };

  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset);

  void extractGaugeGhost(const GaugeField &u, void **ghost, bool extract, int offset) {

    // if number of colors doesn't equal three then we must have
    // coarse-gauge field
    if (u.Ncolor() != 3) {
      extractGaugeGhostMG(u, ghost, extract, offset);
    } else {
      instantiatePrecision<GhostExtract>(u, ghost, extract, offset);
    }
  }

} // namespace quda
