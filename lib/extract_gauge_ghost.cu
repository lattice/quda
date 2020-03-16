#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float>
  void extractGhost(const GaugeField &u, Float **Ghost, bool extract, int offset) {

    const int length = 18;

    QudaFieldLocation location
        = (typeid(u) == typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type G;
        extractGhost<Float, length>(G(u, 0, Ghost), u, location, extract, offset);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location, extract, offset);
#else
        errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_12", QUDA_RECONSTRUCT);
#endif
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location, extract, offset);
#else
        errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_8", QUDA_RECONSTRUCT);
#endif
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_13>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location, extract, offset);
#else
        errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_13", QUDA_RECONSTRUCT);
#endif
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
          typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_9, 18, QUDA_STAGGERED_PHASE_MILC>::type G;
          extractGhost<Float, length>(G(u, 0, Ghost), u, location, extract, offset);
        } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
          typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type G;
          extractGhost<Float, length>(G(u, 0, Ghost), u, location, extract, offset);
        } else {
          errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
        }
#else
        errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_9", QUDA_RECONSTRUCT);
#endif
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      extractGhost<Float,length>(QDPJITOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      extractGhost<Float,length>(CPSOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      extractGhost<Float,length>(MILCOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      extractGhost<Float,length>(BQCDOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      extractGhost<Float,length>(TIFROrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      extractGhost<Float,length>(TIFRPaddedOrder<Float,length>(u, 0, Ghost), u, location, extract, offset);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset);

  void extractGaugeGhost(const GaugeField &u, void **ghost, bool extract, int offset) {

    // if number of colors doesn't equal three then we must have
    // coarse-gauge field
    if (u.Ncolor() != 3) {
      extractGaugeGhostMG(u, ghost, extract, offset);
    } else {
      if (u.Precision() == QUDA_DOUBLE_PRECISION) {
	extractGhost(u, (double**)ghost, extract, offset);
      } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
	extractGhost(u, (float**)ghost, extract, offset);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (u.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
	extractGhost(u, (short**)ghost, extract, offset);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (u.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        extractGhost(u, (char **)ghost, extract, offset);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Unknown precision type %d", u.Precision());
      }
    }
  }

} // namespace quda
