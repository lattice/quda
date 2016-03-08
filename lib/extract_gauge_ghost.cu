#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float>
    void extractGhost(const GaugeField &u, Float **Ghost) {

    const int length = 18;

    QudaFieldLocation location = 
      (typeid(u)==typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<short,length>(FloatNOrder<short,length,2,19>
				     (u, 0, (short**)Ghost), u, location);
	} else {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  extractGhost<Float,length>(G(u, 0, Ghost), u, location);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_13>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_9>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location);
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("QDP interface has not been built\n");
#endif
      
    } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      extractGhost<Float,length>(QDPJITOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      extractGhost<Float,length>(CPSOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      extractGhost<Float,length>(MILCOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      extractGhost<Float,length>(BQCDOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      extractGhost<Float,length>(TIFROrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

  void extractGaugeGhostMG(const GaugeField &u, void **ghost);

  void extractGaugeGhost(const GaugeField &u, void **ghost) {

    // if number of colors doesn't equal three then we must have
    // coarse-gauge field
    if (u.Ncolor() != 3) {
      extractGaugeGhostMG(u, ghost);
    } else {
      if (u.Precision() == QUDA_DOUBLE_PRECISION) {
	extractGhost(u, (double**)ghost);
      } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
	extractGhost(u, (float**)ghost);
      } else if (u.Precision() == QUDA_HALF_PRECISION) {
	extractGhost(u, (short**)ghost);
      } else {
	errorQuda("Unknown precision type %d", u.Precision());
      }
    }
  }

} // namespace quda
