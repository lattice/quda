#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float, int Nc>
    void extractGhost(const GaugeField &u, Float **Ghost) {    

    const int length = 2*Nc*Nc;

    QudaFieldLocation location = 
      (typeid(u)==typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location);
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, location);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }
  }


  /** This is the template driver for extractGhost */
  template <typename Float>
    void extractGhost(const GaugeField &u, Float **Ghost) {

    if (u.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct %d not supported");

    if (u.LinkType() != QUDA_GENERAL_LINKS)
      errorQuda("Link type not supported");

    if (u.Ncolor() == 4) {
      extractGhost<Float, 4>(u, Ghost);
    } else {
      errorQuda("Ncolor = %d not supported", u.Ncolor());
    }
  }

  void extractGaugeGhostMG(const GaugeField &u, void **ghost) {

    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      extractGhost(u, (double**)ghost);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      extractGhost(u, (float**)ghost);
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
  }

} // namespace quda
