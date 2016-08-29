// trove cannot deal with the large matrices that MG uses so we need
// to disable it
#define DISABLE_TROVE

#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float, int Nc>
  void extractGhostMG(const GaugeField &u, Float **Ghost, bool extract) {

    const int length = 2*Nc*Nc;

    QudaFieldLocation location = 
      (typeid(u)==typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.isNative()) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	extractGhost<Float,length>(G(u, 0, Ghost), u, location, extract);
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost), u, location, extract);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }
  }


  /** This is the template driver for extractGhost */
  template <typename Float>
  void extractGhostMG(const GaugeField &u, Float **Ghost, bool extract) {

    if (u.Reconstruct() != QUDA_RECONSTRUCT_NO) 
      errorQuda("Reconstruct %d not supported", u.Reconstruct());

    if (u.LinkType() != QUDA_COARSE_LINKS)
      errorQuda("Link type %d not supported", u.LinkType());

    if (u.Ncolor() == 4) {
      extractGhostMG<Float, 4>(u, Ghost, extract);
    } else if (u.Ncolor() == 32) {
      extractGhostMG<Float, 32>(u, Ghost, extract);
    } else if (u.Ncolor() == 48) {
      extractGhostMG<Float, 48>(u, Ghost, extract);
    } else if (u.Ncolor() == 64) {
      extractGhostMG<Float, 64>(u, Ghost, extract);
    } else {
      errorQuda("Ncolor = %d not supported", u.Ncolor());
    }
  }

  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract) {

    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      extractGhostMG(u, (double**)ghost, extract);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      extractGhostMG(u, (float**)ghost, extract);
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
  }

} // namespace quda
