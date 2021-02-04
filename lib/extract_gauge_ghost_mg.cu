#define FINE_GRAINED_ACCESS

#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>
#include <instantiate.h>

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename storeFloat, int Nc>
  void extractGhostMG(const GaugeField &u, storeFloat **Ghost, bool extract, int offset)
  {
    typedef typename mapper<storeFloat>::type Float;

    if (u.isNative()) {
      using G = typename gauge::FieldOrder<Float,Nc,1,QUDA_FLOAT2_GAUGE_ORDER,false,storeFloat>;
      ExtractGhost<storeFloat, Nc, G>(u, Ghost, extract, offset);
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
      using G = typename gauge::FieldOrder<Float,Nc,1,QUDA_QDP_GAUGE_ORDER,true,storeFloat>;
      ExtractGhost<storeFloat, Nc, G>(u, Ghost, extract, offset);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

      using G = typename gauge::FieldOrder<Float, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, storeFloat>;
      ExtractGhost<storeFloat, Nc, G>(u, Ghost, extract, offset);

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }
  }

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtractMG {
    GhostExtractMG(const GaugeField &u, void **Ghost_, bool extract, int offset)
    {
      Float **Ghost = reinterpret_cast<Float**>(Ghost_);

      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO) 
        errorQuda("Reconstruct %d not supported", u.Reconstruct());

      if (u.LinkType() != QUDA_COARSE_LINKS)
        errorQuda("Link type %d not supported", u.LinkType());

      if (u.Ncolor() == 48) {
        extractGhostMG<Float, 48>(u, Ghost, extract, offset);
#ifdef NSPIN4
      } else if (u.Ncolor() == 12) { // free field Wilson
        extractGhostMG<Float, 12>(u, Ghost, extract, offset);
      } else if (u.Ncolor() == 64) {
        extractGhostMG<Float, 64>(u, Ghost, extract, offset);
#endif // NSPIN4
#ifdef NSPIN1
      } else if (u.Ncolor() == 128) {
        extractGhostMG<Float, 128>(u, Ghost, extract, offset);
      } else if (u.Ncolor() == 192) {
        extractGhostMG<Float, 192>(u, Ghost, extract, offset);
#endif // NSPIN1
      } else {
        errorQuda("Ncolor = %d not supported", u.Ncolor());
      }
    }
  };

#ifdef GPU_MULTIGRID
  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset)
  {
    instantiatePrecisionMG<GhostExtractMG>(u, ghost, extract, offset);
  }
#else
  void extractGaugeGhostMG(const GaugeField &, void **, bool, int)
  {
    errorQuda("Multigrid has not been enabled");
  }
#endif

} // namespace quda
