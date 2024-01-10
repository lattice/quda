#define FINE_GRAINED_ACCESS

#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>
#include <multigrid.h>

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
      
      if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
        using G = typename gauge::FieldOrder<Float,Nc,1,QUDA_QDP_GAUGE_ORDER,true,storeFloat>;
        ExtractGhost<storeFloat, Nc, G>(u, Ghost, extract, offset);
      } else {
        errorQuda("QDP interface has not been built\n");
      }

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

      using G = typename gauge::FieldOrder<Float, Nc, 1, QUDA_MILC_GAUGE_ORDER, true, storeFloat>;
      ExtractGhost<storeFloat, Nc, G>(u, Ghost, extract, offset);

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }
  }

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtractMG {
    template <class static_color>
    GhostExtractMG(const GaugeField &u, void **Ghost_, bool extract, int offset, static_color)
    {
      Float **Ghost = reinterpret_cast<Float**>(Ghost_);

      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO) 
        errorQuda("Reconstruct %d not supported", u.Reconstruct());

      if (u.LinkType() != QUDA_COARSE_LINKS)
        errorQuda("Link type %d not supported", u.LinkType());

      // factor of two from inherit spin in coarse gauge fields
      extractGhostMG<Float, 2 * static_color::value>(u, Ghost, extract, offset);
    }
  };

  template <int nColor> void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset);

  constexpr int nColor = @QUDA_MULTIGRID_NVEC@;

  template <>
  void extractGaugeGhostMG<nColor>(const GaugeField &u, void **ghost, bool extract, int offset)
  {
    if constexpr (is_enabled_multigrid()) {
      instantiatePrecisionMG<GhostExtractMG>(u, ghost, extract, offset, std::integral_constant<int, nColor>());
    } else {
      errorQuda("Multigrid has not been enabled");
    }
  }

} // namespace quda
