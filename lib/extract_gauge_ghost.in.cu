#include <gauge_field_order.h>
#include <extract_gauge_ghost_helper.cuh>
#include "multigrid.h"

namespace quda {

  using namespace gauge;

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtract {
    GhostExtract(const GaugeField &u, void **Ghost_, bool extract, int offset)
    {
      Float **Ghost = reinterpret_cast<Float**>(Ghost_);
      constexpr int nColor = 3;
      constexpr int length = nColor * nColor * 2;

      if (u.isNative()) {
        if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
          ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_12>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type;
            ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_12", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_8>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type;
            ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_8", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_13>::type;
            ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_13", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_9>()) {
            if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
              using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9, 18, QUDA_STAGGERED_PHASE_MILC>::type;
              ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
            } else if (u.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
              using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type;
              ExtractGhost<Float, nColor, G>(u, Ghost, extract, offset);
            } else {
              errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
            }
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_9", QUDA_RECONSTRUCT);
          }
        }
      } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, QDPOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("QDP interface has not been built");
        }

      } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, QDPJITOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("QDPJIT interface has not been built");
        }

      } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_CPS_WILSON_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, CPSOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("CPS interface has not been built");
        }

      } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, MILCOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("MILC interface has not been built");
        }

      } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_BQCD_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, BQCDOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("BQCD interface has not been built");
        }

      } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, TIFROrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("TIFR interface has not been built");
        }

      } else if (u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
          ExtractGhost<Float, nColor, TIFRPaddedOrder<Float,length>>(u, Ghost, extract, offset);
        } else {
          errorQuda("TIFR interface has not been built");
        }

      } else {
        errorQuda("Gauge field %d order not supported", u.Order());
      }

    }
  };

  template <int nColor>
  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset);

  template <int...> struct IntList { };

  template <int nColor, int...N>
  void extractGaugeGhostMG(const GaugeField &u, void **ghost, bool extract, int offset, IntList<nColor, N...>)
  {
    if (u.Ncolor() / 2 == nColor) {
        extractGaugeGhostMG<nColor>(u, ghost, extract, offset);
    } else {
      if constexpr (sizeof...(N) > 0) {
        extractGaugeGhostMG(u, ghost, extract, offset, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", u.Ncolor() / 2);
      }
    }
  }

  void extractGaugeGhost(const GaugeField &u, void **ghost, bool extract, int offset)
  {
    // if number of colors doesn't equal three then we must have
    // coarse-gauge field
    if (u.Ncolor() != 3) {
      if constexpr (is_enabled_multigrid()) {
        extractGaugeGhostMG(u, ghost, extract, offset, IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
      } else {
        errorQuda("Multigrid has not been built");
      }
    } else {
      instantiatePrecision<GhostExtract>(u, ghost, extract, offset);
    }
  }

} // namespace quda
