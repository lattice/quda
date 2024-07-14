#include <quda_internal.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/extract_gauge_ghost_extended.cuh>

namespace quda {

  template <typename Gauge>
  class ExtractGhostEx : TunableKernel3D {
    unsigned int size;
    const GaugeField &u;
    const int dim;
    const lat_dim_t &R;
    void **ghost;
    const bool extract;

    unsigned int minThreads() const { return size; }

  public:
    ExtractGhostEx(const GaugeField &u, int dim, const lat_dim_t &R, void **ghost, bool extract) :
      TunableKernel3D(u, 2, 2),
      u(u),
      dim(dim),
      R(R),
      ghost(ghost),
      extract(extract)
    {
      strcat(aux, extract ? ",extract" : ",inject");
      strcat(aux, ",dim=");
      u32toa(aux + strlen(aux), dim);
      apply(device::get_default_stream());
    }

    template <int dim, bool extract> void Launch(const qudaStream_t &stream)
    {
      constexpr bool enable_host = true;
      ExtractGhostExArg<Gauge, dim, extract> arg(u, R, ghost);
      size = arg.threads.x; // don't call tuneLaunch until after we have set size
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GhostExtractorEx, enable_host>(tp, stream, arg);
    }

    void apply(const qudaStream_t &stream)
    {
      switch (dim) {
      case 0: extract ? Launch<0, true>(stream) : Launch<0, false>(stream); break;
      case 1: extract ? Launch<1, true>(stream) : Launch<1, false>(stream); break;
      case 2: extract ? Launch<2, true>(stream) : Launch<2, false>(stream); break;
      case 3: extract ? Launch<3, true>(stream) : Launch<3, false>(stream); break;
      }
    }

    long long flops() const { return 0; }
    // 2 for i/o, 2 for direction, size/2 for number of active threads
    long long bytes() const { return 2 * 2 * (size/2) * u.Reconstruct() * u.Precision(); }
  };

  /** This is the template driver for extractGhost */
  template <typename Float> struct GhostExtractEx {
    GhostExtractEx(const GaugeField &u, int dim, const lat_dim_t &R, void **ghost, bool extract)
    {
      const int length = 18;

      if (u.isNative()) {
        if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
          ExtractGhostEx<G>(u, dim, R, ghost, extract);
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_12>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type;
            ExtractGhostEx<G>(u, dim, R, ghost, extract);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_12", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_8>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type;
            ExtractGhostEx<G>(u, dim, R, ghost, extract);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_8", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_13>::type;
            ExtractGhostEx<G>(u, dim, R, ghost, extract);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_13", QUDA_RECONSTRUCT);
          }
        } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
          if constexpr (is_enabled<QUDA_RECONSTRUCT_9>()) {
            using G = typename gauge_mapper<Float,QUDA_RECONSTRUCT_9>::type;
            ExtractGhostEx<G>(u, dim, R, ghost, extract);
          } else {
            errorQuda("QUDA_RECONSTRUCT = %d does not enable QUDA_RECONSTRUCT_9", QUDA_RECONSTRUCT);
          }
        }
      } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
          ExtractGhostEx<QDPOrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("QDP interface has not been built");
        }

      } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>()) {
          ExtractGhostEx<QDPJITOrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("QDPJIT interface has not been built");
        }

      } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_CPS_WILSON_GAUGE_ORDER>()) {
          ExtractGhostEx<CPSOrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("CPS interface has not been built");
        }

      } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
          ExtractGhostEx<MILCOrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("MILC interface has not been built");
        }

      } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_BQCD_GAUGE_ORDER>()) {
          ExtractGhostEx<BQCDOrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("BQCD interface has not been built");
        }

      } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

        if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
          ExtractGhostEx<TIFROrder<Float,length>>(u, dim, R, ghost, extract);
        } else {
          errorQuda("TIFR interface has not been built");
        }

      } else {
        errorQuda("Gauge field %d order not supported", u.Order());
      }

    }
  };

  void extractExtendedGaugeGhost(const GaugeField &u, int dim, const lat_dim_t &R,
				 void **ghost, bool extract)
  {
    instantiatePrecision<GhostExtractEx>(u, dim, R, ghost, extract);
  }

} // namespace quda
