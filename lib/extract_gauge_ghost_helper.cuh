#include <tunable_nd.h>
#include <kernels/extract_gauge_ghost.cuh>

namespace quda {

  /**
     Generic gauge ghost extraction and packing (or the converse)
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int nColor, typename Order, bool fine_grain = false> class ExtractGhost : TunableKernel3D
  {
    static constexpr int nDim = 4;
    uint64_t size;
    const GaugeField &u;
    Float **Ghost;
    bool extract;
    int offset;

    unsigned int minThreads() const { return size; }

  public:
    ExtractGhost(const GaugeField &u, Float **Ghost, bool extract, int offset) :
      TunableKernel3D(u, fine_grain ? nColor : 1, 2 * nDim), u(u), Ghost(Ghost), extract(extract), offset(offset)
    {
      if (nDim != u.Ndim()) errorQuda("Require 4-dimensional field");
      int faceMax = 0;
      for (int d = 0; d < nDim; d++) faceMax = (u.SurfaceCB(d) > faceMax) ? u.SurfaceCB(d) : faceMax;
      size = 2 * faceMax * u.Nface(); // factor of comes from parity
      strcat(aux, extract ? ",extract" : ",inject");

      apply(device::get_default_stream());
    }

    template <bool extract, bool fine_grained> using Arg = ExtractGhostArg<Float, nColor, Order, extract, fine_grained>;

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      constexpr bool enable_host = true;
      if constexpr (fine_grain) {
        if (extract) {
          launch<GhostExtractorFineGrained, enable_host>(tp, stream, Arg<true, fine_grain>(u, Ghost, offset, size));
        } else {
          launch<GhostExtractorFineGrained, enable_host>(tp, stream, Arg<false, fine_grain>(u, Ghost, offset, size));
        }
      } else {
        if (extract) {
          launch<GhostExtractor, enable_host>(tp, stream, Arg<true, fine_grain>(u, Ghost, offset, size));
        } else {
          launch<GhostExtractor, enable_host>(tp, stream, Arg<false, fine_grain>(u, Ghost, offset, size));
        }
      }
    }

    long long bytes() const
    {
      uint64_t sites = 0;
      for (int d = 0; d < nDim; d++) sites += 2 * u.SurfaceCB(d) * u.Nface();
      return sites * 2 * (u.Ncolor() == 3 ? u.Reconstruct() : 2 * u.Ncolor() * u.Ncolor()) * u.Precision();
    }
  };

} // namespace quda
