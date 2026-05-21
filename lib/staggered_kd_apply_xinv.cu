#include "gauge_field.h"
#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "multigrid.h"
#include "kernels/staggered_kd_apply_xinv_kernel.cuh"

namespace quda {

  template <typename Float, int nColor> class StaggeredKDBlock : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const GaugeField& Xinv;
    bool dagger;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    StaggeredKDBlock(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                     const GaugeField &Xinv, int dagger) :
      TunableKernel3D(in[0], in.size(), 2), out(out), in(in), Xinv(Xinv), dagger(dagger)
    {
      if (in.Nspin() != 1 || out.Nspin() != 1) errorQuda("Unsupported nSpin=%d %d", out.Nspin(), in.Nspin());
      if (Xinv.Geometry() != QUDA_KDINVERSE_GEOMETRY)
        errorQuda("Unsupported gauge geometry %d , expected %d for Xinv", Xinv.Geometry(), QUDA_KDINVERSE_GEOMETRY);
      if (Xinv.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Unsupported reconstruct %d", Xinv.Reconstruct());

      // the staggered KD block inverse can only be applied to a full field
      if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
        errorQuda("Invalid spinor parities for KD apply %d %d\n", out.SiteSubset(), in.SiteSubset());
    
      if (dagger) strcat(aux, ",dagger");
      setRHSstring(aux, in.size());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (dagger) {
        StaggeredKDBlockArg<Float, nColor, true> arg(out, in, Xinv);
        launch<StaggeredKDBlockApply>(tp, stream, arg);
      } else {
        StaggeredKDBlockArg<Float, nColor, false> arg(out, in, Xinv);
        launch<StaggeredKDBlockApply>(tp, stream, arg);
      }
    }

    // 3x3 mat-vec, gathering from 16 sites (same as asqtad stencil)
    long long flops() const { return in.size() * in.Volume() * out.Ncolor() * (8ll * out.Ncolor() * 16ll - 2ll); }

    // load the input 16 times (gather from each site of 2^4 hypercube), store once
    long long bytes() const { return in.size() * ((16ll + 1ll) * out.Bytes() + Xinv.Bytes()); }
  };

  /**
     @brief Apply the staggered Kahler-Dirac block inverse
     @param out[out] output staggered spinor field
     @param in[in] input staggered spinor field
     @param Xinv[in] KD block inverse gauge field
     @param dagger[in] whether or not we're applying the dagger of the KD block
  */
  void ApplyStaggeredKahlerDiracInverse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                        const GaugeField &Xinv, bool dagger)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>() && is_enabled_multigrid()) {
      // Instantiate based on precision, number of colors
      instantiate_recurse2<StaggeredKDBlock>(out, in, Xinv, dagger);
    } else {
      errorQuda("Staggered fermion multigrid support has not been built");
    }
  }

} //namespace quda
