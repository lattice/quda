#include <color_spinor_field.h>
#include <gauge_field.h>
#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_staggered_local.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType reconstruct_u,
            QudaReconstructType reconstruct_l, bool improved, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_MILC>
  class LocalStaggered : public TunableKernel2D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const GaugeField &U;
    const GaugeField &L;
    const ColorSpinorField &x;
    double a;
    int parity;
    bool xpay;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    LocalStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField& U,
                   const GaugeField &L, double a, const ColorSpinorField &x, int parity,
                   bool xpay) :
      TunableKernel2D(out, 1),
      out(out),
      in(in),
      U(U),
      L(L),
      x(x),
      a(a),
      parity(parity),
      xpay(xpay)
    {
      checkPrecision(out, in, U, L, x);
      checkLocation(out, in, U, L, x);
      if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET || x.SiteSubset() != QUDA_PARITY_SITE_SUBSET)
        errorQuda("Unsupported site subset %d %d %d, expected %d", in.SiteSubset(), out.SiteSubset(), x.SiteSubset(), QUDA_PARITY_SITE_SUBSET);
      if (in.Nspin() != 1 || out.Nspin() != 1 || x.Nspin() != 1) errorQuda("Unsupported nSpin=%d %d %d", out.Nspin(), in.Nspin(), x.Nspin());

      strcat(aux, ",LocalStaggered");
      strcat(aux, comm_dim_partitioned_string());
      char recon[3];
      if (improved) {
        strcat(aux, ",recon_u=");
        u32toa(recon, reconstruct_u);
        strcat(aux, recon);
        strcat(aux, ",recon_l=");
        u32toa(recon, reconstruct_l);
        strcat(aux, recon);
        strcat(aux, ",improved");
      } else {
        strcat(aux, ",recon_u=");
        u32toa(recon, reconstruct_u);
        strcat(aux, recon);
      }
      strcat(aux, parity == QUDA_EVEN_PARITY ? ",even" : ",odd");
      if (xpay)
        strcat(aux, ",xpay");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      (void)stream;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (xpay) {
        LocalStaggeredArg<Float, nColor, reconstruct_u, reconstruct_l, improved, true, phase> arg(out, in, U, L, a, x, parity);
        launch<LocalStaggeredApply>(tp, stream, arg);
      } else {
        LocalStaggeredArg<Float, nColor, reconstruct_u, reconstruct_l, improved, false, phase> arg(out, in, U, L, a, x, parity);
        launch<LocalStaggeredApply>(tp, stream, arg);
      }
    }

    long long flops() const {
      return 0ll; // FIXME
    }
    long long bytes() const {
      return 0ll; // FIXME
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct applyLocalStaggered {
    applyLocalStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField& L,
                           const GaugeField &U, double a, const ColorSpinorField &x, int parity,
                           bool improved, bool xpay) {
      // Template on improved, unimproved, as well as different phase types for unimproved
      if (improved) {
        constexpr bool improved = true;

        constexpr QudaReconstructType recon_u = QUDA_RECONSTRUCT_NO;
        constexpr QudaReconstructType recon_l = recon;

        LocalStaggered<Float, nColor, recon_u, recon_l, improved>(out, in, U, L, a, x, parity, xpay);
      } else {
        if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC || (U.LinkType() == QUDA_GENERAL_LINKS && U.Reconstruct() == QUDA_RECONSTRUCT_NO)) {
#ifdef BUILD_MILC_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          LocalStaggered<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_MILC>(out, in, U, L, a, x, parity, xpay);
#else
          errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
#endif // BUILD_MILC_INTERFACE
        } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          LocalStaggered<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_TIFR>(out, in, U, L, a, x, parity, xpay);
#else
          errorQuda("TIFR interface has not been built so TIFR phase taggered fermions not enabled");
#endif // BUILD_TIFR_INTERFACE
        } else {
          errorQuda("Unsupported combination of staggered phase type %d gauge link type %d and reconstruct %d", U.StaggeredPhase(), U.LinkType(), U.Reconstruct());
        }
      }
    }
  };

#ifdef GPU_STAGGERED_DIRAC
  // Apply a piece of the local staggered operator to a colorspinor field
  void ApplyLocalStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField& U,
                           const GaugeField &L, double a, const ColorSpinorField &x, int parity,
                           bool improved, bool xpay)
  {
    // swizzle L and U to template on reconstruct
    // For naive staggered, U and L both alias to the fine gauge field
    instantiate<applyLocalStaggered, StaggeredReconstruct>(out, in, L, U, a, x, parity, improved, xpay);
  }
#else
  void ApplyLocalStaggered(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
			const GaugeField &, double, const ColorSpinorField &, int, bool, bool)
  {
    errorQuda("Staggered dslash has not been built");
  }
#endif // GPU_STAGGERED_DIRAC

}

