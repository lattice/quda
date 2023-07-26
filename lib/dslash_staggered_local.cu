#include <color_spinor_field.h>
#include <gauge_field.h>
#include <dslash_quda.h>
#include <tunable_nd.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_staggered_local.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType reconstruct_u,
            QudaReconstructType reconstruct_l, bool improved, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_MILC>
  class LocalStaggered : public TunableKernel3D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const GaugeField &U;
    const GaugeField &L;
    const ColorSpinorField &x;
    double a;
    int parity;
    bool boundary_clover;
    bool xpay;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    LocalStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField& U,
                   const GaugeField &L, double a, const ColorSpinorField &x, int parity,
                   bool boundary_clover, bool xpay) :
      TunableKernel3D(out, 1, out.SiteSubset()),
      out(out),
      in(in),
      U(U),
      L(L),
      x(x),
      a(a),
      parity(parity),
      boundary_clover(boundary_clover),
      xpay(xpay)
    {
      checkPrecision(out, in, U, L, x);
      checkLocation(out, in, U, L, x);
      if (in.SiteSubset() != out.SiteSubset() || in.SiteSubset() != x.SiteSubset())
        errorQuda("Site subsets %d %d %d do not agree", in.SiteSubset(), out.SiteSubset(), x.SiteSubset());
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
      if (boundary_clover)
        strcat(aux, ",clover");
      if (xpay)
        strcat(aux, ",xpay");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      (void)stream;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (boundary_clover) {
        LocalStaggeredArg<Float, nColor, reconstruct_u, reconstruct_l, improved, true, phase> arg(out, in, U, L, a, x, parity, xpay);
        launch<LocalStaggeredApply>(tp, stream, arg);
      } else {
        LocalStaggeredArg<Float, nColor, reconstruct_u, reconstruct_l, improved, false, phase> arg(out, in, U, L, a, x, parity, xpay);
        launch<LocalStaggeredApply>(tp, stream, arg);
      }
    }

    long long flops() const {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int mv_add_flops = 8 * in.Ncolor() * in.Ncolor(); // SU(3) vector += matrix-vector flops
      int num_dir = 2 * 4;
      int xpay_flops = 2 * 2 * in.Ncolor();
      long long sites = in.Volume();

      long long flops_ = 0;

      // compute the flops for the full volume, then subtract/add boundary effects
      flops_ = (mv_flops + (num_dir - 1) * mv_add_flops) * sites; // one-hop links
      if (improved) flops_ += num_dir * mv_add_flops * sites; // long links
      if (xpay || boundary_clover) flops_ += xpay_flops * sites;

      // don't assume we have the ghost face sites precomputed in the spinor
      long long ghost_sites[4] = { in.X()[1] * in.X()[2] * in.X()[3],
                                   in.X()[0] * in.X()[2] * in.X()[3],
                                   in.X()[0] * in.X()[1] * in.X()[3],
                                   in.X()[0] * in.X()[1] * in.X()[2] };

      for (int d = 0; d < 4; d++) {
        if (comm_dim_partitioned(d)) {
          if (!boundary_clover) {
            // for each face, subtract off an appropriate number of flops

            // subtract off forward/backwards fat link flops
            flops_ -= 2 * mv_add_flops * ghost_sites[d];

            // subtract off three "slices" worth of forward/backwards long link flops
            if (improved)
              flops_ -= 3 * 2 * mv_add_flops * ghost_sites[d];
          } else {
            // for each face, add on the "clover" flops contribution

            // add on the fat link contribution to the depth-1 term
            flops_ += 2 * mv_flops * ghost_sites[d];
            // add on improved contributions
            if (improved) {
              // add on the long link contribution to the depth-1 term
              flops_ += 2 * mv_add_flops * ghost_sites[d];

              // add on the long link contribution to the depth-1,2,3 term
              flops_ += 3 * 2 * mv_flops * ghost_sites[d];

              // add on the fat link contribution to the depth-3 term
              flops_ += 2 * mv_add_flops * ghost_sites[d];
            } // improved
          } // xpay
        } // partitioned
      } // dimension

      return flops_;
    }

    long long bytes() const {
      int fat_gauge_bytes = (improved ? 2 * in.Ncolor() * in.Ncolor() : reconstruct_u) * in.Precision();
      int long_gauge_bytes = (improved ? reconstruct_l : 0) * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char));
      int spinor_bytes = 2 * in.Ncolor() + isFixed ? sizeof(float) : 0;
      int num_dir = 2 * 4;
      long long sites = in.Volume();

      long long bytes_ = 0;

      // compute the bytes for the full volume, then subtract/add boundary effects
      bytes_ = (fat_gauge_bytes + spinor_bytes) * num_dir * sites;
      if (improved) bytes_ += long_gauge_bytes * num_dir * sites;
      if (xpay || boundary_clover) bytes_ += spinor_bytes * sites;

      // don't assume we have the ghost face sites precomputed in the spinor
      long long ghost_sites[4] = { in.X()[1] * in.X()[2] * in.X()[3],
                                   in.X()[0] * in.X()[2] * in.X()[3],
                                   in.X()[0] * in.X()[1] * in.X()[3],
                                   in.X()[0] * in.X()[1] * in.X()[2] };
      for (int d = 0; d < 4; d++) {
        if (comm_dim_partitioned(d)) {
          if (!boundary_clover) {
            // for each face, subtract off an appropriate number of bytes

            // subtract off forward/backwards fat link bytes
            bytes_ -= 2 * (fat_gauge_bytes + spinor_bytes) * ghost_sites[d];
            // subtract off three "slices" worth of forward/backwards long link bytes
            if (improved)
              bytes_ -= 3 * 2 * (long_gauge_bytes + spinor_bytes) * ghost_sites[d];
          } else {
            // for each face, correct for the "clover" bytes contribution

            // amusingly, we save some bytes on the fat link contribution to the depth-1 term
            // this is because we don't have to load anything extra, and we already have "x"
            bytes_ -= 2 * spinor_bytes * ghost_sites[d];
            // add on improved contributions
            if (improved) {
              // we can subtract off more extra things because we already have "x"
              bytes_ -= 3 * 2 * spinor_bytes * ghost_sites[d];

              // add on the lonk link contribution to the depth-1,2,3 term
              bytes_ += 3 * 2 * (long_gauge_bytes + spinor_bytes) * ghost_sites[d];

              // add on the fat link contribution to the depth-3 term
              bytes_ += 2 * (fat_gauge_bytes + spinor_bytes) * ghost_sites[d];
            } // improved
          } // xpay
        } // partitioned
      } // dimension
      return bytes_;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct applyLocalStaggered {
    applyLocalStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField& L,
                           const GaugeField &U, double a, const ColorSpinorField &x, int parity,
                           bool improved, bool boundary_clover, bool xpay) {
      // Template on improved, unimproved, as well as different phase types for unimproved
      if (improved) {
        constexpr bool improved = true;

        constexpr QudaReconstructType recon_u = QUDA_RECONSTRUCT_NO;
        constexpr QudaReconstructType recon_l = recon;

        LocalStaggered<Float, nColor, recon_u, recon_l, improved>(out, in, U, L, a, x, parity, boundary_clover, xpay);
      } else {
        if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC || (U.LinkType() == QUDA_GENERAL_LINKS && U.Reconstruct() == QUDA_RECONSTRUCT_NO)) {
#ifdef BUILD_MILC_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          LocalStaggered<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_MILC>(out, in, U, L, a, x, parity, boundary_clover, xpay);
#else
          errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
#endif // BUILD_MILC_INTERFACE
        } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          LocalStaggered<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_TIFR>(out, in, U, L, a, x, parity, boundary_clover, xpay);
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
                           bool improved, bool boundary_clover, bool xpay)
  {
    // swizzle L and U to template on reconstruct
    // For naive staggered, U and L both alias to the fine gauge field
    instantiate<applyLocalStaggered, StaggeredReconstruct>(out, in, L, U, a, x, parity, improved, boundary_clover, xpay);
  }
#else
  void ApplyLocalStaggered(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
			const GaugeField &, double, const ColorSpinorField &, int, bool, bool, bool)
  {
    errorQuda("Staggered dslash has not been built");
  }
#endif // GPU_STAGGERED_DIRAC

}

