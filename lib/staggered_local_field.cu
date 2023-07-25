#include <tunable_nd.h>
#include <gauge_field.h>
#include <kernels/staggered_local_field.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_u, QudaReconstructType reconstruct_l,
            bool improved, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_MILC>
  class StaggeredLocalField : TunableKernel2D
  {
    GaugeField &Ulocal;
    GaugeField &Llocal;
    const GaugeField &U;
    const GaugeField &L;
    unsigned int minThreads() const { return U.VolumeCB(); }

  public:
    StaggeredLocalField(GaugeField &Ulocal, GaugeField &Llocal, const GaugeField &U, const GaugeField &L) :
      TunableKernel2D(Ulocal, 2),
      Ulocal(Ulocal),
      Llocal(Llocal),
      U(U),
      L(L)
    {
      strcat(aux, comm_dim_partitioned_string());
      char recon[3];
      if (improved) {
        strcat(aux, ",recon_l=");
        u32toa(recon, reconstruct_l);
        strcat(aux, recon);
        strcat(aux, ",improved");
      } else {
        strcat(aux, ",recon_u=");
        u32toa(recon, reconstruct_u);
        strcat(aux, recon);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeStaggeredLocalField>(tp, stream, StaggeredLocalFieldArg<Float, nColor, reconstruct_u, reconstruct_l, improved, phase>(Ulocal, Llocal, U, L));
    }

    long long flops() const {
      return 0ll; // fixme (2430 + 36) * 6 * f.Volume();
    }

    long long bytes() const {
      return 0ll; // fixme return ((16 * u.Reconstruct() + f.Reconstruct()) * 6 * f.Volume() * f.Precision());
    }
  };

template <typename Float, int nColor, QudaReconstructType recon>
  struct computeStaggeredLocalField {
    computeStaggeredLocalField(const GaugeField &L, GaugeField &Ulocal, GaugeField &Llocal, const GaugeField &U, bool improved) {
      // Template on improved, unimproved, as well as different phase types for unimproved
      if (improved) {
        constexpr bool improved = true;

        constexpr QudaReconstructType recon_u = QUDA_RECONSTRUCT_NO;
        constexpr QudaReconstructType recon_l = recon;

        StaggeredLocalField<Float, nColor, recon_u, recon_l, improved>(Ulocal, Llocal, U, L);
      } else {
        if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC || (U.LinkType() == QUDA_GENERAL_LINKS && U.Reconstruct() == QUDA_RECONSTRUCT_NO)) {
#ifdef BUILD_MILC_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          StaggeredLocalField<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_MILC>(Ulocal, Llocal, U, L);
#else
          errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
#endif // BUILD_MILC_INTERFACE
        } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
          constexpr bool improved = false;

          constexpr QudaReconstructType recon_u = recon;
          constexpr QudaReconstructType recon_l = QUDA_RECONSTRUCT_NO; // ignored

          StaggeredLocalField<Float, nColor, recon_u, recon_l, improved, QUDA_STAGGERED_PHASE_TIFR>(Ulocal, Llocal, U, L);
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
  void ConstructStaggeredLocalField(GaugeField &Ulocal, GaugeField &Llocal, const GaugeField &U, const GaugeField &L, bool improved)
  {
    checkPrecision(Ulocal, Llocal, U, L);
    // swizzle L to the first location for correct template instantiation
    instantiate<computeStaggeredLocalField,ReconstructStaggered>(L, Ulocal, Llocal, U, improved);
  }
#else
  void ConstructStaggeredLocalField(GaugeField&, GaugeField&, const GaugeField&, const GaugeField&, bool)
  {
    errorQuda("Staggered dslash has not been built");
  }
#endif

} // namespace quda
