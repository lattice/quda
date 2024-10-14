#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_staggered.cuh>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Arg> class Staggered : public Dslash<staggered, Arg>
  {
    using Dslash = Dslash<staggered, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;
    const GaugeField &L;

  public:
    Staggered(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
              const ColorSpinorField &halo, const GaugeField &L) :
      Dslash(arg, out, in, halo), L(L)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      // operator is anti-Hermitian so do not instantiate dagger
      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash::template instantiate<packStaggeredShmem, 1, false, true>(tp, stream);
        else
          Dslash::template instantiate<packStaggeredShmem, 1, false, false>(tp, stream);
      } else if (arg.nParity == 2) {
        if (arg.xpay)
          Dslash::template instantiate<packStaggeredShmem, 2, false, true>(tp, stream);
        else
          Dslash::template instantiate<packStaggeredShmem, 2, false, false>(tp, stream);
      }
    }

    /*
      per direction / dimension flops
      SU(3) matrix-vector flops = (8 Nc - 2) * Nc
      xpay = 2 * 2 * Nc * Ns

      So for the full dslash we have
      flops = (2 * 2 * Nd * (8*Nc-2) * Nc)  +  ((2 * 2 * Nd - 1) * 2 * Nc * Ns)
      flops_xpay = flops + 2 * 2 * Nc * Ns

      For Asqtad this should give 1146 for Nc=3,Ns=2 and 1158 for the axpy equivalent
    */
    long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int ghost_flops = (3 + 1) * (mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = 2 * 4;                               // hard code factor of 4 in direction since fields may be 5-d

      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: flops_ = ghost_flops * 2 * halo.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites
          = 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        flops_ = ghost_flops * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume();
        flops_ = (2 * num_dir * mv_flops + // SU(3) matrix-vector multiplies
                  (2 * num_dir - 1) * 2 * in.Ncolor() * in.Nspin())
          * sites;                                  // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites; // axpy is always on interior

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }
      return flops_;
    }

    long long bytes() const
    {
      int gauge_bytes_fat = QUDA_RECONSTRUCT_NO * in.Precision();
      int gauge_bytes_long = L.Reconstruct() * in.Precision();
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
      int ghost_bytes = 3 * (spinor_bytes + gauge_bytes_long) + (spinor_bytes + gauge_bytes_fat)
        + 3 * 2 * spinor_bytes; // last term is the accumulator load/store through the face
      int num_dir = 2 * 4;      // set to 4-d since we take care of 5-d fermions in derived classes where necessary

      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes_ = ghost_bytes * 2 * halo.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites
          = 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume();
        bytes_ = (num_dir * (gauge_bytes_fat + gauge_bytes_long) + // gauge reads
                  num_dir * 2 * spinor_bytes +                     // spinor reads
                  spinor_bytes)
          * sites; // spinor write
        if (arg.xpay) bytes_ += spinor_bytes;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        bytes_ -= ghost_bytes * ghost_sites;

        break;
      }
      }
      return bytes_;
    }

  };

  template <bool distance_pc> struct DistanceType {
  };


  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon_l> struct ImprovedStaggeredApply {
    template <bool distance_pc>
    ImprovedStaggeredApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           cvector_ref<const ColorSpinorField> &x, const GaugeField &L, const GaugeField &U, double a,
                           int parity, bool dagger, const int *comm_override,DistanceType<distance_pc>, TimeProfile &profile)
#ifdef SIGNATURE_ONLY
    ;
#else
    {
      constexpr int nDim = 4;
      constexpr bool improved = true;
      constexpr QudaReconstructType recon_u = QUDA_RECONSTRUCT_NO;
      auto halo = ColorSpinorField::create_comms_batch(in);
      StaggeredArg<Float, nColor, nDim, DDArg, recon_u, recon_l, improved> arg(out, in, halo, U, L, a, x, parity,
                                                                               dagger, comm_override);
      Staggered<decltype(arg)> staggered(arg, out, in, halo, L);
      dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, in, halo, profile);
    }
#endif
  };

} // namespace quda
