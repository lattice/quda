#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <uint_to_char.h>

#include <dslash_policy.hpp>
#include <kernels/covariant_derivative.cuh>

/**
   This is the covariant derivative based on the basic gauged Laplace operator
*/

namespace quda
{

  template <typename Arg> class CovDev : public Dslash<covDev, Arg>
  {
    using Dslash = Dslash<covDev, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;

  public:
    CovDev(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
           const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay) errorQuda("Covariant derivative operator only defined without xpay");
      if (arg.nParity != 2) errorQuda("Covariant derivative operator only defined for full field");

      constexpr bool xpay = false;
      constexpr int nParity = 2;
      Dslash::template instantiate<packStaggeredShmem, nParity, xpay>(tp, stream);
    }

    long long flops() const override
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin();
      int ghost_flops = num_mv_multiply * mv_flops;
      int dim = arg.mu % 4;
      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        if (arg.kernel_type != dim) break;
        flops_ = (ghost_flops)*halo.GhostFace()[dim];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = halo.GhostFace()[dim];
        flops_ = ghost_flops * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume();
        flops_ = num_mv_multiply * mv_flops * sites; // SU(3) matrix-vector multiplies

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = arg.commDim[dim] ? halo.GhostFace()[dim] : 0;
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    long long bytes() const override
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() +
        (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
      int ghost_bytes = gauge_bytes + 3 * spinor_bytes; // 3 since we have to load the partial
      int dim = arg.mu % 4;
      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        if (arg.kernel_type != dim) break;
        bytes_ = ghost_bytes * halo.GhostFace()[dim];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = halo.GhostFace()[dim];
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume();
        bytes_ = (gauge_bytes + 2 * spinor_bytes) * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = arg.commDim[dim] ? halo.GhostFace()[dim] : 0;
        bytes_ -= ghost_bytes * ghost_sites;

        break;
      }
      }
      return bytes_;
    }

    TuneKey tuneKey() const override
    { // add mu to the key
      auto key = Dslash::tuneKey();
      strcat(key.aux, ",mu=");
      u32toa(key.aux + strlen(key.aux), arg.mu);
      return key;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct CovDevApply {

    CovDevApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &, const GaugeField &U, int mu, int parity, bool dagger,
                const int *comm_override, TimeProfile &profile)

    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in, 1, false);
      if (in.Nspin() == 4) {
        CovDevArg<Float, 4, nColor, recon, nDim> arg(out, in, halo, U, mu, parity, dagger, comm_override);
        CovDev<decltype(arg)> covDev(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(covDev)> policy(covDev, in, halo, profile);
      } else if (in.Nspin() == 1) {
        CovDevArg<Float, 1, nColor, recon, nDim> arg(out, in, halo, U, mu, parity, dagger, comm_override);
        CovDev<decltype(arg)> covDev(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(covDev)> policy(covDev, in, halo, profile);
      } else {
        errorQuda("Spin not supported");
      }
    }
  };

  // Apply the covariant derivative operator
  // out(x) = U_{\mu}(x)in(x+mu) for mu = 0...3
  // out(x) = U^\dagger_mu'(x-mu')in(x-mu') for mu = 4...7 and we set mu' = mu-4
  void ApplyCovDev(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                   int mu, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_COVDEV_DSLASH>()) {
      instantiate<CovDevApply>(out, in, in, U, mu, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Covariant derivative kernels have not been built");
    }
  }

} // namespace quda
