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
#include <kernels/laplace.cuh>

/**
   This is the laplacian derivative based on the basic gauged differential operator
*/

namespace quda
{

  template <typename Arg> class Laplace : public Dslash<laplace, Arg>
  {
    using Dslash = Dslash<laplace, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;

  public:
    Laplace(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
            const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);

      // operator is Hermitian so do not instantiate dagger
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

    long long flops() const override
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int ghost_flops = (in.Nspin() * mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);      // 3D or 4D operator

      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * 2 * halo.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites
          = 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume();
        flops_ = (num_dir * in.Nspin() * mv_flops +                  // SU(3) matrix-vector multiplies
                  ((num_dir - 1) * 2 * in.Ncolor() * in.Nspin()))
          * sites; // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites;

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

    virtual long long bytes() const override
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
      int proj_spinor_bytes = in.Nspin() == 4 ? spinor_bytes / 2 : spinor_bytes;
      int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + 2 * spinor_bytes; // 2 since we have to load the partial
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);                           // 3D or 4D operator

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
        bytes_ = (num_dir * gauge_bytes + ((num_dir - 2) * spinor_bytes + 2 * proj_spinor_bytes) + spinor_bytes) * sites;
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
    
    TuneKey tuneKey() const override
    { // add laplace transverse dir to the key
      auto key = Dslash::tuneKey();
      strcat(key.aux, ",laplace=");
      u32toa(key.aux + strlen(key.aux), arg.dir);
      return key;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct LaplaceApply {

    LaplaceApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                 cvector_ref<const ColorSpinorField> &x, const GaugeField &U, int dir, double a, double b, int parity,
                 const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in, 1, false);
      if (in.Nspin() == 1) {
        constexpr int nSpin = 1;
        LaplaceArg<Float, nSpin, nColor, nDim, recon> arg(out, in, halo, U, dir, a, b, x, parity, comm_override);
        Laplace<decltype(arg)> laplace(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(laplace)> policy(laplace, in, halo, profile);
      } else if (in.Nspin() == 4) {
        constexpr int nSpin = 4;
        LaplaceArg<Float, nSpin, nColor, nDim, recon> arg(out, in, halo, U, dir, a, b, x, parity, comm_override);
        Laplace<decltype(arg)> laplace(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(laplace)> policy(laplace, in, halo, profile);
      } else {
        errorQuda("Unsupported nSpin= %d", in.Nspin());
      }
    }
  };

  // Apply the Laplace operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu) + b*in(x)
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                    int dir, double a, double b, cvector_ref<const ColorSpinorField> &x, int parity,
                    const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_LAPLACE_DSLASH>()) {
      instantiate<LaplaceApply>(out, in, x, U, dir, a, b, parity, comm_override, profile);
    } else {
      errorQuda("Laplace operator has not been enabled");
    }
  }
} // namespace quda
