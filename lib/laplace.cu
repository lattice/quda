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

#include <dslash_policy.cuh>
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
    using Dslash::in;

  public:
    Laplace(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
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

    long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin() == 4 ? 2 : 1;
      int ghost_flops = (num_mv_multiply * mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);      // 3D or 4D operator

      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        flops_ = (ghost_flops + (arg.xpay ? xpay_flops : xpay_flops / 2)) * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        flops_ = (num_dir * (in.Nspin() / 4) * in.Ncolor() * in.Nspin() + // spin project (=0 for staggered)
                  num_dir * num_mv_multiply * mv_flops +                  // SU(3) matrix-vector multiplies
                  ((num_dir - 1) * 2 * in.Ncolor() * in.Nspin()))
          * sites; // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    virtual long long bytes() const
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int proj_spinor_bytes = in.Nspin() == 4 ? spinor_bytes / 2 : spinor_bytes;
      int ghost_bytes = (proj_spinor_bytes + gauge_bytes) + 2 * spinor_bytes; // 2 since we have to load the partial
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);                           // 3D or 4D operator

      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes_ = ghost_bytes * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        bytes_ = (num_dir * gauge_bytes + ((num_dir - 2) * spinor_bytes + 2 * proj_spinor_bytes) + spinor_bytes) * sites;
        if (arg.xpay) bytes_ += spinor_bytes;
	
        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes_ -= ghost_bytes * ghost_sites;
	
        break;
      }
      }
      return bytes_;
    }
    
    TuneKey tuneKey() const
    {
      // add laplace transverse dir to the key
      char aux[TuneKey::aux_n];
      strcpy(aux,
             (arg.pack_blocks > 0 && arg.kernel_type == INTERIOR_KERNEL) ? Dslash::aux_pack :
                                                                           Dslash::aux[arg.kernel_type]);
      strcat(aux, ",laplace=");
      char laplace[32];
      u32toa(laplace, arg.dir);
      strcat(aux, laplace);
      return TuneKey(in.VolString(), typeid(*this).name(), aux);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct LaplaceApply {

    inline LaplaceApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir,
                        double a, double b, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile)
    {
      if (in.Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
        constexpr int nDim = 4;
        constexpr int nSpin = 1;
        LaplaceArg<Float, nSpin, nColor, nDim, recon> arg(out, in, U, dir, a, b, x, parity, dagger, comm_override);
        Laplace<decltype(arg)> laplace(arg, out, in);

        dslash::DslashPolicyTune<decltype(laplace)> policy(
          laplace, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
#else
        errorQuda("nSpin=1 Laplace operator required staggered dslash to be enabled");
#endif
      } else if (in.Nspin() == 4) {
#ifdef GPU_WILSON_DIRAC
        constexpr int nDim = 4;
        constexpr int nSpin = 4;
        LaplaceArg<Float, nSpin, nColor, nDim, recon> arg(out, in, U, dir, a, b, x, parity, dagger, comm_override);
        Laplace<decltype(arg)> laplace(arg, out, in);

        dslash::DslashPolicyTune<decltype(laplace)> policy(
          laplace, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
#else
        errorQuda("nSpin=4 Laplace operator required wilson dslash to be enabled");
#endif
      } else {
        errorQuda("Unsupported nSpin= %d", in.Nspin());
      }

      checkCudaError();
    }
  };

  // Apply the Laplace operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu) + b*in(x)
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double a, double b,
                    const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<LaplaceApply>(out, in, U, dir, a, b, x, parity, dagger, comm_override, profile);
  }
} // namespace quda
