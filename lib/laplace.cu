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

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
  */
  template <typename Float, int reg_block_size, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct LaplaceLaunch {

    // kernel name for jit compilation
    static constexpr const char *kernel = "quda::laplaceGPU";

    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(laplaceGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class Laplace : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    Laplace(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash<Float>(arg, out, in, "kernels/laplace.cuh"),
      arg(arg),
      in(in)
    {
    }

    virtual ~Laplace() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(tp, arg);
      Dslash<Float>::template instantiate<LaplaceLaunch, nDim, nColor>(tp, arg, stream);
    }

    long long flops() const
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int num_mv_multiply = in.Nspin() == 4 ? 2 : 1;
      int ghost_flops = (num_mv_multiply * mv_flops + 2 * in.Ncolor() * in.Nspin());
      int xpay_flops = 2 * 2 * in.Ncolor() * in.Nspin(); // multiply and add per real component
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);      // 3D or 4D operator

      long long flops_ = 0;

      // FIXME - should we count the xpay flops in the derived kernels
      // since some kernels require the xpay in the exterior (preconditiond clover)

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
      strcpy(aux, Dslash<Float>::aux[arg.kernel_type]);
      strcat(aux, ",laplace=");
      char laplace[32];
      u32toa(laplace, arg.dir);
      strcat(aux, laplace);
      return TuneKey(in.VolString(), typeid(*this).name(), aux);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct LaplaceApply {

    inline LaplaceApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double a,
                        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile)
    {

      constexpr int nDim = 4;
      LaplaceArg<Float, nColor, recon> arg(out, in, U, dir, a, x, parity, dagger, comm_override);
      Laplace<Float, nDim, nColor, LaplaceArg<Float, nColor, recon>> laplace(arg, out, in);

      dslash::DslashPolicyTune<decltype(laplace)> policy(
        laplace, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the Laplace operator
  // out(x) = M*in = - kappa*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator.
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double kappa,
                    const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {

    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    instantiate<LaplaceApply>(out, in, U, dir, kappa, x, parity, dagger, comm_override, profile);
  }
} // namespace quda
