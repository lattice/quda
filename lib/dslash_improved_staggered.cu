#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.cuh>
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
    using Dslash::in;

  public:
    Staggered(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

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
      case EXTERIOR_KERNEL_T: flops_ = ghost_flops * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        flops_ = ghost_flops * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        long long sites = in.Volume();
        flops_ = (2 * num_dir * mv_flops + // SU(3) matrix-vector multiplies
                  (2 * num_dir - 1) * 2 * in.Ncolor() * in.Nspin())
          * sites;                                  // accumulation
        if (arg.xpay) flops_ += xpay_flops * sites; // axpy is always on interior

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

    long long bytes() const
    {
      int gauge_bytes_fat = QUDA_RECONSTRUCT_NO * in.Precision();
      int gauge_bytes_long = arg.reconstruct * in.Precision();
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      int ghost_bytes = 3 * (spinor_bytes + gauge_bytes_long) + (spinor_bytes + gauge_bytes_fat)
        + 3 * 2 * spinor_bytes; // last term is the accumulator load/store through the face
      int num_dir = 2 * 4;      // set to 4-d since we take care of 5-d fermions in derived classes where necessary

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
        bytes_ = (num_dir * (gauge_bytes_fat + gauge_bytes_long) + // gauge reads
                  num_dir * 2 * spinor_bytes +                     // spinor reads
                  spinor_bytes)
          * sites; // spinor write
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

  };

  template <typename Float, int nColor, QudaReconstructType recon_l> struct ImprovedStaggeredApply {

    inline ImprovedStaggeredApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &L,
                                  const GaugeField &U, double a, const ColorSpinorField &x, int parity, bool dagger,
                                  const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
      constexpr bool improved = true;
      constexpr QudaReconstructType recon_u = QUDA_RECONSTRUCT_NO;
      StaggeredArg<Float, nColor, nDim, recon_u, recon_l, improved> arg(out, in, U, L, a, x, parity, dagger,
                                                                        comm_override);
      Staggered<decltype(arg)> staggered(arg, out, in);

      dslash::DslashPolicyTune<decltype(staggered)> policy(
        staggered, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  void ApplyImprovedStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                              const GaugeField &L, double a, const ColorSpinorField &x, int parity, bool dagger,
                              const int *comm_override, TimeProfile &profile)
  {

#ifdef GPU_STAGGERED_DIRAC
    for (int i = 0; i < 4; i++) {
      if (comm_dim_partitioned(i) && (U.X()[i] < 6)) {
        errorQuda(
          "ERROR: partitioned dimension with local size less than 6 is not supported in improved staggered dslash\n");
      }
    }

    // L must be first gauge field argument since we template on long reconstruct
    instantiate<ImprovedStaggeredApply, StaggeredReconstruct>(out, in, L, U, a, x, parity, dagger, comm_override,
                                                              profile);
#else
    errorQuda("Staggered dslash has not been built");
#endif
  }

} // namespace quda
