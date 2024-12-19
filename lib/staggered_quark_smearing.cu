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
#include <kernels/staggered_quark_smearing.cuh>

/**
   This is the staggered (two-link) Laplacian derivative.
*/

namespace quda
{

  template <typename Arg> class StaggeredQSmear : public Dslash<staggered_qsmear, Arg>
  {
    using Dslash = Dslash<staggered_qsmear, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;

  public:
    StaggeredQSmear(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream) override
    {
      if (arg.is_t0_kernel) {
        arg.exterior_threads = this->Nface()
          * (halo.GhostFaceCB()[0] + halo.GhostFaceCB()[1] + halo.GhostFaceCB()[2] + halo.GhostFaceCB()[3])
          / (in.X(3) * in.size());
        switch (arg.kernel_type) {
        case EXTERIOR_KERNEL_X:
        case EXTERIOR_KERNEL_Y:
        case EXTERIOR_KERNEL_Z:
        case EXTERIOR_KERNEL_T:
          arg.threads = this->Nface() * halo.GhostFaceCB()[arg.kernel_type] / (in.X(3) * in.size()); break;
        case EXTERIOR_KERNEL_ALL: arg.threads = arg.exterior_threads; break;
        case INTERIOR_KERNEL:
        case UBER_KERNEL: arg.threads = in.VolumeCB() / in.X(3); break;
        default: errorQuda("Unexpected kernel type %d", arg.kernel_type);
        }
      }

      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);

      // operator is Hermitian so do not instantiate dagger
      if (arg.nParity == 1) {
        Dslash::template instantiate<packStaggeredShmem, 1, false, false>(tp, stream);
      } else if (arg.nParity == 2) {
        Dslash::template instantiate<packStaggeredShmem, 2, false, false>(tp, stream);
      }
    }

    long long flops() const override
    {
      int mv_flops = (8 * in.Ncolor() - 2) * in.Ncolor(); // SU(3) matrix-vector flops
      int ghost_flops = mv_flops + 2 * in.Ncolor();
      int num_dir = arg.dir == 4 ? 2 * 4 : 2 * 3; // 3D or 4D operator
      long long flops_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        flops_ = ghost_flops * 2 * (halo.GhostFace()[arg.kernel_type] / (arg.is_t0_kernel ? in.X(3) : 1));
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2
          * ((halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3])
             / (arg.is_t0_kernel ? in.X(3) : 1));
        flops_ = ghost_flops * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        long long sites = halo.Volume() / (arg.is_t0_kernel ? in.X(3) : 1);
        // mv products + accumulation
        flops_ = (num_dir * mv_flops + (num_dir - 1) * 2 * in.Ncolor()) * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * (halo.GhostFace()[d] / (arg.is_t0_kernel ? in.X(3) : 1));
        flops_ -= ghost_flops * ghost_sites;

        break;
      }
      }

      return flops_;
    }

    virtual long long bytes() const override
    {
      int gauge_bytes = arg.reconstruct * in.Precision();
      int spinor_bytes = 2 * in.Ncolor() * in.Precision() + (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
      int ghost_bytes = (spinor_bytes + gauge_bytes) + 2 * spinor_bytes;      // 2 since we have to load the partial
      int num_dir = (arg.dir == 4 ? 2 * 4 : 2 * 3);                           // 3D or 4D operator
      int pack_bytes = 2 * 2 * in.Ncolor() * in.Precision();

      long long bytes_ = 0;

      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        bytes_ = ghost_bytes * 2 * (halo.GhostFace()[arg.kernel_type] / (arg.is_t0_kernel ? in.X(3) : 1));
        break;
      case EXTERIOR_KERNEL_ALL: {
        long long ghost_sites = 2
          * ((halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3])
             / (arg.is_t0_kernel ? in.X(3) : 1));
        bytes_ = ghost_bytes * ghost_sites;
        break;
      }
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        if (arg.pack_threads && (arg.kernel_type == INTERIOR_KERNEL || arg.kernel_type == UBER_KERNEL))
          bytes_ += pack_bytes * arg.nParity * halo.getDslashConstant().Ls * arg.pack_threads;
        long long sites = halo.Volume() / (arg.is_t0_kernel ? in.X(3) : 1);
        bytes_ = (num_dir * (gauge_bytes + spinor_bytes) + spinor_bytes) * sites;

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * (halo.GhostFace()[d] / (arg.is_t0_kernel ? in.X(3) : 1));
        bytes_ -= ghost_bytes * ghost_sites;

        break;
      }
      }
      return bytes_;
    }

    TuneKey tuneKey() const override
    {
      // add laplace transverse dir to the key
      char aux[TuneKey::aux_n];
      strcpy(aux,
             (arg.pack_blocks > 0 && arg.kernel_type == INTERIOR_KERNEL) ? Dslash::aux_pack :
                                                                           Dslash::aux[arg.kernel_type]);
      strcat(aux, ",staggered_qsmear=");
      char staggered_qsmear_[32];
      u32toa(staggered_qsmear_, arg.dir);
      strcat(aux, staggered_qsmear_);

      strcat(aux, ",tslice_kernel=");
      char tslice_kernel[4];
      u32toa(tslice_kernel, arg.is_t0_kernel);
      strcat(aux, tslice_kernel);

      return TuneKey(in.VolString().c_str(), typeid(*this).name(), aux);
    }

    /**
       @brief Compute the tuning rank
       @return The rank on which to do kernel tuning
    */
    int32_t getTuneRank() const override {
      int32_t tune_rank = 0;

      if (arg.is_t0_kernel) { // find the minimum rank for tuning
        tune_rank = ( arg.t0 < 0 ) ? comm_size() : comm_rank_global();
        comm_allreduce_min(tune_rank);
      }

      return tune_rank;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct StaggeredQSmearApply {
    StaggeredQSmearApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         cvector_ref<const ColorSpinorField> &, const GaugeField &U, int t0, bool is_tslice_kernel,
                         int parity, int dir, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      if (in.Nspin() == 1) {
        constexpr int nDim = 4;
        constexpr int nSpin = 1;

        auto halo = ColorSpinorField::create_comms_batch(in, 3);
        StaggeredQSmearArg<Float, nSpin, nColor, nDim, recon> arg(out, in, halo, U, t0, is_tslice_kernel, parity, dir,
                                                                  dagger, comm_override);
        StaggeredQSmear<decltype(arg)> staggered_qsmear(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(staggered_qsmear)> policy(staggered_qsmear, in, halo, profile);
      } else {
        errorQuda("Unsupported nSpin = %d", in.Nspin());
      }
    }
  };

  // Apply the StaggeredQSmear operator
  void ApplyStaggeredQSmear(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                            const GaugeField &U, int t0, bool is_tslice_kernel, int parity, int dir, bool dagger,
                            const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      // Local lattice size should be bigger than or equal to 6 in every partitioned direction.
      for (int i = 0; i < 4; i++) {
        if (comm_dim_partitioned(i) && (U.X()[i] < 6)) {
          errorQuda(
            "ERROR: partitioned dimension with local size less than 6 is not supported in two-link Gaussian smearing");
        }
      }
      instantiate<StaggeredQSmearApply>(out, in, in, U, t0, is_tslice_kernel, parity, dir, dagger, comm_override,
                                        profile);
    } else {
      errorQuda("StaggeredQSmear operator requires the staggered operator to be enabled");
    }
  }

} // namespace quda
