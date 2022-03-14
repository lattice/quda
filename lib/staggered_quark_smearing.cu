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
#include <kernels/staggered_quark_smearing.cuh>

/**
   This is the laplacian derivative based on the basic gauged differential operator
*/

namespace quda
{

  template <typename Arg> class StaggeredQSmear : public Dslash<staggered_qsmear, Arg>
  {
    using Dslash = Dslash<staggered_qsmear, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    StaggeredQSmear(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);

      // reset threadDimMapLower and threadDimMapUpper when t0 is given
      // partial replacation of dslash::setFusedParam()
      if( arg.is_t0_kernel )
      {
        int prev = -1;
        for( int i=0; i<4; i++ )
        {
          arg.threadDimMapLower[i] = 0;
          arg.threadDimMapUpper[i] = 0;
          if( !(arg.commDim[i]) ) continue;
          arg.threadDimMapLower[i] = ( prev>=0 ? arg.threadDimMapUpper[prev] : 0 );
          arg.threadDimMapUpper[i] = arg.threadDimMapLower[i] + this->Nface() * (in.GhostFaceCB())[i];
          prev = i;
        }
      }

      // operator is Hermitian so do not instantiate dagger
      if (arg.nParity == 1) {
        if(arg.is_t0_kernel) //FIXME we reuse xpay non-type template parameter for t0 option!
          Dslash::template instantiate<packStaggeredShmem, 1, false, true>(tp, stream);
        else
          Dslash::template instantiate<packStaggeredShmem, 1, false, false>(tp, stream);
      } else if (arg.nParity == 2) {
        if(arg.is_t0_kernel) //FIXME we reuse xpay non-type template parameter for t0 option!
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
      case UBER_KERNEL:
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
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
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
      case UBER_KERNEL:
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
      strcat(aux, ",staggered_qsmear=");
      char staggered_qsmear_[32];
      u32toa(staggered_qsmear_, arg.dir);
      strcat(aux, staggered_qsmear_);
      return TuneKey(in.VolString(), typeid(*this).name(), aux);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct StaggeredQSmearApply {

    inline StaggeredQSmearApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int t0, bool is_tslice_kernel, int parity, int dir,
                        bool dagger, const int *comm_override,
                        TimeProfile &profile)
    {
      if (in.Nspin() == 1) {
#if defined(GPU_STAGGERED_DIRAC) // && defined(GPU_LAPLACE) => GPU_QSMEARING
        constexpr int nDim = 4;
        constexpr int nSpin = 1;
        
        const int volume = is_tslice_kernel ? in.VolumeCB() / in.X(3) : in.VolumeCB();
        
        StaggeredQSmearArg<Float, nSpin, nColor, nDim, recon> arg(out, in, U, t0, is_tslice_kernel, parity, dir, dagger, comm_override);
        
        StaggeredQSmear<decltype(arg)> staggered_qsmear(arg, out, in);

        int faceVolumeCB[nDim];
        for( int i=0; i<nDim; i++ )
        {
          faceVolumeCB[i] = (in.GhostFaceCB())[i];
          if( is_tslice_kernel && i<3 ) faceVolumeCB[i] /= in.X(3);
        }

        dslash::DslashPolicyTune<decltype(staggered_qsmear)> policy(
          staggered_qsmear, in, volume,
          faceVolumeCB, profile);
#else
        errorQuda("nSpin=%d StaggeredQSmear operator required staggered dslash and laplace to be enabled", in.Nspin());
#endif
      } else {
        errorQuda("Unsupported nSpin= %d", in.Nspin());
      }
    }
  };

  // Apply the StaggeredQSmear operator
  void ApplyStaggeredQSmear(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int t0, bool is_tslice_kernel, int parity, int dir,
                    bool dagger, const int *comm_override, TimeProfile &profile)
  {
    // Local lattice size should be bigger than or equal to 6 in every partitioned direction.
    for (int i = 0; i < 4; i++) {
      if (comm_dim_partitioned(i) && (U.X()[i] < 6)) {
        errorQuda(
          "ERROR: partitioned dimension with local size less than 6 is not supported in two-link Gaussian smearing.\n");
      }
    }
    
    instantiate<StaggeredQSmearApply>(out, in, U, t0, is_tslice_kernel, parity, dir, dagger, comm_override, profile);
  }
} // namespace quda

