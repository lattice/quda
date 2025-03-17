#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  template <typename Arg>
  class Wilson : public Dslash<wilson, Arg, /* check_bounds */ false, /* launch_bounds */ wilson_use_reg_realloc>
  {
    using Dslash = Dslash<wilson, Arg, /* check_bounds */ false, /* launch_bounds */ wilson_use_reg_realloc>;

  public:
    Wilson(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
           const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo,
             wilson_use_async ? ",async-p" + std::to_string(pipeline_depth) + "-specialized-reg-realloc"
                 + std::to_string(wilson_use_reg_realloc) :
                                "")
    {
    }

    unsigned int sharedBytesPerThread() const override
    {
      if constexpr (wilson_use_async) {
        using Float = typename Arg::Float;
        int bulk = Arg::F::length * sizeof(Float);
        int norm = isFixed<Float>::value ? sizeof(float) : 0;
        int gauge = Arg::G::reconLen * sizeof(Float);
        return (bulk + norm + gauge) * pipeline_depth / 2;
      } else {
        return 0;
      }
    }

    unsigned int minThreads() const override { return Dslash::minThreads() * 2; }

    int blockStep() const override { return (wilson_use_reg_realloc ? 8 : 2) * device::warp_size(); }

    int blockMin() const override { return (wilson_use_reg_realloc ? 8 : 2) * device::warp_size(); }

    unsigned int maxBlockSize(const TuneParam &) const
    {
      return (wilson_use_reg_realloc ? 8 : 16) * device::warp_size();
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::arg.half_block_dim = tp.block.x / 2;
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <bool distance_pc> struct DistanceType {
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    template <bool distance_pc>
    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double alpha0, int t0,
                int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonArg<Float, nColor, nDim, recon, distance_pc> arg(out, in, halo, U, a, x, parity, dagger, comm_override,
                                                             alpha0, t0);
      Wilson<decltype(arg)> wilson(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

} // namespace quda
