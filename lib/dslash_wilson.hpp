#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  template <typename Arg> class Wilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;

  public:
    Wilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      if (in.Ndim() == 5) { TunableKernel3D::resizeVector(in.X(4), arg.nParity); }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <bool distance_pc> struct DistanceType {
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    template <bool distance_pc>
    inline WilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double alpha0,
                       int t0, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                       DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonArg<Float, nColor, nDim, recon, distance_pc> arg(out, in, U, a, x, parity, dagger, comm_override, alpha0, t0);
      Wilson<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

} // namespace quda
