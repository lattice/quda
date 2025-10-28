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

  template <typename Arg> class Wilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;
    const GaugeField &U;

  public:
    Wilson(Arg &arg, const GaugeField &U, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
           const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo), U(U)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      const_cast<quda::gauge::tensor_desc_t&>(Dslash::arg.U.tensor_desc) = U.get_tensor_descriptor(tp.block.x);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct WilsonApply {

    template <bool distance_pc>
    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double alpha0, int t0,
                int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);

#ifdef QUDA_DSLASH_DOUBLE_STORE
      GaugeField Uback = shift(U, 1);
#else
      const GaugeField &Uback = U;
#endif

      WilsonArg<Float, nColor, nDim, DDArg, recon, distance_pc> arg(out, in, halo, U, Uback, a, x, parity, dagger,
                                                                    comm_override, alpha0, t0);
      Wilson<decltype(arg)> wilson(arg, U, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

} // namespace quda
