#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_overlap_wilson.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Arg> class HWilson : public Dslash<hwilson, Arg>
  {
    using Dslash = Dslash<hwilson, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    HWilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const hipStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay)
        Dslash::template instantiate<packShmem, true>(tp, stream);
      else
        errorQuda("HWilson operator only defined for xpay=true");
    }

  };

  template <typename Float, int nColor, QudaReconstructType recon> struct HWilsonApply {

    inline HWilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        double a, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      hwilsonArg<Float, nColor, nDim, recon> arg(out, in, U, a, x, parity, dagger, comm_override);
      HWilson<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the HWilson operator
  // out(x) = gamma5 * (1 + a*( \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)))
  void ApplyHWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      double a, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_OVERLAP_WILSON_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, x);

    // check all locations match
    checkLocation(out, in, U, x);

    instantiate<HWilsonApply>(out, in, U, a, x, parity, dagger, comm_override, profile);
#else
    errorQuda("HWilson dslash has not been built");
#endif
  }

  template <typename Float, int nColor, typename Arg>
  class OverlapLinop : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &in;
    const ColorSpinorField &out;

    long long flops() const { return 0; }
    long long bytes() const { return arg.out.Bytes() + arg.in.Bytes(); }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    OverlapLinop(Arg &arg, const ColorSpinorField &in, ColorSpinorField &out) : TunableVectorY(arg.nParity), arg(arg), in(in), out(out)
    {
      strcpy(aux, in.AuxString());
    }
    virtual ~OverlapLinop() { }

    void apply(const hipStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        overlapLinop<Float,nColor,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg); 
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); }
    void postTune() { arg.out.load(); }
  };

  template <typename Float, int nColor>
  void ApplyOverlapLinop(ColorSpinorField &out, const ColorSpinorField &in, double k0, double k1, double k2)
  {
    OverlapLinopArg<Float,nColor> arg(out, in, k0, k1, k2);
    OverlapLinop<Float,nColor,OverlapLinopArg<Float,nColor> > linop(arg,in,out);
    linop.apply(streams[Nstream-1]);
  }

  void ApplyOverlapLinop(ColorSpinorField &out, const ColorSpinorField &in, double k0, double k1, double k2)
  {
    checkPrecision(out, in);    // check all precisions match
    checkLocation(out, in);     // check all locations match

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyOverlapLinop<double,3>(out, in, k0, k1, k2);
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyOverlapLinop<float,3>(out, in, k0, k1, k2);
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      ApplyOverlapLinop<short,3>(out, in, k0, k1, k2);
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      ApplyOverlapLinop<char,3>(out, in, k0, k1, k2);
    } else {
      errorQuda("Unsupported precision %d\n", in.Precision());
    }
  }

} // namespace quda
