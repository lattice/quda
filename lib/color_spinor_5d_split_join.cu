#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "kernels/color_spinor_5d_split_join.cuh"

namespace quda {

  template <typename Float, int nColor> class Split5DTo4DApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    const ColorSpinorField &in;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Split5DTo4DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(out[0], out.size(), in[0].SiteSubset()), out(out), in(in[0])
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Split5DTo4D>(tp, stream, Split5DTo4DArgs<Float, nColor>(out, in));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  void Split5DTo4DFields(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>() || is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      instantiate_recurse2<Split5DTo4DApply>(out, cvector_ref<const ColorSpinorField>{in});
    } else {
      errorQuda("Twisted clover operator or domain wall operator has not been built");
    }
  }

  template <typename Float, int nColor> class Join4DTo5DApply : public TunableKernel3D
  {
    ColorSpinorField &out;
    cvector_ref<const ColorSpinorField> &in;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    Join4DTo5DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(in[0], in.size(), out[0].SiteSubset()), out(out[0]), in(in)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Join4DTo5D>(tp, stream, Join4DTo5DArgs<Float, nColor>(out, in));
    }

    void preTune() { out.backup(); }
    void postTune() { out.restore(); }
    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  void Join4DTo5DField(ColorSpinorField &out, cvector_ref<const ColorSpinorField> &in)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>() || is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      instantiate_recurse2<Join4DTo5DApply>(cvector_ref<ColorSpinorField>{out}, in);
    } else {
      errorQuda("Twisted mass operator or domain wall operator has not been built");
    }
  }

}