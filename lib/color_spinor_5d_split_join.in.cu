#include "color_spinor_field.h"
#include "dslash_quda.h"
#include "tunable_nd.h"
#include "instantiate.h"
#include "int_list.hpp"
#include "multigrid.h"
#include "kernels/color_spinor_5d_split_join.cuh"

namespace quda {

  template <typename Float, int nColor, int nSpin> class Split5DTo4DApply : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    const ColorSpinorField &in;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    Split5DTo4DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(out[0], out.size(), in[0].SiteSubset()), out(out), in(in[0])
    {
      strcat(aux,",Ls=");
      char lsstr[3];
      i32toa(lsstr, static_cast<int>(out.size()));
      strcat(aux, lsstr);

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Split5DTo4D>(tp, stream, Split5DTo4DArgs<Float, nColor, nSpin>(out, in));
    }

    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  template <int nColor, int nSpin>
  void Split5DTo4DFields2(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in_)
  {
    auto prec = in_.Precision();
    // wrap
    cvector_ref<const ColorSpinorField> in{in_};

    if constexpr (nColor == 3) {
      // instantiate over all enabled precisions
      if (prec == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled(QUDA_DOUBLE_PRECISION))
          Split5DTo4DApply<double, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
      } else if (prec == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          Split5DTo4DApply<float, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else if (prec == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          Split5DTo4DApply<short, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (prec == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          Split5DTo4DApply<int8_t, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
      } else {
        errorQuda("Unsupported precision %d\n", prec);
      }
    } else {
      if (prec == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          Split5DTo4DApply<double, nColor, nSpin>(out, in);
        else
          errorQuda("double precision multigrid has not been enabled");
      } else if (prec == QUDA_SINGLE_PRECISION) {
        if (is_enabled(QUDA_SINGLE_PRECISION))
          Split5DTo4DApply<float, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else {
        errorQuda("Unsupported precision %d\n", prec);
      }
    }
  }

  template <int nSpin, int nColor, int... N>
  void Split5DTo4DFields(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in, IntList<nColor, N...>)
  {
    if (in.Ncolor() == nColor) {
      Split5DTo4DFields2<nColor, nSpin>(out, in);
    } else {
      if constexpr (sizeof...(N) > 0) {
        Split5DTo4DFields<nSpin>(out, in, IntList<N...>());
      } else {
        errorQuda("Ncolor %d with Nspin %d has not been instantiated", in.Ncolor(), in.Nspin());
      }
    }
  }

  void Split5DTo4DFields(cvector_ref<ColorSpinorField> &out, const ColorSpinorField &in)
  {
    checkSpin(out[0], in);
    checkColor(out[0], in);
    checkPrecision(out[0], in);

    if (in.Nspin() == 4) {
      if constexpr (is_enabled_spin(4))
        Split5DTo4DFields<4>(out, in, IntList<3>());
      else
        errorQuda("Unsupported spin %d", in.Nspin());
    } else if (in.Nspin() == 2) {
      if constexpr (is_enabled_spin(2) && is_enabled_multigrid()) {
        // clang-format off
        IntList<@QUDA_MULTIGRID_NVEC_LIST@> Ncolors;
        // clang-format on
        Split5DTo4DFields<2>(out, in, Ncolors);
      } else {
        errorQuda("Unsupported spin %d", in.Nspin());
      }
    } else {
      errorQuda("Unsupported spin %d", in.Nspin());
    }
  }

  template <typename Float, int nColor, int nSpin> class Join4DTo5DApply : public TunableKernel3D
  {
    ColorSpinorField &out;
    cvector_ref<const ColorSpinorField> &in;
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    Join4DTo5DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) :
      TunableKernel3D(in[0], in.size(), out[0].SiteSubset()), out(out[0]), in(in)
    {
      strcat(aux,",Ls=");
      char lsstr[3];
      i32toa(lsstr, static_cast<int>(in.size()));
      strcat(aux, lsstr);

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Join4DTo5D>(tp, stream, Join4DTo5DArgs<Float, nColor, nSpin>(out, in));
    }

    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  template <int nColor, int nSpin>
  void Join4DTo5DField2(ColorSpinorField &out_, cvector_ref<const ColorSpinorField> &in)
  {
    auto prec = out_.Precision();
    // wrap
    cvector_ref<ColorSpinorField> out{out_};

    if constexpr (nColor == 3) {
      // instantiate over all enabled precisions
      if (prec == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled(QUDA_DOUBLE_PRECISION))
          Join4DTo5DApply<double, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
      } else if (prec == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          Join4DTo5DApply<float, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else if (prec == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          Join4DTo5DApply<short, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (prec == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          Join4DTo5DApply<int8_t, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
      } else {
        errorQuda("Unsupported precision %d\n", prec);
      }
    } else {
      if (prec == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          Join4DTo5DApply<double, nColor, nSpin>(out, in);
        else
          errorQuda("double precision multigrid has not been enabled");
      } else if (prec == QUDA_SINGLE_PRECISION) {
        if (is_enabled(QUDA_SINGLE_PRECISION))
          Join4DTo5DApply<float, nColor, nSpin>(out, in);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else {
        errorQuda("Unsupported precision %d\n", prec);
      }
    }
  }

  template <int nSpin, int nColor, int... N>
  void Join4DTo5DField(ColorSpinorField &out, cvector_ref<const ColorSpinorField> &in, IntList<nColor, N...>)
  {
    if (out.Ncolor() == nColor) {
      Join4DTo5DField2<nColor, nSpin>(out, in);
    } else {
      if constexpr (sizeof...(N) > 0) {
        Join4DTo5DField<nSpin>(out, in, IntList<N...>());
      } else {
        errorQuda("Ncolor %d with Nspin %d has not been instantiated", out.Ncolor(), out.Nspin());
      }
    }
  }

  void Join4DTo5DField(ColorSpinorField &out, cvector_ref<const ColorSpinorField> &in)
  {
    checkSpin(out, in[0]);
    checkColor(out, in[0]);
    checkPrecision(out, in[0]);

    if (out.Nspin() == 4) {
      if constexpr (is_enabled_spin(4))
        Join4DTo5DField<4>(out, in, IntList<3>());
      else
        errorQuda("Unsupported spin %d", out.Nspin());
    } else if (out.Nspin() == 2) {
      if constexpr (is_enabled_spin(2) && is_enabled_multigrid()) {
        // clang-format off
        IntList<@QUDA_MULTIGRID_NVEC_LIST@> Ncolors;
        // clang-format on
        Join4DTo5DField<2>(out, in, Ncolors);
      } else {
        errorQuda("Unsupported spin %d", out.Nspin());
      }
    } else {
      errorQuda("Unsupported spin %d", out.Nspin());
    }
  }

}