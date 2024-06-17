#pragma once

#include <typeinfo>

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <domain_decomposition_helper.cuh>

namespace quda
{

  /**
     @brief This instantiate function is used to instantiate the reconstruct types used
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, typename, QudaReconstructType> class Apply, typename Recon, typename Float,
            int nColor, typename DDArg, typename... Args>
  void instantiate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, Args &&...args)
  {
    if (U.Reconstruct() == Recon::recon[0]) {
      if constexpr (is_enabled<QUDA_RECONSTRUCT_NO>())
        Apply<Float, nColor, DDArg, Recon::recon[0]>(out, in, x, U, args...);
      else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-18", QUDA_RECONSTRUCT);
    } else if (U.Reconstruct() == Recon::recon[1]) {
      if constexpr (is_enabled<QUDA_RECONSTRUCT_12>())
        Apply<Float, nColor, DDArg, Recon::recon[1]>(out, in, x, U, args...);
      else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12/13", QUDA_RECONSTRUCT);
    } else if (U.Reconstruct() == Recon::recon[2]) {
      if constexpr (is_enabled<QUDA_RECONSTRUCT_8>())
        Apply<Float, nColor, DDArg, Recon::recon[2]>(out, in, x, U, args...);
      else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8/9", QUDA_RECONSTRUCT);
    } else {
      errorQuda("Unsupported reconstruct type %d", U.Reconstruct());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the domain decomposition type
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, typename, QudaReconstructType> class Apply, typename Recon, typename Float,
            int nColor, typename... Args>
  inline void instantiate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                          cvector_ref<const ColorSpinorField> &x, const GaugeField &U, Args &&...args)
  {
    if (out.DD().type == QUDA_DD_NO and in.DD().type == QUDA_DD_NO) {
      instantiate<Apply, Recon, Float, 3, DDNo>(out, in, x, U, args...);
#ifdef GPU_DD_DIRAC
    } else if (out.DD().type == QUDA_DD_RED_BLACK or in.DD().type == QUDA_DD_RED_BLACK) {
      instantiate<Apply, Recon, Float, 3, DDRedBlack>(out, in, x, U, args...);
#endif
    } else {
      errorQuda("Unsupported DD type %d\n", out.DD().type);
    }
  }

  /**
     @brief This instantiate function is used to instantiate the colors
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, typename, QudaReconstructType> class Apply, typename Recon, typename Float, typename... Args>
  void instantiate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, Args &&...args)
  {
    if (in.Ncolor() == 3) {
      instantiate<Apply, Recon, Float, 3>(out, in, x, U, args...);
    } else {
      errorQuda("Unsupported number of colors %d", U.Ncolor());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the precisions
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, typename, QudaReconstructType> class Apply, typename Recon = ReconstructWilson,
            typename... Args>
  void instantiate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, Args &&...args)
  {
    if (in.size() > get_max_multi_rhs()) {
      instantiate<Apply, Recon>({out.begin(), out.begin() + out.size() / 2}, {in.begin(), in.begin() + in.size() / 2},
                                {x.begin(), x.begin() + x.size() / 2}, U, args...);
      instantiate<Apply, Recon>({out.begin() + out.size() / 2, out.end()}, {in.begin() + in.size() / 2, in.end()},
                                {x.begin() + x.size() / 2, x.end()}, U, args...);
      return;
    }

    auto precision = checkPrecision(out, in, x, U); // check all precisions match
    if (!is_enabled(precision)) errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, precision);

    if (precision == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled(QUDA_DOUBLE_PRECISION)) instantiate<Apply, Recon, double>(out, in, x, U, args...);
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) instantiate<Apply, Recon, float>(out, in, x, U, args...);
    } else if (precision == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) instantiate<Apply, Recon, short>(out, in, x, U, args...);
    } else if (precision == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) instantiate<Apply, Recon, int8_t>(out, in, x, U, args...);
    } else {
      errorQuda("Unsupported precision %d", precision);
    }
  }

  /**
     @brief This instantiatePrecondtioner function is used to
     instantiate the precisions for a preconditioner.  This is the
     same as the instantiate helper above, except it only handles half
     and quarter precision.
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, typename, QudaReconstructType> class Apply, typename Recon = ReconstructWilson,
            typename... Args>
  void instantiatePreconditioner(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                 cvector_ref<const ColorSpinorField> &x, const GaugeField &U, Args &&...args)
  {
    if (!is_enabled(U.Precision()))
      errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, U.Precision());

    if (U.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) instantiate<Apply, Recon, short>(out, in, x, U, args...);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) instantiate<Apply, Recon, int8_t>(out, in, x, U, args...);
    } else {
      errorQuda("Unsupported precision %d", U.Precision());
    }
  }

} // namespace quda
