#pragma once

#include <typeinfo>

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <instantiate.h>

namespace quda
{

  /**
     @brief This instantiate function is used to instantiate the reconstruct types used
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon, typename Float, int nColor,
            typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&...args)
  {
    if (U.Reconstruct() == Recon::recon[0]) {
#if QUDA_RECONSTRUCT & 4
      Apply<Float, nColor, Recon::recon[0]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-18", QUDA_RECONSTRUCT);
#endif
    } else if (U.Reconstruct() == Recon::recon[1]) {
#if QUDA_RECONSTRUCT & 2
      Apply<Float, nColor, Recon::recon[1]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12/13", QUDA_RECONSTRUCT);
#endif
    } else if (U.Reconstruct() == Recon::recon[2]) {
#if QUDA_RECONSTRUCT & 1
      Apply<Float, nColor, Recon::recon[2]>(out, in, U, args...);
#else
      errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8/9", QUDA_RECONSTRUCT);
#endif
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the colors
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon, typename Float, typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&...args)
  {
    if (in.Ncolor() == 3) {
      instantiate<Apply, Recon, Float, 3>(out, in, U, args...);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  /**
     @brief This instantiate function is used to instantiate the precisions
     @param[out] out Output result field
     @param[in] in Input field
     @param[in] U Gauge field
     @param[in] args Additional arguments for different dslash kernels
  */
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  inline void instantiate(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, Args &&...args)
  {
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8
      instantiate<Apply, Recon, double>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      instantiate<Apply, Recon, float>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      instantiate<Apply, Recon, short>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
      instantiate<Apply, Recon, int8_t>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
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
#if (QUDA_PRECISION & 2) || (QUDA_PRECISION & 1)
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  inline void instantiatePreconditioner(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                        Args &&...args)
  {
    if (U.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      instantiate<Apply, Recon, short>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
      instantiate<Apply, Recon, int8_t>(out, in, U, args...);
#else
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }
#else
  template <template <typename, int, QudaReconstructType> class Apply, typename Recon = WilsonReconstruct, typename... Args>
  inline void instantiatePreconditioner(ColorSpinorField &, const ColorSpinorField &, const GaugeField &U, Args &&...)
  {
    if (U.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (U.Precision() == QUDA_QUARTER_PRECISION) {
      errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
  }
#endif

} // namespace quda
