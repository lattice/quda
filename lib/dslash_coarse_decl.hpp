#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <dslash_quda.h>

/**
   Host-visible declarations of the per-store-precision coarse dslash
   entry points. Definitions live in the precision .cu files.
   This header must not include dslash_coarse.hpp (CUDA templates).
*/

namespace quda
{

  template <bool dagger, int coarseColor, typename Float, typename yFloat, typename ghostFloat>
  void ApplyCoarse_t(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                     cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                     int parity, bool dslash, bool clover, DslashType type, MemoryLocation *halo_location,
                     const ColorSpinorField &halo);

  template <bool dagger, int coarseColor, int nVec, typename Float, typename yFloat, typename ghostFloat>
  void ApplyCoarseMma_t(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                        cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                        int parity, bool dslash, bool clover, DslashType type, MemoryLocation *halo_location,
                        const ColorSpinorField &halo);

} // namespace quda
