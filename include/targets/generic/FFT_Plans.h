#pragma once

#include <quda_internal.h>

// Dummy implementation that does nothing

#define FFT_FORWARD 0
#define FFT_INVERSE 1

namespace quda
{

  typedef struct {
    bool isDouble;
  } FFTPlanHandle;

  inline static constexpr bool HaveFFT() { return false; }

  inline void ApplyFFT(FFTPlanHandle &, float2 *, float2 *, int) { errorQuda("FFTs are disabled"); }

  inline void ApplyFFT(FFTPlanHandle &, double2 *, double2 *, int) { errorQuda("FFTs are disabled"); }

  inline void SetPlanFFTMany(FFTPlanHandle &, int4, int, QudaPrecision) { errorQuda("FFTs are disabled"); }

  inline void SetPlanFFT2DMany(FFTPlanHandle &, int4, int, QudaPrecision) { errorQuda("FFTs are disabled"); }

  inline void FFTDestroyPlan(FFTPlanHandle &) { errorQuda("FFTs are disabled"); }

} // namespace quda
