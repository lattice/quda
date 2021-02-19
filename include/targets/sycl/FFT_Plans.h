#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>

using FFTPlanHandle = int;
#define FFT_FORWARD     0
#define FFT_INVERSE     1

#define CUFFT_SAFE_CALL(call)

inline void ApplyFFT(FFTPlanHandle &, float2 *, float2 *, int)
{
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void ApplyFFT(FFTPlanHandle &, double2 *, double2 *, int)
{
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void SetPlanFFTMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void SetPlanFFT2DMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void FFTDestroyPlan(FFTPlanHandle &)
{
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
