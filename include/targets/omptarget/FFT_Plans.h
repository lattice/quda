#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>

using FFTPlanHandle = int;
/*
#include <cufft.h>

using FFTPlanHandle = cufftHandle;
#define FFT_FORWARD     CUFFT_FORWARD
#define FFT_INVERSE     CUFFT_INVERSE

#ifndef GPU_GAUGE_ALG
*/
#ifdef QUDA_TARGET_OMPTARGET
#define CUFFT_SAFE_CALL(call)

inline void ApplyFFT(FFTPlanHandle &, float2 *, float2 *, int)
{
  errorQuda("unimplemented");
}

inline void ApplyFFT(FFTPlanHandle &, double2 *, double2 *, int)
{
  errorQuda("unimplemented");
}

inline void SetPlanFFTMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("unimplemented");
}

inline void SetPlanFFT2DMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("unimplemented");
}

inline void FFTDestroyPlan(FFTPlanHandle &)
{
  errorQuda("unimplemented");
}
#else

/*-------------------------------------------------------------------------------*/
#define CUFFT_SAFE_CALL( call) {                                      \
    cufftResult err = call;                                         \
    if ( CUFFT_SUCCESS != err ) {                                     \
      fprintf(stderr, "CUFFT error in file '%s' in line %i.\n",   \
              __FILE__, __LINE__);                                \
      exit(EXIT_FAILURE);                                         \
    } }
/*-------------------------------------------------------------------------------*/

/**
 * @brief Call CUFFT to perform a single-precision complex-to-complex
 * transform plan in the transform direction as specified by direction
 * parameter
 * @param[in] CUFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: CUFFT_FORWARD or CUFFT_INVERSE
 */
inline void ApplyFFT(FFTPlanHandle &plan, float2 *data_in, float2 *data_out, int direction){
  CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)data_in, (cufftComplex *)data_out, direction));
}

/**
 * @brief Call CUFFT to perform a double-precision complex-to-complex transform plan in the transform direction
as specified by direction parameter
 * @param[in] CUFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: CUFFT_FORWARD or CUFFT_INVERSE
 */
inline void ApplyFFT(FFTPlanHandle &plan, double2 *data_in, double2 *data_out, int direction){
  CUFFT_SAFE_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex *)data_in, (cufftDoubleComplex *)data_out, direction));
}

/**
 * @brief Creates a CUFFT plan supporting 4D (1D+3D) data layouts for complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] precision The precision of the computation
 */

inline void SetPlanFFTMany(FFTPlanHandle &plan, int4 size, int dim, QudaPrecision precision)
{
  auto type = precision == QUDA_DOUBLE_PRECISION ? CUFFT_Z2Z : CUFFT_C2C;
  switch (dim) {
  case 1:
  {
    int n[1] = { size.w };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, type, size.x * size.y * size.z));
  }
  break;
  case 3:
  {
    int n[3] = { size.x, size.y, size.z };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, type, size.w));
  }
  break;
  }
}

/**
 * @brief Creates a CUFFT plan supporting 4D (2D+2D) data layouts for complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] precision The precision of the computation
 */
inline void SetPlanFFT2DMany(cufftHandle &plan, int4 size, int dim, QudaPrecision precision)
{
  auto type = precision == QUDA_DOUBLE_PRECISION ? CUFFT_Z2Z : CUFFT_C2C;
  switch (dim) {
  case 0:
  {
    int n[2] = { size.w, size.z };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.x * size.y));
  }
  break;
  case 1:
  {
    int n[2] = { size.x, size.y };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.z * size.w));
  }
  break;
  }
}

inline void FFTDestroyPlan( FFTPlanHandle &plan) {
   CUFFT_SAFE_CALL(cufftDestroy(plan));
}

#endif
