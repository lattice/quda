#pragma once

#include <quda_cuda_api.h>
#include <quda_internal.h>
#include <cufft.h>

#define FFT_FORWARD CUFFT_FORWARD
#define FFT_INVERSE CUFFT_INVERSE

namespace quda
{

  using FFTPlanHandle = cufftHandle;

/**
   @brief Helper function for decoding cuFFT return codes
*/
static const char *cufftGetErrorEnum(cufftResult error)
{
  switch (error) {
  case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
  case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
  case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
  case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
  case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
  default: return "<unknown error>";
  }
}

#define CUFFT_SAFE_CALL(call)                                                                                          \
  {                                                                                                                    \
    cufftResult err = call;                                                                                            \
    if (CUFFT_SUCCESS != err) { errorQuda("CUFFT error %s", cufftGetErrorEnum(err)); }                                 \
  }

/**
 * @brief Call CUFFT to perform a single-precision complex-to-complex
 * transform plan in the transform direction as specified by direction
 * parameter
 * @param[in] CUFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: CUFFT_FORWARD or CUFFT_INVERSE
 */
inline void ApplyFFT(FFTPlanHandle &plan, float2 *data_in, float2 *data_out, int direction)
{
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
inline void ApplyFFT(FFTPlanHandle &plan, double2 *data_in, double2 *data_out, int direction)
{
  CUFFT_SAFE_CALL(cufftExecZ2Z(plan, (cufftDoubleComplex *)data_in, (cufftDoubleComplex *)data_out, direction));
}

/**
 * @brief Creates a CUFFT plan supporting 4D (1D+3D) data layouts for complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and
 * Nz with batch size Nt
 * @param[in] precision The precision of the computation
 */

inline void SetPlanFFTMany(FFTPlanHandle &plan, int4 size, int dim, QudaPrecision precision)
{
  auto type = precision == QUDA_DOUBLE_PRECISION ? CUFFT_Z2Z : CUFFT_C2C;
  switch (dim) {
  case 1: {
    int n[1] = {size.w};
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, type, size.x * size.y * size.z));
  } break;
  case 3: {
    int n[3] = {size.x, size.y, size.z};
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, type, size.w));
  } break;
  }
  CUFFT_SAFE_CALL(cufftSetStream(plan, target::cuda::get_stream(device::get_default_stream())));
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
  case 0: {
    int n[2] = {size.w, size.z}; // outer-most dimension is first
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.x * size.y));
  } break;
  case 1: {
    int n[2] = {size.y, size.x}; // outer-most dimension is first
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.z * size.w));
  } break;
  }
  CUFFT_SAFE_CALL(cufftSetStream(plan, target::cuda::get_stream(device::get_default_stream())));
}

inline void FFTDestroyPlan(FFTPlanHandle &plan) { CUFFT_SAFE_CALL(cufftDestroy(plan)); }

} // namespace quda
