#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <quda_sycl_api.h>

#include <oneapi/mkl/dfti.hpp>
using namespace oneapi::mkl::dft;

typedef struct {
  bool isDouble;
  union {
    descriptor<precision::SINGLE,domain::COMPLEX> *s;
    descriptor<precision::DOUBLE,domain::COMPLEX> *d;
  };
} FFTPlanHandle;

#define FFT_FORWARD 0
#define FFT_INVERSE 1


#ifndef GPU_GAUGE_ALG

inline void ApplyFFT(FFTPlanHandle &, float2 *, float2 *, int)
{
  errorQuda("GPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void ApplyFFT(FFTPlanHandle &, double2 *, double2 *, int)
{
  errorQuda("GPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void SetPlanFFTMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("GPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void SetPlanFFT2DMany(FFTPlanHandle &, int4, int, QudaPrecision)
{
  errorQuda("GPU_GAUGE_ALG is disabled so FFTs are also disabled");
}

inline void FFTDestroyPlan(FFTPlanHandle &) { errorQuda("GPU_GAUGE_ALG is disabled so FFTs are also disabled"); }

#else

#if 0
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
#endif

/**
 * @brief Call MKL to perform a single-precision complex-to-complex
 * transform plan in the transform direction as specified by direction
 * parameter
 * @param[in] MKL FFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: CUFFT_FORWARD or CUFFT_INVERSE
 */
inline void ApplyFFT(FFTPlanHandle &plan, float2 *data_in, float2 *data_out, int direction)
{
  if(plan.isDouble) {
    errorQuda("Called single precision FFT with double precision plan\n");
  }
  cl::sycl::event e;
  if(direction == FFT_FORWARD) {
    //warningQuda("Forward FFT");
    e = compute_forward(*plan.s, (float *)data_in, (float *)data_out);
  } else {
    //warningQuda("Backward FFT");
    e = compute_backward(*plan.s, (float *)data_in, (float *)data_out);
  }
  e.wait();
  //warningQuda("Done FFT");
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
  if(!plan.isDouble) {
    errorQuda("Called double precision FFT with single precision plan\n");
  }
  cl::sycl::event e;
  if(direction == FFT_FORWARD) {
    e = compute_forward(*plan.d, (double *)data_in, (double *)data_out);
  } else {
    e = compute_backward(*plan.d, (double *)data_in, (double *)data_out);
  }
  e.wait();
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
  warningQuda("SetPlanFFTMany %i %i", dim, precision);
#if 0
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
#endif
}

/**
 * @brief Creates a CUFFT plan supporting 4D (2D+2D) data layouts for complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] precision The precision of the computation
 */
inline void SetPlanFFT2DMany(FFTPlanHandle &plan, int4 size, int dim, QudaPrecision precision)
{
  warningQuda("SetPlanFFT2DMany %i %i", dim, precision);
  if(precision == QUDA_SINGLE_PRECISION) {
    plan.isDouble = false;
    if(dim == 0) {
      auto q = device::defaultQueue();
      MKL_LONG distance = size.w * size.z;
      plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.w, size.z});
      //plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.z, size.w});
      plan.s->set_value(config_param::NUMBER_OF_TRANSFORMS, size.x * size.y);
      plan.s->set_value(config_param::FWD_DISTANCE, distance);
      plan.s->set_value(config_param::BWD_DISTANCE, distance);
      plan.s->commit(q);
    } else {
      auto q = device::defaultQueue();
      MKL_LONG distance = size.x * size.y;
      plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.x, size.y});
      //plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.y, size.x});
      plan.s->set_value(config_param::NUMBER_OF_TRANSFORMS, size.w * size.z);
      plan.s->set_value(config_param::FWD_DISTANCE, distance);
      plan.s->set_value(config_param::BWD_DISTANCE, distance);
      plan.s->commit(q);
    }
  } else {
    plan.isDouble = true;
  }
#if 0
  auto type = precision == QUDA_DOUBLE_PRECISION ? CUFFT_Z2Z : CUFFT_C2C;
  switch (dim) {
  case 0: {
    int n[2] = {size.w, size.z};
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.x * size.y));
  } break;
  case 1: {
    int n[2] = {size.x, size.y};
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, type, size.z * size.w));
  } break;
  }
  CUFFT_SAFE_CALL(cufftSetStream(plan, target::cuda::get_stream(device::get_default_stream())));
#endif
}

inline void FFTDestroyPlan(FFTPlanHandle &plan) {
  if(plan.isDouble) {
    plan.d->~descriptor();
  } else {
    plan.s->~descriptor();
  }
}

#endif
