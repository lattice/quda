#pragma once

#ifndef NATIVE_FFT_LIB
#include "../generic/FFT_Plans.h"
#else

#include <quda_internal.h>
#include <quda_sycl_api.h>
#include <oneapi/mkl/dfti.hpp>
using namespace oneapi::mkl::dft;

#define FFT_FORWARD 0
#define FFT_INVERSE 1

namespace quda
{

  typedef struct {
    bool isDouble;
    union {
      descriptor<precision::SINGLE,domain::COMPLEX> *s;
      descriptor<precision::DOUBLE,domain::COMPLEX> *d;
    };
  } FFTPlanHandle;

  inline static constexpr bool HaveFFT() { return true; }

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
    sycl::event e;
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
    sycl::event e;
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

  //inline void SetPlanFFTMany(FFTPlanHandle &plan, int4 size, int dim, QudaPrecision precision)
  inline void SetPlanFFTMany(FFTPlanHandle &, int4 , int dim, QudaPrecision precision)
  {
    warningQuda("SetPlanFFTMany %i %i : unimplemented", dim, precision);
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
    //warningQuda("SetPlanFFT2DMany %i %i", dim, precision);
    if(precision == QUDA_SINGLE_PRECISION) {
      plan.isDouble = false;
      if(dim == 0) {
	auto q = quda::device::defaultQueue();
	MKL_LONG distance = size.w * size.z;
	plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.w, size.z});
	//plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.z, size.w});
	plan.s->set_value(config_param::NUMBER_OF_TRANSFORMS, size.x * size.y);
	plan.s->set_value(config_param::FWD_DISTANCE, distance);
	plan.s->set_value(config_param::BWD_DISTANCE, distance);
	plan.s->set_value(config_param::BACKWARD_SCALE, (1.0/distance));
	plan.s->commit(q);
      } else {
	auto q = quda::device::defaultQueue();
	MKL_LONG distance = size.x * size.y;
	//plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.x, size.y});
	plan.s = new std::remove_pointer_t<decltype(plan.s)>({size.y, size.x});
	plan.s->set_value(config_param::NUMBER_OF_TRANSFORMS, size.w * size.z);
	plan.s->set_value(config_param::FWD_DISTANCE, distance);
	plan.s->set_value(config_param::BWD_DISTANCE, distance);
	plan.s->set_value(config_param::BACKWARD_SCALE, (1.0/distance));
	plan.s->commit(q);
      }
    } else {
      plan.isDouble = true;
      if(dim == 0) {
	auto q = quda::device::defaultQueue();
	MKL_LONG distance = size.w * size.z;
	plan.d = new std::remove_pointer_t<decltype(plan.d)>({size.w, size.z});
	//plan.d = new std::remove_pointer_t<decltype(plan.d)>({size.z, size.w});
	plan.d->set_value(config_param::NUMBER_OF_TRANSFORMS, size.x * size.y);
	plan.d->set_value(config_param::FWD_DISTANCE, distance);
	plan.d->set_value(config_param::BWD_DISTANCE, distance);
	plan.d->set_value(config_param::BACKWARD_SCALE, (1.0/distance));
	plan.d->commit(q);
      } else {
	auto q = quda::device::defaultQueue();
	MKL_LONG distance = size.x * size.y;
	//plan.d = new std::remove_pointer_t<decltype(plan.d)>({size.x, size.y});
	plan.d = new std::remove_pointer_t<decltype(plan.d)>({size.y, size.x});
	plan.d->set_value(config_param::NUMBER_OF_TRANSFORMS, size.w * size.z);
	plan.d->set_value(config_param::FWD_DISTANCE, distance);
	plan.d->set_value(config_param::BWD_DISTANCE, distance);
	plan.d->set_value(config_param::BACKWARD_SCALE, (1.0/distance));
	plan.d->commit(q);
      }
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
      //plan.d->~descriptor();
      delete plan.d;
    } else {
      //plan.s->~descriptor();
      delete plan.s;
    }
  }

}

#endif // ifndef NATIVE_FFT_LIB
