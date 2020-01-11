#pragma once

#ifdef CUDA_BACKEND
#include <cufft.h>

//CUFFT TYPES
#define QUFFT_C2C CUFFT_C2C
#define QUFFT_Z2Z CUFFT_Z2Z
#define QUFFT_SUCCESS CUFFT_SUCCESS
#define QUFFT_FORWARD CUFFT_FORWARD
#define QUFFT_INVERSE CUFFT_INVERSE

#define qufftType cufftType
#define qufftResult cufftResult
#define qufftHandle cufftHandle
#define qufftExecC2C cufftExecC2C
#define qufftExecZ2Z cufftExecZ2Z
#define qufftComplex cufftComplex
#define qufftDoubleComplex cufftDoubleComplex
#define qufftPlanMany cufftPlanMany
#define qufftDestroy cufftDestroy

/*
qufftResult qufftPlanMany(qufftHandle *plan, int rank, int *n, int *inembed,
			  int istride, int idist, int *onembed, int ostride,
			  int odist, qufftType type, int batch){

  return cufftResult cufftPlanMany(&plan, ,rank, int *n, int *inembed,
    int istride, int idist, int *onembed, int ostride,
    int odist, cufftType type, int batch);
}
*/

#endif

#ifdef HIP_BACKEND
#include <hipfft.h>
#endif
