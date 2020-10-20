#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <hipfft.h>

#ifndef GPU_GAUGE_ALG

#define HIPFFT_SAFE_CALL(call)

#else

using FFTPlanHandle = hipfftHandle;

#define	FFT_FORWARD	HIPFFT_FORWARD
#define FFT_INVERSE	HIPFFT_BACKWARD

/*-------------------------------------------------------------------------------*/
#define HIPFFT_SAFE_CALL( call) {                                      \
    hipfftResult err = call;                                         \
    if ( HIPFFT_SUCCESS != err ) {                                     \
      fprintf(stderr, "HIPFFT error in file '%s' in line %i.\n",   \
              __FILE__, __LINE__);                                \
      exit(EXIT_FAILURE);                                         \
    } }
/*-------------------------------------------------------------------------------*/

/**
 * @brief Call HIPFFT to perform a single-precision complex-to-complex
 * transform plan in the transform direction as specified by direction
 * parameter
 * @param[in] HIPFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: HIPFFT_FORWARD or HIPFFT_BACKWARD
 */
inline void ApplyFFT(FFTPlanHandle &plan, float2 *data_in, float2 *data_out, int direction){
  HIPFFT_SAFE_CALL(hipfftExecC2C(plan, (hipfftComplex *)data_in, (hipfftComplex *)data_out, direction));
}

/**
 * @brief Call HIPFFT to perform a double-precision complex-to-complex transform plan in the transform direction
as specified by direction parameter
 * @param[in] HIPFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: HIPFFT_FORWARD or HIPFFT_BACKWARD
 */
inline void ApplyFFT(FFTPlanHandle &plan, double2 *data_in, double2 *data_out, int direction){
  HIPFFT_SAFE_CALL(hipfftExecZ2Z(plan, (hipfftDoubleComplex *)data_in, (hipfftDoubleComplex *)data_out, direction));
}


/**
 * @brief Creates a HIPFFT plan supporting 4D (1D+3D) data layouts for single-precision complex-to-complex
 * @param[out] plan, HIPFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 1:
  { int n[1] = { size.w };
    hipfftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_C2C, size.x * size.y * size.z); }
                                                                                             break;
  case 3:
  { int n[3] = { size.x, size.y, size.z };
    hipfftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_C2C, size.w); }
                                                                           break;
  }
  //printf("Created %dD FFT Plan in Single Precision\n", dim);
}

/**
 * @brief Creates a HIPFFT plan supporting 4D (1D+3D) data layouts for double-precision complex-to-complex
 * @param[out] plan, HIPFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 1:
  { int n[1] = { size.w };
    hipfftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_Z2Z, size.x * size.y * size.z); }
                                                                                             break;
  case 3:
  { int n[3] = { size.x, size.y, size.z };
    hipfftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_Z2Z, size.w); }
                                                                           break;
  }
  //printf("Created %dD FFT Plan in Double Precision\n", dim);
}


/**
 * @brief Creates a HIPFFT plan supporting 4D (2D+2D) data layouts for single-precision complex-to-complex
 * @param[out] plan, HIPFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 0:
  { int n[2] = { size.w, size.z };
    hipfftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_C2C, size.x * size.y); }
                                                                                    break;
  case 1:
  { int n[2] = { size.x, size.y };
    hipfftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_C2C, size.z * size.w); }
                                                                                    break;
  }
  //printf("Created 2D FFT Plan in Single Precision\n");
}

/**
 * @brief Creates a HIPFFT plan supporting 4D (2D+2D) data layouts for double-precision complex-to-complex
 * @param[out] plan, HIPFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 0:
  { int n[2] = { size.w, size.z };
    hipfftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_Z2Z, size.x * size.y); }
                                                                                    break;
  case 1:
  { int n[2] = { size.x, size.y };
    hipfftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, HIPFFT_Z2Z, size.z * size.w); }
                                                                                    break;
  }
  //printf("Created 2D FFT Plan in Double Precision\n");
}

inline void FFTDestroyPlan( FFTPlanHandle &plan) {
   HIPFFT_SAFE_CALL(hipfftDestroy(plan));
}

#endif
