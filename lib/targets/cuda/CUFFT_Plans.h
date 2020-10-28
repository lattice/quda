#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <cufft.h>

using FFTPlanHandle = cufftHandle;
#define FFT_FORWARD     CUFFT_FORWARD
#define FFT_INVERSE     CUFFT_INVERSE

#ifndef GPU_GAUGE_ALG

#define CUFFT_SAFE_CALL(call)

inline void ApplyFFT(FFTPlanHandle &plan, float2 *data_in, float2 *data_out, int direction){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
inline void ApplyFFT(FFTPlanHandle &plan, double2 *data_in, double2 *data_out, int direction){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
inline void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
}
inline void FFTDestroyPlan( FFTPlanHandle &plan){
  errorQuda("CPU_GAUGE_ALG is disabled so FFTs are also disabled");
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
 * @brief Creates a CUFFT plan supporting 4D (1D+3D) data layouts for single-precision complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 1:
  {
    int n[1] = { size.w };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y * size.z)); 
  }
  break;
  case 3:
  { 
    int n[3] = { size.x, size.y, size.z };
    CUFFT_SAFE_CALL( cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.w)); 
  }
  break;
  }
}

/**
 * @brief Creates a CUFFT plan supporting 4D (1D+3D) data layouts for double-precision complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFTMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 1:
  { 
    int n[1] = { size.w };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y * size.z));
  }
  break;
  case 3:
  { 
    int n[3] = { size.x, size.y, size.z };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.w));
  }
  break;
  }
}


/**
 * @brief Creates a CUFFT plan supporting 4D (2D+2D) data layouts for single-precision complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 0:
  { 
    int n[2] = { size.w, size.z };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.x * size.y));
  }
  break;
  case 1:
  { 
    int n[2] = { size.x, size.y };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, size.z * size.w));
  }
  break;
  }
  
}

/**
 * @brief Creates a CUFFT plan supporting 4D (2D+2D) data layouts for double-precision complex-to-complex
 * @param[out] plan, CUFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( FFTPlanHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 0:
  { 
    int n[2] = { size.w, size.z };
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.x * size.y));
  }
  break;
  case 1:
  { 
    int n[2] = { size.x, size.y };
										      
    CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, size.z * size.w)); 
  }
  break;
  }
}

inline void FFTDestroyPlan( FFTPlanHandle &plan) {
   CUFFT_SAFE_CALL(cufftDestroy(plan));
}

#endif
