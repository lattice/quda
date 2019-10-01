#pragma once

#include <quda_internal.h>
#include <quda_matrix.h>
#include <quda_fft.h>

/*-------------------------------------------------------------------------------*/
#define QUDAFFT_SAFE_CALL( call) {                                      \
    qudafftResult err = call;                                         \
    if ( QUDAFFT_SUCCESS != err ) {                                     \
      fprintf(stderr, "QUDAFFT error in file '%s' in line %i.\n",   \
              __FILE__, __LINE__);                                \
      exit(EXIT_FAILURE);                                         \
    } }
/*-------------------------------------------------------------------------------*/



/**
 * @brief Call QUDAFFT to perform a single-precision complex-to-complex transform plan in the transform direction 
as specified by direction parameter
 * @param[in] QUDAFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: QUDAFFT_FORWARD or QUDAFFT_INVERSE
 */
inline void ApplyFFT(qudafftHandle &plan, float2 *data_in, float2 *data_out, int direction){
  QUDAFFT_SAFE_CALL(cufftExecC2C(plan, (qudafftComplex *)data_in, (qudafftComplex *)data_out, direction));
}

/**
 * @brief Call QUDAFFT to perform a double-precision complex-to-complex transform plan in the transform direction 
as specified by direction parameter
 * @param[in] QUDAFFT plan
 * @param[in] data_in, pointer to the complex input data (in GPU memory) to transform
 * @param[out] data_out, pointer to the complex output data (in GPU memory)
 * @param[in] direction, the transform direction: QUDAFFT_FORWARD or QUDAFFT_INVERSE
 */
inline void ApplyFFT(qudafftHandle &plan, double2 *data_in, double2 *data_out, int direction){
  QUDAFFT_SAFE_CALL(cufftExecZ2Z(plan, (qudafftDoubleComplex *)data_in, (qudafftDoubleComplex *)data_out, direction));
}


/**
 * @brief Creates a QUDAFFT plan supporting 4D (1D+3D) data layouts for single-precision complex-to-complex
 * @param[out] plan, QUDAFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
void SetPlanFFTMany( qudafftHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 1:
  { int n[1] = { size.w };
    cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_C2C, size.x * size.y * size.z); }
                                                                                             break;
  case 3:
  { int n[3] = { size.x, size.y, size.z };
    cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_C2C, size.w); }
                                                                           break;
  }
  //printf("Created %dD FFT Plan in Single Precision\n", dim);
}

/**
 * @brief Creates a QUDAFFT plan supporting 4D (1D+3D) data layouts for double-precision complex-to-complex
 * @param[out] plan, QUDAFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 1 for 1D plan along the temporal direction with batch size Nx*Ny*Nz, 3 for 3D plan along Nx, Ny and Nz with batch size Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFTMany( qudafftHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 1:
  { int n[1] = { size.w };
    cufftPlanMany(&plan, 1, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_Z2Z, size.x * size.y * size.z); }
                                                                                             break;
  case 3:
  { int n[3] = { size.x, size.y, size.z };
    cufftPlanMany(&plan, 3, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_Z2Z, size.w); }
                                                                           break;
  }
  //printf("Created %dD FFT Plan in Double Precision\n", dim);
}


/**
 * @brief Creates a QUDAFFT plan supporting 4D (2D+2D) data layouts for single-precision complex-to-complex
 * @param[out] plan, QUDAFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( qudafftHandle &plan, int4 size, int dim, float2 *data){
  switch ( dim ) {
  case 0:
  { int n[2] = { size.w, size.z };
    cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_C2C, size.x * size.y); }
                                                                                    break;
  case 1:
  { int n[2] = { size.x, size.y };
    cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_C2C, size.z * size.w); }
                                                                                    break;
  }
  //printf("Created 2D FFT Plan in Single Precision\n");
}

/**
 * @brief Creates a QUDAFFT plan supporting 4D (2D+2D) data layouts for double-precision complex-to-complex
 * @param[out] plan, QUDAFFT plan
 * @param[in] size, int4 with lattice size dimensions, (.x,.y,.z,.w) -> (Nx, Ny, Nz, Nt)
 * @param[in] dim, 0 for 2D plan in Z-T planes with batch size Nx*Ny, 1 for 2D plan in X-Y planes with batch size Nz*Nt
 * @param[in] data, pointer to the double-precision complex data, this is only passed to choose between single and double precision
 */
inline void SetPlanFFT2DMany( qudafftHandle &plan, int4 size, int dim, double2 *data){
  switch ( dim ) {
  case 0:
  { int n[2] = { size.w, size.z };
    cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_Z2Z, size.x * size.y); }
                                                                                    break;
  case 1:
  { int n[2] = { size.x, size.y };
    cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, QUDAFFT_Z2Z, size.z * size.w); }
                                                                                    break;
  }
  //printf("Created 2D FFT Plan in Double Precision\n");
}


#endif

