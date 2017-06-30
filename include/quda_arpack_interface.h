#pragma once

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <algorithm>

#ifdef ARPACK_LIB

#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MULTI_GPU

extern int ARPACK(pcnaupd) (int *fcomm, int *ido, char *bmat, int *n, char *which, int *nev, float *tol,
                         std::complex<float> *resid, int *ncv, std::complex<float> *v, int *ldv,
                         int *iparam, int *ipntr, std::complex<float> *workd, std::complex<float> *workl,
                         int *lworkl, float *rwork, int *info);


extern int ARPACK(pznaupd) (int *fcomm, int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
                         std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv,
                         int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl,
                         int *lworkl, double *rwork, int *info);


extern int ARPACK(pcneupd) (int *fcomm, int *comp_evecs, char *howmany, int *select, std::complex<float> *evals,
                         std::complex<float> *v, int *ldv, std::complex<float> *sigma, std::complex<float> *workev,
                         char *bmat, int *n, char *which, int *nev, float *tol, std::complex<float> *resid,
                         int *ncv, std::complex<float> *v1, int *ldv1, int *iparam, int *ipntr,
                         std::complex<float> *workd, std::complex<float> *workl, int *lworkl, float *rwork, int *info);


extern int ARPACK(pzneupd) (int *fcomm, int *comp_evecs, char *howmany, int *select, std::complex<double> *evals,
                         std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev,
                         char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid,
                         int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr,
                         std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info);

#else

extern int ARPACK(cnaupd) (int *ido, char *bmat, int *n, char *which, int *nev, float *tol,
                         std::complex<float> *resid, int *ncv, std::complex<float> *v, int *ldv,
                         int *iparam, int *ipntr, std::complex<float> *workd, std::complex<float> *workl,
                         int *lworkl, float *rwork, int *info);


extern int ARPACK(znaupd) (int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
                         std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, 
                         int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, 
                         int *lworkl, double *rwork, int *info);


extern int ARPACK(cneupd) (int *comp_evecs, char *howmany, int *select, std::complex<float> *evals, 
			 std::complex<float> *v, int *ldv, std::complex<float> *sigma, std::complex<float> *workev, 
			 char *bmat, int *n, char *which, int *nev, float *tol, std::complex<float> *resid, 
                         int *ncv, std::complex<float> *v1, int *ldv1, int *iparam, int *ipntr, 
                         std::complex<float> *workd, std::complex<float> *workl, int *lworkl, float *rwork, int *info);			


extern int ARPACK(zneupd) (int *comp_evecs, char *howmany, int *select, std::complex<double> *evals, 
			 std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev, 
			 char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, 
                         int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
                         std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info);

#endif

#ifdef __cplusplus
}
#endif

#endif //ARPACK_LIB

namespace quda{

 /**
 *  Interface function to the external ARPACK library. This function utilizes ARPACK implemntation 
 *  of the Implicitly Restarted Arnoldi Method to compute a number of eigenvectors/eigenvalues with 
 *  user specified features such as those with small real part, small magnitude etc. Parallel version
 *  is also supported.
 *  @param[in/out] B           Container of eigenvectors
 *  @param[in/out] evals       A pointer to eigenvalue array.
 *  @param[in]     matEigen    Any QUDA implementation of the matrix-vector operation
 *  @param[in]     matPrec     Precision of the matrix-vector operation
 *  @param[in]     arpackPrec  Precision of IRAM procedures. 
 *  @param[in]     tol         tolerance for computing eigenvalues with ARPACK
 *  @param[in]     nev         number of eigenvectors 
 *  @param[in]     ncv         size of the subspace used by IRAM. ncv must satisfy the two
 *                             inequalities 2 <= ncv-nev and ncv <= *B[0].Length()
 *  @param[in]     target      eigenvector selection criteria:  
 *                             'LM' -> want the nev eigenvalues of largest magnitude.
 *                             'SM' -> want the nev eigenvalues of smallest magnitude.
 *                             'LR' -> want the nev eigenvalues of largest real part.
 *                             'SR' -> want the nev eigenvalues of smallest real part.
 *                             'LI' -> want the nev eigenvalues of largest imaginary part.
 *                             'SI' -> want the nev eigenvalues of smallest imaginary part.                       
 **/

  void arpackSolve( std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision arpackPrec, double tol, int nev, int ncv, char *target );
//  void primmeSolve( std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision primmePrec, double tol, int nev, int ncv, char *target);

  //GPU version of the IRA solver
  using ColorSpinorFieldSet = ColorSpinorField;

// Defines sorting criteria (nothing more) for the algorithm
// Use the "Reverse Communication Interface" to get more options
// by redefining your matrix
// Also note that using LM mode should converge fastest due
// to the power-iteration nature of the method
  typedef enum arnmode_s{
    arnmode_LM = 1, // Largest magnitude
    arnmode_LR,     // Largest real part
    arnmode_SM,     // Smallest magnitude
    arnmode_SR,     // Smallest real part

    arnmode_force32bit = 0x7FFFFFF // Force size of type to be at least 32 bits
  } arnmode;

  typedef struct scomplex_s {
    float re;
    float im;
  } scomplex_t;

  typedef struct dcomplex_s {
    double re;
    double im;
  } dcomplex_t;

/* Do reduction of val across all nodes - for single-node support, just leave this NULL */
  typedef void (*mpi_reductiont) (void* val);


  typedef struct arnoldi_abs_int_s{
    DiracMatrix *mvecmulFun;

    mpi_reductiont scalar_redFun;   // Scalar reduction function - optional
    mpi_reductiont complex_redFun;  // Complex reduction function - optional

    void* reserve1; // For futureproofing
    void* reserve2;
  } arnoldi_abs_int;

#ifndef EXTERN_C_BEGIN
#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif
#endif



// init_vec contains the initial field vector (type depends on fieldAllocT) for Arnoldi iteration (optional)
// rvecs should be a pointer to an array of n_eigs pointers or NULL - if not NULL, will contain the right eigenvectors corresponding to results
// maxIter contains the number of iterations used
  EXTERN_C_BEGIN
  int arnoldiGPUSolve(dcomplex_t* results, const ColorSpinorField *init_vec, ColorSpinorFieldSet* rvecs, int size, int nMulti, int stride, int n_eigs, int n_extend, double tolerance, int* maxIter, 
                       const arnoldi_abs_int* functions, arnmode mode);
  EXTERN_C_END


}//endof namespace quda 

