#pragma once

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <algorithm>

//#ifdef PRIMME_LIB
//#include "primme.h"
//#endif

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

}//endof namespace quda 

