#pragma once

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <algorithm>

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
//#include <qmp.h>
#include <mpi.h>
#endif

#ifdef PRIMME_LIB
#include "primme.h"
#endif

#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(MPI_COMMS) || defined(QMP_COMMS))

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


namespace quda{

  void arpackSolve( std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision arpackPrec, double tol, int nev, int ncv, char *target);
//  void primmeSolve( std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &matEigen, QudaPrecision matPrec, QudaPrecision primmePrec, double tol, int nev, int ncv, char *target);


}//endof namespace quda 

