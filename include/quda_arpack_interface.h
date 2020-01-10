#pragma once

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>

#ifdef ARPACK_LIB

#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  Interface functions to the external ARPACK library. These functions utilize
 *  ARPACK's implemntation of the Implicitly Restarted Arnoldi Method to compute a
 *  number of eigenvectors/eigenvalues with user specified features, such as those
 *  with small real part, small magnitude etc. Parallel (OMP/MPI) versions
 *  are also supported.
 */

#if (defined(QMP_COMMS) || defined(MPI_COMMS))

// Parallel, single prec complex eigenvectors
extern int ARPACK(pcnaupd)(int *fcomm, int *ido, char *bmat, int *n, char *spectrum, int *nev, float *tol,
                           std::complex<float> *resid, int *ncv, std::complex<float> *v, int *ldv, int *iparam,
                           int *ipntr, std::complex<float> *workd, std::complex<float> *workl, int *lworkl,
                           float *rwork, int *info, int bmat_size, int spectrum_size);

// Parallel, double prec complex eigenvectors
extern int ARPACK(pznaupd)(int *fcomm, int *ido, char *bmat, int *n, char *spectrum, int *nev, double *tol,
                           std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, int *iparam,
                           int *ipntr, std::complex<double> *workd, std::complex<double> *workl, int *lworkl,
                           double *rwork, int *info, int bmat_size, int spectrum_size);

// Parallel, single prec complex eigenvalues
extern int ARPACK(pcneupd)(int *fcomm, int *comp_evecs, char *howmany, int *select, std::complex<float> *evals,
                           std::complex<float> *v, int *ldv, std::complex<float> *sigma, std::complex<float> *workev,
                           char *bmat, int *n, char *which, int *nev, float *tol, std::complex<float> *resid, int *ncv,
                           std::complex<float> *v1, int *ldv1, int *iparam, int *ipntr, std::complex<float> *workd,
                           std::complex<float> *workl, int *lworkl, float *rwork, int *info, int howmany_size,
                           int bmat_size, int spectrum_size);

// Parallel, double prec complex eigenvalues
extern int ARPACK(pzneupd)(int *fcomm, int *comp_evecs, char *howmany, int *select, std::complex<double> *evals,
                           std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev,
                           char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid,
                           int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr,
                           std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork,
                           int *info, int howmany_size, int bmat_size, int spectrum_size);

extern int ARPACK(pmcinitdebug)(int *, int *, int *, int *, int *, int *, int *, int *);

#else

// Serial, single prec complex eigenvectors
extern int ARPACK(cnaupd)(int *ido, char *bmat, int *n, char *which, int *nev, float *tol, std::complex<float> *resid,
                          int *ncv, std::complex<float> *v, int *ldv, int *iparam, int *ipntr,
                          std::complex<float> *workd, std::complex<float> *workl, int *lworkl, float *rwork, int *info,
                          int bmat_size, int spectrum_size);

// Serial, double prec complex eigenvectors
extern int ARPACK(znaupd)(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid,
                          int *ncv, std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
                          std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork,
                          int *info, int bmat_size, int spectrum_size);

// Serial, single prec complex eigenvalues
extern int ARPACK(cneupd)(int *comp_evecs, char *howmany, int *select, std::complex<float> *evals,
                          std::complex<float> *v, int *ldv, std::complex<float> *sigma, std::complex<float> *workev,
                          char *bmat, int *n, char *which, int *nev, float *tol, std::complex<float> *resid, int *ncv,
                          std::complex<float> *v1, int *ldv1, int *iparam, int *ipntr, std::complex<float> *workd,
                          std::complex<float> *workl, int *lworkl, float *rwork, int *info, int howmany_size,
                          int bmat_size, int spectrum_size);

// Serial, double prec complex eigenvalues
extern int ARPACK(zneupd)(int *comp_evecs, char *howmany, int *select, std::complex<double> *evals,
                          std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev,
                          char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, int *ncv,
                          std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, std::complex<double> *workd,
                          std::complex<double> *workl, int *lworkl, double *rwork, int *info, int howmany_size,
                          int bmat_size, int spectrum_size);

extern int ARPACK(mcinitdebug)(int *, int *, int *, int *, int *, int *, int *, int *);

#endif

// ARPACK initlog and finilog routines for printing the ARPACK log
#ifdef ARPACK_LOGGING
extern int ARPACK(initlog)(int *, char *, int);
extern int ARPACK(finilog)(int *);
#endif

#ifdef __cplusplus
}
#endif

#endif //ARPACK_LIB
