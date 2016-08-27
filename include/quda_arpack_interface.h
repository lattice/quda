#ifndef _QUDA_ARPACK_INTERFACE_H
#define _QUDA_ARPACK_INTERFACE_H

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <algorithm>

namespace quda{

   class ArpackArgs{

    private:
      //main setup:
      /**Problem matrix **/
      DiracMatrix &matEigen;

      /**Matrix vector precision (may not coincide with arpack IRA routines precision) **/
      QudaPrecision mat_precision;

      /**precision of IRA routines**/
      bool use_full_prec_arpack;

      /**spectrum info**/
      int nev;//number of eigenvecs to be comupted
      int ncv;//search subspace dimension (note that 1 <= NCV-NEV and NCV <= N) 

      char *lanczos_which;// ARPACK which="{S,L}{R,I,M}

      /**general arpack library parameters**/	
      double tol;
      int   info;

      //experimental: for tests only!
      //supported oprtions: 2d fields -> 1 (U(1)), 4d fields -> 2 (SU(2))      
      int reducedColors;

      //experimental: 2d or 4d fields?
      bool _2d_field;

    public:

      ArpackArgs(DiracMatrix &matEigen, QudaPrecision prec, int nev, int ncv, char *which) : matEigen(matEigen), mat_precision(prec), 
            use_full_prec_arpack(true), nev(nev), ncv(ncv), lanczos_which(which), tol(1e-6), info(0), reducedColors(3), _2d_field(false) { };       

      virtual ~ArpackArgs() { };

      //Extra setup:
      void Set2D() {_2d_field = true; } 

      void SetReducedColors(int c) { reducedColors = c;}

      void SetTol(double _tol) {tol = _tol;};
      //Main IRA algorithm driver:
      void operator()(std::vector<ColorSpinorField*> &B, std::complex<double> *evals);     

   };

}//endof namespace quda 
#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif


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


#ifdef __cplusplus
}
#endif

#endif // _QUDA_ARPACK_INTERFACE_H
