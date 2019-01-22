#ifndef _EIGENSOLVE_QUDA_H
#define _EIGENSOLVE_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

#ifdef ARPACK_LIB
#include <quda_arpack_interface.h>
#endif

namespace quda {

  //void lanczosSolve(void *h_evecs, void *h_evals, const Dirac &mat,
  //QudaEigParam *eig_param, ColorSpinorParam *cpuParam);
  
  //void irlmSolve(void *h_evecs, void *h_evals, const Dirac &mat,
  //QudaEigParam *eig_param, ColorSpinorParam *cpuParam);

  void lanczosSolve(std::vector<ColorSpinorField*> kSpace,
		    void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param);
  
  void irlmSolve(std::vector<ColorSpinorField*> kSpace,
		 void *h_evals, const Dirac &mat,
		 QudaEigParam *eig_param);
  
  void arnoldiSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param, ColorSpinorParam *cpuParam);
  
  void iramSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		 QudaEigParam *eig_param, ColorSpinorParam *cpuParam);

}

#endif
  
