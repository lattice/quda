#ifndef _EIGENSOLVE_QUDA_H
#define _EIGENSOLVE_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <quda_arpack_interface.h>
namespace quda {

  void lanczosSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param, ColorSpinorParam *cpuParam);
  
  void irlmSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		 QudaEigParam *eig_param, ColorSpinorParam *cpuParam);

  void arnoldiSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		    QudaEigParam *eig_param, ColorSpinorParam *cpuParam);
  
  void iramSolve(void *h_evecs, void *h_evals, const Dirac &mat,
		 QudaEigParam *eig_param, ColorSpinorParam *cpuParam);
  
}

#endif
  
