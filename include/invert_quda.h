#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda_internal.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <dirac.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern FullGauge cudaGaugePrecise;
  extern FullGauge cudaGaugeSloppy;

  extern FullGauge cudaFatLinkPrecise;
  extern FullGauge cudaFatLinkSloppy;

  extern FullGauge cudaLongLinkPrecise;
  extern FullGauge cudaLongLinkSloppy;


  extern FullClover cudaCloverPrecise;
  extern FullClover cudaCloverSloppy;

  extern FullClover cudaCloverInvPrecise;
  extern FullClover cudaCloverInvSloppy;

  // -- inv_cg_cuda.cpp
  void invertCgCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField &tmp,
		    QudaInvertParam *param);
  int invertCgCudaMultiMass(Dirac & dirac, Dirac& diracSloppy, cudaColorSpinorField** x, cudaColorSpinorField b,
			    QudaInvertParam *invert_param, 
			    double* offsets, int num_offsets, double* residue_sq);
    
  // -- inv_bicgstab_cuda.cpp
  void invertBiCGstabCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField &tmp, 
			  QudaInvertParam *param, DagType dag_type);

#ifdef __cplusplus
}
#endif

#endif // _INVERT_QUDA_H
