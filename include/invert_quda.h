#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

#ifdef __cplusplus
extern "C" {
#endif

  // defined in interface_quda.cpp

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

  // defined in inv_cg_cuda.cpp

  void invertCgCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField &tmp,
		    QudaInvertParam *param);

  // defined in inv_multi_cg_quda.cpp

  int invertMultiShiftCgCuda(Dirac & dirac, Dirac& diracSloppy, cudaColorSpinorField** x, cudaColorSpinorField b,
			     QudaInvertParam *invert_param, 
			     double* offsets, int num_offsets, double* residue_sq);
  
  // defined in inv_bicgstab_cuda.cpp

  void invertBiCGstabCuda(Dirac &dirac, Dirac &diracSloppy, cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField &tmp, 
			  QudaInvertParam *param, DagType dag_type);

#ifdef __cplusplus
}
#endif

#endif // _INVERT_QUDA_H
