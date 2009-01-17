#ifndef _QUDA_FIELD_H
#define _QUDA_FIELD_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  ParitySpinor allocateParitySpinor();
  FullSpinor allocateSpinorField();
  
  void freeGaugeField(FullGauge gauge);
  void freeParitySpinor(ParitySpinor spinor);
  void freeSpinorField(FullSpinor spinor);
  void freeSpinorBuffer();

  FullGauge loadGaugeField(void *gauge);
  
  void loadParitySpinor(ParitySpinor, void *spinor, Precision cpu_prec, Precision cuda_prec, 
			DiracFieldOrder dirac_order);
  void loadSpinorField(FullSpinor, void *spinor, Precision cpu_prec,  Precision cuda_prec, 
		       DiracFieldOrder dirac_order);
  
  void retrieveParitySpinor(void *res, ParitySpinor spinor, Precision cpu_prec,  Precision cuda_prec, 
			    DiracFieldOrder dirac_order);
  void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec,  Precision cuda_prec, 
			   DiracFieldOrder dirac_order);
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_FIELD_H
