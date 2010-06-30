#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"

#ifdef __cplusplus
extern "C"{
#endif

  void llfat_cuda(void* fatLink, void* siteLink,
		  FullGauge cudaFatLink, FullGauge cudaSiteLink, 
		  FullStaple cudaStaple, FullStaple cudaStaple1,
		  QudaGaugeParam* param, double* act_path_coeff);

  void llfat_init_cuda(QudaGaugeParam* param);
    
#ifdef __cplusplus
}
#endif

#endif // _LLFAT_QUDA_H
