#ifndef __LLFAT_QUDA_H__
#define __LLFAT_QUDA_H__

#include "quda.h"
#ifdef __cplusplus
extern "C"{
#endif

    void llfat_cuda(void* fatLink, void* siteLink,
		    FullGauge cudaFatLink, FullGauge cudaSiteLink, 
		    FullStaple cudaStaple, FullStaple cudaStaple1,
		    QudaGaugeParam* param, void* act_path_coeff);
    void llfat_init_cuda(QudaGaugeParam* param, void* act_path_coeff);
    
#ifdef __cplusplus
}
#endif


#endif
