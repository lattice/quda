#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"

#ifdef __cplusplus
extern "C"{
#endif

  void llfat_cuda(FullGauge cudaFatLink, FullGauge cudaSiteLink, 
		  FullStaple cudaStaple, FullStaple cudaStaple1,
		  QudaGaugeParam* param, double* act_path_coeff);

  void llfat_init_cuda(QudaGaugeParam* param);

  void computeGenStapleFieldParityKernel(void* staple_even, void* staple_odd, 
					 void* sitelink_even, void* sitelink_odd,
					 void* fatlink_even, void* fatlink_odd,			    
					 void* mulink_even, void* mulink_odd, 
					 int mu, int nu, int odd_bit, int save_staple,
					 double mycoeff,
					 QudaReconstructType recon, QudaPrecision prec,
					 int2 tloc, dim3 halfGridDim, 
					 cudaStream_t* stream);
  
  void siteComputeGenStapleParityKernel(void* staple_even, void* staple_odd, 
					void* sitelink_even, void* sitelink_odd, 
					void* fatlink_even, void* fatlink_odd,	
					int mu, int nu,int odd_bit,
					double mycoeff,
					QudaReconstructType recon, QudaPrecision prec,
					int2 tloc, dim3 halfGridDim, 
					cudaStream_t* stream); 

  void llfatOneLinkKernel(FullGauge cudaFatLink, FullGauge cudaSiteLink,
			  FullStaple cudaStaple, FullStaple cudaStaple1,
			  QudaGaugeParam* param, double* act_path_coeff);
  
#ifdef __cplusplus
}
#endif

#endif // _LLFAT_QUDA_H
