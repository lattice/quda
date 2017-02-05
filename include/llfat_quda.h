#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"
#include "quda_internal.h"

namespace quda {
  
  struct llfat_kernel_param_t;

  void llfat_cuda_ex(cudaGaugeField* cudaFatLink, 
                     cudaGaugeField* cudaLongLink,
                     cudaGaugeField& cudaSiteLink, 
		     QudaGaugeParam* param, double* act_path_coeff);
  
  void computeLongLinkCuda(void* outEven, void* outOdd,
                           const void* const inEven, const void* const inOdd,
                           double coeff, QudaReconstructType recon, QudaPrecision prec,
                           dim3 halfGridDim, llfat_kernel_param_t kparam);           
                

  void computeGenStapleFieldParityKernel_ex(void* staple_even, void* staple_odd, 
					    const void* sitelink_even, const void* sitelink_odd,
					    void* fatlink_even, void* fatlink_odd,			    
					    const void* mulink_even, const void* mulink_odd, 
					    int mu, int nu, int save_staple,
					    double mycoeff,
					    QudaReconstructType recon, QudaPrecision prec,
					    llfat_kernel_param_t kparam);  

  void siteComputeGenStapleParityKernel_ex(void* staple_even, void* staple_odd, 
					   const void* sitelink_even, const void* sitelink_odd, 
					   void* fatlink_even, void* fatlink_odd,	
					   int mu, int nu,	double mycoeff,
					   QudaReconstructType recon, QudaPrecision prec,
					   llfat_kernel_param_t kparam);
  void llfatOneLinkKernel_ex(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink,
			     cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
			     QudaGaugeParam* param, double* act_path_coeff,
			     llfat_kernel_param_t kparam);


  void computeFatLinkCore(cudaGaugeField* cudaSiteLink, double* act_path_coeff,
			  QudaGaugeParam* qudaGaugeParam, cudaGaugeField* cudaFatLink,
			  cudaGaugeField* cudaLongLink, TimeProfile& profile);
  
} // namespace quda

#endif // _LLFAT_QUDA_H
