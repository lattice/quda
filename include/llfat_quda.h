#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"
#include "quda_internal.h"

#define LLFAT_INTERIOR_KERNEL 0
#define LLFAT_EXTERIOR_KERNEL_FWD_X 1
#define LLFAT_EXTERIOR_KERNEL_BACK_X 2
#define LLFAT_EXTERIOR_KERNEL_FWD_Y 3
#define LLFAT_EXTERIOR_KERNEL_BACK_Y 4
#define LLFAT_EXTERIOR_KERNEL_FWD_Z 5
#define LLFAT_EXTERIOR_KERNEL_BACK_Z 6
#define LLFAT_EXTERIOR_KERNEL_FWD_T 7
#define LLFAT_EXTERIOR_KERNEL_BACK_T 8

namespace quda {
  
  typedef struct llfat_kernel_param_s {
    unsigned long threads;
    int ghostDim[4]; // Whether a ghost zone has been allocated for a given dimension
    int kernel_type;
    
    //use in extended kernels
    int D1, D2,D3, D4, D1h;
    dim3 blockDim;
    dim3 halfGridDim;
    int base_idx;
    
  } llfat_kernel_param_t;
  

  void llfat_cuda_ex(cudaGaugeField* cudaFatLink, 
                     cudaGaugeField* cudaLongLink,
                     cudaGaugeField& cudaSiteLink, 
		     cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
		     QudaGaugeParam* param, double* act_path_coeff);
  
  void llfat_init_cuda_ex(QudaGaugeParam* param_ex);

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
