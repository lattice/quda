#ifndef _LLFAT_QUDA_H
#define _LLFAT_QUDA_H

#include "quda.h"

#ifdef __cplusplus
extern "C"{
#endif

#define LLFAT_INTERIOR_KERNEL 0
#define LLFAT_EXTERIOR_KERNEL_FWD_X 1
#define LLFAT_EXTERIOR_KERNEL_BACK_X 2
#define LLFAT_EXTERIOR_KERNEL_FWD_Y 3
#define LLFAT_EXTERIOR_KERNEL_BACK_Y 4
#define LLFAT_EXTERIOR_KERNEL_FWD_Z 5
#define LLFAT_EXTERIOR_KERNEL_BACK_Z 6
#define LLFAT_EXTERIOR_KERNEL_FWD_T 7
#define LLFAT_EXTERIOR_KERNEL_BACK_T 8

typedef struct llfat_kernel_param_s{
        unsigned long threads;
	int ghostDim[4]; // Whether a ghost zone has been allocated for a given dimension
        int kernel_type;
}llfat_kernel_param_t;


  void llfat_cuda(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink, 
		  FullStaple cudaStaple, FullStaple cudaStaple1,
		  QudaGaugeParam* param, double* act_path_coeff);

  void llfat_init_cuda(QudaGaugeParam* param);

  void computeGenStapleFieldParityKernel(void* staple_even, void* staple_odd, 
					 void* sitelink_even, void* sitelink_odd,
					 void* fatlink_even, void* fatlink_odd,			    
					 void* mulink_even, void* mulink_odd, 
					 int mu, int nu, int save_staple,
					 double mycoeff,
					 QudaReconstructType recon, QudaPrecision prec,
					 dim3 halfGridDim,  llfat_kernel_param_t kparam,
					 cudaStream_t* stream);
  
  void siteComputeGenStapleParityKernel(void* staple_even, void* staple_odd, 
					void* sitelink_even, void* sitelink_odd, 
					void* fatlink_even, void* fatlink_odd,	
					int mu, int nu,	double mycoeff,
					QudaReconstructType recon, QudaPrecision prec,
					dim3 halfGridDim, llfat_kernel_param_t kparam,
					cudaStream_t* stream); 

  void llfatOneLinkKernel(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink,
			  FullStaple cudaStaple, FullStaple cudaStaple1,
			  QudaGaugeParam* param, double* act_path_coeff);
  
#ifdef __cplusplus
}
#endif

#endif // _LLFAT_QUDA_H
