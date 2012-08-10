
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <quda_internal.h>
#include <read_gauge.h>
#include "gauge_field.h"
#include <force_common.h>
#include "llfat_quda.h"
#include <face_quda.h>

#define BLOCK_DIM 64

void
llfat_cuda(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink, 
	   cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
	   QudaGaugeParam* param, double* act_path_coeff)
{
  int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  int Vh = volume/2;
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 halfGridDim(Vh/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM , 1, 1);
  
  QudaPrecision prec = cudaSiteLink.Precision();
  QudaReconstructType recon = cudaSiteLink.Reconstruct();
  
  if( ((param->X[0] % 2 != 0)
       ||(param->X[1] % 2 != 0)
       ||(param->X[2] % 2 != 0)
       ||(param->X[3] % 2 != 0))
      && (recon  == QUDA_RECONSTRUCT_12)){
    errorQuda("12 reconstruct and odd dimensionsize is not supported by link fattening code (yet)\n");
    
  }
      
  int nStream=9;
  cudaStream_t stream[nStream];
  for(int i = 0;i < nStream; i++){
    cudaStreamCreate(&stream[i]);
  }


  llfatOneLinkKernel(cudaFatLink, cudaSiteLink,cudaStaple, cudaStaple1,
		     param, act_path_coeff); 

  
  llfat_kernel_param_t kparam;
  for(int i=0;i < 4;i++){
     kparam.ghostDim[i] = commDimPartitioned(i);
  }
#ifdef MULTI_GPU
  int ktype[8] = {
		LLFAT_EXTERIOR_KERNEL_BACK_X, 
		LLFAT_EXTERIOR_KERNEL_FWD_X, 
		LLFAT_EXTERIOR_KERNEL_BACK_Y, 
		LLFAT_EXTERIOR_KERNEL_FWD_Y, 
		LLFAT_EXTERIOR_KERNEL_BACK_Z, 
		LLFAT_EXTERIOR_KERNEL_FWD_Z, 
		LLFAT_EXTERIOR_KERNEL_BACK_T, 
		LLFAT_EXTERIOR_KERNEL_FWD_T, 
  };
#endif

  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){

#ifdef MULTI_GPU
	//start of one call
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  
	  kparam.kernel_type = ktype[2*k];
	  siteComputeGenStapleParityKernel((void*)cudaStaple.Even_p(), (void*)cudaStaple.Odd_p(),
					   (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					   (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(),
					   dir, nu,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[2*k]);
	  
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]);
	  
	  kparam.kernel_type = ktype[2*k+1];
	  siteComputeGenStapleParityKernel((void*)cudaStaple.Even_p(), (void*)cudaStaple.Odd_p(),
					   (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					   (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(),
					   dir, nu,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[2*k+1]);
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]);
	}
#endif
        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	siteComputeGenStapleParityKernel((void*)cudaStaple.Even_p(), (void*)cudaStaple.Odd_p(),
					 (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					 (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
					 dir, nu,
					 act_path_coeff[2],
					 recon, prec, halfGridDim, 
					 kparam, &stream[nStream-1]);

#ifdef MULTI_GPU	
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]);
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]);
	}	
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]);
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]);
	}
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  cudaStreamSynchronize(stream[2*k]);
	  cudaStreamSynchronize(stream[2*k+1]);
	}	
#endif
	//end

	//start of one call
        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	if(act_path_coeff[5] != 0.0){
	  computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL,
					    (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					    (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
					    (const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
					    dir, nu, 0,
					    act_path_coeff[5],
					    recon, prec,  halfGridDim, kparam, &stream[nStream-1]);
	}
	//end
	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){

	    //start of one call
#ifdef MULTI_GPU	
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      kparam.kernel_type = ktype[2*k];	    
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.Even_p(), (void*)cudaStaple1.Odd_p(),
						(const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
						(void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
						(const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
						dir, rho, 1,
						act_path_coeff[3],
						recon, prec, halfGridDim, kparam, &stream[2*k]);
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]);
	      kparam.kernel_type = ktype[2*k+1];	    
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.Even_p(), (void*)cudaStaple1.Odd_p(),
						(const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
						(void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
						(const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
						dir, rho, 1,
						act_path_coeff[3],
						recon, prec, halfGridDim, kparam, &stream[2*k+1]);
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]);
	    }	    
#endif

	    kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	    computeGenStapleFieldParityKernel((void*)cudaStaple1.Even_p(), (void*)cudaStaple1.Odd_p(),
					      (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					      (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
					      (const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
					      dir, rho, 1,
					      act_path_coeff[3],
					      recon, prec, halfGridDim, kparam, &stream[nStream-1]);

#ifdef MULTI_GPU
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]);
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]);
	    }
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_BACKWARDS, &stream[2*k]);
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_FORWARDS, &stream[2*k+1]);
	    }
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      cudaStreamSynchronize(stream[2*k]);
	      cudaStreamSynchronize(stream[2*k+1]);
	    }	
#endif	    
	    //end

	    
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){						
		
		//start of one call
		kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
		computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL, 
						  (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
						  (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
						  (const void*)cudaStaple1.Even_p(), (const void*)cudaStaple1.Odd_p(),
						  dir, sig, 0,
						  act_path_coeff[4],
						  recon, prec, halfGridDim, kparam, &stream[nStream-1]);

		//end
		
	      }			    
	    }//sig
	  }
	}//rho	
      }
    }//nu
  }//dir
  
  
  cudaDeviceSynchronize(); 
  checkCudaError();
  
  for(int i=0;i < nStream; i++){
    cudaStreamDestroy(stream[i]);
  }

  return;
}



void
llfat_cuda_ex(cudaGaugeField& cudaFatLink, cudaGaugeField& cudaSiteLink, 
	      cudaGaugeField& cudaStaple, cudaGaugeField& cudaStaple1,
	      QudaGaugeParam* param, double* act_path_coeff)
{

  dim3 blockDim(BLOCK_DIM, 1,1);
  
  int volume = (param->X[0])*(param->X[1])*(param->X[2])*(param->X[3]);
  int Vh = volume/2;
  dim3 halfGridDim(Vh/blockDim.x,1,1);
  if(Vh % blockDim.x != 0){
    halfGridDim.x +=1;
  }


  int volume_1g = (param->X[0]+2)*(param->X[1]+2)*(param->X[2]+2)*(param->X[3]+2);
  int Vh_1g = volume_1g/2;
  dim3 halfGridDim_1g(Vh_1g/blockDim.x,1,1);
  if(Vh_1g % blockDim.x != 0){
    halfGridDim_1g.x +=1;
  }
  
  int volume_2g = (param->X[0]+4)*(param->X[1]+4)*(param->X[2]+4)*(param->X[3]+4);
  int Vh_2g = volume_2g/2;
  dim3 halfGridDim_2g(Vh_2g/blockDim.x,1,1);
  if(Vh_2g % blockDim.x != 0){
    halfGridDim_2g.x +=1;
  }

  QudaPrecision prec = cudaSiteLink.Precision();
  QudaReconstructType recon = cudaSiteLink.Reconstruct();
  
  if( ((param->X[0] % 2 != 0)
       ||(param->X[1] % 2 != 0)
       ||(param->X[2] % 2 != 0)
       ||(param->X[3] % 2 != 0))
      && (recon  == QUDA_RECONSTRUCT_12)){
    errorQuda("12 reconstruct and odd dimensionsize is not supported by link fattening code (yet)\n");
    
  }
      
  
  llfat_kernel_param_t kparam;
  llfat_kernel_param_t kparam_1g;
  llfat_kernel_param_t kparam_2g;
  
  kparam.threads= Vh;
  kparam.halfGridDim = halfGridDim;
  kparam.D1 = param->X[0];
  kparam.D2 = param->X[1];
  kparam.D3 = param->X[2];
  kparam.D4 = param->X[3];
  kparam.D1h = param->X[0]/2;
  kparam.base_idx = 2;
  
  kparam_1g.threads= Vh_1g;
  kparam_1g.halfGridDim = halfGridDim_1g;
  kparam_1g.D1 = param->X[0] + 2;
  kparam_1g.D2 = param->X[1] + 2;
  kparam_1g.D3 = param->X[2] + 2;
  kparam_1g.D4 = param->X[3] + 2;
  kparam_1g.D1h = (param->X[0] + 2)/2;
  kparam_1g.base_idx = 1;

  kparam_2g.threads= Vh_2g;
  kparam_2g.halfGridDim = halfGridDim_2g;
  kparam_2g.D1 = param->X[0] + 4;
  kparam_2g.D2 = param->X[1] + 4;
  kparam_2g.D3 = param->X[2] + 4;
  kparam_2g.D4 = param->X[3] + 4;
  kparam_2g.D1h = (param->X[0] + 4)/2;
  kparam_2g.base_idx = 0;
  
  kparam_1g.blockDim = kparam_2g.blockDim = kparam.blockDim = blockDim;

  /*
  {
    static dim3 blocks[3]={{64, 1, 1}, {64,1,1}, {64,1,1}};
    QudaVerbosity verbose = QUDA_DEBUG_VERBOSE;
    TuneLinkFattening fatTune(cudaFatLink, cudaSiteLink, cudaStaple, cudaStaple1,
			      kparam, kparam_1g, verbose);
    fatTune.BenchmarkMulti(blocks, 3);
    
  }
  */

  llfatOneLinkKernel_ex(cudaFatLink, cudaSiteLink,cudaStaple, cudaStaple1,
			param, act_path_coeff, kparam);

  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){
	

	siteComputeGenStapleParityKernel_ex((void*)cudaStaple.Even_p(), (void*)cudaStaple.Odd_p(),
					    (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					    (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
					    dir, nu,
					    act_path_coeff[2],
					    recon, prec, kparam_1g); 

        if(act_path_coeff[5] != 0.0){
	  computeGenStapleFieldParityKernel_ex((void*)NULL, (void*)NULL,
					       (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
					       (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
					       (const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
					       dir, nu, 0,
					       act_path_coeff[5],
					       recon, prec, kparam);
	} 

	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){
	    
	    computeGenStapleFieldParityKernel_ex((void*)cudaStaple1.Even_p(), (void*)cudaStaple1.Odd_p(),
						 (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
						 (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
						 (const void*)cudaStaple.Even_p(), (const void*)cudaStaple.Odd_p(),
						 dir, rho, 1,
						 act_path_coeff[3],
						 recon, prec, kparam_1g);
	    
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){						
		
		computeGenStapleFieldParityKernel_ex((void*)NULL, (void*)NULL, 
						     (const void*)cudaSiteLink.Even_p(), (const void*)cudaSiteLink.Odd_p(),
						     (void*)cudaFatLink.Even_p(), (void*)cudaFatLink.Odd_p(), 
						     (const void*)cudaStaple1.Even_p(), (const void*)cudaStaple1.Odd_p(),
						     dir, sig, 0,
						     act_path_coeff[4],
						     recon, prec, kparam);
		
	      }			    
	    }//sig
	  }
	}//rho	
      }
    }//nu
  }//dir
  
  
  cudaDeviceSynchronize(); 
  checkCudaError();
  
  return;
}
