
#include <stdio.h>
#include <quda_internal.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <read_gauge.h>
#include <gauge_quda.h>
#include <force_common.h>
#include "llfat_quda.h"
#include <face_quda.h>


#define BLOCK_DIM 64

void
llfat_cuda(FullGauge cudaFatLink, FullGauge cudaSiteLink, 
	   FullStaple cudaStaple, FullStaple cudaStaple1,
	   QudaGaugeParam* param, double* act_path_coeff)
{
  
  int Vs = param->X[0]*param->X[1]*param->X[2];
  int Vsh = Vs /2;
  
  int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  int Vh = volume/2;
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 halfGridDim(Vh/BLOCK_DIM,1,1);
#ifdef MULTI_GPU
  dim3 halfInteriorGridDim((Vh - 2*Vsh)/BLOCK_DIM, 1, 1);
#else
  //for single GPU, half interior kernel covers engire half
  dim3 halfInteriorGridDim(Vh/BLOCK_DIM, 1, 1); 
#endif
  
  dim3 halfExteriorGridDim(2*Vsh/BLOCK_DIM, 1,1);
  dim3 blockDim(BLOCK_DIM , 1, 1);
  

  QudaPrecision prec = cudaSiteLink.precision;
  QudaReconstructType recon = cudaSiteLink.reconstruct;
  int even = 0;
  int odd  = 1;

  int nStream=9;
  cudaStream_t stream[nStream];
  for(int i = 0;i < nStream; i++){
    cudaStreamCreate(&stream[i]);
  }
  
  llfatOneLinkKernel(cudaFatLink, cudaSiteLink,cudaStaple, cudaStaple1,
		     param, act_path_coeff); CUERR;
  
  llfat_kernel_param_t kparam;
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


  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){

	
	for (int k = 0; k <8;k++){
	  kparam.kernel_type = ktype[k];
	  //even kernel
	  siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					   (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					   (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					   dir, nu,0,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[k]); CUERR;
	  /*
	  //odd kernel
	  siteComputeGenStapleParityKernel((void*)cudaStaple.odd, (void*)cudaStaple.even,
					   (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					   (void*)cudaFatLink.odd, (void*)cudaFatLink.even,
					   dir, nu,1,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[k]); CUERR;
	  */

	  
	}

 	for(int k=0; k < 4 ;k++){
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]);  CUERR;
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]);  CUERR;
	}
        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	//even kernel
	siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					 (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					 (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					 dir, nu,0,
					 act_path_coeff[2],
					 recon, prec, halfGridDim, 
					 kparam, &stream[nStream-1]); CUERR;
	/*
	//odd kernel
	siteComputeGenStapleParityKernel((void*)cudaStaple.odd, (void*)cudaStaple.even,
					 (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					 (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
					 dir, nu,1,
					 act_path_coeff[2],
					 recon, prec, halfGridDim, 
					 kparam, &stream[nStream-1]); CUERR;	
	*/

	for(int k = 0; k  < 4; k++){
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	}	
	for(int k = 0; k  < 4; k++){
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	}
	for(int k = 0; k  < 4; k++){
	  cudaStreamSynchronize(stream[2*k]);
	  cudaStreamSynchronize(stream[2*k+1]);
	}	

        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	//even
	computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL,
					  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					  (void*)cudaStaple.even, (void*)cudaStaple.odd,
					  dir, nu,even,0,
					  act_path_coeff[5],
					  recon, prec,  halfGridDim, kparam, &stream[nStream-1]); CUERR;
	//odd 
	/*
	computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL, 
					  (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					  (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
					  (void*)cudaStaple.odd, (void*)cudaStaple.even,
					  dir, nu,odd,0,
					  act_path_coeff[5],
					  recon, prec, halfGridDim, kparam, &stream[nStream-1]);	 CUERR;
	*/

	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){

	    for (int k = 0; k <8;k++){
	      kparam.kernel_type = ktype[k];	    
	      //even 
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						(void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						(void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						(void*)cudaStaple.even, (void*)cudaStaple.odd,
						dir, rho,even,1,
						act_path_coeff[3],
						recon, prec, halfGridDim, kparam, &stream[k]); CUERR;
	      /*
	      //odd 
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.odd, (void*)cudaStaple1.even,
						(void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
						(void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
						(void*)cudaStaple.odd, (void*)cudaStaple.even,
						dir, rho,odd,1,
						act_path_coeff[3],
						recon, prec,  halfGridDim, kparam, &stream[k]); CUERR;
	      */
	    }
	    for(int k=0; k < 4 ;k++){
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]);  CUERR;
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]);  CUERR;
	    }	    

	    kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	    //even 
	    computeGenStapleFieldParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
					      (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					      (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					      (void*)cudaStaple.even, (void*)cudaStaple.odd,
					      dir, rho,even,1,
					      act_path_coeff[3],
					      recon, prec, halfGridDim, kparam, &stream[nStream-1]); CUERR;
	    /*
	    //odd 
	    computeGenStapleFieldParityKernel((void*)cudaStaple1.odd, (void*)cudaStaple1.even,
					      (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
					      (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
					      (void*)cudaStaple.odd, (void*)cudaStaple.even,
					      dir, rho,odd,1,
					      act_path_coeff[3],
					      recon, prec, halfGridDim, kparam, &stream[nStream-1]); CUERR;
	    */

#ifdef MULTI_GPU
	    for(int k = 0; k  < 4; k++){
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	    }
	    for(int k=0; k < 4; k++){
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_BACKWARDS, &stream[2*k]); CUERR;
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	    }
#endif	    
	    for(int k = 0; k  < 4; k++){
	      cudaStreamSynchronize(stream[2*k]);
	      cudaStreamSynchronize(stream[2*k+1]);
	    }	
	    
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){				
		
		
		kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
		//even				
		computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL, 
						  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						  (void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						  dir, sig, even, 0,
						  act_path_coeff[4],
						  recon, prec, halfGridDim, kparam, &stream[nStream-1]);	 CUERR;
		/*
		//odd
		computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL,
						  (void*)cudaSiteLink.odd, (void*)cudaSiteLink.even,
						  (void*)cudaFatLink.odd, (void*)cudaFatLink.even, 
						  (void*)cudaStaple1.odd, (void*)cudaStaple1.even,
						  dir, sig, odd, 0,
						  act_path_coeff[4],
						  recon, prec, halfGridDim, kparam, &stream[nStream-1]);	 CUERR;
		*/
		
	      }			    
	    }//sig
	  }
	}//rho	
      }
    }//nu
  }//dir
  
  
  cudaThreadSynchronize(); 
  checkCudaError();
  
  for(int i=0;i < nStream; i++){
    cudaStreamDestroy(stream[i]);
  }

  return;
}

