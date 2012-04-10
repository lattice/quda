#include <read_gauge.h>
#include <gauge_field.h>

#include "gauge_force_quda.h"
#ifdef MULTI_GPU
#include "face_quda.h"
#endif


__constant__ int path_max_length;

#define GF_SITE_MATRIX_LOAD_TEX 1

//single precsison, 12-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle_recon, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle_recon, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkOdd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, mom_ga_stride)
#define RECONSTRUCT_MATRIX(sign, var) RECONSTRUCT_LINK_12(sign,var)
#define DECLARE_LINK_VARS(var) FloatN var##0, var##1, var##2, var##3, var##4
#define N_IN_FLOATN 4
#define GAUGE_FORCE_KERN_NAME parity_compute_gauge_force_kernel_sp12
#include "gauge_force_core.h"
#undef LOAD_EVEN_MATRIX 
#undef LOAD_ODD_MATRIX
#undef LOAD_ANTI_HERMITIAN 
#undef RECONSTRUCT_MATRIX
#undef DECLARE_LINK_VARS
#undef N_IN_FLOATN 
#undef GAUGE_FORCE_KERN_NAME

//double precsison, 12-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkOdd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, mom_ga_stride)
#define RECONSTRUCT_MATRIX(sign, var) RECONSTRUCT_LINK_12(sign,var)
#define DECLARE_LINK_VARS(var) FloatN var##0, var##1, var##2, var##3, var##4, var##5, var##6, var##7, var##8 
#define N_IN_FLOATN 2
#define GAUGE_FORCE_KERN_NAME parity_compute_gauge_force_kernel_dp12
#include "gauge_force_core.h"
#undef LOAD_EVEN_MATRIX 
#undef LOAD_ODD_MATRIX
#undef LOAD_ANTI_HERMITIAN 
#undef RECONSTRUCT_MATRIX
#undef DECLARE_LINK_VARS
#undef N_IN_FLOATN 
#undef GAUGE_FORCE_KERN_NAME

//single precision, 18-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(siteLink0TexSingle, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_SINGLE_TEX(siteLink1TexSingle, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var,mom_ga_stride)
#define RECONSTRUCT_MATRIX(sign, var) 
#define DECLARE_LINK_VARS(var) FloatN var##0, var##1, var##2, var##3, var##4, var##5, var##6, var##7, var##8 
#define N_IN_FLOATN 2
#define GAUGE_FORCE_KERN_NAME parity_compute_gauge_force_kernel_sp18
#include "gauge_force_core.h"
#undef LOAD_EVEN_MATRIX
#undef LOAD_ODD_MATRIX
#undef LOAD_ANTI_HERMITIAN 
#undef RECONSTRUCT_MATRIX
#undef DECLARE_LINK_VARS
#undef N_IN_FLOATN 
#undef GAUGE_FORCE_KERN_NAME

//double precision, 18-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, mom_ga_stride)
#define RECONSTRUCT_MATRIX(sign, var) 
#define DECLARE_LINK_VARS(var) FloatN var##0, var##1, var##2, var##3, var##4, var##5, var##6, var##7, var##8 
#define N_IN_FLOATN 2
#define GAUGE_FORCE_KERN_NAME parity_compute_gauge_force_kernel_dp18
#include "gauge_force_core.h"
#undef LOAD_EVEN_MATRIX
#undef LOAD_ODD_MATRIX
#undef LOAD_ANTI_HERMITIAN 
#undef RECONSTRUCT_MATRIX
#undef DECLARE_LINK_VARS
#undef N_IN_FLOATN 
#undef GAUGE_FORCE_KERN_NAME

void
gauge_force_init_cuda(QudaGaugeParam* param, int path_max_length)
{    
  
  static int gauge_force_init_cuda_flag = 0;
  if (gauge_force_init_cuda_flag){
    return;
  }
  gauge_force_init_cuda_flag=1;


#ifdef MULTI_GPU
  int E1 = param->X[0] + 4;
  int E1h = E1/2;
  int E2 = param->X[1] + 4;
  int E3 = param->X[2] + 4;
  int E4 = param->X[3] + 4;
  int E2E1 =E2*E1;
  int E3E2E1=E3*E2*E1;
  int Vh_ex = E1*E2*E3*E4/2;
  
  cudaMemcpyToSymbol("E1", &E1, sizeof(int));
  cudaMemcpyToSymbol("E1h", &E1h, sizeof(int));
  cudaMemcpyToSymbol("E2", &E2, sizeof(int));
  cudaMemcpyToSymbol("E3", &E3, sizeof(int));
  cudaMemcpyToSymbol("E4", &E4, sizeof(int));
  cudaMemcpyToSymbol("E2E1", &E2E1, sizeof(int));
  cudaMemcpyToSymbol("E3E2E1", &E3E2E1, sizeof(int));
  
  cudaMemcpyToSymbol("Vh_ex", &Vh_ex, sizeof(int));
#endif    

  int* X = param->X;
  int Vh = X[0]*X[1]*X[2]*X[3]/2;
  cudaMemcpyToSymbol("path_max_length", &path_max_length, sizeof(int));
  
#ifdef MULTI_GPU
  int site_ga_stride = param->site_ga_pad + Vh_ex;
#else  
  int site_ga_stride = param->site_ga_pad + Vh;
#endif

  cudaMemcpyToSymbol("site_ga_stride", &site_ga_stride, sizeof(int));
  int mom_ga_stride = param->mom_ga_pad + Vh;
  cudaMemcpyToSymbol("mom_ga_stride", &mom_ga_stride, sizeof(int));
     
}


void
gauge_force_cuda_dir(cudaGaugeField&  cudaMom, int dir, double eb3, cudaGaugeField& cudaSiteLink,
		     QudaGaugeParam* param, int** input_path, 
		     int* length, void* path_coeff, int num_paths, int max_length)
{
  int i, j;
    //input_path
    int bytes = num_paths*max_length* sizeof(int);
    int* input_path_d;
   
    if(cudaMalloc((void**)&input_path_d, bytes) != cudaSuccess){
      errorQuda("cudaMalloc failed for input_path_d\n");
    }

    cudaMemset(input_path_d, 0, bytes);checkCudaError();

    int* input_path_h = (int*)malloc(bytes);
    if (input_path_h == NULL){
	printf("ERROR: malloc failed for input_path_h in function %s\n", __FUNCTION__);
	exit(1);
    }
        
    memset(input_path_h, 0, bytes);
    for(i=0;i < num_paths;i++){
	for(j=0; j < length[i]; j++){
	    input_path_h[i*max_length + j] =input_path[i][j];
	}
    }

    cudaMemcpy(input_path_d, input_path_h, bytes, cudaMemcpyHostToDevice); 
    
    //length
    int* length_d;
    if(cudaMalloc((void**)&length_d, num_paths*sizeof(int)) != cudaSuccess){
      errorQuda("cudaMalloc failed for length_d\n");
    }
    cudaMemcpy(length_d, length, num_paths*sizeof(int), cudaMemcpyHostToDevice);
    
    //path_coeff
    int gsize = param->cuda_prec;
    void* path_coeff_d;
    if(cudaMalloc((void**)&path_coeff_d, num_paths*gsize) != cudaSuccess){
      errorQuda("cudaMalloc failed for path_coeff_d\n");
    }
    cudaMemcpy(path_coeff_d, path_coeff, num_paths*gsize, cudaMemcpyHostToDevice); 

    //compute the gauge forces
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 blockDim(BLOCK_DIM, 1,1);
    dim3 gridDim(volume/blockDim.x, 1, 1);
    dim3 halfGridDim(volume/(2*blockDim.x), 1, 1);
        
    void* momEven = (void*)cudaMom.Even_p();
    void* momOdd = (void*)cudaMom.Odd_p();

    void* linkEven = (void*)cudaSiteLink.Even_p();
    void* linkOdd = (void*)cudaSiteLink.Odd_p();        
    
    kernel_param_t kparam;
#ifdef MULTI_GPU
    for(int i =0;i < 4;i++){
      kparam.ghostDim[i] = commDimPartitioned(i);
    }
#endif

    kparam.threads  = volume/2;

    if(param->cuda_prec == QUDA_DOUBLE_PRECISION){
      cudaBindTexture(0, siteLink0TexDouble, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
      cudaBindTexture(0, siteLink1TexDouble, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);			      
    }else{ //QUDA_SINGLE_PRECISION
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);		
      }else{//QUDA_RECONSTRUCT_12
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes()/2);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes()/2);	
      }
    }
    
    if(param->cuda_prec == QUDA_DOUBLE_PRECISION){      
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	parity_compute_gauge_force_kernel_dp18<0><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									dir, eb3,
									(double2*)linkEven, (double2*)linkOdd, 
									input_path_d, length_d, (double*)path_coeff_d,
									     num_paths, kparam);   
	parity_compute_gauge_force_kernel_dp18<1><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									dir, eb3,
									(double2*)linkEven, (double2*)linkOdd, 
									input_path_d, length_d, (double*)path_coeff_d,
									     num_paths, kparam);  
		
      }else{ //QUDA_RECONSTRUCT_12
   	parity_compute_gauge_force_kernel_dp12<0><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									     dir, eb3,
									     (double2*)linkEven, (double2*)linkOdd, 
									     input_path_d, length_d, (double*)path_coeff_d,
									     num_paths, kparam);   
	parity_compute_gauge_force_kernel_dp12<1><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									     dir, eb3,
									     (double2*)linkEven, (double2*)linkOdd, 
									     input_path_d, length_d, (double*)path_coeff_d,
									     num_paths, kparam);    
      }
    }else{ //QUDA_SINGLE_PRECISION
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	
	parity_compute_gauge_force_kernel_sp18<0><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float2*)linkEven, (float2*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths, kparam);   
	parity_compute_gauge_force_kernel_sp18<1><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float2*)linkEven, (float2*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths, kparam); 
	
      }else{ //QUDA_RECONSTRUCT_12
	parity_compute_gauge_force_kernel_sp12<0><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float4*)linkEven, (float4*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths, kparam);   
	//odd
	/* The reason we do not switch the even/odd function input paramemters and the texture binding
	 * is that we use the oddbit to decided where to load, in the kernel function
	 */
	parity_compute_gauge_force_kernel_sp12<1><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float4*)linkEven, (float4*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths, kparam);  
      }
      
    }
    

    if(param->cuda_prec == QUDA_DOUBLE_PRECISION){
      cudaUnbindTexture(siteLink0TexDouble);
      cudaUnbindTexture(siteLink1TexDouble);
    }else{ //QUDA_SINGLE_PRECISION
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);
      }else{//QUDA_RECONSTRUCT_12
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);
      }
    }

    
    checkCudaError();
    
    cudaFree(input_path_d); checkCudaError();
    free(input_path_h);
    cudaFree(length_d);
    cudaFree(path_coeff_d);

    

}


void
gauge_force_cuda(cudaGaugeField&  cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
		 QudaGaugeParam* param, int*** input_path, 
		 int* length, void* path_coeff, int num_paths, int max_length)
{
  
  for(int dir=0; dir < 4; dir++){
    gauge_force_cuda_dir(cudaMom, dir, eb3, cudaSiteLink, param, input_path[dir], 
			 length, path_coeff, num_paths, max_length);
  }
  
}
