#include <read_gauge.h>
#include <gauge_field.h>

#include "gauge_force_quda.h"



__constant__ int path_max_length;

#define GF_SITE_MATRIX_LOAD_TEX 1

//single precsison, 12-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle_recon, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle_recon, dir, idx, var, Vh)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkOdd, dir, idx, var, Vh)
#endif
#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_DIRECT
#define RECONSTRUCT_MATRIX(dir, idx, sign, var) RECONSTRUCT_LINK_12(dir,idx,sign,var)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, Vh)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkOdd, dir, idx, var, Vh)
#endif
#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_DIRECT
#define RECONSTRUCT_MATRIX(dir, idx, sign, var) RECONSTRUCT_LINK_12(dir,idx,sign,var)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(siteLink0TexSingle, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_SINGLE_TEX(siteLink1TexSingle, dir, idx, var, Vh)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, Vh)
#endif
#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_DIRECT
#define RECONSTRUCT_MATRIX(dir, idx, sign, var) 
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, Vh)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, Vh)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, Vh)
#endif
#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_DIRECT
#define RECONSTRUCT_MATRIX(dir, idx, sign, var) 
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
  
#ifdef MULTI_GPU
#error "multi gpu is not supported for gauge force computation"  
#endif
  
    static int gauge_force_init_cuda_flag = 0;
    if (gauge_force_init_cuda_flag){
	return;
    }
    gauge_force_init_cuda_flag=1;

    init_kernel_cuda(param);
    
    cudaMemcpyToSymbol("path_max_length", &path_max_length, sizeof(int));

}


void
gauge_force_cuda(cudaGaugeField&  cudaMom, int dir, double eb3, cudaGaugeField& cudaSiteLink,
                 QudaGaugeParam* param, int** input_path, 
		 int* length, void* path_coeff, int num_paths, int max_length)
{
    int i, j;
    //input_path
    int bytes = num_paths*max_length* sizeof(int);
    int* input_path_d;
    cudaMalloc((void**)&input_path_d, bytes); checkCudaError();    
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

    cudaMemcpy(input_path_d, input_path_h, bytes, cudaMemcpyHostToDevice); checkCudaError();
    
    //length
    int* length_d;
    cudaMalloc((void**)&length_d, num_paths*sizeof(int)); checkCudaError();
    cudaMemcpy(length_d, length, num_paths*sizeof(int), cudaMemcpyHostToDevice); checkCudaError();
    
    //path_coeff
    int gsize;
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
	gsize = sizeof(double);
    }else{
	gsize= sizeof(float);
    }     
    void* path_coeff_d;
    cudaMalloc((void**)&path_coeff_d, num_paths*gsize); checkCudaError();
    cudaMemcpy(path_coeff_d, path_coeff, num_paths*gsize, cudaMemcpyHostToDevice); checkCudaError();

    //compute the gauge forces
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 blockDim(BLOCK_DIM, 1,1);
    dim3 gridDim(volume/blockDim.x, 1, 1);
    dim3 halfGridDim(volume/(2*blockDim.x), 1, 1);
        
    void* momEven = (void*)cudaMom.Even_p();
    void* momOdd = (void*)cudaMom.Odd_p();

    void* linkEven = (void*)cudaSiteLink.Even_p();
    void* linkOdd = (void*)cudaSiteLink.Odd_p();        
    
    
    if(param->cuda_prec == QUDA_DOUBLE_PRECISION){
      cudaBindTexture(0, siteLink0TexDouble, cudaSiteLink.Even_p(), cudaSiteLink.Bytes());
      cudaBindTexture(0, siteLink1TexDouble, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes());			      
    }else{ //QUDA_SINGLE_PRECISION
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.Even_p(), cudaSiteLink.Bytes());
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes());		
      }else{//QUDA_RECONSTRUCT_12
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.Even_p(), cudaSiteLink.Bytes());
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.Odd_p(), cudaSiteLink.Bytes());	
      }
    }
    
    if(param->cuda_prec == QUDA_DOUBLE_PRECISION){      
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	parity_compute_gauge_force_kernel_dp18<0><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									dir, eb3,
									(double2*)linkEven, (double2*)linkOdd, 
									input_path_d, length_d, (double*)path_coeff_d,
									num_paths);   
	parity_compute_gauge_force_kernel_dp18<1><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									dir, eb3,
									(double2*)linkEven, (double2*)linkOdd, 
									input_path_d, length_d, (double*)path_coeff_d,
									num_paths);  
		
      }else{ //QUDA_RECONSTRUCT_12
   	parity_compute_gauge_force_kernel_dp12<0><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									     dir, eb3,
									     (double2*)linkEven, (double2*)linkOdd, 
									     input_path_d, length_d, (double*)path_coeff_d,
									     num_paths);   
	parity_compute_gauge_force_kernel_dp12<1><<<halfGridDim, blockDim>>>((double2*)momEven, (double2*)momOdd,
									     dir, eb3,
									     (double2*)linkEven, (double2*)linkOdd, 
									     input_path_d, length_d, (double*)path_coeff_d,
									     num_paths);    
      }
    }else{ //QUDA_SINGLE_PRECISION
      if(param->reconstruct == QUDA_RECONSTRUCT_NO){
	
	parity_compute_gauge_force_kernel_sp18<0><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float2*)linkEven, (float2*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths);   
	parity_compute_gauge_force_kernel_sp18<1><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float2*)linkEven, (float2*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths); 
	
      }else{ //QUDA_RECONSTRUCT_12
	parity_compute_gauge_force_kernel_sp12<0><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float4*)linkEven, (float4*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths);   
	//odd
	/* The reason we do not switch the even/odd function input paramemters and the texture binding
	 * is that we use the oddbit to decided where to load, in the kernel function
	 */
	parity_compute_gauge_force_kernel_sp12<1><<<halfGridDim, blockDim>>>((float2*)momEven, (float2*)momOdd,
									     dir, eb3,
									     (float4*)linkEven, (float4*)linkOdd, 
									     input_path_d, length_d, (float*)path_coeff_d,
									     num_paths);  
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


