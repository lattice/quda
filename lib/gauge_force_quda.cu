#include <read_gauge.h>
#include <gauge_field.h>

#include "gauge_force_quda.h"
#ifdef MULTI_GPU
#include "face_quda.h"
#endif

namespace quda {

#define GF_SITE_MATRIX_LOAD_TEX 1

  //single precsison, 12-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle_recon, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle_recon, dir, idx, var, gf.site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkOdd, dir, idx, var, gf.site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, gf.mom_ga_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, gf.site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkOdd, dir, idx, var, gf.site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, gf.mom_ga_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(siteLink0TexSingle, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_SINGLE_TEX(siteLink1TexSingle, dir, idx, var, gf.site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, gf.site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var,gf.mom_ga_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, gf.site_ga_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, gf.site_ga_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, gf.site_ga_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, gf.mom_ga_stride)
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

    int* X = param->X;
  
    int Vh = X[0]*X[1]*X[2]*X[3]/2;
    fat_force_const_t gf_h;
    gf_h.path_max_length = path_max_length;  
#ifdef MULTI_GPU  
    int Vh_ex = (X[0]+4)*(X[1]+4)*(X[2]+4)*(X[3]+4)/2;
    gf_h.site_ga_stride = param->site_ga_pad + Vh_ex;
#else  
    gf_h.site_ga_stride = param->site_ga_pad + Vh;
#endif
  
    gf_h.mom_ga_stride = param->mom_ga_pad + Vh;  
    cudaMemcpyToSymbol(gf, &gf_h, sizeof(fat_force_const_t));     
  }


  class GaugeForceCuda : public Tunable {

  private:
    cudaGaugeField &mom;
    const int dir;
    const double &eb3;
    const cudaGaugeField &link;
    const int *input_path;
    const int *length;
    const double *path_coeff;
    const int num_paths;
    const kernel_param_t &kparam;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }
  
    // don't tune the grid dimension
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return kparam.threads; }

  public:
    GaugeForceCuda(cudaGaugeField &mom, const int dir, const double &eb3, const cudaGaugeField &link,
		   const int *input_path, const int *length, const double *path_coeff, 
		   const int num_paths, const kernel_param_t &kparam) :
      mom(mom), dir(dir), eb3(eb3), link(link), input_path(input_path), length(length), 
      path_coeff(path_coeff), num_paths(num_paths), kparam(kparam) { 

      if(link.Precision() == QUDA_DOUBLE_PRECISION){
	cudaBindTexture(0, siteLink0TexDouble, link.Even_p(), link.Bytes()/2);
	cudaBindTexture(0, siteLink1TexDouble, link.Odd_p(), link.Bytes()/2);			      
      }else{ //QUDA_SINGLE_PRECISION
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  cudaBindTexture(0, siteLink0TexSingle, link.Even_p(), link.Bytes()/2);
	  cudaBindTexture(0, siteLink1TexSingle, link.Odd_p(), link.Bytes()/2);		
	}else{//QUDA_RECONSTRUCT_12
	  cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
	  cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);	
	}
      }
    }

    virtual ~GaugeForceCuda() {
      if(link.Precision() == QUDA_DOUBLE_PRECISION){
	cudaBindTexture(0, siteLink0TexDouble, link.Even_p(), link.Bytes()/2);
	cudaBindTexture(0, siteLink1TexDouble, link.Odd_p(), link.Bytes()/2);			      
      }else{ //QUDA_SINGLE_PRECISION
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  cudaBindTexture(0, siteLink0TexSingle, link.Even_p(), link.Bytes()/2);
	  cudaBindTexture(0, siteLink1TexSingle, link.Odd_p(), link.Bytes()/2);		
	}else{//QUDA_RECONSTRUCT_12
	  cudaBindTexture(0, siteLink0TexSingle_recon, link.Even_p(), link.Bytes()/2);
	  cudaBindTexture(0, siteLink1TexSingle_recon, link.Odd_p(), link.Bytes()/2);	
	}
      }
    }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());    
      if(link.Precision() == QUDA_DOUBLE_PRECISION){      
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  parity_compute_gauge_force_kernel_dp18<0,double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
									   dir, eb3,
									   (double2*)link.Even_p(), (double2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);   
	  parity_compute_gauge_force_kernel_dp18<1,double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
									   dir, eb3,
									   (double2*)link.Even_p(), (double2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);  
	
	}else{ //QUDA_RECONSTRUCT_12
	  parity_compute_gauge_force_kernel_dp12<0,double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
									   dir, eb3,
									   (double2*)link.Even_p(), (double2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);   
	  parity_compute_gauge_force_kernel_dp12<1,double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
									   dir, eb3,
									   (double2*)link.Even_p(), (double2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);    
	}
      }else{ //QUDA_SINGLE_PRECISION
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	
	  parity_compute_gauge_force_kernel_sp18<0,float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									   dir, eb3,
									   (float2*)link.Even_p(), (float2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);   
	  parity_compute_gauge_force_kernel_sp18<1,float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									   dir, eb3,
									   (float2*)link.Even_p(), (float2*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam); 
	
	}else{ //QUDA_RECONSTRUCT_12
	  parity_compute_gauge_force_kernel_sp12<0,float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									   dir, eb3,
									   (float4*)link.Even_p(), (float4*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);   
	  //odd
	  /* The reason we do not switch the even/odd function input paramemters and the texture binding
	   * is that we use the oddbit to decided where to load, in the kernel function
	   */
	  parity_compute_gauge_force_kernel_sp12<1,float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									   dir, eb3,
									   (float4*)link.Even_p(), (float4*)link.Odd_p(), 
									   input_path, length, path_coeff,
									   num_paths, kparam);  
	}
      }
    }
  
    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); } 
  
    long long flops() const { return 0; } // FIXME: add flops counter
  
    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << link.X()[0] << "x";
      vol << link.X()[1] << "x";
      vol << link.X()[2] << "x";
      vol << link.X()[3] << "x";
      aux << "threads=" << link.Volume() << ",prec=" << link.Precision();
      aux << "stride=" << link.Stride() << ",recon=" << link.Reconstruct();
      aux << "dir=" << dir << "num_paths=" << num_paths;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }  
  
  };
  
  void
  gauge_force_cuda_dir(cudaGaugeField& cudaMom, const int dir, const double eb3, const cudaGaugeField& cudaSiteLink,
		       const QudaGaugeParam* param, int** input_path, const int* length, const double* path_coeff, 
		       const int num_paths, const int max_length)
  {
    //input_path
    size_t bytes = num_paths*max_length*sizeof(int);

    int *input_path_d = (int *) device_malloc(bytes);
    cudaMemset(input_path_d, 0, bytes);
    checkCudaError();

    int* input_path_h = (int *) safe_malloc(bytes);
    memset(input_path_h, 0, bytes);

    for(int i=0; i < num_paths; i++) {
      for(int j=0; j < length[i]; j++) {
	input_path_h[i*max_length + j] = input_path[i][j];
      }
    }

    cudaMemcpy(input_path_d, input_path_h, bytes, cudaMemcpyHostToDevice); 
    
    //length
    int* length_d = (int *) device_malloc(num_paths*sizeof(int));
    cudaMemcpy(length_d, length, num_paths*sizeof(int), cudaMemcpyHostToDevice);
    
    //path_coeff
    void* path_coeff_d = device_malloc(num_paths*sizeof(double));
    cudaMemcpy(path_coeff_d, path_coeff, num_paths*sizeof(double), cudaMemcpyHostToDevice); 

    //compute the gauge forces
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
        
    kernel_param_t kparam;
#ifdef MULTI_GPU
    for(int i=0; i<4; i++) {
      kparam.ghostDim[i] = commDimPartitioned(i);
    }
#endif
    kparam.threads = volume/2;

    GaugeForceCuda gaugeForce(cudaMom, dir, eb3, cudaSiteLink, input_path_d, 
			      length_d, reinterpret_cast<double*>(path_coeff_d), num_paths, kparam);
    gaugeForce.apply(0);
    checkCudaError();
    
    host_free(input_path_h);
    device_free(input_path_d);
    device_free(length_d);
    device_free(path_coeff_d);
  }


  void
  gauge_force_cuda(cudaGaugeField&  cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
		   QudaGaugeParam* param, int*** input_path, 
		   int* length, double* path_coeff, int num_paths, int max_length)
  {  
    for(int dir=0; dir < 4; dir++){
      gauge_force_cuda_dir(cudaMom, dir, eb3, cudaSiteLink, param, input_path[dir], 
			   length, path_coeff, num_paths, max_length);
    }  
  }

} // namespace quda
