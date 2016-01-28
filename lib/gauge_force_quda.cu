#include <read_gauge.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <force_common.h>
#include <gauge_force_quda.h>
#ifdef MULTI_GPU
#include "face_quda.h"
#endif
#include <index_helper.cuh>

namespace quda {

#ifdef GPU_GAUGE_FORCE

  namespace gaugeforce {
#include <dslash_constants.h> // FIXME dependency of dslash_textures.h
#include <dslash_textures.h> // needed for texture references
  } // namespace gaugeforce

  using namespace gaugeforce;

  struct GaugeForceArg {
    unsigned long threads;
    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int ghostDim[4]; // Whether a ghost zone has been allocated for a given dimension

    int gauge_stride;
    int mom_stride;

    int num_paths;
    int path_max_length;
    
    double coeff;
    int dir;

    const int *length;
    const int *input_path;
    const double *path_coeff;

    int count; // equal to sum of all path lengths.  Used a convenience for computing perf

  };
 
  std::ostream& operator<<(std::ostream& output, const GaugeForceArg& arg) {
    std::cout << "threads         = " << arg.threads << std::endl;
    std::cout << "gauge_stride    = " << arg.gauge_stride << std::endl;
    std::cout << "mom_stride      = " << arg.mom_stride << std::endl;
    std::cout << "num_paths       = " << arg.num_paths << std::endl;
    std::cout << "path_max_length = " << arg.path_max_length << std::endl;
    std::cout << "coeff           = " << arg.coeff << std::endl;
    std::cout << "dir             = " << arg.dir << std::endl;
    std::cout << "count           = " << arg.count << std::endl;
  }

#define GF_SITE_MATRIX_LOAD_TEX 1

  //single precsison, 12-reconstruct
#if (GF_SITE_MATRIX_LOAD_TEX == 1)
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(siteLink0TexSingle_recon, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_SINGLE_TEX(siteLink1TexSingle_recon, dir, idx, var, arg.gauge_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_SINGLE(linkOdd, dir, idx, var, arg.gauge_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, arg.mom_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_12_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, arg.gauge_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_12_DOUBLE(linkOdd, dir, idx, var, gf.gauge_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, arg.mom_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_SINGLE_TEX(siteLink0TexSingle, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_SINGLE_TEX(siteLink1TexSingle, dir, idx, var, arg.gauge_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, arg.gauge_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var,arg.mom_stride)
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
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18_DOUBLE_TEX(siteLink0TexDouble, linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) 	LOAD_MATRIX_18_DOUBLE_TEX(siteLink1TexDouble, linkOdd, dir, idx, var, arg.gauge_stride)
#else
#define LOAD_EVEN_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkEven, dir, idx, var, arg.gauge_stride)
#define LOAD_ODD_MATRIX(dir, idx, var) LOAD_MATRIX_18(linkOdd, dir, idx, var, arg.gauge_stride)
#endif
#define LOAD_ANTI_HERMITIAN(src, dir, idx, var) LOAD_ANTI_HERMITIAN_DIRECT(src, dir, idx, var, arg.mom_stride)
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


  class GaugeForceCuda : public TunableLocalParity {

  private:
    cudaGaugeField &mom;
    const cudaGaugeField &link;
    const GaugeForceArg &arg;

    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugeForceCuda(cudaGaugeField &mom, const cudaGaugeField &link, const GaugeForceArg &arg) :
      mom(mom), link(link), arg(arg) { 

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
	cudaUnbindTexture(siteLink0TexDouble);
	cudaUnbindTexture(siteLink1TexDouble);
      }else{ //QUDA_SINGLE_PRECISION
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  cudaUnbindTexture(siteLink0TexSingle);
	  cudaUnbindTexture(siteLink1TexSingle);
	}else{//QUDA_RECONSTRUCT_12
	  cudaUnbindTexture(siteLink0TexSingle_recon);
	  cudaUnbindTexture(siteLink1TexSingle_recon);
	}
      }
    }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());    
      if(link.Precision() == QUDA_DOUBLE_PRECISION){      
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  parity_compute_gauge_force_kernel_dp18<double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
										(double2*)link.Even_p(), (double2*)link.Odd_p(), arg);
	}else{ //QUDA_RECONSTRUCT_12
	  parity_compute_gauge_force_kernel_dp12<double><<<tp.grid, tp.block>>>((double2*)mom.Even_p(), (double2*)mom.Odd_p(),
										(double2*)link.Even_p(), (double2*)link.Odd_p(), arg);
	}
      }else{ //QUDA_SINGLE_PRECISION
	if(link.Reconstruct() == QUDA_RECONSTRUCT_NO){
	  parity_compute_gauge_force_kernel_sp18<float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									       (float2*)link.Even_p(), (float2*)link.Odd_p(), arg);
	}else{ //QUDA_RECONSTRUCT_12
	  parity_compute_gauge_force_kernel_sp12<float><<<tp.grid, tp.block>>>((float2*)mom.Even_p(), (float2*)mom.Odd_p(),
									       (float4*)link.Even_p(), (float4*)link.Odd_p(), arg);
	}
      }
    }
  
    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); } 
  
    long long flops() const { return (arg.count - arg.num_paths + 1) * 198ll * mom.Volume(); }
    long long bytes() const { return ((arg.count + 1ll) * link.Reconstruct() + 2ll*mom.Reconstruct()) * mom.Volume() * mom.Precision(); }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << link.X()[0] << "x"  << link.X()[1] << "x"  << link.X()[2] << "x" << link.X()[3];
      aux << "threads=" << link.Volume() << ",prec=" << link.Precision();
      aux << ",stride=" << link.Stride() << ",recon=" << link.Reconstruct();
      aux << ",dir=" << arg.dir << ",num_paths=" << arg.num_paths;
      return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
    }
  
  };
  
  void
  gauge_force_cuda_dir(cudaGaugeField& cudaMom, const int dir, const double eb3, const cudaGaugeField& cudaSiteLink,
		       const QudaGaugeParam* param, int** input_path, const int* length, const double* path_coeff, 
		       const int num_paths, const int max_length)
  {
    GaugeForceArg arg;

    //input_path
    size_t bytes = num_paths*max_length*sizeof(int);

    int *input_path_d = (int *) device_malloc(bytes);
    cudaMemset(input_path_d, 0, bytes);
    checkCudaError();

    int* input_path_h = (int *) safe_malloc(bytes);
    memset(input_path_h, 0, bytes);

    // use this to estimate bytes and flops
    arg.count = 0;
    for(int i=0; i < num_paths; i++) {
      for(int j=0; j < length[i]; j++) {
	input_path_h[i*max_length + j] = input_path[i][j];
	arg.count++;
      }
    }

    cudaMemcpy(input_path_d, input_path_h, bytes, cudaMemcpyHostToDevice); 
    
    //length
    int* length_d = (int *) device_malloc(num_paths*sizeof(int));
    cudaMemcpy(length_d, length, num_paths*sizeof(int), cudaMemcpyHostToDevice);
    
    //path_coeff
    void* path_coeff_d = device_malloc(num_paths*sizeof(double));
    cudaMemcpy(path_coeff_d, path_coeff, num_paths*sizeof(double), cudaMemcpyHostToDevice);

    for(int i=0; i<4; i++) {
      arg.X[i] = cudaMom.X()[i];
      arg.E[i] = cudaSiteLink.X()[i];
      arg.ghostDim[i] = commDimPartitioned(i);
    }
    arg.threads = cudaMom.VolumeCB();
    arg.gauge_stride = cudaSiteLink.Stride();
    arg.mom_stride = cudaMom.Stride();
    arg.num_paths = num_paths;
    arg.path_max_length = max_length;
    arg.coeff = eb3;
    arg.dir = dir;
    arg.length = length_d;
    arg.input_path = input_path_d;
    arg.path_coeff = static_cast<double*>(path_coeff_d);

    GaugeForceCuda gaugeForce(cudaMom, cudaSiteLink, arg);
    gaugeForce.apply(0);
    checkCudaError();
    
    host_free(input_path_h);
    device_free(input_path_d);
    device_free(length_d);
    device_free(path_coeff_d);
  }
#endif // GPU_GAUGE_FORCE


  void
  gauge_force_cuda(cudaGaugeField&  cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
		   QudaGaugeParam* param, int*** input_path, 
		   int* length, double* path_coeff, int num_paths, int max_length)
  {  
#ifdef GPU_GAUGE_FORCE
    for(int dir=0; dir < 4; dir++){
      gauge_force_cuda_dir(cudaMom, dir, eb3, cudaSiteLink, param, input_path[dir], 
			   length, path_coeff, num_paths, max_length);
    }
#else
    errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  }

} // namespace quda
