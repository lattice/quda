#include <unistd.h>
#include <fast_intdiv.h>
#include <convert.h>

enum KernelType {
  INTERIOR_KERNEL = 5,
  EXTERIOR_KERNEL_ALL = 6,
  EXTERIOR_KERNEL_X = 0,
  EXTERIOR_KERNEL_Y = 1,
  EXTERIOR_KERNEL_Z = 2,
  EXTERIOR_KERNEL_T = 3,
  KERNEL_POLICY = 7
};

  struct DslashParam {
    int threads; // the desired number of active threads
    int parity;  // Even-Odd or Odd-Even

    int_fastdiv block[4]; // dslash tile block parameter
    int_fastdiv grid[4]; // dslash tile grid parameter
    int_fastdiv swizzle; // block index swizzle factor

    DslashConstant dc;

    KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T

    int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
    int ghostDim[QUDA_MAX_DIM]; // Whether a ghost zone has been allocated for a given dimension
    int ghostOffset[QUDA_MAX_DIM+1][2];
    int ghostNormOffset[QUDA_MAX_DIM+1][2];
    int sp_stride; // spinor stride

#ifdef GPU_CLOVER_DIRAC
    int cl_stride; // clover stride
#endif
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
    int fl_stride; // twisted-mass flavor stride
#endif
    int gauge_stride;
#ifdef GPU_STAGGERED_DIRAC
    int long_gauge_stride;
    float fat_link_max;
#endif 

    int gauge_fixed; // whether the gauge field is fixed to axial gauge

    double t_boundary;
    float t_boundary_f;

    bool Pt0;
    bool PtNm1;

    double anisotropy;
    float anisotropy_f;

    float2 An2;
    float2 TB2;
    float2 No2;

    int threadDimMapLower[4];
    int threadDimMapUpper[4];

    double coeff; // used as a gauge field scaling factor by the staggered kernels
    float coeff_f;

    double a;
    float a_f;

    double b;
    float b_f;

    double c;
    float c_f;

    double d;
    float d_f;

    double a_inv;
    float a_inv_f;

    double rho;
    float rho_f;

    double mferm;
    float mferm_f;

    // domain wall constants
    double m5_d;
    float m5_f;

    // the coefficients used in MDWF
    double mdwf_b5_d[QUDA_MAX_DWF_LS];
    double mdwf_c5_d[QUDA_MAX_DWF_LS];

    float mdwf_b5_f[QUDA_MAX_DWF_LS];
    float mdwf_c5_f[QUDA_MAX_DWF_LS];

    double tProjScale;
    float tProjScale_f;

    void *out;
    float *outNorm;
    
    void *in;
    float *inNorm;

    void *ghost[2*QUDA_MAX_DIM];
    float *ghostNorm[2*QUDA_MAX_DIM];

    void *x;
    float *xNorm;

    void *gauge0;
    void *gauge1;

    void *longGauge0;
    void *longGauge1;

    void *longPhase0;
    void *longPhase1;

    void *clover;
    float *cloverNorm;

    void *cloverInv;
    float *cloverInvNorm;

    double twist_a;
    double twist_b;

    int Vsh; // used by contraction kernels

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t inTex;
    cudaTextureObject_t inTexNorm;
    cudaTextureObject_t ghostTex[2*QUDA_MAX_DIM];
    cudaTextureObject_t ghostTexNorm[2*QUDA_MAX_DIM];
    cudaTextureObject_t xTex;
    cudaTextureObject_t xTexNorm;
    cudaTextureObject_t outTex;
    cudaTextureObject_t outTexNorm;
    cudaTextureObject_t gauge0Tex; // also applies to fat gauge
    cudaTextureObject_t gauge1Tex; // also applies to fat gauge
    cudaTextureObject_t longGauge0Tex;
    cudaTextureObject_t longGauge1Tex;
    cudaTextureObject_t longPhase0Tex;
    cudaTextureObject_t longPhase1Tex;
    cudaTextureObject_t cloverTex;
    cudaTextureObject_t cloverNormTex;
    cudaTextureObject_t cloverInvTex;
    cudaTextureObject_t cloverInvNormTex;
#endif

    // used by the autotuner to switch on/off remote writing vs using copy engines
    bool remote_write;

    void print() {
      printfQuda("threads = %d\n", threads);
      printfQuda("parity = %d\n", parity);
      printfQuda("X = {%d, %d, %d, %d}\n", (int)dc.X[0], (int)dc.X[1], (int)dc.X[2], (int)dc.X[3]);
      printfQuda("Xh = {%d, %d, %d, %d}\n", (int)dc.Xh[0], (int)dc.Xh[1], (int)dc.Xh[2], (int)dc.Xh[3]);
      printfQuda("volume4CB = %d\n", (int)dc.volume_4d_cb);
      printfQuda("Ls = %d\n", dc.Ls);
      printfQuda("kernel_type = %d\n", kernel_type);
      printfQuda("commDim = {%d, %d, %d, %d}\n", commDim[0], commDim[1], commDim[2], commDim[3]);
      printfQuda("ghostDim = {%d, %d, %d, %d}\n", ghostDim[0], ghostDim[1], ghostDim[2], ghostDim[3]);
      printfQuda("ghostOffset = {{%d, %d}, {%d, %d}, {%d, %d}, {%d, %d}}\n", ghostOffset[0][0], ghostOffset[0][1], 
                                                                              ghostOffset[1][0], ghostOffset[1][1],
                                                                              ghostOffset[2][0], ghostOffset[2][1],
                                                                              ghostOffset[3][0], ghostOffset[3][1]);
      printfQuda("ghostNormOffset = {{%d, %d}, {%d, %d}, {%d, %d}, {%d, %d}}\n", ghostNormOffset[0][0], ghostNormOffset[0][1], 
                                                                                 ghostNormOffset[1][0], ghostNormOffset[1][1],
                                                                                 ghostNormOffset[2][0], ghostNormOffset[2][1],
                                                                                 ghostNormOffset[3][0], ghostNormOffset[3][1]);
      printfQuda("sp_stride = %d\n", sp_stride);
#ifdef GPU_CLOVER_DIRAC
      printfQuda("cl_stride = %d\n", cl_stride);
#endif
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      printfQuda("fl_stride = %d\n", fl_stride);
#endif
#ifdef GPU_STAGGERED_DIRAC
      printfQuda("gauge_stride = %d\n", gauge_stride);
      printfQuda("long_gauge_stride = %d\n", long_gauge_stride);
      printfQuda("fat_link_max = %e\n", fat_link_max);
#endif
      printfQuda("threadDimMapLower = {%d, %d, %d, %d}\n", threadDimMapLower[0], threadDimMapLower[1],
		 threadDimMapLower[2], threadDimMapLower[3]);
      printfQuda("threadDimMapUpper = {%d, %d, %d, %d}\n", threadDimMapUpper[0], threadDimMapUpper[1],
		 threadDimMapUpper[2], threadDimMapUpper[3]);
      printfQuda("a = %e\n", a);
      printfQuda("b = %e\n", b);
      printfQuda("c = %e\n", c);
      printfQuda("d = %e\n", d);
      printfQuda("a_inv = %e\n", a_inv);
      printfQuda("rho = %e\n", rho);
      printfQuda("mferm = %e\n", mferm);
      printfQuda("tProjScale = %e\n", tProjScale);
      printfQuda("twist_a = %e\n", twist_a);
      printfQuda("twist_b = %e\n", twist_b);
    }
  };

typedef struct fat_force_stride_s {
  int fat_ga_stride;
  int long_ga_stride;
  int site_ga_stride;
  int staple_stride;
  int mom_ga_stride;
  int path_max_length;
  int color_matrix_stride;
} fat_force_const_t;

//for link fattening/gauge force/fermion force code
__constant__ int Vh;
__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;

__constant__ int X2X1;
__constant__ int X1h;
__constant__ int X1m1;
__constant__ int X2m1;
__constant__ int X3m1;
__constant__ int X4m1;

__constant__ int X3X2;
__constant__ int X3X2X1;

__constant__ int X2X1mX1;
__constant__ int X3X2X1mX2X1;
__constant__ int X4X3X2X1mX3X2X1;

__constant__ int E1, E2, E3, E4, E1h;
__constant__ int Vh_ex;
__constant__ int E2E1;
__constant__ int E3E2E1;

__constant__ fat_force_const_t fl; //fatlink
__constant__ fat_force_const_t gf; //gauge force
__constant__ fat_force_const_t hf; //hisq force

void initLatticeConstants(const LatticeField &lat, TimeProfile &profile)
{
  profile.TPSTART(QUDA_PROFILE_CONSTANT);

  checkCudaError();

  //constants used by fatlink/gauge force/hisq force code
  int volumeCB = lat.VolumeCB();
  cudaMemcpyToSymbol(Vh, &volumeCB, sizeof(int));  

  int L1 = lat.X()[0];
  cudaMemcpyToSymbol(X1, &L1, sizeof(int));  

  int L2 = lat.X()[1];
  cudaMemcpyToSymbol(X2, &L2, sizeof(int));  

  int L3 = lat.X()[2];
  cudaMemcpyToSymbol(X3, &L3, sizeof(int));  

  int L4 = lat.X()[3];
  cudaMemcpyToSymbol(X4, &L4, sizeof(int));  

  int L2L1 = L2*L1;
  cudaMemcpyToSymbol(X2X1, &L2L1, sizeof(int));  

  int L3L2 = L3*L2;
  cudaMemcpyToSymbol(X3X2, &L3L2, sizeof(int));  

  int L3L2L1 = L3*L2*L1;
  cudaMemcpyToSymbol(X3X2X1, &L3L2L1, sizeof(int));  
  
  int L1h = L1/2;
  cudaMemcpyToSymbol(X1h, &L1h, sizeof(int));  

  int L1m1 = L1 - 1;
  cudaMemcpyToSymbol(X1m1, &L1m1, sizeof(int));  

  int L2m1 = L2 - 1;
  cudaMemcpyToSymbol(X2m1, &L2m1, sizeof(int));  

  int L3m1 = L3 - 1;
  cudaMemcpyToSymbol(X3m1, &L3m1, sizeof(int));  

  int L4m1 = L4 - 1;
  cudaMemcpyToSymbol(X4m1, &L4m1, sizeof(int));  
  
  int L2L1mL1 = L2L1 - L1;
  cudaMemcpyToSymbol(X2X1mX1, &L2L1mL1, sizeof(int));  

  int L3L2L1mL2L1 = L3L2L1 - L2L1;
  cudaMemcpyToSymbol(X3X2X1mX2X1, &L3L2L1mL2L1, sizeof(int));  

  int L4L3L2L1mL3L2L1 = (L4-1)*L3L2L1;
  cudaMemcpyToSymbol(X4X3X2X1mX3X2X1, &L4L3L2L1mL3L2L1, sizeof(int));  

  int E1_h  = L1+4;
  int E1h_h = E1_h/2;
  int E2_h  = L2+4;
  int E3_h  = L3+4;
  int E4_h  = L4+4;
  int E2E1_h   = E2_h*E1_h;
  int E3E2E1_h = E3_h*E2_h*E1_h;
  int Vh_ex_h  = E1_h*E2_h*E3_h*E4_h/2;

  cudaMemcpyToSymbol(E1, &E1_h, sizeof(int));
  cudaMemcpyToSymbol(E1h, &E1h_h, sizeof(int));
  cudaMemcpyToSymbol(E2, &E2_h, sizeof(int));
  cudaMemcpyToSymbol(E3, &E3_h, sizeof(int));
  cudaMemcpyToSymbol(E4, &E4_h, sizeof(int));
  cudaMemcpyToSymbol(E2E1, &E2E1_h, sizeof(int));
  cudaMemcpyToSymbol(E3E2E1, &E3E2E1_h, sizeof(int));
  cudaMemcpyToSymbol(Vh_ex, &Vh_ex_h, sizeof(int));  

  checkCudaError();

  profile.TPSTOP(QUDA_PROFILE_CONSTANT);
}
