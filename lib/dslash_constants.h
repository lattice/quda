#include <unistd.h>
#include <fast_intdiv.h>
#include <convert.h>

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

    bool spin_project; // If using covDev, turn off spin projection.

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
    double twist_c;

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
      printfQuda("spin_project = %s\n", spin_project ? "true" : "false");
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
      printfQuda("twist_c = %e\n", twist_c);
    }
  };
