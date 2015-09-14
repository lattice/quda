enum KernelType {
  INTERIOR_KERNEL = 5,
  EXTERIOR_KERNEL_ALL = 6,
  EXTERIOR_KERNEL_X = 0,
  EXTERIOR_KERNEL_Y = 1,
  EXTERIOR_KERNEL_Z = 2,
  EXTERIOR_KERNEL_T = 3
};

  struct DslashParam {
#ifndef STAGGERED_TESLA_HACK
    char do_not_delete; // work around for bug in CUDA 6.5
#endif
    int threads; // the desired number of active threads
    int parity;  // Even-Odd or Odd-Even
    int X[4];
#ifdef GPU_DOMAIN_WALL_DIRAC 
    int Ls;
#endif
    KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T
#ifndef STAGGERED_TESLA_HACK
    int commDim[4]; // Whether to do comms or not
#endif
    int ghostDim[4]; // Whether a ghost zone has been allocated for a given dimension
    int ghostOffset[QUDA_MAX_DIM];
    int ghostNormOffset[QUDA_MAX_DIM];
    int sp_stride; // spinor stride
#ifdef GPU_CLOVER_DIRAC
    int cl_stride; // clover stride
#endif
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
    int fl_stride; // twisted-mass flavor stride
#endif
#ifdef GPU_STAGGERED_DIRAC
    int gauge_stride;
    int long_gauge_stride;
    float fat_link_max;
#endif 
#ifdef MULTI_GPU
    int threadDimMapLower[4];
    int threadDimMapUpper[4];
#endif

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t inTex;
    cudaTextureObject_t inTexNorm;
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

    void print() {
      printfQuda("threads = %d\n", threads);
      printfQuda("parity = %d\n", parity);
      printfQuda("X = {%d, %d, %d, %d}\n", X[0], X[1], X[2], X[3]);
#ifdef GPU_DOMAIN_WALL_DIRAC
      printfQuda("Ls = %d\n", Ls);
#endif
      printfQuda("commDim = {%d, %d, %d, %d}\n", commDim[0], commDim[1], commDim[2], commDim[3]);
      printfQuda("ghostDim = {%d, %d, %d, %d}\n", ghostDim[0], ghostDim[1], ghostDim[2], ghostDim[3]);
      printfQuda("ghostOffset = {%d, %d, %d, %d}\n", ghostOffset[0], ghostOffset[1], ghostOffset[2], ghostOffset[3]);
      printfQuda("ghostNormOffset = {%d, %d, %d, %d}\n", ghostNormOffset[0], ghostNormOffset[1], ghostNormOffset[2], ghostNormOffset[3]);
      printfQuda("kernel_type = %d\n", kernel_type);
      printfQuda("sp_stride = %d\n", sp_stride);
#ifdef GPU_CLOVER_DIRAC
      printfQuda("cl_stride = %d\n", cl_stride);
#endif
    }
  };

  static DslashParam dslashParam;



#ifdef MULTI_GPU
  static double twist_a = 0.0;
  static double twist_b = 0.0;
#endif


#define MAX(a,b) ((a)>(b) ? (a):(b))

typedef struct fat_force_stride_s {
  int fat_ga_stride;
  int long_ga_stride;
  int site_ga_stride;
  int staple_stride;
  int mom_ga_stride;
  int path_max_length;
  int color_matrix_stride;
} fat_force_const_t;

__constant__ int X1h;
__constant__ int X2h;
__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;

__constant__ int X1_3;
__constant__ int X2_3;
__constant__ int X3_3;
__constant__ int X4_3;

__constant__ int X1m1;
__constant__ int X2m1;
__constant__ int X3m1;
__constant__ int X4m1;

__constant__ int X1m3;
__constant__ int X2m3;
__constant__ int X3m3;
__constant__ int X4m3;

__constant__ int X2X1mX1;
__constant__ int X3X2X1mX2X1;
__constant__ int X4X3X2X1mX3X2X1;
__constant__ int X4X3X2X1hmX3X2X1h;

__constant__ int X2X1m3X1;
__constant__ int X3X2X1m3X2X1;
__constant__ int X4X3X2X1m3X3X2X1;
__constant__ int X4X3X2X1hm3X3X2X1h;

__constant__ int X2X1;
__constant__ int X3X1;
__constant__ int X3X2;
__constant__ int X3X2X1;
__constant__ int X4X2X1;
__constant__ int X4X2X1h;
__constant__ int X4X3X1;
__constant__ int X4X3X1h;
__constant__ int X4X3X2;
__constant__ int X4X3X2h;

__constant__ int Vh_2d_max;

__constant__ int X2X1_3;
__constant__ int X3X2X1_3;

__constant__ int Vh;
__constant__ int Vs;
__constant__ int Vsh;
//__constant__ int sp_stride;
__constant__ int ga_stride;
//__constant__ int cl_stride;
__constant__ int ghostFace[QUDA_MAX_DIM+1];

__constant__ int fat_ga_stride;
__constant__ int long_ga_stride;
__constant__ float fat_ga_max;

__constant__ int gauge_fixed;

// domain wall constants
//__constant__ int Ls;
__constant__ double m5_d;
__constant__ float m5_f;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float coeff_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;
__constant__ double coeff;

__constant__ float2 An2;
__constant__ float2 TB2;
__constant__ float2 No2;

// Are we processor 0 in time?
__constant__ bool Pt0;

// Are we processor Nt-1 in time?
__constant__ bool PtNm1;

// factor of 2 (or 1) for T-dimensional spin projection
__constant__ double tProjScale;
__constant__ float tProjScale_f;

//for link fattening/gauge force/fermion force code
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

  int volumeCB = lat.VolumeCB();
  cudaMemcpyToSymbol(Vh, &volumeCB, sizeof(int));  

  int Vspatial = lat.X()[0]*lat.X()[1]*lat.X()[2]/2; // FIXME - this should not be called Vs, rather Vsh
  cudaMemcpyToSymbol(Vs, &Vspatial, sizeof(int));

  int half_Vspatial = Vspatial;
  cudaMemcpyToSymbol(Vsh, &half_Vspatial, sizeof(int));

  int L1 = lat.X()[0];
  cudaMemcpyToSymbol(X1, &L1, sizeof(int));  

  int L2 = lat.X()[1];
  cudaMemcpyToSymbol(X2, &L2, sizeof(int));  

  int L3 = lat.X()[2];
  cudaMemcpyToSymbol(X3, &L3, sizeof(int));  

  int L4 = lat.X()[3];
  cudaMemcpyToSymbol(X4, &L4, sizeof(int));  

  int ghostFace_h[4];
  ghostFace_h[0] = L2*L3*L4/2;
  ghostFace_h[1] = L1*L3*L4/2;
  ghostFace_h[2] = L1*L2*L4/2;
  ghostFace_h[3] = L1*L2*L3/2;
  cudaMemcpyToSymbol(ghostFace, ghostFace_h, 4*sizeof(int));  

  int L1_3 = 3*L1;
  cudaMemcpyToSymbol(X1_3, &L1_3, sizeof(int));  

  int L2_3 = 3*L2;
  cudaMemcpyToSymbol(X2_3, &L2_3, sizeof(int));  

  int L3_3 = 3*L3;
  cudaMemcpyToSymbol(X3_3, &L3_3, sizeof(int));  

  int L4_3 = 3*L4;
  cudaMemcpyToSymbol(X4_3, &L4_3, sizeof(int));  

  int L2L1 = L2*L1;
  cudaMemcpyToSymbol(X2X1, &L2L1, sizeof(int));  

  int L3L1 = L3*L1;
  cudaMemcpyToSymbol(X3X1, &L3L1, sizeof(int));  

  int L3L2 = L3*L2;
  cudaMemcpyToSymbol(X3X2, &L3L2, sizeof(int));  

  int L3L2L1 = L3*L2*L1;
  cudaMemcpyToSymbol(X3X2X1, &L3L2L1, sizeof(int));  
  
  int L4L2L1 = L4*L2*L1;
  cudaMemcpyToSymbol(X4X2X1, &L4L2L1, sizeof(int));  

  int L4L2L1h = L4*L2*L1/2;
  cudaMemcpyToSymbol(X4X2X1h, &L4L2L1h, sizeof(int));  

  int L4L3L1 = L4*L3*L1;
  cudaMemcpyToSymbol(X4X3X1, &L4L3L1, sizeof(int));  

  int L4L3L1h = L4*L3*L1/2;
  cudaMemcpyToSymbol(X4X3X1h, &L4L3L1h, sizeof(int));  

  int L4L3L2 = L4*L3*L2;
  cudaMemcpyToSymbol(X4X3X2, &L4L3L2, sizeof(int));  

  int L4L3L2h = L4*L3*L2/2;
  cudaMemcpyToSymbol(X4X3X2h, &L4L3L2h, sizeof(int));  

  int L2L1_3 = 3*L2*L1;
  cudaMemcpyToSymbol(X2X1_3, &L2L1_3, sizeof(int));  
  
  int L3L2L1_3 = 3*L3*L2*L1;
  cudaMemcpyToSymbol(X3X2X1_3, &L3L2L1_3, sizeof(int)); 

  int L1h = L1/2;
  cudaMemcpyToSymbol(X1h, &L1h, sizeof(int));  

  int L2h = L2/2;
  cudaMemcpyToSymbol(X2h, &L2h, sizeof(int));  

  int L1m1 = L1 - 1;
  cudaMemcpyToSymbol(X1m1, &L1m1, sizeof(int));  

  int L2m1 = L2 - 1;
  cudaMemcpyToSymbol(X2m1, &L2m1, sizeof(int));  

  int L3m1 = L3 - 1;
  cudaMemcpyToSymbol(X3m1, &L3m1, sizeof(int));  

  int L4m1 = L4 - 1;
  cudaMemcpyToSymbol(X4m1, &L4m1, sizeof(int));  
  
  int L1m3 = L1 - 3;
  cudaMemcpyToSymbol(X1m3, &L1m3, sizeof(int));  

  int L2m3 = L2 - 3;
  cudaMemcpyToSymbol(X2m3, &L2m3, sizeof(int));  

  int L3m3 = L3 - 3;
  cudaMemcpyToSymbol(X3m3, &L3m3, sizeof(int));  

  int L4m3 = L4 - 3;
  cudaMemcpyToSymbol(X4m3, &L4m3, sizeof(int));  

  int L2L1mL1 = L2L1 - L1;
  cudaMemcpyToSymbol(X2X1mX1, &L2L1mL1, sizeof(int));  

  int L3L2L1mL2L1 = L3L2L1 - L2L1;
  cudaMemcpyToSymbol(X3X2X1mX2X1, &L3L2L1mL2L1, sizeof(int));  

  int L4L3L2L1mL3L2L1 = (L4-1)*L3L2L1;
  cudaMemcpyToSymbol(X4X3X2X1mX3X2X1, &L4L3L2L1mL3L2L1, sizeof(int));  

  int L4L3L2L1hmL3L2L1h = (L4-1)*L3*L2*L1h;
  cudaMemcpyToSymbol(X4X3X2X1hmX3X2X1h, &L4L3L2L1hmL3L2L1h, sizeof(int));  

  int L2L1m3L1 = L2L1 - 3*L1;
  cudaMemcpyToSymbol(X2X1m3X1, &L2L1m3L1, sizeof(int));  

  int L3L2L1m3L2L1 = L3L2L1 - 3*L2L1;
  cudaMemcpyToSymbol(X3X2X1m3X2X1, &L3L2L1m3L2L1, sizeof(int));  

  int L4L3L2L1m3L3L2L1 = (L4-3)*L3L2L1;
  cudaMemcpyToSymbol(X4X3X2X1m3X3X2X1, &L4L3L2L1m3L3L2L1, sizeof(int));  

  int L4L3L2L1hm3L3L2L1h = (L4-3)*L3*L2*L1h;
  cudaMemcpyToSymbol(X4X3X2X1hm3X3X2X1h, &L4L3L2L1hm3L3L2L1h, sizeof(int)); 
  int Vh_2d_max_h = MAX(L1*L2/2, L1*L3/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L1*L4/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L2*L3/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L2*L4/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L3*L4/2);
  cudaMemcpyToSymbol(Vh_2d_max, &Vh_2d_max_h, sizeof(int));

#ifdef MULTI_GPU
  bool first_node_in_t = (commCoords(3) == 0);
  bool last_node_in_t = (commCoords(3) == commDim(3)-1);
#else
  bool first_node_in_t = true;
  bool last_node_in_t = true;
#endif

  cudaMemcpyToSymbol(Pt0, &(first_node_in_t), sizeof(bool)); 
  cudaMemcpyToSymbol(PtNm1, &(last_node_in_t), sizeof(bool)); 

  //constants used by fatlink/gauge force/hisq force code
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


void initGaugeConstants(const cudaGaugeField &gauge, TimeProfile &profile) 
{
  profile.TPSTART(QUDA_PROFILE_CONSTANT);

  int ga_stride_h = gauge.Stride();
  cudaMemcpyToSymbol(ga_stride, &ga_stride_h, sizeof(int));  

  // set fat link stride and max (used by naive staggered)
  cudaMemcpyToSymbol(fat_ga_stride, &ga_stride_h, sizeof(int)); 
  float link_max_h = gauge.LinkMax();
  cudaMemcpyToSymbol(fat_ga_max, &link_max_h, sizeof(float));

  int gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES);
  cudaMemcpyToSymbol(gauge_fixed, &(gf), sizeof(int));

  double anisotropy_ = gauge.Anisotropy();
  cudaMemcpyToSymbol(anisotropy, &(anisotropy_), sizeof(double));

  double t_bc = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol(t_boundary, &(t_bc), sizeof(double));

  float anisotropy_fh = gauge.Anisotropy();
  cudaMemcpyToSymbol(anisotropy_f, &(anisotropy_fh), sizeof(float));

  float t_bc_f = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol(t_boundary_f, &(t_bc_f), sizeof(float));


  // constants used by the READ_GAUGE() macros in read_gauge.h
  float2 An2_h = make_float2(gauge.Anisotropy(), 1.0 / (gauge.Anisotropy()*gauge.Anisotropy()));
  cudaMemcpyToSymbol(An2, &(An2_h), sizeof(float2));
  float2 TB2_h = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol(TB2, &(TB2_h), sizeof(float2));
  float2 No2_h = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol(No2, &(No2_h), sizeof(float2));

  checkCudaError();

  profile.TPSTOP(QUDA_PROFILE_CONSTANT);
}

void initDslashConstants(TimeProfile &profile)
{
  profile.TPSTART(QUDA_PROFILE_CONSTANT);

  float pi_f_h = M_PI;
  cudaMemcpyToSymbol(pi_f, &pi_f_h, sizeof(float));

  // temporary additions (?) for checking Ron's T-packing kernel with old multi-gpu kernel

  double tProjScale_h = (getKernelPackT() ? 1.0 : 2.0);
  cudaMemcpyToSymbol(tProjScale, &tProjScale_h, sizeof(double));

  float tProjScale_fh = (float)tProjScale_h;
  cudaMemcpyToSymbol(tProjScale_f, &tProjScale_fh, sizeof(float));

  // set these for naive staggered
  float coeff_fh = 1.0;
  cudaMemcpyToSymbol(coeff_f, &(coeff_fh), sizeof(float));

  double coeff_h = 1.0;
  cudaMemcpyToSymbol(coeff, &(coeff_h), sizeof(double));
  
  checkCudaError();

  profile.TPSTOP(QUDA_PROFILE_CONSTANT);
}

void initStaggeredConstants(const cudaGaugeField &fatgauge, const cudaGaugeField &longgauge,
			    TimeProfile &profile)
{
  profile.TPSTART(QUDA_PROFILE_CONSTANT);

  int fat_ga_stride_h = fatgauge.Stride();
  int long_ga_stride_h = longgauge.Stride();
  float fat_link_max_h = fatgauge.LinkMax();
  
  float coeff_fh = 1.0/longgauge.Scale();
  cudaMemcpyToSymbol(coeff_f, &(coeff_fh), sizeof(float));

  double coeff_h = 1.0/longgauge.Scale();
  cudaMemcpyToSymbol(coeff, &(coeff_h), sizeof(double));

  cudaMemcpyToSymbol(fat_ga_stride, &fat_ga_stride_h, sizeof(int));  
  cudaMemcpyToSymbol(long_ga_stride, &long_ga_stride_h, sizeof(int));  
  cudaMemcpyToSymbol(fat_ga_max, &fat_link_max_h, sizeof(float));

  checkCudaError();

  profile.TPSTOP(QUDA_PROFILE_CONSTANT);
}

//For initializing the coefficients used in MDWF
__constant__ double mdwf_b5_d[QUDA_MAX_DWF_LS];
__constant__ double mdwf_c5_d[QUDA_MAX_DWF_LS];

__constant__ float mdwf_b5_f[QUDA_MAX_DWF_LS];
__constant__ float mdwf_c5_f[QUDA_MAX_DWF_LS];

void initMDWFConstants(const double *b_5, const double *c_5, int dim_s, const double m5h, TimeProfile &profile)
{
  profile.TPSTART(QUDA_PROFILE_CONSTANT);

  static int last_Ls = -1;
  if (dim_s != last_Ls) {
    float b_5_f[QUDA_MAX_DWF_LS];
    float c_5_f[QUDA_MAX_DWF_LS];
    for (int i=0; i<dim_s; i++) {
      b_5_f[i] = (float)b_5[i];
      c_5_f[i] = (float)c_5[i];
    }

    cudaMemcpyToSymbol(mdwf_b5_d, b_5, dim_s*sizeof(double));
    cudaMemcpyToSymbol(mdwf_c5_d, c_5, dim_s*sizeof(double));
    cudaMemcpyToSymbol(mdwf_b5_f, b_5_f, dim_s*sizeof(float));
    cudaMemcpyToSymbol(mdwf_c5_f, c_5_f, dim_s*sizeof(float));
    checkCudaError();
    last_Ls = dim_s;
  }

  static double last_m5 = 99999;
  if (m5h != last_m5) {
    float m5h_f = (float)m5h;
    cudaMemcpyToSymbol(m5_d, &m5h, sizeof(double));
    cudaMemcpyToSymbol(m5_f, &m5h_f, sizeof(float));
    checkCudaError();
    last_m5 = m5h;
  }

  profile.TPSTOP(QUDA_PROFILE_CONSTANT);
}

void setTwistParam(double &a, double &b, const double &kappa, const double &mu, 
		   const int dagger, const QudaTwistGamma5Type twist) {
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu;
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    a = -2.0 * kappa * mu;
    b = 1.0 / (1.0 + a*a);
  } else {
    errorQuda("Twist type %d not defined\n", twist);
  }
  if (dagger) a *= -1.0;

}
