#define MAX(a,b) ((a)>(b) ? (a):(b))

typedef struct fat_force_stride_s{
  int fat_ga_stride;
  int site_ga_stride;
  int staple_stride;
  int mom_ga_stride;
  int path_max_length;
  int color_matrix_stride;
}fat_force_const_t;

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
__constant__ int sp_stride;
__constant__ int ga_stride;
__constant__ int cl_stride;
__constant__ int ghostFace[QUDA_MAX_DIM];

__constant__ int fat_ga_stride;
__constant__ int long_ga_stride;
__constant__ float fat_ga_max;

__constant__ int gauge_fixed;

// domain wall constants
__constant__ int Ls;

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


void initLatticeConstants(const LatticeField &lat)
{
  int Vh = lat.VolumeCB();
  cudaMemcpyToSymbol("Vh", &Vh, sizeof(int));  
  
  Vspatial = lat.X()[0]*lat.X()[1]*lat.X()[2]/2; // FIXME - this should not be called Vs, rather Vsh
  cudaMemcpyToSymbol("Vs", &Vspatial, sizeof(int));

  int half_Vspatial = Vspatial;
  cudaMemcpyToSymbol("Vsh", &half_Vspatial, sizeof(int));

  int X1 = lat.X()[0];
  cudaMemcpyToSymbol("X1", &X1, sizeof(int));  

  int X2 = lat.X()[1];
  cudaMemcpyToSymbol("X2", &X2, sizeof(int));  

  int X3 = lat.X()[2];
  cudaMemcpyToSymbol("X3", &X3, sizeof(int));  

  int X4 = lat.X()[3];
  cudaMemcpyToSymbol("X4", &X4, sizeof(int));  

  int ghostFace[4];
  ghostFace[0] = X2*X3*X4/2;
  ghostFace[1] = X1*X3*X4/2;
  ghostFace[2] = X1*X2*X4/2;
  ghostFace[3] = X1*X2*X3/2;
  cudaMemcpyToSymbol("ghostFace", ghostFace, 4*sizeof(int));  

  int X1_3 = 3*X1;
  cudaMemcpyToSymbol("X1_3", &X1_3, sizeof(int));  

  int X2_3 = 3*X2;
  cudaMemcpyToSymbol("X2_3", &X2_3, sizeof(int));  

  int X3_3 = 3*X3;
  cudaMemcpyToSymbol("X3_3", &X3_3, sizeof(int));  

  int X4_3 = 3*X4;
  cudaMemcpyToSymbol("X4_3", &X4_3, sizeof(int));  


  int X2X1 = X2*X1;
  cudaMemcpyToSymbol("X2X1", &X2X1, sizeof(int));  

  int X3X1 = X3*X1;
  cudaMemcpyToSymbol("X3X1", &X3X1, sizeof(int));  

  int X3X2 = X3*X2;
  cudaMemcpyToSymbol("X3X2", &X3X2, sizeof(int));  


  int X3X2X1 = X3*X2*X1;
  cudaMemcpyToSymbol("X3X2X1", &X3X2X1, sizeof(int));  
  
  int X4X2X1 = X4*X2*X1;
  cudaMemcpyToSymbol("X4X2X1", &X4X2X1, sizeof(int));  

  int X4X2X1h = X4*X2*X1/2;
  cudaMemcpyToSymbol("X4X2X1h", &X4X2X1h, sizeof(int));  

  int X4X3X1 = X4*X3*X1;
  cudaMemcpyToSymbol("X4X3X1", &X4X3X1, sizeof(int));  

  int X4X3X1h = X4*X3*X1/2;
  cudaMemcpyToSymbol("X4X3X1h", &X4X3X1h, sizeof(int));  

  int X4X3X2 = X4*X3*X2;
  cudaMemcpyToSymbol("X4X3X2", &X4X3X2, sizeof(int));  

 int X4X3X2h = X4*X3*X2/2;
  cudaMemcpyToSymbol("X4X3X2h", &X4X3X2h, sizeof(int));  

  int X2X1_3 = 3*X2*X1;
  cudaMemcpyToSymbol("X2X1_3", &X2X1_3, sizeof(int));  
  
  int X3X2X1_3 = 3*X3*X2*X1;
  cudaMemcpyToSymbol("X3X2X1_3", &X3X2X1_3, sizeof(int)); 


  int X1h = X1/2;
  cudaMemcpyToSymbol("X1h", &X1h, sizeof(int));  

  int X2h = X2/2;
  cudaMemcpyToSymbol("X2h", &X2h, sizeof(int));  

  int X1m1 = X1 - 1;
  cudaMemcpyToSymbol("X1m1", &X1m1, sizeof(int));  

  int X2m1 = X2 - 1;
  cudaMemcpyToSymbol("X2m1", &X2m1, sizeof(int));  

  int X3m1 = X3 - 1;
  cudaMemcpyToSymbol("X3m1", &X3m1, sizeof(int));  

  int X4m1 = X4 - 1;
  cudaMemcpyToSymbol("X4m1", &X4m1, sizeof(int));  
  
  int X1m3 = X1 - 3;
  cudaMemcpyToSymbol("X1m3", &X1m3, sizeof(int));  

  int X2m3 = X2 - 3;
  cudaMemcpyToSymbol("X2m3", &X2m3, sizeof(int));  

  int X3m3 = X3 - 3;
  cudaMemcpyToSymbol("X3m3", &X3m3, sizeof(int));  

  int X4m3 = X4 - 3;
  cudaMemcpyToSymbol("X4m3", &X4m3, sizeof(int));  


  int X2X1mX1 = X2X1 - X1;
  cudaMemcpyToSymbol("X2X1mX1", &X2X1mX1, sizeof(int));  

  int X3X2X1mX2X1 = X3X2X1 - X2X1;
  cudaMemcpyToSymbol("X3X2X1mX2X1", &X3X2X1mX2X1, sizeof(int));  

  int X4X3X2X1mX3X2X1 = (X4-1)*X3X2X1;
  cudaMemcpyToSymbol("X4X3X2X1mX3X2X1", &X4X3X2X1mX3X2X1, sizeof(int));  

  int X4X3X2X1hmX3X2X1h = (X4-1)*X3*X2*X1h;
  cudaMemcpyToSymbol("X4X3X2X1hmX3X2X1h", &X4X3X2X1hmX3X2X1h, sizeof(int));  

  int X2X1m3X1 = X2X1 - 3*X1;
  cudaMemcpyToSymbol("X2X1m3X1", &X2X1m3X1, sizeof(int));  

  int X3X2X1m3X2X1 = X3X2X1 - 3*X2X1;
  cudaMemcpyToSymbol("X3X2X1m3X2X1", &X3X2X1m3X2X1, sizeof(int));  

  int X4X3X2X1m3X3X2X1 = (X4-3)*X3X2X1;
  cudaMemcpyToSymbol("X4X3X2X1m3X3X2X1", &X4X3X2X1m3X3X2X1, sizeof(int));  

  int X4X3X2X1hm3X3X2X1h = (X4-3)*X3*X2*X1h;
  cudaMemcpyToSymbol("X4X3X2X1hm3X3X2X1h", &X4X3X2X1hm3X3X2X1h, sizeof(int)); 
  
  int Vh_2d_max = MAX(X1*X2/2, X1*X3/2);
  Vh_2d_max = MAX(Vh_2d_max, X1*X4/2);
  Vh_2d_max = MAX(Vh_2d_max, X2*X3/2);
  Vh_2d_max = MAX(Vh_2d_max, X2*X4/2);
  Vh_2d_max = MAX(Vh_2d_max, X3*X4/2);
  cudaMemcpyToSymbol("Vh_2d_max", &Vh_2d_max, sizeof(int));

#ifdef MULTI_GPU
  bool first_node_in_t = (commCoords(3) == 0);
  bool last_node_in_t = (commCoords(3) == commDim(3)-1);
#else
  bool first_node_in_t = true;
  bool last_node_in_t = true;
#endif

  cudaMemcpyToSymbol("Pt0", &(first_node_in_t), sizeof(bool)); 
  cudaMemcpyToSymbol("PtNm1", &(last_node_in_t), sizeof(bool)); 

  //constants used by fatlink/gauge force/hisq force code
  int E1  = X1+4;
  int E1h = E1/2;
  int E2  = X2+4;
  int E3  = X3+4;
  int E4  = X4+4;
  int E2E1   = E2*E1;
  int E3E2E1 = E3*E2*E1;
  int Vh_ex  = E1*E2*E3*E4/2;
  
  cudaMemcpyToSymbol("E1", &E1, sizeof(int));
  cudaMemcpyToSymbol("E1h", &E1h, sizeof(int));
  cudaMemcpyToSymbol("E2", &E2, sizeof(int));
  cudaMemcpyToSymbol("E3", &E3, sizeof(int));
  cudaMemcpyToSymbol("E4", &E4, sizeof(int));
  cudaMemcpyToSymbol("E2E1", &E2E1, sizeof(int));
  cudaMemcpyToSymbol("E3E2E1", &E3E2E1, sizeof(int));
  cudaMemcpyToSymbol("Vh_ex", &Vh_ex, sizeof(int));  

  // copy a few of the constants needed by tuneLaunch()
  dslashConstants.x[0] = X1;
  dslashConstants.x[1] = X2;
  dslashConstants.x[2] = X3;
  dslashConstants.x[3] = X4;

  checkCudaError();
}


void initGaugeConstants(const cudaGaugeField &gauge) 
{
  int ga_stride = gauge.Stride();
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  int gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES);
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  double anisotropy_ = gauge.Anisotropy();
  cudaMemcpyToSymbol("anisotropy", &(anisotropy_), sizeof(double));

  double t_bc = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  double coeff = -24.0*gauge.Tadpole()*gauge.Tadpole();
  cudaMemcpyToSymbol("coeff", &(coeff), sizeof(double));

  float anisotropy_f = gauge.Anisotropy();
  cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));

  float t_bc_f = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

  float coeff_f = -24.0*gauge.Tadpole()*gauge.Tadpole();
  cudaMemcpyToSymbol("coeff_f", &(coeff_f), sizeof(float));

  // constants used by the READ_GAUGE() macros in read_gauge.h
  float2 An2 = make_float2(gauge.Anisotropy(), 1.0 / (gauge.Anisotropy()*gauge.Anisotropy()));
  cudaMemcpyToSymbol("An2", &(An2), sizeof(float2));
  float2 TB2 = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol("TB2", &(TB2), sizeof(float2));
  float2 No2 = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol("No2", &(No2), sizeof(float2));

  checkCudaError();
}


/**
 * This routine gets called often, so be sure not to set any constants unnecessarily
 * or introduce synchronization (e.g., via checkCudaError()).
 */
void initSpinorConstants(const cudaColorSpinorField &spinor)
{
  static int last_sp_stride = -1;
  static int last_Ls = -1;

  int sp_stride = spinor.Stride();
  if (sp_stride != last_sp_stride) {
    cudaMemcpyToSymbol("sp_stride", &sp_stride, sizeof(int));
    checkCudaError();
  }
  
  // for domain wall:
  if (spinor.Ndim() == 5) {
    int Ls = spinor.X(4);
    if (Ls != last_Ls) {
      cudaMemcpyToSymbol("Ls", &Ls, sizeof(int));  
      dslashConstants.Ls = Ls; // needed by tuneLaunch()
      checkCudaError();
    }
  }
}


void initDslashConstants()
{
  float pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &pi_f, sizeof(float));

  // temporary additions (?) for checking Ron's T-packing kernel with old multi-gpu kernel

  double tProjScale = (kernelPackT ? 1.0 : 2.0);
  cudaMemcpyToSymbol("tProjScale", &tProjScale, sizeof(double));

  float tProjScale_f = (float)tProjScale;
  cudaMemcpyToSymbol("tProjScale_f", &tProjScale_f, sizeof(float));

  checkCudaError();
}


void initCloverConstants (const cudaCloverField &clover)
{
  int cl_stride = clover.Stride();
  cudaMemcpyToSymbol("cl_stride", &cl_stride, sizeof(int));  

  checkCudaError();
}


void initStaggeredConstants(const cudaGaugeField &fatgauge, const cudaGaugeField &longgauge)
{
  int fat_ga_stride = fatgauge.Stride();
  int long_ga_stride = longgauge.Stride();
  float fat_link_max = fatgauge.LinkMax();
  
  cudaMemcpyToSymbol("fat_ga_stride", &fat_ga_stride, sizeof(int));  
  cudaMemcpyToSymbol("long_ga_stride", &long_ga_stride, sizeof(int));  
  cudaMemcpyToSymbol("fat_ga_max", &fat_link_max, sizeof(float));

  checkCudaError();
}
