#define MAX(a,b) ((a) > (b)?(a): (b))
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

//for link fattening code
__constant__ int site_ga_stride;
__constant__ int staple_stride;
__constant__ int llfat_ga_stride;


int initDslash = 0;
int initClover = 0;
int initDomainWall = 0;
int initStaggered = 0;

bool qudaPt0 = true;   // Single core versions always to Boundary
bool qudaPtNm1 = true;

void initCommonConstants(const LatticeField &lat) {
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

  cudaMemcpyToSymbol("Pt0", &(qudaPt0), sizeof(bool)); 
  cudaMemcpyToSymbol("PtNm1", &(qudaPtNm1), sizeof(bool)); 

  checkCudaError();
}

void initGaugeFieldConstants(const cudaGaugeField &gauge) 
{
  initCommonConstants(gauge);

  int ga_stride = gauge.Stride();
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  int gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  double anisotropy_ = gauge.Anisotropy();
  cudaMemcpyToSymbol("anisotropy", &(anisotropy_), sizeof(double));

  double t_bc = (gauge.TBoundary() == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  double coeff = -24.0*gauge.Tadpole()*gauge.Tadpole();
  cudaMemcpyToSymbol("coeff", &(coeff), sizeof(double));

  return;
}



void initDslashConstants(const cudaGaugeField &gauge, const int sp_stride) 
{

  initCommonConstants(gauge);

  cudaMemcpyToSymbol("sp_stride", &sp_stride, sizeof(int));  
  
  int ga_stride = gauge.Stride();
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  int gf = (gauge.GaugeFixed() == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
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


  float2 An2 = make_float2(gauge.Anisotropy(), 1.0 / (gauge.Anisotropy()*gauge.Anisotropy()));
  cudaMemcpyToSymbol("An2", &(An2), sizeof(float2));
  float2 TB2 = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol("TB2", &(TB2), sizeof(float2));
  float2 No2 = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol("No2", &(No2), sizeof(float2));

  float h_pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));

  double TProjScale = (kernelPackT ? 1.0 : 2.0);
  // temporary additions (?) for checking Ron's T-packing kernel with old multi-gpu kernel
  cudaMemcpyToSymbol("tProjScale", &(TProjScale), sizeof(double));

  float TProjScale_f = (float)TProjScale;
  cudaMemcpyToSymbol("tProjScale_f", &(TProjScale_f), sizeof(float));

  checkCudaError();

  initDslash = 1;

  // create events
#ifndef DSLASH_PROFILING
  // add cudaEventDisableTiming for lower sync overhead
  for (int i=0; i<Nstream; i++) {
    cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
  }
#else
  cudaEventCreate(&dslashStart);
  cudaEventCreate(&dslashEnd);
  for (int i=0; i<Nstream; i++) {
    cudaEventCreate(&packStart[i]);
    cudaEventCreate(&packEnd[i]);

    cudaEventCreate(&gatherStart[i]);
    cudaEventCreate(&gatherEnd[i]);

    cudaEventCreate(&scatterStart[i]);
    cudaEventCreate(&scatterEnd[i]);

    cudaEventCreate(&kernelStart[i]);
    cudaEventCreate(&kernelEnd[i]);

    kernelTime[i][0] = 0.0;
    kernelTime[i][1] = 0.0;

    gatherTime[i][0] = 0.0;
    gatherTime[i][1] = 0.0;

    commsTime[i][0] = 0.0;
    commsTime[i][1] = 0.0;

    scatterTime[i][0] = 0.0;
    scatterTime[i][1] = 0.0;
  }
#endif
}

void initCloverConstants (const int cl_stride) {
  cudaMemcpyToSymbol("cl_stride", &cl_stride, sizeof(int));  

  initClover = 1;
}

void initDomainWallConstants(const int Ls) {
  cudaMemcpyToSymbol("Ls", &Ls, sizeof(int));  

  initDomainWall = 1;
}

void
initStaggeredConstants(const cudaGaugeField &fatgauge, const cudaGaugeField &longgauge)
{
  
  int fat_ga_stride = fatgauge.Stride();
  int long_ga_stride = longgauge.Stride();
  float fat_link_max = fatgauge.LinkMax();
  
  cudaMemcpyToSymbol("fat_ga_stride", &fat_ga_stride, sizeof(int));  
  cudaMemcpyToSymbol("long_ga_stride", &long_ga_stride, sizeof(int));  
  
  cudaMemcpyToSymbol("fat_ga_max", &fat_link_max, sizeof(float));
  initStaggered = 1;
  return;
}
  
