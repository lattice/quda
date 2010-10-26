__constant__ int X1h;
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
__constant__ int X3X2X1;

__constant__ int X2X1_3;
__constant__ int X3X2X1_3;

__constant__ int Vh;
__constant__ int sp_stride;
__constant__ int ga_stride;
__constant__ int cl_stride;

__constant__ int fat_ga_stride;
__constant__ int long_ga_stride;

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

int initDslash = 0;

void initCommonConstants(FullGauge gauge) {
  int Vh = gauge.volume;
  cudaMemcpyToSymbol("Vh", &Vh, sizeof(int));  
  
  if (Vh%BLOCK_DIM != 0) {
    errorQuda("Error, Volume not a multiple of the thread block size");
  }

  int X1 = 2*gauge.X[0];
  cudaMemcpyToSymbol("X1", &X1, sizeof(int));  

  int X2 = gauge.X[1];
  cudaMemcpyToSymbol("X2", &X2, sizeof(int));  

  int X3 = gauge.X[2];
  cudaMemcpyToSymbol("X3", &X3, sizeof(int));  

  int X4 = gauge.X[3];
  cudaMemcpyToSymbol("X4", &X4, sizeof(int));  


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

  int X3X2X1 = X3*X2*X1;
  cudaMemcpyToSymbol("X3X2X1", &X3X2X1, sizeof(int));  
  
  int X2X1_3 = 3*X2*X1;
  cudaMemcpyToSymbol("X2X1_3", &X2X1_3, sizeof(int));  
  
  int X3X2X1_3 = 3*X3*X2*X1;
  cudaMemcpyToSymbol("X3X2X1_3", &X3X2X1_3, sizeof(int)); 


  int X1h = X1/2;
  cudaMemcpyToSymbol("X1h", &X1h, sizeof(int));  

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

  checkCudaError();
}


void initDslashConstants(FullGauge gauge, int sp_stride, int cl_stride, int Ls) 
{

  initCommonConstants(gauge);

  cudaMemcpyToSymbol("sp_stride", &sp_stride, sizeof(int));  
  
  int ga_stride = gauge.stride;
  cudaMemcpyToSymbol("ga_stride", &ga_stride, sizeof(int));  

  int fat_ga_stride = gauge.stride;
  int long_ga_stride = gauge.stride;
    
  cudaMemcpyToSymbol("fat_ga_stride", &fat_ga_stride, sizeof(int));
  cudaMemcpyToSymbol("long_ga_stride", &long_ga_stride, sizeof(int));



  cudaMemcpyToSymbol("cl_stride", &cl_stride, sizeof(int));  

  int gf = (gauge.gauge_fixed == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  cudaMemcpyToSymbol("anisotropy", &(gauge.anisotropy), sizeof(double));

  double t_bc = (gauge.t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  double coeff = -24.0*gauge.tadpole_coeff*gauge.tadpole_coeff;
  cudaMemcpyToSymbol("coeff", &(coeff), sizeof(double));


  float anisotropy_f = gauge.anisotropy;
  cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));

  float t_bc_f = (gauge.t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

  float coeff_f = -24.0*gauge.tadpole_coeff*gauge.tadpole_coeff;
  cudaMemcpyToSymbol("coeff_f", &(coeff_f), sizeof(float));


  float2 An2 = make_float2(gauge.anisotropy, 1.0 / (gauge.anisotropy*gauge.anisotropy));
  cudaMemcpyToSymbol("An2", &(An2), sizeof(float2));
  float2 TB2 = make_float2(t_bc_f, 1.0 / (t_bc_f * t_bc_f));
  cudaMemcpyToSymbol("TB2", &(TB2), sizeof(float2));
  float2 No2 = make_float2(1.0, 1.0);
  cudaMemcpyToSymbol("No2", &(No2), sizeof(float2));

  float h_pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));

  cudaMemcpyToSymbol("Ls", &Ls, sizeof(int));  

  checkCudaError();

  initDslash = 1;
}

