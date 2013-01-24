#ifndef DIMENSION_CONSTANTS_H
#define DIMENSION_CONSTANTS_H

#define MAX(a,b) ((a)>(b) ? (a):(b))

struct LatticeConstants {
  int X1h;
  int X2h;
  int X1;
  int X2;
  int X3;
  int X4;

  int X1_3;
  int X2_3;
  int X3_3;
  int X4_3;

  int X1m1;
  int X2m1;
  int X3m1;
  int X4m1;

  int X1m3;
  int X2m3;
  int X3m3;
  int X4m3;

  int X2X1mX1;
  int X3X2X1mX2X1;
  int X4X3X2X1mX3X2X1;
  int X4X3X2X1hmX3X2X1h;
  
  int X2X1m3X1;
  int X3X2X1m3X2X1;
  int X4X3X2X1m3X3X2X1;
  int X4X3X2X1hm3X3X2X1h;

  int X2X1;
  int X3X1;
  int X3X2;
  int X3X2X1;
  int X4X2X1;
  int X4X2X1h;
  int X4X3X1;
  int X4X3X1h;
  int X4X3X2;
  int X4X3X2h;
  
  int X2X1_3;
  int X3X2X1_3;
  int X3X3X1_3;

  int sp_stride;
  int ghostFace[QUDA_MAX_DIM];

  int Vh;
  int Vs; // spatial volume
  int Vsh; // half spatial volume
};


__constant__ LatticeConstants  cudaConstants;
LatticeConstants constants;



__constant__ int Y1h;
__constant__ int Y1;
__constant__ int Y2;
__constant__ int Y3;
__constant__ int Y4;

__constant__ int Y3Y2Y1;
__constant__ int Y2Y1;

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

// domain wall constants
__constant__ int Ls;

//for link fattening/gauge force/fermion force code
__constant__ int E1, E2, E3, E4, E1h;
__constant__ int Vh_ex;
__constant__ int E2E1;
__constant__ int E3E2E1;




void initDimensionConstants(const LatticeField &lat)
{
  checkCudaError();

  int volumeCB = lat.VolumeCB();
  cudaMemcpyToSymbol(Vh, &volumeCB, sizeof(int));  

  constants.Vh = volumeCB;

  Vspatial = lat.X()[0]*lat.X()[1]*lat.X()[2]/2; // FIXME - this should not be called Vs, rather Vsh
  cudaMemcpyToSymbol(Vs, &Vspatial, sizeof(int));
  constants.Vs = Vspatial;

  int half_Vspatial = Vspatial;
  cudaMemcpyToSymbol(Vsh, &half_Vspatial, sizeof(int));
  constants.Vsh = half_Vspatial;


  int L1 = lat.X()[0];
  cudaMemcpyToSymbol(X1, &L1, sizeof(int));  
  constants.X1 = L1;

  int L2 = lat.X()[1];
  cudaMemcpyToSymbol(X2, &L2, sizeof(int));  
  constants.X2 = L2;

  int L3 = lat.X()[2];
  cudaMemcpyToSymbol(X3, &L3, sizeof(int));  
  constants.X3 = L3;
 
  int L4 = lat.X()[3];
  cudaMemcpyToSymbol(X4, &L4, sizeof(int));  
  constants.X4 = L4;

//  int ghostFace_h[4];
//  ghostFace_h[0] = L2*L3*L4/2;
//  ghostFace_h[1] = L1*L3*L4/2;
//  ghostFace_h[2] = L1*L2*L4/2;
//  ghostFace_h[3] = L1*L2*L3/2;
//  constants.ghostFace[0] = ghostFace_h[0];
//  constants.ghostFace[1] = ghostFace_h[1];
//  constants.ghostFace[2] = ghostFace_h[2];
//  constants.ghostFace[3] = ghostFace_h[3];
//  cudaMemcpyToSymbol(ghostFace, ghostFace_h, 4*sizeof(int));  

  int L1_3 = 3*L1;
  cudaMemcpyToSymbol(X1_3, &L1_3, sizeof(int)); 
  constants.X1_3 = L1_3; 

  int L2_3 = 3*L2;
  cudaMemcpyToSymbol(X2_3, &L2_3, sizeof(int));  
  constants.X2_3 = L2_3;

  int L3_3 = 3*L3;
  cudaMemcpyToSymbol(X3_3, &L3_3, sizeof(int));  
  constants.X3_3 = L3_3;

  int L4_3 = 3*L4;
  cudaMemcpyToSymbol(X4_3, &L4_3, sizeof(int));  
  constants.X4_3 = L4_3;

  int L2L1 = L2*L1;
  cudaMemcpyToSymbol(X2X1, &L2L1, sizeof(int));  
  constants.X2X1 = L2L1;

  int L3L1 = L3*L1;
  cudaMemcpyToSymbol(X3X1, &L3L1, sizeof(int));  
  constants.X3X1 = L3L1;

  int L3L2 = L3*L2;
  cudaMemcpyToSymbol(X3X2, &L3L2, sizeof(int));  
  constants.X3X2 = L3L2;

  int L3L2L1 = L3*L2*L1;
  cudaMemcpyToSymbol(X3X2X1, &L3L2L1, sizeof(int));  
  constants.X3X2X1 = L3L2L1;
  
  int L4L2L1 = L4*L2*L1;
  cudaMemcpyToSymbol(X4X2X1, &L4L2L1, sizeof(int));  
  constants.X4X2X1 = L4L2L1;

  int L4L2L1h = L4*L2*L1/2;
  cudaMemcpyToSymbol(X4X2X1h, &L4L2L1h, sizeof(int));  
  constants.X4X2X1h = L4L2L1h;

  int L4L3L1 = L4*L3*L1;
  cudaMemcpyToSymbol(X4X3X1, &L4L3L1, sizeof(int));  
  constants.X4X3X1 = L4L3L1;

  int L4L3L1h = L4*L3*L1/2;
  cudaMemcpyToSymbol(X4X3X1h, &L4L3L1h, sizeof(int));  
  constants.X4X3X1h = L4L3L1h;

  int L4L3L2 = L4*L3*L2;
  cudaMemcpyToSymbol(X4X3X2, &L4L3L2, sizeof(int));  
  constants.X4X3X2 = L4L3L2;

  int L4L3L2h = L4*L3*L2/2;
  cudaMemcpyToSymbol(X4X3X2h, &L4L3L2h, sizeof(int));  
  constants.X4X3X2h = L4L3L2h;

  int L2L1_3 = 3*L2*L1;
  cudaMemcpyToSymbol(X2X1_3, &L2L1_3, sizeof(int));  
  constants.X2X1_3 = L2L1_3;
  
  int L3L2L1_3 = 3*L3*L2*L1;
  cudaMemcpyToSymbol(X3X2X1_3, &L3L2L1_3, sizeof(int)); 
  constants.X3X2X1_3 = L3L2L1_3;

  int L1h = L1/2;
  cudaMemcpyToSymbol(X1h, &L1h, sizeof(int));  
  constants.X1h = L1h;

  int L2h = L2/2;
  cudaMemcpyToSymbol(X2h, &L2h, sizeof(int));  
  constants.X2h = L2h;

  int L1m1 = L1 - 1;
  cudaMemcpyToSymbol(X1m1, &L1m1, sizeof(int));  
  constants.X1m1 = L1m1;

  int L2m1 = L2 - 1;
  cudaMemcpyToSymbol(X2m1, &L2m1, sizeof(int));  
  constants.X2m1 = L2m1;

  int L3m1 = L3 - 1;
  cudaMemcpyToSymbol(X3m1, &L3m1, sizeof(int));  
  constants.X3m1 = L3m1;

  int L4m1 = L4 - 1;
  cudaMemcpyToSymbol(X4m1, &L4m1, sizeof(int));  
  constants.X4m1 = L4m1;
  
  int L1m3 = L1 - 3;
  cudaMemcpyToSymbol(X1m3, &L1m3, sizeof(int));  
  constants.X1m3 = L1m3;

  int L2m3 = L2 - 3;
  cudaMemcpyToSymbol(X2m3, &L2m3, sizeof(int));  
  constants.X2m3 = L2m3;

  int L3m3 = L3 - 3;
  cudaMemcpyToSymbol(X3m3, &L3m3, sizeof(int));  
  constants.X3m3 = L3m3;

  int L4m3 = L4 - 3;
  cudaMemcpyToSymbol(X4m3, &L4m3, sizeof(int));  
  constants.X4m3 = L4m3;

  int L2L1mL1 = L2L1 - L1;
  cudaMemcpyToSymbol(X2X1mX1, &L2L1mL1, sizeof(int));  
  constants.X2X1mX1 = L2L1mL1;  

  int L3L2L1mL2L1 = L3L2L1 - L2L1;
  cudaMemcpyToSymbol(X3X2X1mX2X1, &L3L2L1mL2L1, sizeof(int));  
  constants.X3X2X1mX2X1 = L3L2L1mL2L1;

  int L4L3L2L1mL3L2L1 = (L4-1)*L3L2L1;
  cudaMemcpyToSymbol(X4X3X2X1mX3X2X1, &L4L3L2L1mL3L2L1, sizeof(int));  
  constants.X4X3X2X1mX3X2X1 = L4L3L2L1mL3L2L1;

  int L4L3L2L1hmL3L2L1h = (L4-1)*L3*L2*L1h;
  cudaMemcpyToSymbol(X4X3X2X1hmX3X2X1h, &L4L3L2L1hmL3L2L1h, sizeof(int));  
  constants.X4X3X2X1hmX3X2X1h = L4L3L2L1hmL3L2L1h;

  int L2L1m3L1 = L2L1 - 3*L1;
  cudaMemcpyToSymbol(X2X1m3X1, &L2L1m3L1, sizeof(int));  
  constants.X2X1m3X1 = L2L1m3L1;

  int L3L2L1m3L2L1 = L3L2L1 - 3*L2L1;
  cudaMemcpyToSymbol(X3X2X1m3X2X1, &L3L2L1m3L2L1, sizeof(int));  
  constants.X3X2X1m3X2X1 = L3L2L1m3L2L1;

  int L4L3L2L1m3L3L2L1 = (L4-3)*L3L2L1;
  cudaMemcpyToSymbol(X4X3X2X1m3X3X2X1, &L4L3L2L1m3L3L2L1, sizeof(int));  
  constants.X4X3X2X1m3X3X2X1 = L4L3L2L1m3L3L2L1;  

  int L4L3L2L1hm3L3L2L1h = (L4-3)*L3*L2*L1h;
  cudaMemcpyToSymbol(X4X3X2X1hm3X3X2X1h, &L4L3L2L1hm3L3L2L1h, sizeof(int)); 
  constants.X4X3X2X1hm3X3X2X1h = L4L3L2L1hm3L3L2L1h;
 
  int Vh_2d_max_h = MAX(L1*L2/2, L1*L3/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L1*L4/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L2*L3/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L2*L4/2);
  Vh_2d_max_h = MAX(Vh_2d_max_h, L3*L4/2);
  cudaMemcpyToSymbol(Vh_2d_max, &Vh_2d_max_h, sizeof(int));

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

  // copy a few of the constants needed by tuneLaunch()
  dslashConstants.x[0] = L1;
  dslashConstants.x[1] = L2;
  dslashConstants.x[2] = L3;
  dslashConstants.x[3] = L4;

  cudaMemcpyToSymbol(cudaConstants, &constants, sizeof(LatticeConstants));  
  
  checkCudaError();
}

#undef MAX

#endif // LATTICE_CONSTANTS_H
