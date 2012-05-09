#ifndef _TEST_UTIL_H
#define _TEST_UTIL_H

#include <quda.h>

#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 24 // real numbers per spinor
#define cloverSiteSize 72 // real numbers per block-diagonal clover matrix
#define momSiteSize    10 // real numbers per momentum
#define hwSiteSize    12 // real numbers per half wilson

#ifdef __cplusplus
extern "C" {
#endif

  extern int Z[4];
  extern int V;
  extern int Vh;
  extern int Vs_x, Vs_y, Vs_z, Vs_t;
  extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
  extern int faceVolume[4];
  extern int E1, E1h, E2, E3, E4; 
  extern int E[4];
  extern int V_ex, Vh_ex;

  extern int Ls;
  extern int V5;
  extern int V5h;
  
  extern int mySpinorSiteSize;

  void setDims(int *X);
  void dw_setDims(int *X, const int L5);
  void setSpinorSiteSize(int n);

  int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
  int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1) ;
  int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
  int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1);

  void printSpinorElement(void *spinor, int X, QudaPrecision precision);
  void printGaugeElement(void *gauge, int X, QudaPrecision precision);
  
  int fullLatticeIndex(int i, int oddBit);
  int getOddBit(int X);

  void construct_gauge_field(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param);
    void construct_fat_long_gauge_field(void **fatlink, void** longlink, int type, QudaPrecision precision, QudaGaugeParam*);
    void construct_clover_field(void *clover, double norm, double diag, QudaPrecision precision);
  void construct_spinor_field(void *spinor, int type, int i0, int s0, int c0, QudaPrecision precision);
  void createSiteLinkCPU(void** link,  QudaPrecision precision, int phase) ;

  void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision);
  void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision, QudaGaugeParam *param);
  //void su3_construct_8_half(float *mat, short *mat_half);
  //void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx, QudaGaugeParam *param);

  void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, QudaPrecision precision);
  void strong_check(void *spinor, void *spinorGPU, int len, QudaPrecision precision);
  int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision);

  void check_gauge(void **, void **, double epsilon, QudaPrecision precision);

  int strong_check_link(void ** linkA, const char* msgA,  void **linkB, const char* msgB, int len, QudaPrecision prec);
  int strong_check_mom(void * momA, void *momB, int len, QudaPrecision prec);
  
  void createMomCPU(void* mom,  QudaPrecision precision);
  void createHwCPU(void* hw,  QudaPrecision precision);
  
  //used by link fattening code
  int x4_from_full_index(int i);
  // ---------- gauge_read.cpp ----------
  
  //void readGaugeField(char *filename, float *gauge[], int argc, char *argv[]);

  // additions for dw (quickly hacked on)
  int fullLatticeIndex_4d(int i, int oddBit);
  int fullLatticeIndex_5d(int i, int oddBit);
  int process_command_line_option(int argc, char** argv, int* idx);
 
#ifdef __cplusplus
}
#endif

#endif // _TEST_UTIL_H
