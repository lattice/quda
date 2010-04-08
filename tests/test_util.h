#ifndef _TEST_UTIL_H
#define _TEST_UTIL_H

#include <quda.h>

#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 24 // real numbers per spinor
#define cloverSiteSize 72 // real numbers per block-diagonal clover matrix

#ifdef __cplusplus
extern "C" {
#endif
  
  void printSpinorElement(void *spinor, int X, QudaPrecision precision);
  void printGaugeElement(void *gauge, int X, QudaPrecision precision);
  
  int fullLatticeIndex(int i, int oddBit);
  int getOddBit(int X);

  void construct_gauge_field(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param);
    void construct_fat_long_gauge_field(void **fatlink, void** longlink, int type, QudaPrecision precision, QudaGaugeParam*);
    void construct_clover_field(void *clover, double norm, double diag, QudaPrecision precision);
  void construct_spinor_field(void *spinor, int type, int i0, int s0, int c0, QudaPrecision precision);
  
  void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision);
  void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision, QudaGaugeParam *param);
  //void su3_construct_8_half(float *mat, short *mat_half);
  //void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx, QudaGaugeParam *param);

  void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, QudaPrecision precision);
  void strong_check(void *spinor, void *spinorGPU, int len, QudaPrecision precision);
  int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision);

  void check_gauge(void **, void **, double epsilon, QudaPrecision precision);

  // ---------- gauge_read.cpp ----------
  
  //void readGaugeField(char *filename, float *gauge[], int argc, char *argv[]);
 
#ifdef __cplusplus
}
#endif

#endif // _TEST_UTIL_H
