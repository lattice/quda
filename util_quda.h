#ifndef _QUDA_UTIL_H
#define _QUDA_UTIL_H

#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  // ---------- qcd.cpp ----------
  
  int compare_floats(void *a, void *b, int len, double tol, Precision precision);
  
  void stopwatchStart();
  double stopwatchReadSeconds();

  void printSpinorElement(void *spinor, int X, Precision precision);
  void printGaugeElement(void *gauge, int X, Precision precision);
  
  int fullLatticeIndex(int i, int oddBit);
  int getOddBit(int X);
  
  void construct_gauge_field(void **gauge, int type, Precision precision);
  void construct_spinor_field(void *spinor, int type, int i0, int s0, int c0, Precision precision);
  
  void su3_construct(void *mat, ReconstructType reconstruct, Precision precision);
  void su3_reconstruct(void *mat, int dir, int ga_idx, ReconstructType reconstruct, Precision precision);
  //void su3_construct_8_half(float *mat, short *mat_half);
  //void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx);

  void apply_gamma5(void *out, void *in, int len, Precision precision);
  
  void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, Precision precision);

  // ---------- gauge_read.cpp ----------
  
  void readGaugeField(char *filename, float *gauge[], int argc, char *argv[]);
 
#ifdef __cplusplus
}
#endif

#endif // _QUDA_UTIL_H
