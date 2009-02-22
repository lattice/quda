#ifndef _QUDA_UTIL_H
#define _QUDA_UTIL_H

#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  // ---------- qcd.cpp ----------
  
  int compareFloats(float *a, float *b, int len, float tol);
  
  void stopwatchStart();
  double stopwatchReadSeconds();

  void printSpinor(float *spinor);
  void printSpinorElement(float *spinor, int X);
  void printGauge(float *gauge);
  void printGaugeElement(float *gauge, int X);
  
  int fullLatticeIndex(int i, int oddBit);
  int getOddBit(int X);
  
  void applyGaugeFieldScaling(float **gauge);
  void constructUnitGaugeField(float **gauge);
  void constructGaugeField(float **gauge);
  void constructPointSpinorField(float *spinor, int i0, int s0, int c0);
  void constructSpinorField(float *res);
  
  void applyGamma5(float *out, float *in, int len);
  
  void su3_construct_12(float *mat);
  void su3_reconstruct_12(float *mat, int dir, int ga_idx);
  void su3_construct_8(float *mat);
  void su3_reconstruct_8(float *mat, int dir, int ga_idx);
  void su3_construct_8_half(float *mat, short *mat_half);
  void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx);

  void su3_construct_8_bunk(float *mat, int dir);
  void su3_reconstruct_8_bunk(float *mat, int dir, int ga_idx);
  
  // ---------- gauge_read.cpp ----------
  
  void readGaugeField(char *filename, float *gauge[], int argc, char *argv[]);
 
#ifdef __cplusplus
}
#endif

#endif // _QUDA_UTIL_H
