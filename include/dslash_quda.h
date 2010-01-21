#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int initDslash;
  extern unsigned long long dslash_quda_flops;
  extern unsigned long long dslash_quda_bytes;

  int dslashCudaSharedBytes(QudaPrecision spinor_prec, int blockDim);

  void initDslashConstants(FullGauge gauge, int sp_stride, int cl_stride);

  // plain wilson
  
  void dslashDCuda(double2 *out, FullGauge gauge, double2 *in, int oddBit, int daggerBit,
		   int volume, int length);
  void dslashSCuda(float4 *out, FullGauge gauge, float4 *in, int oddBit, int daggerBit,
		   int volume, int length);
  void dslashHCuda(short4 *out, float* outNorm, FullGauge gauge, short4* in, float* inNorm,
		   int oddBit, int daggerBit, int volume, int length);
  
  void dslashXpayDCuda(double2 *out, FullGauge gauge, double2 *in, int oddBit,
		       int daggerBit, double2 *x, double a,int volume, int length);
  void dslashXpaySCuda(float4 *out, FullGauge gauge, float4 *in, int oddBit,
		       int daggerBit, float4 *x, double a, int volume, int length);
  void dslashXpayHCuda(short4 *out, float *outNorm, FullGauge gauge, short4* in, float *inNorm, 
		       int oddBit, int daggerBit, short4 *x, float *xNorm, double a,
		       int volume, int length);
  
  // clover dslash

  void cloverDslashDCuda(double2 *out, FullGauge gauge, FullClover cloverInv, double2 *in,
			 int oddBit, int daggerBit, int volume, int length);

  void cloverDslashSCuda(float4 *out, FullGauge gauge, FullClover cloverInv, float4 *in,
			 int oddBit, int daggerBit, int volume, int length);

  void cloverDslashHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover cloverInv, 
			 short4 *in, float *inNorm, int oddBit, int daggerBit,
			 int volume, int length);
  
  void cloverDslashXpayDCuda(double2 *out, FullGauge gauge, FullClover cloverInv, double2 *in,
			     int oddBit, int daggerBit, double2 *x, double a, int volume, int length);
  
  void cloverDslashXpaySCuda(float4 *out, FullGauge gauge, FullClover cloverInv, float4 *in,
			     int oddBit, int daggerBit, float4 *x, double a, int volume, int length);

  void cloverDslashXpayHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover cloverInv, 
			     short4 *in, float *inNorm, int oddBit, int daggerBit, 
			     short4 *x, float*xNorm, double a, int volume, int length);

  // solo clover term
  void cloverDCuda(double2 *out, FullGauge gauge, FullClover clover,
		   double2 *in, int oddBit, int volume, int length);
  void cloverSCuda(float4 *out, FullGauge gauge, FullClover clover,
		   float4 *in, int oddBit, int volume, int length);
  void cloverHCuda(short4 *out, float *outNorm, FullGauge gauge, FullClover clover,
		   short4 *in, float *inNorm, int oddBit, int volume, int length);

#ifdef __cplusplus
}
#endif

#endif // _DLASH_QUDA_H
