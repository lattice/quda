#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <color_spinor_field.h>

namespace quda
{
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);  
  void evecProjectSumQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::complex<double> *result);
  void colorCrossQuda(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result);
  void colorContractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);
} // namespace quda



#ifndef __CUDACC_RTC__
#define double_complex double _Complex
#else // keep NVRTC happy since it can't handle C types
#define double_complex double2
#endif

#ifdef __cplusplus
extern "C" {
#endif


  void laphSinkProject(void *host_quark, void **host_evec, double _Complex *host_sinks,
		       QudaInvertParam inv_param, unsigned int nEv, const int X[4]);

  void laphBaryonKernel(int n1, int n2, int n3, int nMom,
			double _Complex *host_coeffs1, 
			double _Complex *host_coeffs2, 
			double _Complex *host_coeffs3,
			double _Complex *host_mom, 
			int nEv, void **host_evec, 
			void *retArr,
			int blockSizeMomProj,
			const int X[4]);

  void laphBaryonKernelComputeModeTripletA(int nMom, int nEv, void **host_evec, 
					   double _Complex *host_mom,
					   void *retArr,
					   int blockSizeMomProj,
					   const int X[4]);

  void laphBaryonKernelComputeModeTripletB(int n1, int n2, int n3, int nMom,
					   double _Complex *host_coeffs1, 
					   double _Complex *host_coeffs2, 
					   double _Complex *host_coeffs3,
					   double _Complex *host_mom, 
					   double _Complex *host_mode_trip_buf,
					   int nEv, void **host_evec, 
					   void *retArr,
					   const int X[4]);

#ifdef __cplusplus
}
#endif

// remove NVRTC WAR
#undef double_complex
