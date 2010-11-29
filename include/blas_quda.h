#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#include <quda_internal.h>
#include <color_spinor_field.h>


// keep these with C-linkage for the moment

#ifdef __cplusplus
extern "C" {
#endif

  // ---------- blas_quda.cu ---------- 

  // creates and destroys reduction buffers  
  void initBlas(void); 
  void endBlas(void);

  void setBlasTuning(int tuning);
  void setBlasParam(int kernel, int prec, int threads, int blocks);

  extern unsigned long long blas_quda_flops;
  extern unsigned long long blas_quda_bytes;

#ifdef __cplusplus
}
#endif

// C++ linkage

// Generic variants

double norm2(const ColorSpinorField&);

// CUDA variants

void zeroCuda(cudaColorSpinorField &a);
void copyCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src);

double axpyNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
double sumCuda(cudaColorSpinorField &b);
double normCuda(const cudaColorSpinorField &b);
double reDotProductCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
double xmyNormCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);

void axpbyCuda(const double &a, cudaColorSpinorField &x, const double &b, cudaColorSpinorField &y);
void axpyCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
void axCuda(const double &a, cudaColorSpinorField &x);
void xpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y);
void xpayCuda(const cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y);
void mxpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y);

void axpyZpbxCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y, cudaColorSpinorField &z, const double &b);
void axpyBzpcxCuda(const double &a, cudaColorSpinorField& x, cudaColorSpinorField& y, const double &b, cudaColorSpinorField& z, const double &c); 

void caxpbyCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, cudaColorSpinorField &y);
void caxpyCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
void cxpaypbzCuda(cudaColorSpinorField &, const Complex &b, cudaColorSpinorField &y, const Complex &c, cudaColorSpinorField &z);
void caxpbypzYmbwCuda(const Complex &, cudaColorSpinorField &, const Complex &, cudaColorSpinorField &, cudaColorSpinorField &, cudaColorSpinorField &);

Complex cDotProductCuda(cudaColorSpinorField &, cudaColorSpinorField &);
Complex xpaycDotzyCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y, cudaColorSpinorField &z);

double3 cDotProductNormACuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
double3 cDotProductNormBCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
double3 caxpbypzYmbwcDotProductWYNormYCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, cudaColorSpinorField &y, 
					   cudaColorSpinorField &z, cudaColorSpinorField &w, cudaColorSpinorField &u);

// CPU variants

double normCpu(const cpuColorSpinorField &b);


#endif // _QUDA_BLAS_H
