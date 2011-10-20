#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#include <quda_internal.h>
#include <color_spinor_field.h>

  // ---------- blas_quda.cu ---------- 

namespace quda {
  // creates and destroys reduction buffers  
  void initBlas(void); 
  void endBlas(void);

  void setBlasTuning(QudaTune tune);
  void setBlasParam(int kernel, int prec, int threads, int blocks);

  extern unsigned long long blas_flops;
  extern unsigned long long blas_bytes;
}


// C++ linkage

// Generic variants

double norm2(const ColorSpinorField&);

// CUDA variants

void zeroCuda(cudaColorSpinorField &a);
void copyCuda(cudaColorSpinorField &dst, const cudaColorSpinorField &src);

double axpyNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
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

void caxpbyCuda(const quda::Complex &a, cudaColorSpinorField &x, const quda::Complex &b, cudaColorSpinorField &y);
void caxpyCuda(const quda::Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
void cxpaypbzCuda(cudaColorSpinorField &, const quda::Complex &b, cudaColorSpinorField &y, const quda::Complex &c, cudaColorSpinorField &z);
void caxpbypzYmbwCuda(const quda::Complex &, cudaColorSpinorField &, const quda::Complex &, cudaColorSpinorField &, cudaColorSpinorField &, cudaColorSpinorField &);

quda::Complex cDotProductCuda(cudaColorSpinorField &, cudaColorSpinorField &);
quda::Complex xpaycDotzyCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y, cudaColorSpinorField &z);

double3 cDotProductNormACuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
double3 cDotProductNormBCuda(cudaColorSpinorField &a, cudaColorSpinorField &b);
double3 caxpbypzYmbwcDotProductUYNormYCuda(const quda::Complex &a, cudaColorSpinorField &x, const quda::Complex &b, cudaColorSpinorField &y, 
					   cudaColorSpinorField &z, cudaColorSpinorField &w, cudaColorSpinorField &u);

void cabxpyAxCuda(const double &a, const quda::Complex &b, cudaColorSpinorField &x, cudaColorSpinorField &y);
double caxpyNormCuda(const quda::Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
double caxpyXmazNormXCuda(const quda::Complex &a, cudaColorSpinorField &x, 
			  cudaColorSpinorField &y, cudaColorSpinorField &z);
double cabxpyAxNormCuda(const double &a, const quda::Complex &b, cudaColorSpinorField &x, cudaColorSpinorField &y);

void caxpbypzCuda(const quda::Complex &, cudaColorSpinorField &, const quda::Complex &, cudaColorSpinorField &, 
		  cudaColorSpinorField &);
void caxpbypczpwCuda(const quda::Complex &, cudaColorSpinorField &, const quda::Complex &, cudaColorSpinorField &, 
		     const quda::Complex &, cudaColorSpinorField &, cudaColorSpinorField &);
quda::Complex caxpyDotzyCuda(const quda::Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y,
		       cudaColorSpinorField &z);

// CPU variants

double axpyNormCpu(const double &a, const cpuColorSpinorField &x, cpuColorSpinorField &y);
double normCpu(const cpuColorSpinorField &b);
double reDotProductCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b);
double xmyNormCpu(const cpuColorSpinorField &a, cpuColorSpinorField &b);
void axpbyCpu(const double &a, const cpuColorSpinorField &x, const double &b, cpuColorSpinorField &y);
void axpyCpu(const double &a, const cpuColorSpinorField &x, cpuColorSpinorField &y);
void axCpu(const double &a, cpuColorSpinorField &x);
void xpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y);
void xpayCpu(const cpuColorSpinorField &x, const double &a, cpuColorSpinorField &y);
void mxpyCpu(const cpuColorSpinorField &x, cpuColorSpinorField &y);
void axpyZpbxCpu(const double &a, cpuColorSpinorField &x, cpuColorSpinorField &y, 
		 const cpuColorSpinorField &z, const double &b);
void axpyBzpcxCpu(const double &a, cpuColorSpinorField& x, cpuColorSpinorField& y,
		  const double &b, const cpuColorSpinorField& z, const double &c); 

void caxpbyCpu(const quda::Complex &a, const cpuColorSpinorField &x, const quda::Complex &b, cpuColorSpinorField &y);
void caxpyCpu(const quda::Complex &a, const cpuColorSpinorField &x, cpuColorSpinorField &y);
void cxpaypbzCpu(const cpuColorSpinorField &x, const quda::Complex &b, const cpuColorSpinorField &y, 
		 const quda::Complex &c, cpuColorSpinorField &z);
void caxpbypzYmbwCpu(const quda::Complex &, const cpuColorSpinorField &, const quda::Complex &, cpuColorSpinorField &, 
		     cpuColorSpinorField &, const cpuColorSpinorField &); 
quda::Complex cDotProductCpu(const cpuColorSpinorField &, const cpuColorSpinorField &);
quda::Complex xpaycDotzyCpu(const cpuColorSpinorField &x, const double &a, cpuColorSpinorField &y, 
		      const cpuColorSpinorField &z);
double3 cDotProductNormACpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b);
double3 cDotProductNormBCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b);
double3 caxpbypzYmbwcDotProductUYNormYCpu(const quda::Complex &a, const cpuColorSpinorField &x, 
					  const quda::Complex &b, cpuColorSpinorField &y, 
					  cpuColorSpinorField &z, const cpuColorSpinorField &w, 
					  const cpuColorSpinorField &u);

void cabxpyAxCpu(const double &a, const quda::Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y);

double caxpyNormCpu(const quda::Complex &a, cpuColorSpinorField &x, cpuColorSpinorField &y);

double caxpyXmazNormXCpu(const quda::Complex &a, cpuColorSpinorField &x, 
			 cpuColorSpinorField &y, cpuColorSpinorField &z);
double cabxpyAxNormCpu(const double &a, const quda::Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y);

void caxpbypzCpu(const quda::Complex &, cpuColorSpinorField &, const quda::Complex &, cpuColorSpinorField &, 
		 cpuColorSpinorField &);

void caxpbypczpwCpu(const quda::Complex &, cpuColorSpinorField &, const quda::Complex &, cpuColorSpinorField &, 
		    const quda::Complex &, cpuColorSpinorField &, cpuColorSpinorField &);
quda::Complex caxpyDotzyCpu(const quda::Complex &a, cpuColorSpinorField &x, cpuColorSpinorField &y,
		      cpuColorSpinorField &z);

#endif // _QUDA_BLAS_H
