#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#include <quda_internal.h>
#include <color_spinor_field.h>

// ---------- blas_quda.cu ---------- 

namespace quda {
  // creates and destroys reduction buffers  
  void initBlas(); 
  void endBlas(void);

  void setBlasParam(int kernel, int prec, int threads, int blocks);

  extern unsigned long long blas_flops;
  extern unsigned long long blas_bytes;

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

  double2 reDotProductNormACuda(cudaColorSpinorField &a, cudaColorSpinorField &b);


  void axpbyCuda(const double &a, cudaColorSpinorField &x, const double &b, cudaColorSpinorField &y);
  void axpyCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  void axCuda(const double &a, cudaColorSpinorField &x);
  void xpyCuda(cudaColorSpinorField &x, cudaColorSpinorField &y);
  void xpayCuda(cudaColorSpinorField &x, const double &a, cudaColorSpinorField &y);
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
  double3 caxpbypzYmbwcDotProductUYNormYCuda(const Complex &a, cudaColorSpinorField &x, const Complex &b, cudaColorSpinorField &y, 
					     cudaColorSpinorField &z, cudaColorSpinorField &w, cudaColorSpinorField &u);

  void cabxpyAxCuda(const double &a, const Complex &b, cudaColorSpinorField &x, cudaColorSpinorField &y);
  double caxpyNormCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  void caxpyXmazCuda(const Complex &a, cudaColorSpinorField &x,
		     cudaColorSpinorField &y, cudaColorSpinorField &z);
  double caxpyXmazNormXCuda(const Complex &a, cudaColorSpinorField &x, 
			    cudaColorSpinorField &y, cudaColorSpinorField &z);
  double cabxpyAxNormCuda(const double &a, const Complex &b, cudaColorSpinorField &x, cudaColorSpinorField &y);

  void caxpbypzCuda(const Complex &, cudaColorSpinorField &, const Complex &, cudaColorSpinorField &, 
		    cudaColorSpinorField &);
  void caxpbypczpwCuda(const Complex &, cudaColorSpinorField &, const Complex &, cudaColorSpinorField &, 
		       const Complex &, cudaColorSpinorField &, cudaColorSpinorField &);
  Complex caxpyDotzyCuda(const Complex &a, cudaColorSpinorField &x, cudaColorSpinorField &y,
			       cudaColorSpinorField &z);
  Complex axpyCGNormCuda(const double &a, cudaColorSpinorField &x, cudaColorSpinorField &y);
  double3 HeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &r);
  double3 xpyHeavyQuarkResidualNormCuda(cudaColorSpinorField &x, cudaColorSpinorField &y, cudaColorSpinorField &r);

  void tripleCGUpdateCuda(const double &alpha, const double &beta, cudaColorSpinorField &q,
			  cudaColorSpinorField &r, cudaColorSpinorField &x, cudaColorSpinorField &p);
  double3 tripleCGReductionCuda(cudaColorSpinorField &x, cudaColorSpinorField &y, cudaColorSpinorField &z);

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

  void caxpbyCpu(const Complex &a, const cpuColorSpinorField &x, const Complex &b, cpuColorSpinorField &y);
  void caxpyCpu(const Complex &a, const cpuColorSpinorField &x, cpuColorSpinorField &y);
  void cxpaypbzCpu(const cpuColorSpinorField &x, const Complex &b, const cpuColorSpinorField &y, 
		   const Complex &c, cpuColorSpinorField &z);
  void caxpbypzYmbwCpu(const Complex &, const cpuColorSpinorField &, const Complex &, cpuColorSpinorField &, 
		       cpuColorSpinorField &, const cpuColorSpinorField &); 
  Complex cDotProductCpu(const cpuColorSpinorField &, const cpuColorSpinorField &);
  Complex xpaycDotzyCpu(const cpuColorSpinorField &x, const double &a, cpuColorSpinorField &y, 
			      const cpuColorSpinorField &z);
  double3 cDotProductNormACpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b);
  double3 cDotProductNormBCpu(const cpuColorSpinorField &a, const cpuColorSpinorField &b);
  double3 caxpbypzYmbwcDotProductUYNormYCpu(const Complex &a, const cpuColorSpinorField &x, 
					    const Complex &b, cpuColorSpinorField &y, 
					    cpuColorSpinorField &z, const cpuColorSpinorField &w, 
					    const cpuColorSpinorField &u);

  void cabxpyAxCpu(const double &a, const Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y);

  double caxpyNormCpu(const Complex &a, cpuColorSpinorField &x, cpuColorSpinorField &y);

  void caxpyXmazCpu(const Complex &a, cpuColorSpinorField &x,
		    cpuColorSpinorField &y, cpuColorSpinorField &z);
  double caxpyXmazNormXCpu(const Complex &a, cpuColorSpinorField &x, 
			   cpuColorSpinorField &y, cpuColorSpinorField &z);
  double cabxpyAxNormCpu(const double &a, const Complex &b, cpuColorSpinorField &x, cpuColorSpinorField &y);

  void caxpbypzCpu(const Complex &, cpuColorSpinorField &, const Complex &, cpuColorSpinorField &, 
		   cpuColorSpinorField &);

  void caxpbypczpwCpu(const Complex &, cpuColorSpinorField &, const Complex &, cpuColorSpinorField &, 
		      const Complex &, cpuColorSpinorField &, cpuColorSpinorField &);
  Complex caxpyDotzyCpu(const Complex &a, cpuColorSpinorField &x, cpuColorSpinorField &y,
			      cpuColorSpinorField &z);
  double3 HeavyQuarkResidualNormCpu(cpuColorSpinorField &x, cpuColorSpinorField &r);
  double3 xpyHeavyQuarkResidualNormCpu(cpuColorSpinorField &x, cpuColorSpinorField &y, cpuColorSpinorField &r);

} // namespace quda

#endif // _QUDA_BLAS_H
