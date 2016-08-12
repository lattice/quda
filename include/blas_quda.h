#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#include <quda_internal.h>
#include <color_spinor_field.h>

// ---------- blas_quda.cu ----------

// these defitions are used to avoid calling
// std::complex<type>::real/imag which have C++11 ABI incompatibility
// issues with certain versions of GCC

#define REAL(a) (*((double*)&a))
#define IMAG(a) (*((double*)&a+1))

namespace quda {

  namespace blas {

    // creates and destroys reduction buffers
    void init();
    void end(void);

    void* getDeviceReduceBuffer();
    void* getMappedHostReduceBuffer();
    void* getHostReduceBuffer();

    void setParam(int kernel, int prec, int threads, int blocks);

    extern unsigned long long flops;
    extern unsigned long long bytes;

    double norm2(const ColorSpinorField &a);
    double norm1(const ColorSpinorField &b);

    void zero(ColorSpinorField &a);
    void copy(ColorSpinorField &dst, const ColorSpinorField &src);

    double axpyNorm(const double &a, ColorSpinorField &x, ColorSpinorField &y);
    double axpyReDot(const double &a, ColorSpinorField &x, ColorSpinorField &y);

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y);
    double2 reDotProductNormA(ColorSpinorField &a, ColorSpinorField &b);

    double xmyNorm(ColorSpinorField &x, ColorSpinorField &y);

    void axpby(const double &a, ColorSpinorField &x, const double &b, ColorSpinorField &y);
    void axpy(const double &a, ColorSpinorField &x, ColorSpinorField &y);
    void ax(const double &a, ColorSpinorField &x);
    void xpy(ColorSpinorField &x, ColorSpinorField &y);
    void xpay(ColorSpinorField &x, const double &a, ColorSpinorField &y);
    void mxpy(ColorSpinorField &x, ColorSpinorField &y);

    void axpyZpbx(const double &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, const double &b);
    void axpyBzpcx(const double &a, ColorSpinorField& x, ColorSpinorField& y, const double &b, ColorSpinorField& z, const double &c);

    void caxpby(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y);
    void caxpy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y);
    void cxpaypbz(ColorSpinorField &, const Complex &b, ColorSpinorField &y, const Complex &c, ColorSpinorField &z);
    void caxpbypzYmbw(const Complex &, ColorSpinorField &, const Complex &, ColorSpinorField &, ColorSpinorField &, ColorSpinorField &);
    void multcaxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, int N);


    Complex cDotProduct(ColorSpinorField &, ColorSpinorField &);
    Complex xpaycDotzy(ColorSpinorField &x, const double &a, ColorSpinorField &y, ColorSpinorField &z);

    double3 cDotProductNormA(ColorSpinorField &a, ColorSpinorField &b);
    double3 cDotProductNormB(ColorSpinorField &a, ColorSpinorField &b);
    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y,
					   ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &u);

    void cabxpyAx(const double &a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y);
    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpyXmaz(const Complex &a, ColorSpinorField &x,
		   ColorSpinorField &y, ColorSpinorField &z);
    void caxpyXmazMR(const Complex &a, ColorSpinorField &x,
		     ColorSpinorField &y, ColorSpinorField &z);
    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x,
			  ColorSpinorField &y, ColorSpinorField &z);
    double cabxpyAxNorm(const double &a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y);

    void caxpbypz(const Complex &, ColorSpinorField &, const Complex &, ColorSpinorField &,
		  ColorSpinorField &);
    void caxpbypczpw(const Complex &, ColorSpinorField &, const Complex &, ColorSpinorField &,
		     const Complex &, ColorSpinorField &, ColorSpinorField &);
    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y,
		       ColorSpinorField &z);
    Complex axpyCGNorm(const double &a, ColorSpinorField &x, ColorSpinorField &y);
    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r);
    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &r);

    void tripleCGUpdate(const double &alpha, const double &beta, ColorSpinorField &q,
			ColorSpinorField &r, ColorSpinorField &x, ColorSpinorField &p);
    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    void reDotProduct(double* result, std::vector<cudaColorSpinorField*>& a, std::vector<cudaColorSpinorField*>& b);
    void cDotProduct(Complex* result, std::vector<cudaColorSpinorField*>& a, std::vector<cudaColorSpinorField*>& b);
  } // namespace blas

} // namespace quda

#endif // _QUDA_BLAS_H
