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
    void caxpyBzpx(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);
    void caxpyBxpz(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);

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
    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    /**
       @brief Compute the block "caxpy" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       The dimensions of a can be rectangular, e.g., the width of x
       and y need not be same, though the maximum width for both is
       16.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y);

    /**
       @brief This is a wrapper for calling the block "caxpy" with a
       composite ColorSpinorField.  E.g., it computes

       y = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the vectorized "axpyBzpcx" with over the set of
       ColorSpinorFields, where the third vector, z, is constant over the
       batch.  E.g., it computes

       y = a * x + y
       x = b * z + c * x

       The dimensions of a, b, c are the same as the size of x and y,
       with a maximum size of 16.

       @param a[in] Array of coefficients
       @param b[in] Array of coefficients
       @param c[in] Array of coefficients
       @param x[in,out] vector of ColorSpinorFields
       @param y[in,out] vector of ColorSpinorFields
       @param z[in] input ColorSpinorField
    */
    void axpyBzpcx(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
		   const double *b, ColorSpinorField &z, const double *c);

    /**
       @brief Compute the vectorized "caxpyBxpz" over the set of
       ColorSpinorFields, where the second and third vector, y and z, is constant over the
       batch.  E.g., it computes

       y = a * x + y
       z = b * x + z

       The dimensions of a, b are the same as the size of x,
       with a maximum size of 16.

       @param a[in] Array of coefficients
       @param b[in] Array of coefficients
       @param x[in] vector of ColorSpinorFields
       @param y[in,out] input ColorSpinorField
       @param z[in,out] input ColorSpinorField
    */
    void caxpyBxpz(const Complex *a_, std::vector<ColorSpinorField*> &x_, ColorSpinorField &y_,
		   const Complex *b_, ColorSpinorField &z_);

    void reDotProduct(double* result, std::vector<cudaColorSpinorField*>& a, std::vector<cudaColorSpinorField*>& b);
    void cDotProduct(Complex* result, std::vector<cudaColorSpinorField*>& a, std::vector<cudaColorSpinorField*>& b);

  } // namespace blas

} // namespace quda

#endif // _QUDA_BLAS_H
