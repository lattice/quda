#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#include <quda_internal.h>
#include <color_spinor_field.h>

// ---------- blas_quda.cu ----------

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

    void zero(ColorSpinorField &a);

    inline void copy(ColorSpinorField &dst, const ColorSpinorField &src)
    {
      if (&dst == &src) return;

      if (dst.Location() == QUDA_CUDA_FIELD_LOCATION && src.Location() == QUDA_CUDA_FIELD_LOCATION) {
        static_cast<cudaColorSpinorField &>(dst).copy(static_cast<const cudaColorSpinorField &>(src));
      } else if (dst.Location() == QUDA_CPU_FIELD_LOCATION && src.Location() == QUDA_CPU_FIELD_LOCATION) {
        static_cast<cpuColorSpinorField &>(dst).copy(static_cast<const cpuColorSpinorField &>(src));
      } else {
        errorQuda("Cannot call copy with fields with different locations");
      }
    }

    void ax(double a, ColorSpinorField &x);

    void axpbyz(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z);

    inline void xpy(ColorSpinorField &x, ColorSpinorField &y) { axpbyz(1.0, x, 1.0, y, y); }
    inline void mxpy(ColorSpinorField &x, ColorSpinorField &y) { axpbyz(-1.0, x, 1.0, y, y); }
    inline void axpy(double a, ColorSpinorField &x, ColorSpinorField &y) { axpbyz(a, x, 1.0, y, y); }
    inline void axpby(double a, ColorSpinorField &x, double b, ColorSpinorField &y) { axpbyz(a, x, b, y, y); }
    inline void xpay(ColorSpinorField &x, double a, ColorSpinorField &y) { axpbyz(1.0, x, a, y, y); }
    inline void xpayz(ColorSpinorField &x, double a, ColorSpinorField &y, ColorSpinorField &z) { axpbyz(1.0, x, a, y, z); }

    void axpyZpbx(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, double b);
    void axpyBzpcx(double a, ColorSpinorField& x, ColorSpinorField& y, double b, ColorSpinorField& z, double c);

    void caxpby(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y);
    void caxpy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpbypczw(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y, const Complex &c,
                    ColorSpinorField &z, ColorSpinorField &w);
    void cxpaypbz(ColorSpinorField &, const Complex &b, ColorSpinorField &y, const Complex &c, ColorSpinorField &z);
    void caxpbypzYmbw(const Complex &, ColorSpinorField &, const Complex &, ColorSpinorField &, ColorSpinorField &, ColorSpinorField &);
    void caxpyBzpx(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);
    void caxpyBxpz(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);

    void cabxpyAx(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y);
    void caxpyXmaz(const Complex &a, ColorSpinorField &x,
		   ColorSpinorField &y, ColorSpinorField &z);
    void caxpyXmazMR(const Complex &a, ColorSpinorField &x,
		     ColorSpinorField &y, ColorSpinorField &z);

    void tripleCGUpdate(double alpha, double beta, ColorSpinorField &q,
			ColorSpinorField &r, ColorSpinorField &x, ColorSpinorField &p);
    void doubleCG3Init(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    void doubleCG3Update(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);


    // reduction kernels - defined in reduce_quda.cu

    double norm1(const ColorSpinorField &b);
    double norm2(const ColorSpinorField &a);

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y);

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y);

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z);
    inline double axpyNorm(double a, ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(a, x, 1.0, y, y); }
    inline double xmyNorm(ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(1.0, x, -1.0, y, y); }

    Complex cDotProduct(ColorSpinorField &, ColorSpinorField &);
    double3 cDotProductNormA(ColorSpinorField &a, ColorSpinorField &b);

    /**
       @brief Return (a,b) and ||b||^2 - implemented using cDotProductNormA
     */
    inline double3 cDotProductNormB(ColorSpinorField &a, ColorSpinorField &b) {
      double3 a3 = cDotProductNormA(b, a);
      return make_double3(a3.x, -a3.y, a3.z);
    }

    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y,
					   ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &u);

    double caxpyNorm(const Complex &a, ColorSpinorField &x, ColorSpinorField &y);
    double caxpyXmazNormX(const Complex &a, ColorSpinorField &x,
			  ColorSpinorField &y, ColorSpinorField &z);
    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    Complex caxpyDotzy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y,
		       ColorSpinorField &z);
    Complex axpyCGNorm(double a, ColorSpinorField &x, ColorSpinorField &y);
    double3 HeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &r);
    double3 xpyHeavyQuarkResidualNorm(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &r);

    double3 tripleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    double4 quadrupleCGReduction(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v);
    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v);

    double doubleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    double doubleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    // multi-blas kernels - defined in multi_blas.cu

    /**
       @brief Compute the block "axpy" with over the set of
              ColorSpinorFields.  E.g., it computes y = x * a + y
              The dimensions of a can be rectangular, e.g., the width of x and y need not be same.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
      @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y);

    /**
       @brief This is a wrapper for calling the block "axpy" with a
       composite ColorSpinorField.  E.g., it computes
       y = x * a + y
       @param a[in] Matrix of real coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void axpy(const double *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the block "axpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy_U(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);

    /**
       @brief This is a wrapper for calling the block "axpy_U" with a
       composite ColorSpinorField.  E.g., it computes

       y = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void axpy_U(const double *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the block "axpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy_L(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);

    /**
       @brief This is a wrapper for calling the block "axpy_U" with a
       composite ColorSpinorField.  E.g., it computes

       y = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void axpy_L(const double *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the block "caxpy" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       The dimensions of a can be rectangular, e.g., the width of x
       and y need not be same.

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
       @brief Compute the block "caxpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y);

    /**
       @brief This is a wrapper for calling the block "caxpy_U" with a
       composite ColorSpinorField.  E.g., it computes

       y = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the block "caxpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y);

    /**
       @brief This is a wrapper for calling the block "caxpy_U" with a
       composite ColorSpinorField.  E.g., it computes

       y = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in,out] Computed output matrix
    */
    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the block "caxpyz" with over the set of
       ColorSpinorFields.  E.g., it computes

       z = x * a + y

       The dimensions of a can be rectangular, e.g., the width of x
       and y need not be same, though the maximum width for both is
       16.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z);

    /**
       @brief This is a wrapper for calling the block "caxpyz" with a
       composite ColorSpinorField.  E.g., it computes

       z = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in] Computed output matrix
       @param z[out] vector of input/output ColorSpinorFields
    */
    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    /**
       @brief Compute the block "caxpyz" with over the set of
       ColorSpinorFields.  E.g., it computes

       z = x * a + y

       Where 'a' is assumed to be upper triangular.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z);

    /**
       @brief This is a wrapper for calling the block "caxpyz" with a
       composite ColorSpinorField.  E.g., it computes

       z = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in] Computed output matrix
       @param z[out] vector of input/output ColorSpinorFields
    */
    void caxpyz_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    /**
       @brief Compute the block "caxpyz" with over the set of
       ColorSpinorFields.  E.g., it computes

       z = x * a + y

       Where 'a' is assumed to be lower triangular

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z);

    /**
       @brief This is a wrapper for calling the block "caxpyz" with a
       composite ColorSpinorField.  E.g., it computes

       z = x * a + y

       @param a[in] Matrix of coefficients
       @param x[in] Input matrix
       @param y[in] Computed output matrix
       @param z[out] vector of input/output ColorSpinorFields
    */
    void caxpyz_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

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


    // multi-reduce kernels - defined in multi_reduce.cu

    void reDotProduct(double* result, std::vector<ColorSpinorField*>& a, std::vector<ColorSpinorField*>& b);

    /**
       @brief Computes the matrix of inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void cDotProduct(Complex* result, std::vector<ColorSpinorField*>& a, std::vector<ColorSpinorField*>& b);

    /**
       @brief Computes the matrix of inner products between the vector
       set a and the vector set b.  This routine is specifically for
       the case where the result matrix is guarantted to be Hermitian.
       Requires a.size()==b.size().

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void hDotProduct(Complex* result, std::vector<ColorSpinorField*>& a, std::vector<ColorSpinorField*>& b);

    /**
       @brief Computes the matrix of inner products between the vector
       set a and the vector set b.  This routine is specifically for
       the case where the result matrix is guarantted to be Hermitian.
       Uniquely defined for cases like (p, Ap) where the output is Hermitian,
       but there's an A-norm instead of an L2 norm.
       Requires a.size()==b.size().

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void hDotProduct_Anorm(Complex* result, std::vector<ColorSpinorField*>& a, std::vector<ColorSpinorField*>& b);


    /**
       @brief Computes the matrix of inner products between the vector set a and the vector set b, and copies b into c

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
       @param c[out] set of output ColorSpinorFields
    */
    void cDotProductCopy(Complex* result, std::vector<ColorSpinorField*>& a, std::vector<ColorSpinorField*>& b, std::vector<ColorSpinorField*>& c);

  } // namespace blas

} // namespace quda

#endif // _QUDA_BLAS_H
