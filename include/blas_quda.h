#pragma once
#include <quda_internal.h>
#include <color_spinor_field.h>

// ---------- blas_quda.cu ----------

namespace quda {

  namespace reducer
  {

    /**
       @brief Free any persistent allocations associated with global reduction
     */
    void destroy();

  } // namespace reducer

  namespace blas
  {

    void setParam(int kernel, int prec, int threads, int blocks);

    extern unsigned long long flops;
    extern unsigned long long bytes;

    void zero(ColorSpinorField &a);

    inline void copy(ColorSpinorField &dst, const ColorSpinorField &src)
    {
      if (&dst == &src) return;
      dst.copy(src);
    }

    void ax(double a, ColorSpinorField &x);
    void ax(double a, std::vector<ColorSpinorField *> &x);
    void ax(double *a, std::vector<ColorSpinorField *> &x); // not a true block-blas routine

    void axpbyz(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z);

    inline void xpy(ColorSpinorField &x, ColorSpinorField &y) { axpbyz(1.0, x, 1.0, y, y); }
    inline void mxpy(ColorSpinorField &x, ColorSpinorField &y) { axpbyz(-1.0, x, 1.0, y, y); }
    inline void axpy(double a, ColorSpinorField &x, ColorSpinorField &y) { axpbyz(a, x, 1.0, y, y); }
    inline void axpby(double a, ColorSpinorField &x, double b, ColorSpinorField &y) { axpbyz(a, x, b, y, y); }
    inline void xpay(ColorSpinorField &x, double a, ColorSpinorField &y) { axpbyz(1.0, x, a, y, y); }
    inline void xpayz(ColorSpinorField &x, double a, ColorSpinorField &y, ColorSpinorField &z) { axpbyz(1.0, x, a, y, z); }

    void axpyZpbx(double a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, double b);
    void axpyBzpcx(double a, ColorSpinorField& x, ColorSpinorField& y, double b, ColorSpinorField& z, double c);
    void axpbypczw(double a, ColorSpinorField &x, double b, ColorSpinorField &y, double c, ColorSpinorField &z,
                   ColorSpinorField &w);

    void caxpby(const Complex &a, ColorSpinorField &x, const Complex &b, ColorSpinorField &y);
    void caxpy(const Complex &a, ColorSpinorField &x, ColorSpinorField &y);
    void cxpaypbz(ColorSpinorField &, const Complex &b, ColorSpinorField &y, const Complex &c, ColorSpinorField &z);
    void caxpbypzYmbw(const Complex &, ColorSpinorField &, const Complex &, ColorSpinorField &, ColorSpinorField &, ColorSpinorField &);
    void caxpyBzpx(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);
    void caxpyBxpz(const Complex &, ColorSpinorField &, ColorSpinorField &, const Complex &, ColorSpinorField &);

    void cabxpyAx(double a, const Complex &b, ColorSpinorField &x, ColorSpinorField &y);
    void caxpyXmaz(const Complex &a, ColorSpinorField &x,
		   ColorSpinorField &y, ColorSpinorField &z);
    void caxpyXmazMR(const double &a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);

    void tripleCGUpdate(double alpha, double beta, ColorSpinorField &q,
			ColorSpinorField &r, ColorSpinorField &x, ColorSpinorField &p);

    // reduction kernels - defined in reduce_quda.cu

    double norm1(const ColorSpinorField &b);
    double norm2(const ColorSpinorField &a);

    double axpyReDot(double a, ColorSpinorField &x, ColorSpinorField &y);

    double reDotProduct(ColorSpinorField &x, ColorSpinorField &y);

    double axpbyzNorm(double a, ColorSpinorField &x, double b, ColorSpinorField &y, ColorSpinorField &z);
    inline double axpyNorm(double a, ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(a, x, 1.0, y, y); }
    inline double xmyNorm(ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(1.0, x, -1.0, y, y); }

    Complex cDotProduct(ColorSpinorField &, ColorSpinorField &);

    /**
       @brief Return (a,b), ||a||^2 and ||b||^2
    */
    double4 cDotProductNormAB(ColorSpinorField &a, ColorSpinorField &b);

    /**
       @brief Return (a,b) and ||a||^2 - implemented using cDotProductNormAB
     */
    inline double3 cDotProductNormA(ColorSpinorField &a, ColorSpinorField &b)
    {
      auto a4 = cDotProductNormAB(a, b);
      return make_double3(a4.x, a4.y, a4.z);
    }

    /**
       @brief Return (a,b) and ||b||^2 - implemented using cDotProductNormAB
     */
    inline double3 cDotProductNormB(ColorSpinorField &a, ColorSpinorField &b)
    {
      auto a4 = cDotProductNormAB(a, b);
      return make_double3(a4.x, a4.y, a4.w);
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
    
    double4 quadrupleEigCGUpdate(double a, double b, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w, ColorSpinorField &v);
    
    // multi-blas kernels - defined in multi_blas.cu
    /**
       @brief Compute the block "axpy" with over the set of
       ColorSpinorFields.  E.g., it computes y = x * a + y
       The dimensions of a can be rectangular, e.g., the width of x and y need not be same.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x, std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Overloaded version of block axpy that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void axpy(const std::vector<double> &a, T1 &&x, T2 &&y)
    {
      axpy(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "axpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy_U(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
                std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Wrapper function for block axpy_U that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void axpy_U(const std::vector<double> &a, T1 &&x, T2 &&y)
    {
      axpy_U(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "axpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void axpy_L(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
                std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Wrapper function for block axpy_L that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void axpy_L(const std::vector<double> &a, T1 &&x, T2 &&y)
    {
      axpy_L(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "caxpy" with over the s
et of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       The dimensions of a can be rectangular, e.g., the width of x
       and y need not be same.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
               std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Wrapper function for block caxpy that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void caxpy(const std::vector<Complex> &a, T1 &&x, T2 &&y)
    {
      caxpy(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "caxpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_U(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
                 std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Wrapper function for block caxpy_U that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void caxpy_U(const std::vector<Complex> &a, T1 &&x, T2 &&y)
    {
      caxpy_U(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "caxpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_L(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
                 std::vector<ColorSpinorField_ref> &&y);

    /**
       @brief Wrapper function for block caxpy_L that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T1, typename T2> void caxpy_L(const std::vector<Complex> &a, T1 &&x, T2 &&y)
    {
      caxpy_L(a, make_set(x), make_set(y));
    }

    /**
       @brief Compute the block "axpyz" with over the set of
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
    void axpyz(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
               std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block axpyz that allows us to call
       with any or all arguments being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3> void axpyz(const std::vector<double> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      axpyz(a, make_set(x), make_set(y), make_set(z));
    }

    /**
       @brief Compute the block "axpyz" with over the set of
       ColorSpinorFields.  E.g., it computes

       z = x * a + y

       Where 'a' is assumed to be upper triangular.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    void axpyz_U(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
                 std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block axpyz_U that allows us to call
       with any or all arguments being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3> void axpyz_U(const std::vector<double> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      axpyz_U(a, make_set(x), make_set(y), make_set(z));
    }

    /**
       @brief Compute the block "axpyz" with over the set of
       ColorSpinorFields.  E.g., it computes

       z = x * a + y

       Where 'a' is assumed to be lower triangular

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    void axpyz_L(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
                 std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block axpyz_L that allows us to call
       with any or all arguments being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3> void axpyz_L(const std::vector<double> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      axpyz_L(a, make_set(x), make_set(y), make_set(z));
    }

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
    void caxpyz(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
                std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block caxpyz that allows us to call
       with any or all arguments being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3> void caxpyz(const std::vector<Complex> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      caxpyz(a, make_set(x), make_set(y), make_set(z));
    }

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
    void caxpyz_U(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
                  std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block caxpyz_U that allows us to call
       with any or all arguments being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3>
    void caxpyz_U(const std::vector<Complex> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      caxpyz_U(a, make_set(x), make_set(y), make_set(z));
    }

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
    void caxpyz_L(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x,
                  std::vector<ColorSpinorField_ref> &&y, std::vector<ColorSpinorField_ref> &&z);

    /**
       @brief Wrapper function for block caxpyz_L that allows us to call
       with any or all arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2, typename T3>
    void caxpyz_L(const std::vector<Complex> &a, T1 &&x, T2 &&y, T3 &&z)
    {
      caxpyz_L(a, make_set(x), make_set(y), make_set(z));
    }

    /**
       @brief Compute the vectorized "axpyBzpcx" with over the set of
       ColorSpinorFields, where the third vector, z, is constant over the
       batch.  E.g., it computes

       y = a * x + y
       x = b * z + c * x

       The dimensions of a, b, c are the same as the size of x and y,
       with a maximum size of 16.

       @param a[in] Array of coefficients
       @param x[in,out] vector of ColorSpinorFields
       @param y[in,out] vector of ColorSpinorFields
       @param b[in] Array of coefficients
       @param z[in] input ColorSpinorField
       @param c[in] Array of coefficients
    */
    void axpyBzpcx(const std::vector<double> &a, std::vector<ColorSpinorField_ref> &&x,
                   std::vector<ColorSpinorField_ref> &&y, const std::vector<double> &b, ColorSpinorField &z,
                   const std::vector<double> &c);

    /**
       @brief Wrapper function for axpyBzpcx that allows us to call
       with either x or y, or both arguments, being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T1, typename T2>
    void axpyBzpcx(const std::vector<double> &a, T1 &&x, T2 &&y, const std::vector<double> &b, ColorSpinorField &z,
                   const std::vector<double> &c)
    {
      axpyBzpcx(a, make_set(x), make_set(y), b, z, c);
    }

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
    void caxpyBxpz(const std::vector<Complex> &a, std::vector<ColorSpinorField_ref> &&x, ColorSpinorField &y,
                   const std::vector<Complex> &b, ColorSpinorField &z);

    /**
       @brief Wrapper function for axpyBxpz that allows us to call
       with the x argument being std::vector<ColorSpinorField>.
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in] vector of input ColorSpinorFields
       @param z[out] vector of output ColorSpinorFields
    */
    template <typename T>
    void caxpyBxpz(const std::vector<Complex> &a, T &&x, ColorSpinorField &y, const std::vector<Complex> &b,
                   ColorSpinorField &z)
    {
      caxpyBxpz(a, make_set(x), y, b, z);
    }

    // multi-reduce kernels - defined in multi_reduce.cu

    /**
       @brief Computes the matrix of real inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void reDotProduct(std::vector<double> &result, std::vector<ColorSpinorField_ref> &&a,
                      std::vector<ColorSpinorField_ref> &&b);

    /**
       @brief Wrapper function for reDotProduct that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    template <typename T1, typename T2> void reDotProduct(std::vector<double> &result, T1 &&a, T2 &&b)
    {
      reDotProduct(result, make_set(a), make_set(b));
    }

    /**
       @brief Computes the matrix of inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void cDotProduct(std::vector<Complex> &result, std::vector<ColorSpinorField_ref> &&a,
                     std::vector<ColorSpinorField_ref> &&b);

    /**
       @brief Wrapper function for cDotProduct that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    template <typename T1, typename T2> void cDotProduct(std::vector<Complex> &result, T1 &&a, T2 &&b)
    {
      cDotProduct(result, make_set(a), make_set(b));
    }

    /**
       @brief Computes the matrix of inner products between the vector
       set a and the vector set b.  This routine is specifically for
       the case where the result matrix is guaranteed to be Hermitian.
       Requires a.size()==b.size().

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void hDotProduct(std::vector<Complex> &result, std::vector<ColorSpinorField_ref> &&a,
                     std::vector<ColorSpinorField_ref> &&b);

    /**
       @brief Wrapper function for hDotProduct that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    template <typename T1, typename T2> void hDotProduct(std::vector<Complex> &result, T1 &&a, T2 &&b)
    {
      hDotProduct(result, make_set(a), make_set(b));
    }

    /**
        @brief Computes the matrix of inner products between the vector
        set a and the vector set b.  This routine is specifically for
        the case where the result matrix is guaranteed to be Hermitian.
        Uniquely defined for cases like (p, Ap) where the output is Hermitian,
        but there's an A-norm instead of an L2 norm.
        Requires a.size()==b.size().

        @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
        @param a[in] set of input ColorSpinorFields
        @param b[in] set of input ColorSpinorFields
     */
    void hDotProduct_Anorm(std::vector<Complex> &result, std::vector<ColorSpinorField_ref> &&a,
                           std::vector<ColorSpinorField_ref> &&b);

    /**
       @brief Wrapper function for hDotProduct_Anorm that allows us to call
       with either, or both arguments, being std::vector<ColorSpinorField>.

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    template <typename T1, typename T2> void hDotProduct_Anorm(std::vector<Complex> &result, T1 &&a, T2 &&b)
    {
      hDotProduct_Anorm(result, make_set(a), make_set(b));
    }

    // compatibility wrappers until we switch to
    // std::vector<ColorSpinorField> and
    // std::vector<std::reference_wrapper<...>> more broadly

    void axpy(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy(const double *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpy_U(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy_U(const double *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpy_L(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy_L(const double *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy_U(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy_L(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpyz(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
               std::vector<ColorSpinorField *> &z);
    void axpyz(const double *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    void axpyz_U(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                 std::vector<ColorSpinorField *> &z);
    void axpyz_L(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                 std::vector<ColorSpinorField *> &z);
    void caxpyz(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                std::vector<ColorSpinorField *> &z);
    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z);
    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z);
    void axpyBzpcx(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                   const double *b, ColorSpinorField &z, const double *c);
    void caxpyBxpz(const Complex *a_, std::vector<ColorSpinorField *> &x_, ColorSpinorField &y_, const Complex *b_,
                   ColorSpinorField &z_);

    void reDotProduct(double *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);
    void cDotProduct(Complex *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);
    void hDotProduct(Complex *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);

  } // namespace blas

} // namespace quda
