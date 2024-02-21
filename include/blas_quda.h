#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

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

    inline void zero(cvector_ref<ColorSpinorField> &x)
    {
      for (auto i = 0u; i < x.size(); i++) x[i].zero();
    }

    inline void copy(cvector_ref<ColorSpinorField> &dst, cvector_ref<const ColorSpinorField> &src)
    {
      for (auto i = 0u; i < src.size(); i++) { dst[i].copy(src[i]); }
    }

    /**
       @brief Apply the operation y = a * x
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[out] y output vector set
    */
    void axy(cvector_ref<const double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Apply the rescale operation x = a * x
       @param[in] a scalar multiplier set
       @param[in] x set of vectors
    */
    inline void ax(cvector_ref<const double> &a, cvector_ref<ColorSpinorField> &x) { axy(a, x, x); }

    /**
       @brief Apply the operation z = a * x + b * y
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] b scalar multiplier set
       @param[in] y input vector set
       @param[out] z output vector set
    */
    void axpbyz(cvector_ref<const double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const double> &b,
                cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

    /**
       @brief Apply the operation y += x
       @param[in] x input vector set
       @param[in,out] y update vector set
    */
    inline void xpy(cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      axpbyz(1.0, x, 1.0, y, y);
    }

    /**
       @brief Apply the operation y -= x
       @param[in] x input vector set
       @param[in,out] y update vector set
    */
    inline void mxpy(cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      axpbyz(-1.0, x, 1.0, y, y);
    }

    /**
       @brief Apply the operation y += a * x
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in,out] y update vector set
    */
    inline void axpy(cvector_ref<const double> &a, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<ColorSpinorField> &y)
    {
      axpbyz(a, x, 1.0, y, y);
    }

    /**
       @brief Apply the operation y = a * x + b * y
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] b scalar multiplier set
       @param[in,out] y update vector set
    */
    inline void axpby(cvector_ref<const double> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const double> &b, cvector_ref<ColorSpinorField> &y)
    {
      axpbyz(a, x, b, y, y);
    }

    /**
       @brief Apply the operation y = x + a * y
       @param[in] x input vector set
       @param[in] a scalar multiplier set
       @param[in,out] y update vector set
    */
    inline void xpay(cvector_ref<const ColorSpinorField> &x, cvector_ref<const double> &a,
                     cvector_ref<ColorSpinorField> &y)
    {
      axpbyz(1.0, x, a, y, y);
    }

    /**
       @brief Apply the operation z = x + a * y
       @param[in] x input vector set
       @param[in] a scalar multiplier set
       @param[in] y update vector set
       @param[out] z output vector set
    */
    inline void xpayz(cvector_ref<const ColorSpinorField> &x, cvector_ref<const double> &a,
                      cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      axpbyz(1.0, x, a, y, z);
    }

    /**
       @brief Apply the operation y = a * x + y, x = z + b * x
       @param[in] a scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
       @param[in] z input vector set
       @param[in] b scalar multiplier set
    */
    void axpyZpbx(cvector_ref<const double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                  cvector_ref<const ColorSpinorField> &z, cvector_ref<const double> &b);

    /**
       @brief Apply the operation y = a * x + y, x = b * z + c * x
       @param[in] a scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
       @param[in] b scalar multiplier set
       @param[in] z input vector set
       @param[in] c scalar multiplier set
    */
    void axpyBzpcx(cvector_ref<const double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const double> &b, cvector_ref<const ColorSpinorField> &z, cvector_ref<const double> &c);

    /**
       @brief Apply the operation w = a * x + b * y + c * z
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] b scalar multiplier set
       @param[in] y input vector set
       @param[in] c scalar multiplier set
       @param[in] z input vector set
       @param[out] w output vector set
    */
    void axpbypczw(cvector_ref<const double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const double> &b,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<const double> &c,
                   cvector_ref<const ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w);

    /**
       @brief Apply the operation y = a * x + b * y
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] b scalar multiplier set
       @param[in] y input vector set
       @param[out] z output vector set
    */
    void caxpby(cvector_ref<const Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const Complex> &b,
                cvector_ref<ColorSpinorField> &y);

    /**
       @brief Apply the operation y += a * x
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] y update vector set
    */
    void caxpy(cvector_ref<const Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Apply the operation z = x + a * y + b * z
       @param[in] x input vector set
       @param[in] a scalar multiplier set
       @param[in] y input vector set
       @param[in] b scalar multiplier set
       @param[in,out] z update vector set
    */
    void cxpaypbz(cvector_ref<const ColorSpinorField> &x, cvector_ref<const Complex> &a,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<const Complex> &b,
                  cvector_ref<ColorSpinorField> &z);

    /**
       @brief Apply the operation z += a * x + b * y, y-= b * w
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in] b scalar multiplier set
       @param[in,out] y update vector set
       @param[in,out] z update vector set
       @param[in] w input vector set
    */
    void caxpbypzYmbw(cvector_ref<const Complex> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const Complex> &b, cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                      cvector_ref<const ColorSpinorField> &w);

    /**
       @brief Apply the operation y += a * x, x += b * z
       @param[in] a scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
       @param[in] b scalar multiplier set
       @param[in] z input vector set
    */
    void caxpyBzpx(cvector_ref<const Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const Complex> &b, cvector_ref<const ColorSpinorField> &z);

    /**
       @brief Apply the operation y += a * x, z += b * x
       @param[in] a scalar multiplier set
       @param[in] x input vector set
       @param[in,out] y update vector set
       @param[in] b scalar multiplier set
       @param[in,out] z update vector set
    */
    void caxpyBxpz(cvector_ref<const Complex> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y, cvector_ref<const Complex> &b, cvector_ref<ColorSpinorField> &z);

    /**
       @brief Apply the operation y += a * b * x, x = a * x
       @param[in] a real scalar multiplier set
       @param[in] b complex scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
    */
    void cabxpyAx(cvector_ref<const double> &a, cvector_ref<const Complex> &b, cvector_ref<ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y);

    /**
       @brief Apply the operation y += a * x, x -= a * z
       @param[in] a scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
       @param[in] z input vector set
    */
    void caxpyXmaz(cvector_ref<const Complex> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const ColorSpinorField> &z);

    /**
       @brief Apply the operation y += a * x, x = x - a * z.  Special
       variant employed by MR solver where the scalar multiplier is
       computed on the fly from device memory.
       @param[in] a real scalar multiplier set
       @param[in,out] x update vector set
       @param[in,out] y update vector set
       @param[in] z input vector set
    */
    void caxpyXmazMR(cvector_ref<const double> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z);

    /**
       @brief Apply the operation y += a * w, z -= a * x, w = z + b * w
       @param[in] a scalar multiplier set
       @param[in] b scalar multiplier set
       @param[in] x input vector set
       @param[in,out] y update vector set
       @param[in,out] z update vector set
       @param[in,out] w update vector set
    */
    void tripleCGUpdate(cvector_ref<const double> &a, cvector_ref<const double> &b,
                        cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                        cvector_ref<ColorSpinorField> &z, cvector_ref<ColorSpinorField> &w);

    // reduction kernels - defined in reduce_quda.cu

    /**
       @brief Compute the maximum absolute real element of a field
       @param[in] a The field we are reducing
    */
    double max(const ColorSpinorField &x);

    /**
       @brief Compute the maximum real-valued deviation between two
       fields.
       @param[in] x The field we want to compare
       @param[in] y The reference field to which we are comparing against
    */
    array<double, 2> max_deviation(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Compute the L1 norm of a field
       @param[in] x The field we are reducing
    */
    double norm1(const ColorSpinorField &x);

    /**
       @brief Compute the L2 norm (||x||^2) of a field
       @param[in] x The field we are reducing
    */
    double norm2(const ColorSpinorField &x);

    /**
       @brief Compute y += a * x and then (x, y)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    double axpyReDot(double a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the real-valued inner product (x, y)
       @param[in] x input vector
       @param[in] y input vector
    */
    double reDotProduct(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Compute z = a * x + b * y and then ||z||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in] y input vector
       @param[in,out] z update vector
    */
    double axpbyzNorm(double a, const ColorSpinorField &x, double b, const ColorSpinorField &y, ColorSpinorField &z);

    /**
       @brief Compute y += a * x and then ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline double axpyNorm(double a, const ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(a, x, 1.0, y, y); }

    /**
       @brief Compute the complex-valued inner product (x, y)
       @param[in] x input vector
       @param[in] y input vector
    */
    Complex cDotProduct(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Return complex-valued inner product (x,y), ||x||^2 and ||y||^2
       @param[in] x input vector
       @param[in] y input vector
    */
    double4 cDotProductNormAB(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Return complex-valued inner product (x,y) and ||x||^2
       @param[in] x input vector
       @param[in] y input vector
     */
    inline double3 cDotProductNormA(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto a4 = cDotProductNormAB(x, y);
      return make_double3(a4.x, a4.y, a4.z);
    }

    /**
       @brief Return complex-valued inner product (x,y) and ||y||^2
       @param[in] x input vector
       @param[in] y input vector
     */
    inline double3 cDotProductNormB(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto a4 = cDotProductNormAB(x, y);
      return make_double3(a4.x, a4.y, a4.w);
    }

    /**
       @brief Apply the operation z += a * x + b * y, y -= b * w,
       compute complex-valued inner product (u, y) and ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in] w input vector
       @param[in] v input vector
    */
    double3 caxpbypzYmbwcDotProductUYNormY(const Complex &a, const ColorSpinorField &x, const Complex &b,
                                           ColorSpinorField &y, ColorSpinorField &z,
                                           const ColorSpinorField &w, const ColorSpinorField &u);

    /**
       @brief Compute y = a * x + b * y and then ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in,out] y update vector
    */
    double caxpbyNorm(const Complex &a, const ColorSpinorField &x, const Complex &b, ColorSpinorField &y);

    /**
       @brief Compute y += a * x and then ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline double caxpyNorm(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y)
    {
      return caxpbyNorm(a, x, 1.0, y);
    }

    /**
       @brief Compute y -= x and then ||y||^2
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline double xmyNorm(const ColorSpinorField &x, ColorSpinorField &y) { return caxpbyNorm(1.0, x, -1.0, y); }

    /**
       @brief Compute z = a * b * x + y, x = a * x, and then ||x||^2
       @param[in] a scalar multiplier
       @param[in] b scalar multiplier
       @param[in,out] x update vector
       @param[in] y input vector
       @param[in,out] z update vector
    */
    double cabxpyzAxNorm(double a, const Complex &b, ColorSpinorField &x, const ColorSpinorField &y,
                         ColorSpinorField &z);

    /**
       @brief Compute y += a * x and the resulting complex-valued inner product (z, y)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in] z input vector
    */
    Complex caxpyDotzy(const Complex &a, const ColorSpinorField &x, ColorSpinorField &y,
		       const ColorSpinorField &z);

    /**
       @brief Compute y += a * x and then compute ||y||^2 and
       real-valued inner product (y_out, y_out-y_in)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    double2 axpyCGNorm(double a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Computes ||x||^2, ||r||^2 and the MILC/FNAL heavy quark
       residual norm
       @param[in] x input vector
       @param[in] r input vector (residual vector)
    */
    double3 HeavyQuarkResidualNorm(const ColorSpinorField &x, const ColorSpinorField &r);

    /**
       @brief Computes y += x, ||y||^2, ||r||^2 and the MILC/FNAL heavy quark
       residual norm
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in] r input vector (residual vector)
    */
    double3 xpyHeavyQuarkResidualNorm(const ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &r);

    /**
       @brief Computes ||x||^2, ||y||^2, and real-valued inner product (y, z)
       @param[in] x input vector
       @param[in] y input vector
       @param[in] z input vector
    */
    double3 tripleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Computes ||x||^2, ||y||^2, the real-valued inner product (y, z), and ||z||^2
       @param[in] x input vector
       @param[in] y input vector
       @param[in] z input vector
    */
    double4 quadrupleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Computes z = x, w = y, x += a * y, y -= a * v and ||y||^2
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in,out] w update vector
       @param[in] v input vector
    */
    double quadrupleCG3InitNorm(double a, ColorSpinorField &x, ColorSpinorField &y,
                                ColorSpinorField &z, ColorSpinorField &w, const ColorSpinorField &v);

    /**
       @brief Computes x = b * (x + a * y) + ( 1 - b) * z,
       y = b * (y + a * v) + (1 - b) * w, z = x_in, w = y_in, and
       ||y||^2
       @param[in] a scalar multiplier
       @param[in] b scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in,out] w update vector
       @param[in] v input vector
    */
    double quadrupleCG3UpdateNorm(double a, double b, ColorSpinorField &x, ColorSpinorField &y,
                                  ColorSpinorField &z, ColorSpinorField &w, const ColorSpinorField &v);

    // multi-blas kernels - defined in multi_blas.cu

    /**
       @brief Compute the block "axpy" with over the set of
              ColorSpinorFields.  E.g., it computes y = x * a + y
              The dimensions of a can be rectangular, e.g., the width of x and y need not be same.
       @tparam T The type of a coefficients (double or Complex)
       @param a[in] Matrix of real coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T>
    void axpy(const std::vector<T> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "axpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @tparam T The type of a coefficients (double or Complex)
       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T>
    void axpy_U(const std::vector<T> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "axpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @tparam T The type of a coefficients (double or Complex)
       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    template <typename T>
    void axpy_L(const std::vector<T> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

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
    void caxpy(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "caxpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_U(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "caxpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_L(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

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
    void axpyz(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x,
               cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void axpyz_U(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x,
                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void axpyz_L(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x,
                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void caxpyz(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void caxpyz_U(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void caxpyz_L(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z);

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
    void axpyBzpcx(const std::vector<double> &a, cvector_ref<ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y, const std::vector<double> &b, ColorSpinorField &z,
                   const std::vector<double> &c);

    /**
       @brief Compute the vectorized "caxpyBxpz" over the set of
       ColorSpinorFields, where the second and third vector, y and z, is constant over the
       batch.  E.g., it computes

       y = a * x + y
       z = b * x + z

       The dimensions of a, b are the same as the size of x,
       with a maximum size of 16.

       @param a[in] Array of coefficients
       @param x[in] vector of ColorSpinorFields
       @param y[in,out] input ColorSpinorField
       @param b[in] Array of coefficients
       @param z[in,out] input ColorSpinorField
    */
    void caxpyBxpz(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, ColorSpinorField &y,
                   const std::vector<Complex> &b, ColorSpinorField &z);

    // multi-reduce kernels - defined in multi_reduce.cu

    /**
       @brief Computes the matrix of real inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void reDotProduct(std::vector<double> &result, cvector_ref<const ColorSpinorField> &a,
                      cvector_ref<const ColorSpinorField> &b);

    /**
       @brief Computes the matrix of inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void cDotProduct(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &a,
                     cvector_ref<const ColorSpinorField> &b);

    /**
       @brief Computes the matrix of inner products between the vector
       set a and the vector set b.  This routine is specifically for
       the case where the result matrix is guaranteed to be Hermitian.
       Requires a.size()==b.size().

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void hDotProduct(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &a,
                     cvector_ref<const ColorSpinorField> &b);

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
    void hDotProduct_Anorm(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &a,
                           cvector_ref<const ColorSpinorField> &b);

    // compatibility wrappers until we switch to
    // std::vector<ColorSpinorField> and
    // std::vector<std::reference_wrapper<...>> more broadly

    namespace legacy
    {
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
    } // namespace legacy
  } // namespace blas

} // namespace quda

#endif // _QUDA_BLAS_H
