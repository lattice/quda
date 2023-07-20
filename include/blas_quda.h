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

    extern unsigned long long flops;
    extern unsigned long long bytes;

    inline void zero(cvector_ref<ColorSpinorField> &x)
    {
      for (auto i = 0u; i < x.size(); i++) x[i].zero();
    }

    inline void copy(ColorSpinorField &dst, const ColorSpinorField &src)
    {
      if (dst.data() == src.data()) {
        // check the fields are equivalent else error
        if (ColorSpinorField::are_compatible(dst, src))
          return;
        else
          errorQuda("Aliasing pointers with incompatible fields");
      }
      dst.copy(src);
    }

    /**
       @brief Apply the operation y = a * x
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[out] y output vector
    */
    void axy(real_t a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Apply the rescale operation x = a * x
       @param[in] a scalar multiplier
       @param[in] x input vector
    */
    inline void ax(real_t a, ColorSpinorField &x) { axy(a, x, x); }

    /**
       @brief Apply the operation z = a * x + b * y
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in] y input vector
       @param[out] z output vector
    */
    void axpbyz(real_t a, const ColorSpinorField &x, real_t b, const ColorSpinorField &y, ColorSpinorField &z);

    /**
       @brief Apply the operation y += x
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline void xpy(const ColorSpinorField &x, ColorSpinorField &y) { axpbyz(1.0, x, 1.0, y, y); }

    /**
       @brief Apply the operation y -= x
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline void mxpy(const ColorSpinorField &x, ColorSpinorField &y) { axpbyz(-1.0, x, 1.0, y, y); }

    /**
       @brief Apply the operation y += a * x
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline void axpy(real_t a, const ColorSpinorField &x, ColorSpinorField &y) { axpbyz(a, x, 1.0, y, y); }

    /**
       @brief Apply the operation y = a * x + b * y
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in,out] y update vector
    */
    inline void axpby(real_t a, const ColorSpinorField &x, real_t b, ColorSpinorField &y) { axpbyz(a, x, b, y, y); }

    /**
       @brief Apply the operation y = x + a * y
       @param[in] x input vector
       @param[in] a scalar multiplier
       @param[in,out] y update vector
    */
    inline void xpay(const ColorSpinorField &x, real_t a, ColorSpinorField &y) { axpbyz(1.0, x, a, y, y); }

    /**
       @brief Apply the operation z = x + a * y
       @param[in] x input vector
       @param[in] a scalar multiplier
       @param[in] y update vector
       @param[out] z output vector
    */
    inline void xpayz(const ColorSpinorField &x, real_t a, const ColorSpinorField &y, ColorSpinorField &z) { axpbyz(1.0, x, a, y, z); }

    /**
       @brief Apply the operation y = a * x + y, x = z + b * x
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in] z input vector
       @param[in] b scalar multiplier
    */
    void axpyZpbx(real_t a, ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z, real_t b);

    /**
       @brief Apply the operation y = a * x + y, x = b * z + c * x
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in] b scalar multiplier
       @param[in] z input vector
       @param[in] c scalar multiplier
    */
    void axpyBzpcx(real_t a, ColorSpinorField& x, ColorSpinorField& y, real_t b, const ColorSpinorField& z, real_t c);

    /**
       @brief Apply the operation w = a * x + b * y + c * z
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in] y input vector
       @param[in] c scalar multiplier
       @param[in] z input vector
       @param[out] w output vector
    */
    void axpbypczw(real_t a, const ColorSpinorField &x, real_t b, const ColorSpinorField &y,
                   real_t c, const ColorSpinorField &z, ColorSpinorField &w);

    /**
       @brief Apply the operation y = a * x + b * y
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in] y input vector
       @param[out] z output vector
    */
    void caxpby(const complex_t &a, const ColorSpinorField &x, const complex_t &b, ColorSpinorField &y);

    /**
       @brief Apply the operation y += a * x
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] y update vector
    */
    void caxpy(const complex_t &a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Apply the operation z = x + a * y + b * z
       @param[in] x input vector
       @param[in] a scalar multiplier
       @param[in] y input vector
       @param[in] b scalar multiplier
       @param[in,out] z update vector
    */
    void cxpaypbz(const ColorSpinorField &x, const complex_t &a, const ColorSpinorField &y,
                  const complex_t &b, ColorSpinorField &z);

    /**
       @brief Apply the operation z += a * x + b * y, y-= b * w
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in] w input vector
    */
    void caxpbypzYmbw(const complex_t &a, const ColorSpinorField &x, const complex_t &b, ColorSpinorField &y,
                      ColorSpinorField &z, const ColorSpinorField &w);

    /**
       @brief Apply the operation y += a * x, x += b * z
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in] b scalar multiplier
       @param[in] z input vector
    */
    void caxpyBzpx(const complex_t &a, ColorSpinorField &x, ColorSpinorField &y,
                   const complex_t &b, const ColorSpinorField &z);

    /**
       @brief Apply the operation y += a * x, z += b * x
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in] b scalar multiplier
       @param[in,out] z update vector
    */
    void caxpyBxpz(const complex_t &a, const ColorSpinorField &x, ColorSpinorField &y,
                   const complex_t &b, ColorSpinorField &z);

    /**
       @brief Apply the operation y += a * b * x, x = a * x
       @param[in] a real scalar multiplier
       @param[in] b complex scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
    */
    void cabxpyAx(real_t a, const complex_t &b, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Apply the operation y += a * x, x -= a * z
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in] z input vector
    */
    void caxpyXmaz(const complex_t &a, ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Apply the operation y += a * x, x = x - a * z.  Special
       variant employed by MR solver where the scalar multiplier is
       computed on the fly from device memory.
       @param[in] a real scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in] z input vector
    */
    void caxpyXmazMR(const real_t &a, ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Apply the operation y += a * w, z -= a * x, w = z + b * w
       @param[in] a scalar multiplier
       @param[in] b scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in,out] w update vector
    */
    void tripleCGUpdate(real_t a, real_t b, const ColorSpinorField &x,
			ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w);

    // reduction kernels - defined in reduce_quda.cu

    /**
       @brief Compute the maximum absolute real element of a field
       @param[in] a The field we are reducing
    */
    real_t max(const ColorSpinorField &x);

    /**
       @brief Compute the maximum real-valued deviation between two
       fields.
       @param[in] x The field we want to compare
       @param[in] y The reference field to which we are comparing against
    */
    array<real_t, 2> max_deviation(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Compute the L1 norm of a field
       @param[in] x The field we are reducing
    */
    real_t norm1(const ColorSpinorField &x);

    /**
       @brief Compute the L2 norm (||x||^2) of a field
       @param[in] x The field we are reducing
    */
    real_t norm2(const ColorSpinorField &x);

    /**
       @brief Compute y += a * x and then (x, y)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    real_t axpyReDot(real_t a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute the real-valued inner product (x, y)
       @param[in] x input vector
       @param[in] y input vector
    */
    real_t reDotProduct(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Compute z = a * x + b * y and then ||z||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in] b scalar multiplier
       @param[in] y input vector
       @param[in,out] z update vector
    */
    real_t axpbyzNorm(real_t a, const ColorSpinorField &x, real_t b, const ColorSpinorField &y, ColorSpinorField &z);

    /**
       @brief Compute y += a * x and then ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline real_t axpyNorm(real_t a, const ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(a, x, 1.0, y, y); }

    /**
       @brief Compute y -= x and then ||y||^2
       @param[in] x input vector
       @param[in,out] y update vector
    */
    inline real_t xmyNorm(const ColorSpinorField &x, ColorSpinorField &y) { return axpbyzNorm(1.0, x, -1.0, y, y); }

    /**
       @brief Compute the complex-valued inner product (x, y)
       @param[in] x input vector
       @param[in] y input vector
    */
    complex_t cDotProduct(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Return complex-valued inner product (x,y), ||x||^2 and ||y||^2
       @param[in] x input vector
       @param[in] y input vector
    */
    array<real_t, 4> cDotProductNormAB(const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Return complex-valued inner product (x,y) and ||x||^2
       @param[in] x input vector
       @param[in] y input vector
     */
    inline array<real_t, 3> cDotProductNormA(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto a4 = cDotProductNormAB(x, y);
      return {a4[0], a4[1], a4[2]};
    }

    /**
       @brief Return complex-valued inner product (x,y) and ||y||^2
       @param[in] x input vector
       @param[in] y input vector
     */
    inline array<real_t, 3> cDotProductNormB(const ColorSpinorField &x, const ColorSpinorField &y)
    {
      auto a4 = cDotProductNormAB(x, y);
      return {a4[0], a4[1], a4[2]};
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
    array<real_t, 3> caxpbypzYmbwcDotProductUYNormY(const complex_t &a, const ColorSpinorField &x, const complex_t &b,
                                           ColorSpinorField &y, ColorSpinorField &z,
                                           const ColorSpinorField &w, const ColorSpinorField &u);

    /**
       @brief Compute y += a * x and then ||y||^2
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    real_t caxpyNorm(const complex_t &a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute z = a * b * x + y, x = a * x, and then ||x||^2
       @param[in] a scalar multiplier
       @param[in] b scalar multiplier
       @param[in,out] x update vector
       @param[in] y input vector
       @param[in,out] z update vector
    */
    real_t cabxpyzAxNorm(real_t a, const complex_t &b, ColorSpinorField &x, const ColorSpinorField &y,
                         ColorSpinorField &z);

    /**
       @brief Compute y += a * x and the resulting complex-valued inner product (z, y)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in] z input vector
    */
    complex_t caxpyDotzy(const complex_t &a, const ColorSpinorField &x, ColorSpinorField &y,
		       const ColorSpinorField &z);

    /**
       @brief Compute y += a * x and then compute ||y||^2 and
       real-valued inner product (y_out, y_out-y_in)
       @param[in] a scalar multiplier
       @param[in] x input vector
       @param[in,out] y update vector
    */
    array<real_t, 2> axpyCGNorm(real_t a, const ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Computes ||x||^2, ||r||^2 and the MILC/FNAL heavy quark
       residual norm
       @param[in] x input vector
       @param[in] r input vector (residual vector)
    */
    array<real_t, 3> HeavyQuarkResidualNorm(const ColorSpinorField &x, const ColorSpinorField &r);

    /**
       @brief Computes y += x, ||y||^2, ||r||^2 and the MILC/FNAL heavy quark
       residual norm
       @param[in] x input vector
       @param[in,out] y update vector
       @param[in] r input vector (residual vector)
    */
    array<real_t, 3> xpyHeavyQuarkResidualNorm(const ColorSpinorField &x, ColorSpinorField &y, const ColorSpinorField &r);

    /**
       @brief Computes ||x||^2, ||y||^2, and real-valued inner product (y, z)
       @param[in] x input vector
       @param[in] y input vector
       @param[in] z input vector
    */
    array<real_t, 3> tripleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Computes ||x||^2, ||y||^2, the real-valued inner product (y, z), and ||z||^2
       @param[in] x input vector
       @param[in] y input vector
       @param[in] z input vector
    */
    array<real_t, 4> quadrupleCGReduction(const ColorSpinorField &x, const ColorSpinorField &y, const ColorSpinorField &z);

    /**
       @brief Computes z = x, w = y, x += a * y, y -= a * v and ||y||^2
       @param[in] a scalar multiplier
       @param[in,out] x update vector
       @param[in,out] y update vector
       @param[in,out] z update vector
       @param[in,out] w update vector
       @param[in] v input vector
    */
    real_t quadrupleCG3InitNorm(real_t a, ColorSpinorField &x, ColorSpinorField &y,
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
    real_t quadrupleCG3UpdateNorm(real_t a, real_t b, ColorSpinorField &x, ColorSpinorField &y,
                                  ColorSpinorField &z, ColorSpinorField &w, const ColorSpinorField &v);

    // multi-blas kernels - defined in multi_blas.cu

    /**
       @brief Compute the block "axpy" with over the set of
              ColorSpinorFields.  E.g., it computes y = x * a + y
              The dimensions of a can be rectangular, e.g., the width of x and y need not be same.
       @tparam T The type of a coefficients (real_t or complex_t)
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

       @tparam T The type of a coefficients (real_t or complex_t)
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

       @tparam T The type of a coefficients (real_t or complex_t)
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
    void caxpy(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "caxpy_U" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, upper triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_U(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

    /**
       @brief Compute the block "caxpy_L" with over the set of
       ColorSpinorFields.  E.g., it computes

       y = x * a + y

       Where 'a' must be a square, lower triangular matrix.

       @param a[in] Matrix of coefficients
       @param x[in] vector of input ColorSpinorFields
       @param y[in,out] vector of input/output ColorSpinorFields
    */
    void caxpy_L(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y);

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
    void axpyz(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void axpyz_U(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void axpyz_L(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void caxpyz(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void caxpyz_U(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void caxpyz_L(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
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
    void axpyBzpcx(const std::vector<real_t> &a, cvector_ref<ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y, const std::vector<real_t> &b, ColorSpinorField &z,
                   const std::vector<real_t> &c);

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
    void caxpyBxpz(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, ColorSpinorField &y,
                   const std::vector<complex_t> &b, ColorSpinorField &z);

    // multi-reduce kernels - defined in multi_reduce.cu

    /**
       @brief Computes the matrix of real inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void reDotProduct(std::vector<real_t> &result, cvector_ref<const ColorSpinorField> &a,
                      cvector_ref<const ColorSpinorField> &b);

    /**
       @brief Computes the matrix of inner products between the vector set a and the vector set b

       @param result[out] Matrix of inner product result[i][j] = (a[j],b[i])
       @param a[in] set of input ColorSpinorFields
       @param b[in] set of input ColorSpinorFields
    */
    void cDotProduct(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &a,
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
    void hDotProduct(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &a,
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
    void hDotProduct_Anorm(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &a,
                           cvector_ref<const ColorSpinorField> &b);

    // compatibility wrappers until we switch to
    // std::vector<ColorSpinorField> and
    // std::vector<std::reference_wrapper<...>> more broadly

    void axpy(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy(const real_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpy_U(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy_U(const real_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpy_L(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void axpy_L(const real_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy(const complex_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy_U(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy_U(const complex_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void caxpy_L(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y);
    void caxpy_L(const complex_t *a, ColorSpinorField &x, ColorSpinorField &y);
    void axpyz(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
               std::vector<ColorSpinorField *> &z);
    void axpyz(const real_t *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    void axpyz_U(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                 std::vector<ColorSpinorField *> &z);
    void axpyz_L(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                 std::vector<ColorSpinorField *> &z);
    void caxpyz(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                std::vector<ColorSpinorField *> &z);
    void caxpyz(const complex_t *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z);
    void caxpyz_U(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z);
    void caxpyz_L(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z);
    void axpyBzpcx(const real_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                   const real_t *b, ColorSpinorField &z, const real_t *c);
    void caxpyBxpz(const complex_t *a_, std::vector<ColorSpinorField *> &x_, ColorSpinorField &y_, const complex_t *b_,
                   ColorSpinorField &z_);

    void reDotProduct(real_t *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);
    void cDotProduct(complex_t *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);
    void hDotProduct(complex_t *result, std::vector<ColorSpinorField *> &a, std::vector<ColorSpinorField *> &b);

  } // namespace blas

} // namespace quda

#endif // _QUDA_BLAS_H
