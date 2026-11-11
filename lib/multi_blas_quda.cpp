#include <cstring>
#include "reduce_store.hpp"
#include "multi_blas_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      template <>
      void axpy<real_t>(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                        cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_t<store_t>(a, x, y); });
      }

      template <>
      void axpy_U<real_t>(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_U_t<store_t>(a, x, y); });
      }

      template <>
      void axpy_L<real_t>(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                          cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_L_t<store_t>(a, x, y); });
      }

      template <>
      void axpy<complex_t>(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                           cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_t<store_t>(a, x, y); });
      }

      void caxpy(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                 cvector_ref<ColorSpinorField> &y)
      {
        axpy(a, std::move(x), std::move(y));
      }

      template <>
      void axpy_U<complex_t>(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                             cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_U_t<store_t>(a, x, y); });
      }

      void caxpy_U(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y)
      {
        axpy_U(a, std::move(x), std::move(y));
      }

      template <>
      void axpy_L<complex_t>(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                             cvector_ref<ColorSpinorField> &y)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpy_L_t<store_t>(a, x, y); });
      }

      void caxpy_L(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<ColorSpinorField> &y)
      {
        axpy_L(a, std::move(x), std::move(y));
      }

      void axpyz(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                 cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpyz_t<store_t>(a, x, y, z); });
      }

      void axpyz_U(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpyz_U_t<store_t>(a, x, y, z); });
      }

      void axpyz_L(const std::vector<real_t> &a, cvector_ref<const ColorSpinorField> &x,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { axpyz_L_t<store_t>(a, x, y, z); });
      }

      void caxpyz(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                  cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { caxpyz_t<store_t>(a, x, y, z); });
      }

      void caxpyz_U(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x,
                    cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { caxpyz_U_t<store_t>(a, x, y, z); });
      }

      void axpyBzpcx(const std::vector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     const std::vector<real_t> &b, ColorSpinorField &z, const std::vector<real_t> &c)
      {
        // instantiate<> uses z as the X store (swizzled into x[0]).
        dispatch_reduce_store(z, [&]<typename store_t>() { axpyBzpcx_t<store_t>(a, x, y, b, z, c); });
      }

      void caxpyBxpz(const std::vector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, ColorSpinorField &y,
                     const std::vector<complex_t> &b, ColorSpinorField &z)
      {
        dispatch_reduce_store(x, [&]<typename store_t>() { caxpyBxpz_t<store_t>(a, x, y, b, z); });
      }

      // temporary wrappers
      void axpy(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<real_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(real_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        axpy(a_, x_, y_);
      }

      void axpy_U(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<real_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(real_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        axpy_U(a_, x_, y_);
      }

      void axpy_L(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<real_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(real_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        axpy_L(a_, x_, y_);
      }

    } // namespace block

    namespace legacy
    {
      void caxpy(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<complex_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(complex_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::caxpy(a_, x_, y_);
      }

      void caxpy_U(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<complex_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(complex_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::caxpy_U(a_, x_, y_);
      }

      void caxpy_L(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<complex_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(complex_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::caxpy_L(a_, x_, y_);
      }

      void axpyz(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                 std::vector<ColorSpinorField *> &z)
      {
        std::vector<real_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(real_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<const ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        vector_ref<ColorSpinorField> z_;
        for (auto &zi : z) z_.push_back(*zi);
        blas::block::axpyz(a_, x_, y_, z_);
      }

      void caxpyz(const complex_t *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z)
      {
        std::vector<complex_t> a_(x.size() * y.size());
        memcpy(a_.data(), a, x.size() * y.size() * sizeof(complex_t));
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<const ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        vector_ref<ColorSpinorField> z_;
        for (auto &zi : z) z_.push_back(*zi);
        blas::block::caxpyz(a_, x_, y_, z_);
      }

      void axpyBzpcx(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                     const double *b, ColorSpinorField &z, const double *c)
      {
        std::vector<real_t> a_(x.size());
        memcpy(a_.data(), a, x.size() * sizeof(real_t));
        std::vector<real_t> b_(x.size());
        memcpy(b_.data(), b, x.size() * sizeof(real_t));
        std::vector<real_t> c_(x.size());
        memcpy(c_.data(), c, x.size() * sizeof(real_t));

        vector_ref<ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::axpyBzpcx(a_, x_, y_, b_, z, c_);
      }

      void caxpyBxpz(const complex_t *a, std::vector<ColorSpinorField *> &x, ColorSpinorField &y, const complex_t *b,
                     ColorSpinorField &z)
      {
        std::vector<complex_t> a_(x.size());
        memcpy(a_.data(), a, x.size() * sizeof(complex_t));
        std::vector<complex_t> b_(x.size());
        memcpy(b_.data(), b, x.size() * sizeof(complex_t));

        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        blas::block::caxpyBxpz(a_, x_, y, b_, z);
      }

    } // namespace legacy

  } // namespace blas

} // namespace quda
