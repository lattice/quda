#include <cstring>
#include "reduce_store.hpp"
#include "multi_reduce_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    namespace block
    {

      void reDotProduct(std::vector<real_t> &result, cvector_ref<const ColorSpinorField> &x,
                        cvector_ref<const ColorSpinorField> &y)
      {
        dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { reDotProduct_t<store_t>(result, x, y); });
      }

      void cDotProduct(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                       cvector_ref<const ColorSpinorField> &y)
      {
        dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { cDotProduct_t<store_t>(result, x, y); });
      }

      void hDotProduct(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                       cvector_ref<const ColorSpinorField> &y)
      {
        dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { hDotProduct_t<store_t>(result, x, y); });
      }

      void hDotProduct_Anorm(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                             cvector_ref<const ColorSpinorField> &y)
      {
        dispatch_reduce_prec(x.Precision(), [&]<typename store_t>() { hDotProduct_Anorm_t<store_t>(result, x, y); });
      }

    } // namespace block

    namespace legacy
    {

      void reDotProduct(real_t *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<real_t> result_(x.size() * y.size());
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<const ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::reDotProduct(result_, std::move(x_), std::move(y_));
        memcpy(result, result_.data(), x.size() * y.size() * sizeof(real_t));
      }

      void cDotProduct(complex_t *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<complex_t> result_(x.size() * y.size());
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<const ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::cDotProduct(result_, std::move(x_), std::move(y_));
        memcpy(result, result_.data(), x.size() * y.size() * sizeof(complex_t));
      }

      void hDotProduct(complex_t *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
      {
        std::vector<complex_t> result_(x.size() * y.size());
        vector_ref<const ColorSpinorField> x_;
        for (auto &xi : x) x_.push_back(*xi);
        vector_ref<const ColorSpinorField> y_;
        for (auto &yi : y) y_.push_back(*yi);
        blas::block::hDotProduct(result_, std::move(x_), std::move(y_));
        memcpy(result, result_.data(), x.size() * y.size() * sizeof(complex_t));
      }

    } // namespace legacy

  } // namespace blas

} // namespace quda
