#include "reduce_store.hpp"
#include "blas_quda_decl.hpp"

namespace quda
{

  namespace blas
  {

    void axpbyz(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { axpbyz_t<store_t>(a, x, b, y, z); });
    }

    void axy(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { axy_t<store_t>(a, x, y); });
    }

    void caxpyz(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y,
                cvector_ref<ColorSpinorField> &z)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpyz_t<store_t>(a, x, y, z); });
    }

    void caxpby(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                cvector_ref<ColorSpinorField> &y)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpby_t<store_t>(a, x, b, y); });
    }

    void axpbypczw(cvector<real_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<real_t> &b,
                   cvector_ref<const ColorSpinorField> &y, cvector<real_t> &c, cvector_ref<const ColorSpinorField> &z,
                   cvector_ref<ColorSpinorField> &w)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { axpbypczw_t<store_t>(a, x, b, y, c, z, w); });
    }

    void caxpbypzw(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                   cvector_ref<const ColorSpinorField> &y, cvector_ref<const ColorSpinorField> &z,
                   cvector_ref<ColorSpinorField> &w)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpbypzw_t<store_t>(a, x, b, y, z, w); });
    }

    void axpyBzpcx(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<real_t> &b, cvector_ref<const ColorSpinorField> &z, cvector<real_t> &c)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { axpyBzpcx_t<store_t>(a, x, y, b, z, c); });
    }

    void axpyZpbx(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                  cvector_ref<const ColorSpinorField> &z, cvector<real_t> &b)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { axpyZpbx_t<store_t>(a, x, y, z, b); });
    }

    void caxpyBzpx(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<complex_t> &b, cvector_ref<const ColorSpinorField> &z)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpyBzpx_t<store_t>(a, x, y, b, z); });
    }

    void caxpyBxpz(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector<complex_t> &b, cvector_ref<ColorSpinorField> &z)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpyBxpz_t<store_t>(a, x, y, b, z); });
    }

    void caxpbypzYmbw(cvector<complex_t> &a, cvector_ref<const ColorSpinorField> &x, cvector<complex_t> &b,
                      cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                      cvector_ref<const ColorSpinorField> &w)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpbypzYmbw_t<store_t>(a, x, b, y, z, w); });
    }

    void cabxpyAx(cvector<real_t> &a, cvector<complex_t> &b, cvector_ref<ColorSpinorField> &x,
                  cvector_ref<ColorSpinorField> &y)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { cabxpyAx_t<store_t>(a, b, x, y); });
    }

    void caxpyXmaz(cvector<complex_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                   cvector_ref<const ColorSpinorField> &z)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpyXmaz_t<store_t>(a, x, y, z); });
    }

    void caxpyXmazMR(cvector<real_t> &a, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y,
                     cvector_ref<const ColorSpinorField> &z)
    {
      if (!commAsyncReduction()) errorQuda("This kernel requires asynchronous reductions to be set");
      if (x.Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("This kernel cannot be run on CPU fields");
      dispatch_reduce_store(x, [&]<typename store_t>() { caxpyXmazMR_t<store_t>(a, x, y, z); });
    }

    void tripleCGUpdate(cvector<real_t> &a, cvector<real_t> &b, cvector_ref<const ColorSpinorField> &x,
                        cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                        cvector_ref<ColorSpinorField> &w)
    {
      dispatch_reduce_store(x, [&]<typename store_t>() { tripleCGUpdate_t<store_t>(a, b, x, y, z, w); });
    }

  } // namespace blas

} // namespace quda
