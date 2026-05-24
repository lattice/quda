#include <util_quda.h>
#include <reference_wrapper_helper.h>
#include <color_spinor_field.h>
#include <array>
#include <algorithm>
#include <int_list.hpp>

namespace quda
{

  template <int... Values> auto sort_values()
  {
    std::array<int, sizeof...(Values)> arr = {Values...};
    // std::sort is NOT constexpr until C++20
    std::sort(arr.begin(), arr.end());
    return arr;
  }

  /**
     @brief Return the supported number of nVec (right hand sides) to use for
     MMA-enabled MRHS kernels.  This correponds to the smallest size
     at least as large as the requested vector size.
     @return The instantiated nVec to use
  */
  inline int instantiated_nVec_to_use(int input_nVec)
  {
    // clang-format off
    auto sorted_nVecs = sort_values<@QUDA_MULTIGRID_MRHS_LIST@>();
    // clang-format on
    for (int nVec : sorted_nVecs) {
      if (input_nVec <= nVec) { return nVec; }
    }
    if (sorted_nVecs.size() > 0) { return sorted_nVecs.back(); }
    errorQuda("No nVec instantiated");
    return 0;
  }

  /**
     @brief Create a temporary container that corresponds to an
     MMA-ordered copy of an input ColorSpinorField set.  If the input
     set is already in the appropriate order we instead create a set
     that is a reference to input (which allows us to avoid the
     subsequent copy overhead).  Since MMA-ordered ColorSpinorFields
     do not support fixed-point storage, precision is upgraded to
     single if necesary.  This function is the inverse of
     create_color_spinor_expand.
     @param[in,out] fs The input vector set
     @param[in] nVec The number of vectors packed into the returned
     container
     @param[in] order The field order we employ in the returned container
  */
  template <class F> auto create_color_spinor_collapse(cvector_ref<F> &fs, int nVec, QudaFieldOrder order)
  {
    ColorSpinorParam param(fs[0]);
    if (fs.size() == 1 && fs.FieldOrder() == order && fs[0].Nvec() == nVec) {
      // if already in the right order, then we can just wrap it
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.v = fs[0].data();
    } else {
      param.nColor = fs[0].Ncolor() * nVec;
      param.nVec = nVec;
      param.nVec_actual = fs.size();
      param.create = QUDA_NULL_FIELD_CREATE;
      if (param.Precision() < QUDA_SINGLE_PRECISION) param.setPrecision(QUDA_SINGLE_PRECISION);
      param.fieldOrder = order;
    }
    return getFieldTmp<ColorSpinorField>(param);
  }

  /**
     @brief Create a temporary vector container of native-ordered
     ColorSpinorFields from the MMA-ordered input set.  If the input
     set is not in the expected order, or its true number of colors
     doesn't match the requested, then we instead create a reference
     wrapper around the input.  This function is the inverse of
     create_color_spinor_collapse.
     @param[in,out] fs The input vector set
     @param[in] nColor The true number of colors of an unpacked field
   */
  template <class F> auto create_color_spinor_expand(cvector_ref<F> &fs, int nColor)
  {
    if (fs.size() == 1 && fs.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER
        && fs[0].Ncolor() / fs[0].Nvec() == nColor) {
      ColorSpinorParam param(fs[0]);
      param.nColor = nColor;
      param.nVec = 1;
      param.setPrecision(param.Precision(), param.Precision(), true);
      return getFieldTmp<ColorSpinorField>(fs[0].Nvec_actual(), param);
    } else {
      // if already in the right order, then we can just wrap it
      return getFieldTmp(fs, true);
    }
  }

  template <class Op>
  void divide_and_conquer(Op &op, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in)
  {
    if (out.size_actual() != in.size_actual()) {
      errorQuda("divide_and_conquer out.size() != in.size(): %lu != %lu", out.size(), in.size());
    }
    size_t instantiated_nVec = instantiated_nVec_to_use(out.size_actual());
    size_t size = out.size_actual();
    logQuda(QUDA_DEBUG_VERBOSE, "MG divide_and_conquer nVec/out.size() = %lu/%lu\n", instantiated_nVec, size);
    if (size <= instantiated_nVec) {
      op(out, in, instantiated_nVec);
    } else {
      // divide and conquer not supported for fields already in mma order
      if (in.size() != in.size_actual()) errorQuda("Unexpected sizes %lu != %lu", in.size(), in.size_actual());
      if (out.size() != out.size_actual()) errorQuda("Unexpected sizes %lu != %lu", out.size(), out.size_actual());

      for (size_t offset = 0; offset < size; offset += instantiated_nVec) {
        cvector_ref<ColorSpinorField> out_offseted {out.begin() + offset,
                                                    out.begin() + std::min(offset + instantiated_nVec, size)};
        cvector_ref<const ColorSpinorField> in_offseted {in.begin() + offset,
                                                         in.begin() + std::min(offset + instantiated_nVec, size)};
        op(out_offseted, in_offseted, instantiated_nVec);
      }
    }
  }

} // namespace quda
