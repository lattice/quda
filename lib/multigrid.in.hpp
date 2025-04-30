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

  template <class F> auto create_color_spinor_copy(cvector_ref<F> &fs, int nVec, QudaFieldOrder order)
  {
    ColorSpinorParam param(fs[0]);
    param.nColor = fs[0].Ncolor() * nVec;
    param.nVec = nVec;
    param.nVec_actual = fs.size();
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return getFieldTmp<ColorSpinorField>(param);
  }

  template <class Op>
  void divide_and_conquer(Op &op, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in)
  {
    if (out.size() != in.size()) {
      errorQuda("divide_and_conquer out.size() != in.size(): %lu != %lu", out.size(), in.size());
    }
    int instantiated_nVec = instantiated_nVec_to_use(out.size());
    size_t size = out.size();
    logQuda(QUDA_DEBUG_VERBOSE, "MG divide_and_conquer nVec/out.size() = %d/%lu\n", instantiated_nVec, size);
    for (size_t offset = 0; offset < size; offset += instantiated_nVec) {
      cvector_ref<ColorSpinorField> out_offseted {out.begin() + offset,
                                                  out.begin() + std::min(offset + instantiated_nVec, size)};
      cvector_ref<const ColorSpinorField> in_offseted {in.begin() + offset,
                                                       in.begin() + std::min(offset + instantiated_nVec, size)};
      op(out_offseted, in_offseted, instantiated_nVec);
    }
  }

  inline auto create_color_spinor_copy(const ColorSpinorField &f, QudaFieldOrder order)
  {
    ColorSpinorParam param(f);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return getFieldTmp<ColorSpinorField>(param);
  }

} // namespace quda
