#include <blas_quda.h>
#include <staggered_sign_convention.h>

namespace quda
{

  namespace
  {
    template <typename Float> void negate_host(ColorSpinorField &x)
    {
      auto v = x.data<Float *>();
      const size_t n = x.Length();
#pragma omp parallel for
      for (size_t i = 0; i < n; i++) v[i] = -v[i];
    }

    void negate_odd_sites(cvector_ref<ColorSpinorField> &fields, const char *from, const char *to)
    {
      for (auto i = 0u; i < fields.size(); i++) {
        const ColorSpinorField &x = fields[i];
        if (x.IsComposite())
          errorQuda("Field %u of %lu is a composite field, so it cannot be converted from %s to %s sign convention", i,
                    fields.size(), from, to);
        if (x.SiteSubset() != QUDA_FULL_SITE_SUBSET || x.Nspin() != 1 || x.Ndim() != 4)
          errorQuda("Field %u of %lu is not a full 4-d staggered field (it is %s, with %d spins and %d dimensions), so "
                    "it cannot be converted from %s to %s sign convention",
                    i, fields.size(), x.SiteSubset() == QUDA_FULL_SITE_SUBSET ? "full" : "single-parity", x.Nspin(),
                    x.Ndim(), from, to);
        if (x.Location() == QUDA_CPU_FIELD_LOCATION && x.Precision() != QUDA_DOUBLE_PRECISION
            && x.Precision() != QUDA_SINGLE_PRECISION)
          errorQuda("Field %u of %lu is a host field of precision %d, not double or single, so it cannot be converted "
                    "from %s to %s sign convention",
                    i, fields.size(), x.Precision(), from, to);
      }

      for (auto i = 0u; i < fields.size(); i++) {
        ColorSpinorField &odd = fields[i].Odd();
        if (odd.Location() == QUDA_CUDA_FIELD_LOCATION) {
          blas::ax(-1.0, odd);
        } else {
          switch (odd.Precision()) {
          case QUDA_DOUBLE_PRECISION: negate_host<double>(odd); break;
          case QUDA_SINGLE_PRECISION: negate_host<float>(odd); break;
          default: errorQuda("Unsupported precision %d of host field %u", odd.Precision(), i);
          }
        }
      }

      logQuda(QUDA_SUMMARIZE, "Converted %lu staggered field%s from %s to %s sign convention (odd sites negated)\n",
              fields.size(), fields.size() == 1 ? "" : "s", from, to);
    }
  } // namespace

  const char *get_staggered_sign_convention_str(StaggeredSignConvention convention)
  {
    switch (convention) {
    case StaggeredSignConvention::QUDA: return "QUDA";
    case StaggeredSignConvention::MILC: return "MILC";
    }
    return "";
  }

  void staggeredQUDAToMILCSignConvention(cvector_ref<ColorSpinorField> &fields)
  {
    negate_odd_sites(fields, "QUDA's", "MILC's");
  }

  void staggeredMILCToQUDASignConvention(cvector_ref<ColorSpinorField> &fields)
  {
    negate_odd_sites(fields, "MILC's", "QUDA's");
  }

} // namespace quda
