#pragma once

#include <mdw_ls_list.hpp>
#include <domain_wall_helper.h>

namespace quda
{

  template <int Ls>
  void apply_dslash5_mma_impl(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger);

  struct apply_dslash5_mma_t
  {
    template <int Ls, class... Args>
    void operator()(Args &&...args)
    {
      apply_dslash5_mma_impl<Ls>(args...);
    }
  };

#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)
  void inline ApplyDslash5Mma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
    // clang-format off
    IntList<@QUDA_MDW_FUSED_LS_LIST@> int_list;
    // clang-format on
    if (in.PCType() != QUDA_4D_PC) { errorQuda("Only 4-d preconditioned fields are supported"); }
    if (type != Dslash5Type::M5_INV_MOBIUS) { errorQuda("Currently only M5INV_MOBIUS is supported"); }
    checkLocation(out, in, x); // check all locations match
    apply_dslash5_mma_t d;
    instantiateLsList<decltype(d)>(d, out, int_list, in, x, m_f, m_5, b_5, c_5, a, dagger);
  }
#else
  void inline ApplyDslash5Mma(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &, double,
      double, const Complex *, const Complex *, double, bool, Dslash5Type)
  {
    errorQuda("Domain wall dslash has not been built");
  }
#endif

}

