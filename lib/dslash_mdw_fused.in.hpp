#pragma once

#include <quda_internal.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <worker.h>
#include <domain_wall_helper.h>
#include <fast_intdiv.h>
#include <mdw_ls_list.hpp>

// Tensor core functions for Mobius DWF
namespace quda
{

  namespace mobius_tensor_core
  {

    template <int Ls>
    void apply_fused_dslash_impl(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                 ColorSpinorField &y, const ColorSpinorField &x, double m_f, double m_5,
                                 const Complex *b_5, const Complex *c_5, bool dagger, int parity, int shift[4],
                                 int halo_shift[4], MdwfFusedDslashType type);

    struct ApplyFusedDslash
    {
      template <int Ls, class... Args>
      void operator()(Args &&...args)
      {
        apply_fused_dslash_impl<Ls>(args...);
      }
    };

#if defined(GPU_DOMAIN_WALL_DIRAC) && defined(QUDA_MMA_AVAILABLE)
    void inline apply_fused_dslash(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                   ColorSpinorField &y, const ColorSpinorField &x, double m_f, double m_5,
                                   const Complex *b_5, const Complex *c_5, bool dagger, int parity, int shift[4],
                                   int halo_shift[4], MdwfFusedDslashType type)
    {
      // clang-format off
      IntList<@QUDA_MDW_FUSED_LS_LIST@> int_list;
      // clang-format on
      ApplyFusedDslash afd;
      instantiateLsList<decltype(afd)>(afd, out, int_list, in, U, y, x, m_f, m_5, b_5, c_5, dagger, parity, shift, halo_shift, type);
    }
#else
    void inline apply_fused_dslash(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                   ColorSpinorField &, const ColorSpinorField &, double, double,
                                   const Complex *, const Complex *, bool, int, int[4],
                                   int[4], MdwfFusedDslashType)
    {
      errorQuda("Domain wall dslash with tensor cores has not been built");
    }
#endif

  } // namespace mobius_tensor_core
} // namespace quda
