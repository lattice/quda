#include <dslash_mdw_fused.hpp>
#include <dslash_mdw_fused_impl.hpp>

namespace quda
{
  namespace mobius_tensor_core
  {
    // clang-format off
    constexpr int Ls = @QUDA_MDW_FUSED_LS@;
    // clang-format on
#if defined(GPU_DOMAIN_WALL_DIRAC) && defined(QUDA_MMA_AVAILABLE)
    template <>
    void apply_fused_dslash_impl<Ls>(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                     ColorSpinorField &y, const ColorSpinorField &x, double m_f, double m_5,
                                     const Complex *b_5, const Complex *c_5, bool dagger, int parity, int shift[4],
                                     int halo_shift[4], MdwfFusedDslashType type)
    {
      checkLocation(out, in); // check all locations match
      instantiatePreconditioner<FusedDslashLs<Ls>::type>(out, in, U, y, x, m_f, m_5, b_5, c_5, dagger, parity, shift,
                                                         halo_shift, type);
    }
#else
    template <>
    void apply_fused_dslash_impl<Ls>(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                     ColorSpinorField &, const ColorSpinorField &, double, double, const Complex *,
                                     const Complex *, bool, int, int[4], int[4], MdwfFusedDslashType)
    {
      errorQuda("Domain wall dslash with tensor cores has not been built");
    }
#endif
  } // namespace mobius_tensor_core
} // namespace quda
