#include <dslash5_domain_wall_mma.hpp>
#include <dslash5_domain_wall_mma_impl.hpp>

namespace quda
{

  constexpr int Ls = @QUDA_MDW_FUSED_LS@;
  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5 * in
#if defined(GPU_DOMAIN_WALL_DIRAC) && (CUDA_VERSION >= 11000 && __COMPUTE_CAPABILITY__ >= 800)
  template <>
  void apply_dslash5_mma_impl<Ls>(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger)
  {
    if (in.PCType() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
    checkLocation(out, in, x); // check all locations match
    instantiate<Dslash5MmaLs<Ls>::type>(out, in, x, m_f, m_5, b_5, c_5, a, dagger);
  }
#else
  template <>
  void apply_dslash5_mma_impl<Ls>(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &, double,
      double, const Complex *, const Complex *, double, bool)
  {
    errorQuda("Domain wall dslash with tensor core has not been built");
  }
#endif

} // namespace quda
