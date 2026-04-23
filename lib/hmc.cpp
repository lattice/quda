/**
 * @file hmc.cpp
 * @brief Even-odd preconditioned HMC using QUDA internal primitives.
 *
 * Implements the EO fermion action, force, and pseudofermion generation
 * for 2-flavour Wilson fermions. The algorithm follows the verified
 * Schwinger_MG reference (Schwinger_MG/src/hmc.cpp) translated to QUDA's
 * 4D Dirac operator classes and GPU kernels.
 *
 * No existing upstream functions are modified. Internal C++ functions
 * in namespace quda are called by thin C-API wrappers in interface_quda.cpp.
 */

#include <hmc_quda.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <gauge_tools.h>
#include <gauge_path_quda.h>
#include <gauge_update_quda.h>
#include <momentum.h>
#include <clover_field.h>
#include <quda_internal.h>
#include <ctime>

// Globals from interface_quda.cpp (file scope, not in quda namespace)
extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSloppy;
extern quda::GaugeField *gaugePrecondition;
extern quda::CloverField *cloverPrecise;
extern quda::GaugeField momResident;

namespace quda
{

  // Forward declarations for internal functions in solve.cpp and interface_quda.cpp
  void solve(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &b, Dirac &dirac, Dirac &diracSloppy,
             Dirac &diracPre, Dirac &diracEig, QudaInvertParam &param);
  void createDiracWithEig(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dEig, QudaInvertParam &param,
                          bool pc_solve, bool use_smeared_gauge);

  /**
     @brief Create a parity spinor field matching an existing field's metadata.
     @param ref Reference field whose metadata is copied
     @param create Field creation type (default: null allocation)
  */
  static ColorSpinorField createParityField(const ColorSpinorField &ref,
                                            QudaFieldCreate create = QUDA_NULL_FIELD_CREATE)
  {
    ColorSpinorParam param(ref);
    param.create = create;
    return ColorSpinorField(param);
  }

  /**
     @brief Create a full (both-parity) spinor field from a parity field's metadata.
     @param parityRef Parity field whose metadata is used as template
  */
  static ColorSpinorField createFullField(const ColorSpinorField &parityRef)
  {
    ColorSpinorParam param(parityRef);
    param.siteSubset = QUDA_FULL_SITE_SUBSET;
    param.x[0] *= 2;
    param.create = QUDA_ZERO_FIELD_CREATE;
    return ColorSpinorField(param);
  }

  /**
     @brief Create a zero-initialized force field matching a momentum field's metadata.
     @param mom Momentum field whose metadata is used as template
  */
  static GaugeField createForceField(const GaugeField &mom)
  {
    GaugeFieldParam param(mom);
    param.link_type = QUDA_GENERAL_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    return GaugeField(param);
  }

  double computeGaugeActionHMC(double beta)
  {
    lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
    GaugeField *gaugeEx = createExtendedGauge(*gaugePrecise, R, getProfile());
    double3 plaq3 = plaquette(*gaugeEx);
    int V = 1;
    for (int d = 0; d < 4; d++) V *= gaugePrecise->X()[d];
    delete gaugeEx;
    return beta * (1.0 - plaq3.x) * V * 6;
  }

  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, unsigned long long seed)
  {
    ColorSpinorParam csParam(nullptr, inv_param, gaugePrecise->X(), true, QUDA_CUDA_FIELD_LOCATION);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(gaugePrecise->Precision());
    csParam.fieldOrder = QUDA_NATIVE_FIELD_ORDER;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    ColorSpinorField eta(csParam);
    ColorSpinorField phi(csParam);

    if (seed == 0) seed = static_cast<unsigned long long>(std::time(nullptr));
    spinorNoise(eta, seed, QUDA_NOISE_GAUSS);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, true);
    Dirac *dirac = Dirac::create(diracParam);
    dirac->M(phi, eta);
    delete dirac;

    return phi;
  }

  double computeEOFermionAction(ColorSpinorField &phi_even, QudaInvertParam &inv_param)
  {
    QudaInvertParam ip = inv_param;
    ip.solve_type = QUDA_NORMOP_PC_SOLVE;
    ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;

    Dirac *dirac = nullptr, *diracSloppy = nullptr, *diracPre = nullptr, *diracEig = nullptr;
    createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, ip, true, false);

    ColorSpinorField x_even = createParityField(phi_even, QUDA_ZERO_FIELD_CREATE);
    std::vector<ColorSpinorField> x(1, std::move(x_even));
    std::vector<ColorSpinorField> b(1, ColorSpinorField(phi_even));

    solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, ip);

    Complex dot = blas::cDotProduct(phi_even, x[0]);

    delete dirac;
    delete diracSloppy;
    if (diracPre != diracSloppy) delete diracPre;
    if (diracEig != diracPre) delete diracEig;

    double sf = dot.real();

    return sf;
  }

  /**
     @brief Accumulate the EO fermion force into momentum (Wilson hopping).

     Schwinger-style reconstruction of the full-site bilinear from the
     even-parity CG solution, followed by the outer-product kernel.

     @param mom       Momentum field (accumulated into)
     @param x_even    CG solution: D_hat^dag D_hat x = phi (even parity)
     @param inv_param Inverter parameters (provides kappa, dslash type)
     @param dt        Integration step size
  */
  void computeEOFermionForce(GaugeField &mom, ColorSpinorField &x_even,
                             QudaInvertParam &inv_param, double dt)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    double kappa = inv_param.kappa;
    double kappa2 = kappa * kappa;

    QudaInvertParam force_ip = inv_param;
    DiracParam diracParam;
    setDiracParam(diracParam, &force_ip, true);
    Dirac *dirac = Dirac::create(diracParam);

    // --- Common fields: y = M̂x, psi_odd, chi_odd ---
    ColorSpinorField y_even = createParityField(x_even);
    dirac->M(y_even, x_even);

    ColorSpinorField psi_odd = createParityField(x_even);
    dirac->Dslash(psi_odd, x_even, QUDA_ODD_PARITY);

    ColorSpinorField g5y = createParityField(x_even);
    gamma5(g5y, y_even);
    ColorSpinorField tmp = createParityField(x_even);
    dirac->Dslash(tmp, g5y, QUDA_ODD_PARITY);
    ColorSpinorField chi_odd = createParityField(x_even);
    gamma5(chi_odd, tmp);

    // Hopping fields (Schwinger reconstruction, same as Wilson)
    ColorSpinorField psi_full = createFullField(x_even);
    ColorSpinorField chi_full = createFullField(x_even);
    psi_full[QUDA_EVEN_PARITY] = x_even;
    psi_full[QUDA_ODD_PARITY] = psi_odd;
    chi_full[QUDA_EVEN_PARITY] = y_even;
    chi_full[QUDA_ODD_PARITY] = chi_odd;

    // --- Hopping force: coeff = 2κ² (verified for Wilson) ---
    GaugeField force = createForceField(mom);
    computeCloverOprod(force, *gaugePrecise, {chi_full}, {psi_full}, {2.0 * kappa2});

    // Momentum update: p += dt × TA(force)
    updateMomentum(mom, dt, force, "eo_fermion");

    delete dirac;
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
