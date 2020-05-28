#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracMobius::DiracMobius(const DiracParam &param) : DiracDomainWall(param), zMobius(false)
  {
    memcpy(b_5, param.b_5, sizeof(Complex) * param.Ls);
    memcpy(c_5, param.c_5, sizeof(Complex) * param.Ls);

    double b = b_5[0].real();
    double c = c_5[0].real();
    mobius_kappa_b = 0.5 / (b * (m5 + 4.) + 1.);
    mobius_kappa_c = 0.5 / (c * (m5 + 4.) - 1.);

    mobius_kappa = mobius_kappa_b / mobius_kappa_c;

    // check if doing zMobius
    for (int i = 0; i < Ls; i++) {
      if (b_5[i].imag() != 0.0 || c_5[i].imag() != 0.0 || (i < Ls - 1 && (b_5[i] != b_5[i + 1] || c_5[i] != c_5[i + 1]))) {
        zMobius = true;
      }
    }

    if (getVerbosity() > QUDA_VERBOSE) {
      if (zMobius) {
        printfQuda("%s: Detected variable or complex cofficients: using zMobius\n", __func__);
      } else {
        printfQuda("%s: Detected fixed real cofficients: using regular Mobius\n", __func__);
      }
    }

    if (zMobius) { errorQuda("zMobius has NOT been fully tested in QUDA.\n"); }
  }

  // Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobius::Dslash4(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, parity, dagger, commDim, profile);

    flops += 1320LL * (long long)in.Volume();
  }

  void DiracMobius::Dslash4pre(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 72LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  // Unlike DWF-4d, the Mobius variant here applies the full M5 operator and not just D5
  void DiracMobius::Dslash5(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 48LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  // Modification for the 4D preconditioned Mobius domain wall operator
  void DiracMobius::Dslash4Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                                const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDomainWall4D(out, in, *gauge, k, m5, b_5, c_5, x, parity, dagger, commDim, profile);

    flops += 1320LL * (long long)in.Volume();
  }

  void DiracMobius::Dslash4preXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                                   const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, DSLASH5_MOBIUS_PRE);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;

    flops += 72LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  // The xpay operator bakes in a factor of kappa_b^2
  void DiracMobius::Dslash5Xpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                                const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, DSLASH5_MOBIUS);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 96LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  void DiracMobius::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    // zMobius breaks the following code. Refer to the zMobius check in DiracMobius::DiracMobius(param)
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

    // cannot use Xpay variants since it will scale incorrectly for this operator

    ColorSpinorField *tmp = nullptr;
    if (tmp2 && tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = tmp2;
    bool reset = newTmp(&tmp, in);

    if (dagger == QUDA_DAG_NO) {
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(*tmp, out, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS);
    } else {
      // the third term is added, not multiplied, so we only need to swap the first two in the dagger
      ApplyDomainWall4D(out, in, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      ApplyDslash5(*tmp, out, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS);
    }
    blas::axpy(-mobius_kappa_b, *tmp, out);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 72LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // pre
    flops += 1320LL * (long long)in.Volume();                            // dslash4
    flops += 48LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // dslash5

    deleteTmp(&tmp, reset);
  }

  void DiracMobius::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracMobius::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x, ColorSpinorField &b,
                            const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracMobius::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // do nothing
  }

  DiracMobiusPC::DiracMobiusPC(const DiracParam &param) : DiracMobius(param), extended_gauge(nullptr)
  {
    // do nothing
  }

  DiracMobiusPC::DiracMobiusPC(const DiracMobiusPC &dirac) : DiracMobius(dirac), extended_gauge(nullptr)
  {
    // do nothing
  }

  DiracMobiusPC::~DiracMobiusPC()
  {
    if (extended_gauge) delete extended_gauge;
  }

  DiracMobiusPC &DiracMobiusPC::operator=(const DiracMobiusPC &dirac)
  {
    if (&dirac != this) {
      DiracMobius::operator=(dirac);
      extended_gauge = nullptr;
    }

    return *this;
  }

  void DiracMobiusPC::Dslash5inv(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, zMobius ? M5_INV_ZMOBIUS : M5_INV_MOBIUS);

    if (0) {
      // M5 = 1 + 0.5*kappa_b/kappa_c * D5
      using namespace blas;
      cudaColorSpinorField A(out);
      Dslash5(A, out, parity);
      printfQuda("Dslash5Xpay = %e M5inv = %e in = %e\n", norm2(A), norm2(out), norm2(in));
      exit(0);
    }

    long long Ls = in.X(4);
    flops += 144LL * (long long)in.Volume() * Ls + 3LL * Ls * (Ls - 1LL);
  }

  // The xpay operator bakes in a factor of kappa_b^2
  void DiracMobiusPC::Dslash5invXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
                                     const ColorSpinorField &x, const double &k) const
  {
    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ApplyDslash5(out, in, x, mass, m5, b_5, c_5, k, dagger, zMobius ? M5_INV_ZMOBIUS : M5_INV_MOBIUS);

    long long Ls = in.X(4);
    flops += (144LL * Ls + 48LL) * (long long)in.Volume() + 3LL * Ls * (Ls - 1LL);
  }

  // Apply the even-odd preconditioned mobius DWF operator
  // Actually, Dslash5 will return M5 operation and M5 = 1 + 0.5*kappa_b/kappa_c * D5
  void DiracMobiusPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp1, in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    // QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : M5 - kappa_b^2 * D4_{eo}D4pre_{oe}D5inv_{ee}D4_{eo}D4pre_{oe}
    // QUDA_MATPC_ODD_ODD_ASYMMETRIC : M5 - kappa_b^2 * D4_{oe}D4pre_{eo}D5inv_{oo}D4_{oe}D4pre_{eo}
    if (symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5invXpay(out, *tmp1, parity[1], in, -1.0);
    } else if (symmetric && dagger) {
      Dslash5inv(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash4pre(*tmp1, out, parity[0]);
      Dslash5inv(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash4preXpay(out, *tmp1, parity[1], in, -1.0);
    } else if (!symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, -1.0);
    } else if (!symmetric && dagger) {
      Dslash4(*tmp1, in, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash5inv(*tmp1, out, parity[0]);
      Dslash4(out, *tmp1, parity[1]);
      Dslash4pre(*tmp1, out, parity[1]);
      Dslash5Xpay(out, in, parity[1], *tmp1, -1.0);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracMobiusPC::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x, ColorSpinorField &b,
                              const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else { // we desire solution to full system
      // prepare function in MDWF is not tested yet.
      bool reset = newTmp(&tmp1, b.Even());

      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = D5^-1 (b_e + k D4_eo * D4pre * D5^-1 b_o)
        src = &(x.Odd());
        Dslash5inv(*tmp1, b.Odd(), QUDA_ODD_PARITY);
        Dslash4pre(*src, *tmp1, QUDA_ODD_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), 1.0);
        Dslash5inv(*src, *tmp1, QUDA_EVEN_PARITY);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        Dslash5inv(*tmp1, b.Even(), QUDA_EVEN_PARITY);
        Dslash4pre(*src, *tmp1, QUDA_EVEN_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), 1.0);
        Dslash5inv(*src, *tmp1, QUDA_ODD_PARITY);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo * D4pre * D5inv b_o
        src = &(x.Odd());
        Dslash5inv(*src, b.Odd(), QUDA_ODD_PARITY);
        Dslash4pre(*tmp1, *src, QUDA_ODD_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), 1.0);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        Dslash5inv(*src, b.Even(), QUDA_EVEN_PARITY);
        Dslash4pre(*tmp1, *src, QUDA_EVEN_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), 1.0);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want

      deleteTmp(&tmp1, reset);
    }
  }

  void DiracMobiusPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    bool reset1 = newTmp(&tmp1, x.Even());

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // psi_o = M5^-1 (b_o + k_b D4_oe D4pre x_e)
      Dslash4pre(x.Odd(), x.Even(), QUDA_EVEN_PARITY);
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_ODD_PARITY, b.Odd(), 1.0);
      Dslash5inv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
    } else if (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // psi_e = M5^-1 (b_e + k_b D4_eo D4pre x_o)
      Dslash4pre(x.Even(), x.Odd(), QUDA_ODD_PARITY);
      Dslash4Xpay(*tmp1, x.Even(), QUDA_EVEN_PARITY, b.Even(), 1.0);
      Dslash5inv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
    } else {
      errorQuda("MatPCType %d not valid for DiracMobiusPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusPC::MdagMLocal(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (zMobius) { errorQuda("DiracMobiusPC::MdagMLocal doesn't currently support zMobius.\n"); }

    if (extended_gauge == nullptr) {
      const int R[] = {2, 2, 2, 2};
      extended_gauge = createExtendedGauge(*gauge, R, profile, true);
    }

    checkDWF(in, out);
    // checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    ColorSpinorParam csParam(out);
    csParam.create = QUDA_NULL_FIELD_CREATE;

    ColorSpinorField *unextended_tmp1 = ColorSpinorField::Create(csParam);
    ColorSpinorField *unextended_tmp2 = ColorSpinorField::Create(csParam);

    csParam.x[0] += 2; // x direction is checkerboarded
    for (int i = 1; i < 4; ++i) { csParam.x[i] += 4; }
    ColorSpinorField *extended_tmp1 = ColorSpinorField::Create(csParam);
    ColorSpinorField *extended_tmp2 = ColorSpinorField::Create(csParam);

    int shift0[4] = {0, 0, 0, 0};
    int shift1[4] = {1, 1, 1, 1};
    int shift2[4] = {2, 2, 2, 2};

    int odd_bit = (getMatPCType() == QUDA_MATPC_ODD_ODD) ? 1 : 0;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};
    if (out.Precision() == QUDA_HALF_PRECISION || out.Precision() == QUDA_QUARTER_PRECISION) {
      mobius_tensor_core::apply_fused_dslash(*unextended_tmp2, in, *extended_gauge, *unextended_tmp2, in, mass, m5, b_5,
                                             c_5, dagger, parity[1], shift0, shift0, MdwfFusedDslashType::D5PRE);

      mobius_tensor_core::apply_fused_dslash(*extended_tmp2, *unextended_tmp2, *extended_gauge, *extended_tmp2,
                                             *unextended_tmp2, mass, m5, b_5, c_5, dagger, parity[0], shift1, shift2,
                                             MdwfFusedDslashType::D4_D5INV_D5PRE);

      mobius_tensor_core::apply_fused_dslash(*extended_tmp1, *extended_tmp2, *extended_gauge, *unextended_tmp1, in,
                                             mass, m5, b_5, c_5, dagger, parity[1], shift0, shift1,
                                             MdwfFusedDslashType::D4_D5INV_D5INVDAG);

      mobius_tensor_core::apply_fused_dslash(*extended_tmp2, *extended_tmp1, *extended_gauge, *extended_tmp2,
                                             *extended_tmp1, mass, m5, b_5, c_5, dagger, parity[0], shift1, shift1,
                                             MdwfFusedDslashType::D4DAG_D5PREDAG_D5INVDAG);

      mobius_tensor_core::apply_fused_dslash(out, *extended_tmp2, *extended_gauge, out, *unextended_tmp1, mass, m5, b_5,
                                             c_5, dagger, parity[1], shift2, shift2, MdwfFusedDslashType::D4DAG_D5PREDAG);

      const long long Ls = in.X(4);
      const long long mat = 2ll * 4ll * Ls - 1ll; // (multiplicaiton-add) * (spin) * Ls - 1
      const long long hop = 7ll * 8ll;            // 8 for eight directions

      long long vol;
      long long halo_vol;

      vol = (2 * in.X(0)) * in.X(1) * in.X(2) * in.X(3) * Ls / 2ll;
      flops += vol * 24ll * mat;

      vol = (2 * in.X(0) + 2 * 1) * (in.X(1) + 2 * 1) * (in.X(2) + 2 * 1) * (in.X(3) + 2 * 1) * Ls / 2ll;
      halo_vol = (2 * in.X(0)) * in.X(1) * in.X(2) * in.X(3) * Ls / 2ll;
      flops += halo_vol * 24ll * hop + vol * 24ll * mat;

      vol = (2 * in.X(0) + 2 * 2) * (in.X(1) + 2 * 2) * (in.X(2) + 2 * 2) * (in.X(3) + 2 * 2) * Ls / 2ll;
      halo_vol = (2 * in.X(0) + 2 * 1) * (in.X(1) + 2 * 1) * (in.X(2) + 2 * 1) * (in.X(3) + 2 * 1) * Ls / 2ll;
      flops += halo_vol * 24ll * hop + vol * 24ll * mat * 2ll;

      vol = (2 * in.X(0) + 2 * 1) * (in.X(1) + 2 * 1) * (in.X(2) + 2 * 1) * (in.X(3) + 2 * 1) * Ls / 2ll;
      flops += vol * 24ll * (hop + mat);

      vol = (2 * in.X(0)) * in.X(1) * in.X(2) * in.X(3) * Ls / 2ll;
      flops += vol * 24ll * (hop + mat);

      delete extended_tmp2;
      delete extended_tmp1;

      delete unextended_tmp1;
      delete unextended_tmp2;

    } else {
      errorQuda("DiracMobiusPC::MdagMLocal(...) only supports half and quarter precision");
    }
  }

  // Copy the EOFA specific parameters
  DiracMobiusEofa::DiracMobiusEofa(const DiracParam &param) :
    DiracMobius(param), eofa_shift(param.eofa_shift), eofa_pm(param.eofa_pm), mq1(param.mq1), mq2(param.mq2), mq3(param.mq3)
  {
    // Initiaize the EOFA parameters here: u, x, y

    if (zMobius) { errorQuda("DiracMobiusEofa doesn't currently support zMobius.\n"); }

    double b = b_5[0].real();
    double c = c_5[0].real();

    double alpha = b + c;

    double eofa_norm = alpha * (mq3 - mq2) * std::pow(alpha + 1., 2. * Ls)
      / (std::pow(alpha + 1., Ls) + mq2 * std::pow(alpha - 1., Ls))
      / (std::pow(alpha + 1., Ls) + mq3 * std::pow(alpha - 1., Ls));

    double N = (eofa_pm ? +1. : -1.) * (2. * this->eofa_shift * eofa_norm)
      * (std::pow(alpha + 1., Ls) + this->mq1 * std::pow(alpha - 1., Ls)) / (b * (m5 + 4.) + 1.);

    // Here the signs are somewhat mixed:
    // There is one -1 from N for eofa_pm = minus, thus the u_- here is actually -u_- in the document
    // It turns out this actually simplies things.
    for (int s = 0; s < Ls; s++) {
      eofa_u[eofa_pm ? s : Ls - 1 - s]
        = N * std::pow(-1., s) * std::pow(alpha - 1., s) / std::pow(alpha + 1., Ls + s + 1);
    }

    double factor = -mobius_kappa * mass;
    if (eofa_pm) {
      // eofa_pm = plus
      // Computing x
      eofa_x[0] = eofa_u[0];
      for (int s = Ls - 1; s > 0; s--) {
        eofa_x[0] -= factor * eofa_u[s];
        factor *= -mobius_kappa;
      }
      eofa_x[0] /= 1. + factor;
      for (int s = 1; s < Ls; s++) { eofa_x[s] = eofa_x[s - 1] * (-mobius_kappa) + eofa_u[s]; }
      // Computing y
      eofa_y[Ls - 1] = 1. / (1. + factor);
      sherman_morrison_fac = eofa_x[Ls - 1];
      for (int s = Ls - 1; s > 0; s--) { eofa_y[s - 1] = eofa_y[s] * (-mobius_kappa); }
    } else {
      // eofa_pm = minus
      // Computing x
      eofa_x[Ls - 1] = eofa_u[Ls - 1];
      for (int s = 0; s < Ls - 1; s++) {
        eofa_x[Ls - 1] -= factor * eofa_u[s];
        factor *= -mobius_kappa;
      }
      eofa_x[Ls - 1] /= 1. + factor;
      for (int s = Ls - 1; s > 0; s--) { eofa_x[s - 1] = eofa_x[s] * (-mobius_kappa) + eofa_u[s - 1]; }
      // Computing y
      eofa_y[0] = 1. / (1. + factor);
      sherman_morrison_fac = eofa_x[0];
      for (int s = 1; s < Ls; s++) { eofa_y[s] = eofa_y[s - 1] * (-mobius_kappa); }
    }
    m5inv_fac = 0.5 / (1. + factor);                           // 0.5 for the spin project factor
    sherman_morrison_fac = -0.5 / (1. + sherman_morrison_fac); // 0.5 for the spin project factor
  }

  void DiracMobiusEofa::m5_eofa(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkDWF(in, out);
    checkSpinorAlias(in, out);

    mobius_eofa::apply_dslash5(out, in, in, mass, m5, b_5, c_5, 0., eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                               eofa_y, sherman_morrison_fac, dagger, M5_EOFA);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;

    // 96 = 48 + 48, the second 48 from EOFA
    flops += 96LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  void DiracMobiusEofa::m5_eofa_xpay(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
                                     double a) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkDWF(in, out);
    checkSpinorAlias(in, out);

    a *= mobius_kappa_b * mobius_kappa_b; // a = a * kappa_b^2
    // The kernel will actually do (m5 * in - kappa_b^2 * x)
    mobius_eofa::apply_dslash5(out, in, x, mass, m5, b_5, c_5, a, eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                               eofa_y, sherman_morrison_fac, dagger, M5_EOFA);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;

    // 144 = 96 + 48, the 48 from EOFA
    flops += 144LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  void DiracMobiusEofa::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    // FIXME broken for variable coefficients
    double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

    // cannot use Xpay variants since it will scale incorrectly for this operator

    ColorSpinorField *tmp = nullptr;
    if (tmp2 && tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = tmp2;
    bool reset = newTmp(&tmp, in);

    if (dagger == QUDA_DAG_NO) {
      ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);
      ApplyDomainWall4D(*tmp, out, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      mobius_eofa::apply_dslash5(out, in, in, mass, m5, b_5, c_5, 0., eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                                 eofa_y, sherman_morrison_fac, dagger, M5_EOFA);
    } else {
      ApplyDomainWall4D(out, in, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, dagger, commDim, profile);
      ApplyDslash5(*tmp, out, out, mass, m5, b_5, c_5, 0.0, dagger, DSLASH5_MOBIUS_PRE);
      mobius_eofa::apply_dslash5(out, in, in, mass, m5, b_5, c_5, 0., eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                                 eofa_y, sherman_morrison_fac, dagger, M5_EOFA);
    }
    blas::axpy(-mobius_kappa_b, *tmp, out);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += 72LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // pre
    flops += 1320LL * (long long)in.Volume();                            // dslash4

    // 96 = 48 + 48, the second 48 from EOFA
    flops += 96LL * (long long)in.Volume() + 96LL * bulk + 120LL * wall; // dslash5

    deleteTmp(&tmp, reset);
  }

  void DiracMobiusEofa::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracMobiusEofa::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracMobiusEofa::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // do nothing
  }

  DiracMobiusEofaPC::DiracMobiusEofaPC(const DiracParam &param) : DiracMobiusEofa(param) { }

  void DiracMobiusEofaPC::m5inv_eofa(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkDWF(in, out);
    // checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    mobius_eofa::apply_dslash5(out, in, in, mass, m5, b_5, c_5, 0., eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                               eofa_y, sherman_morrison_fac, dagger, M5INV_EOFA);

    long long Ls = in.X(4);
    flops += (192LL * Ls + 96LL) * (long long)in.Volume() + 3LL * Ls * (Ls - 1LL);
  }

  void DiracMobiusEofaPC::m5inv_eofa_xpay(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x,
                                          double a) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");

    checkDWF(in, out);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    a *= mobius_kappa_b * mobius_kappa_b; // a = a * kappa_b^2
    // The kernel will actually do (x - kappa_b^2 * m5inv * in)
    mobius_eofa::apply_dslash5(out, in, x, mass, m5, b_5, c_5, a, eofa_pm, m5inv_fac, mobius_kappa, eofa_u, eofa_x,
                               eofa_y, sherman_morrison_fac, dagger, M5INV_EOFA);

    long long Ls = in.X(4);
    flops += (192LL * Ls + 48LL + 96LL) * (long long)in.Volume() + 3LL * Ls * (Ls - 1LL);
  }

  // Apply the even-odd preconditioned mobius DWF EOFA operator
  void DiracMobiusEofaPC::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset1 = newTmp(&tmp1, in);

    int odd_bit = (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) ? 1 : 0;
    bool symmetric = (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_ODD_ODD) ? true : false;
    QudaParity parity[2] = {static_cast<QudaParity>((1 + odd_bit) % 2), static_cast<QudaParity>((0 + odd_bit) % 2)};

    // QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : M5 - kappa_b^2 * D4_{eo}D4pre_{oe}D5inv_{ee}D4_{eo}D4pre_{oe}
    // QUDA_MATPC_ODD_ODD_ASYMMETRIC : M5 - kappa_b^2 * D4_{oe}D4pre_{eo}D5inv_{oo}D4_{oe}D4pre_{eo}
    if (symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      m5inv_eofa(*tmp1, out);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      m5inv_eofa_xpay(out, *tmp1, in, -1.);
    } else if (symmetric && dagger) {
      m5inv_eofa(*tmp1, in);
      Dslash4(out, *tmp1, parity[0]);
      Dslash4pre(*tmp1, out, parity[0]);
      m5inv_eofa(out, *tmp1);
      Dslash4(*tmp1, out, parity[1]);
      Dslash4preXpay(out, *tmp1, parity[1], in, -1.);
    } else if (!symmetric && !dagger) {
      Dslash4pre(*tmp1, in, parity[1]);
      Dslash4(out, *tmp1, parity[0]);
      m5inv_eofa(*tmp1, out);
      Dslash4pre(out, *tmp1, parity[0]);
      Dslash4(*tmp1, out, parity[1]);
      m5_eofa_xpay(out, in, *tmp1, -1.);
    } else if (!symmetric && dagger) {
      Dslash4(*tmp1, in, parity[0]);
      Dslash4pre(out, *tmp1, parity[0]);
      m5inv_eofa(*tmp1, out);
      Dslash4(out, *tmp1, parity[1]);
      Dslash4pre(*tmp1, out, parity[1]);
      m5_eofa_xpay(out, in, *tmp1, -1.);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusEofaPC::prepare(ColorSpinorField *&src, ColorSpinorField *&sol, ColorSpinorField &x,
                                  ColorSpinorField &b, const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
    } else {
      // we desire solution to full system
      bool reset = newTmp(&tmp1, b.Even());
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = D5^-1 (b_e + k D4_eo * D4pre * D5^-1 b_o)
        src = &(x.Odd());
        m5inv_eofa(*tmp1, b.Odd());
        Dslash4pre(*src, *tmp1, QUDA_ODD_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), 1.0);
        m5inv_eofa(*src, *tmp1);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        m5inv_eofa(*tmp1, b.Even());
        Dslash4pre(*src, *tmp1, QUDA_EVEN_PARITY);
        Dslash4Xpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), 1.0);
        m5inv_eofa(*src, *tmp1);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D4_eo * D4pre * D5inv b_o
        src = &(x.Odd());
        m5inv_eofa(*src, b.Odd());
        Dslash4pre(*tmp1, *src, QUDA_ODD_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), 1.0);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D4_oe * D4pre * D5inv b_e
        src = &(x.Even());
        m5inv_eofa(*src, b.Even());
        Dslash4pre(*tmp1, *src, QUDA_EVEN_PARITY);
        Dslash4Xpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), 1.0);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracMobiusEofaPC", matpcType);
      }
      // here we use final solution to store parity solution and parity source
      // b is now up for grabs if we want
      deleteTmp(&tmp1, reset);
    }
  }

  void DiracMobiusEofaPC::reconstruct(ColorSpinorField &x, const ColorSpinorField &b, const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }

    bool reset1 = newTmp(&tmp1, x.Even());

    // create full solution
    checkFullSpinor(x, b);
    if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // psi_o = M5^-1 (b_o + k_b D4_oe D4pre x_e)
      Dslash4pre(x.Odd(), x.Even(), QUDA_EVEN_PARITY);
      Dslash4Xpay(*tmp1, x.Odd(), QUDA_ODD_PARITY, b.Odd(), 1.0);
      m5inv_eofa(x.Odd(), *tmp1);
    } else if (matpcType == QUDA_MATPC_ODD_ODD || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // psi_e = M5^-1 (b_e + k_b D4_eo D4pre x_o)
      Dslash4pre(x.Even(), x.Odd(), QUDA_ODD_PARITY);
      Dslash4Xpay(*tmp1, x.Even(), QUDA_EVEN_PARITY, b.Even(), 1.0);
      m5inv_eofa(x.Even(), *tmp1);
    } else {
      errorQuda("MatPCType %d not valid for DiracMobiusPC", matpcType);
    }

    deleteTmp(&tmp1, reset1);
  }

  void DiracMobiusEofaPC::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void
  DiracMobiusEofaPC::full_dslash(ColorSpinorField &out,
                                 const ColorSpinorField &in) const // ye = Mee * xe + Meo * xo, yo = Moo * xo + Moe * xe
  {
    checkFullSpinor(out, in);
    bool reset1 = newTmp(&tmp1, in.Odd());
    bool reset2 = newTmp(&tmp2, in.Odd());
    if (!dagger) {
      // Even
      m5_eofa(*tmp1, in.Even());
      Dslash4pre(*tmp2, in.Odd(), QUDA_ODD_PARITY);
      Dslash4Xpay(out.Even(), *tmp2, QUDA_EVEN_PARITY, *tmp1, -1.);
      // Odd
      m5_eofa(*tmp1, in.Odd());
      Dslash4pre(*tmp2, in.Even(), QUDA_EVEN_PARITY);
      Dslash4Xpay(out.Odd(), *tmp2, QUDA_ODD_PARITY, *tmp1, -1.);
    } else {
      printfQuda("Quda EOFA full dslash dagger=yes\n");
      // Even
      m5_eofa(*tmp1, in.Even());
      Dslash4(*tmp2, in.Odd(), QUDA_EVEN_PARITY);
      Dslash4preXpay(out.Even(), *tmp2, QUDA_EVEN_PARITY, *tmp1, -1. / mobius_kappa_b);
      // Odd
      m5_eofa(*tmp1, in.Odd());
      Dslash4(*tmp2, in.Even(), QUDA_ODD_PARITY);
      Dslash4preXpay(out.Odd(), *tmp2, QUDA_ODD_PARITY, *tmp1, -1. / mobius_kappa_b);
    }
    deleteTmp(&tmp1, reset1);
    deleteTmp(&tmp2, reset2);
  }
} // namespace quda
#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>
