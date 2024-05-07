#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>

namespace quda {

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracParam &param) :
    Dirac(param),
    fatGauge(param.fatGauge),
    longGauge(param.longGauge)
  {
  }

  DiracImprovedStaggered::DiracImprovedStaggered(const DiracImprovedStaggered &dirac)
    : Dirac(dirac), fatGauge(dirac.fatGauge), longGauge(dirac.longGauge) { }

  DiracImprovedStaggered::~DiracImprovedStaggered() { }

  DiracImprovedStaggered& DiracImprovedStaggered::operator=(const DiracImprovedStaggered &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      fatGauge = dirac.fatGauge;
      longGauge = dirac.longGauge;
    }
    return *this;
  }

  void DiracImprovedStaggered::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                      const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 0., in, parity, dagger, commDim.data, profile);
  }

  void DiracImprovedStaggered::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                          const QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {    
    checkParitySpinor(in, out);

    // Need to catch the zero mass case.
    if (k == 0.0) {
      // There's a sign convention difference for Dslash vs DslashXpay, which is
      // triggered by looking for k == 0. We need to hack around this.
      if (dagger == QUDA_DAG_YES) {
        ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 0., x, parity, QUDA_DAG_NO, commDim.data, profile);
      } else {
        ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 0., x, parity, QUDA_DAG_YES, commDim.data, profile);
      }
    } else {
      ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, k, x, parity, dagger, commDim.data, profile);
    }
  }

  // Full staggered operator
  void DiracImprovedStaggered::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    checkFullSpinor(out, in);
    // Need to flip sign via dagger convention if mass == 0.
    if (mass == 0.0) {
      if (dagger == QUDA_DAG_YES) {
        ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim.data,
                               profile);
      } else {
        ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data,
                               profile);
      }
    } else {
      ApplyImprovedStaggered(out, in, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim.data,
                             profile);
    }
  }

  void DiracImprovedStaggered::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out.Even());

    //even
    Dslash(tmp, in.Even(), QUDA_ODD_PARITY);
    DslashXpay(out.Even(), tmp, QUDA_EVEN_PARITY, in.Even(), 4 * mass * mass);

    // odd
    Dslash(tmp, in.Odd(), QUDA_EVEN_PARITY);
    DslashXpay(out.Odd(), tmp, QUDA_ODD_PARITY, in.Odd(), 4 * mass * mass);
  }

  void DiracImprovedStaggered::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                       cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      sol[i] = x[i].create_alias();
    }
  }

  void DiracImprovedStaggered::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                           const QudaSolutionType) const
  {
    // do nothing
  }
  
  void DiracImprovedStaggered::SmearOp(ColorSpinorField &out, const ColorSpinorField &in, const double, const double,
                             const int t0, const QudaParity parity) const
  {
    checkSpinorAlias(in, out);

    bool is_time_slice = t0 >= 0 && t0 < comm_dim(3)*in.X(3) ? true : false;
    if( is_time_slice && laplace3D > 3 )
    {
      if (getVerbosity() == QUDA_DEBUG_VERBOSE) warningQuda("t0 will be ignored for d>3 dimensional Laplacian.");
      is_time_slice = false;
    }

    int t0_local = t0 - comm_coord(3)*in.X(3);
    if( is_time_slice && ( t0_local < 0 || t0_local >= in.X(3) ) ) t0_local = -1; // when source is not in this local lattice

    int comm_dim[4] = {};
    // only switch on comms needed for directions with a derivative
    for (int i = 0; i < 4; i++) {
      comm_dim[i] = comm_dim_partitioned(i);
      if (laplace3D == i) comm_dim[i] = 0;
    }
 
    if (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET){
      errorQuda( "Single parity site smearing is not supported yet." );
    } else {
      ApplyStaggeredQSmear(out, in, *gauge, t0_local, is_time_slice, parity, laplace3D, dagger, comm_dim, profile);
    }
  }  

  void DiracImprovedStaggered::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double, double mass,
                                              double, double, bool allow_truncation) const
  {
    if (T.getTransferType() == QUDA_TRANSFER_OPTIMIZED_KD || T.getTransferType() == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
      errorQuda("The optimized improved Kahler-Dirac operator is not built through createCoarseOp");

    StaggeredCoarseOp(Y, X, T, *fatGauge, *longGauge, *fatGauge, mass, allow_truncation, QUDA_ASQTAD_DIRAC,
                      QUDA_MATPC_INVALID);
  }

  void DiracImprovedStaggered::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    fatGauge->prefetch(mem_space, stream);
    longGauge->prefetch(mem_space, stream);
  }

  DiracImprovedStaggeredPC::DiracImprovedStaggeredPC(const DiracParam &param)
    : DiracImprovedStaggered(param)
  {

  }

  DiracImprovedStaggeredPC::DiracImprovedStaggeredPC(const DiracImprovedStaggeredPC &dirac) 
    : DiracImprovedStaggered(dirac)
  {

  }

  DiracImprovedStaggeredPC::~DiracImprovedStaggeredPC()
  {

  }

  DiracImprovedStaggeredPC& DiracImprovedStaggeredPC::operator=(const DiracImprovedStaggeredPC &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
    }
 
    return *this;
  }

  // Unlike with clover, for ex, we don't need a custom Dslash or DslashXpay.
  // That's because the convention for preconditioned staggered is to
  // NOT divide out the factor of "2m", i.e., for the even system we invert
  // (4m^2 - D_eo D_oe), not (1 - (1/(4m^2)) D_eo D_oe).

  void DiracImprovedStaggeredPC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    // Convention note: Dslash applies D_eo, DslashXpay applies 4m^2 - D_oe!
    // Note the minus sign convention in the Xpay version.
    // This applies equally for the e <-> o permutation.

    auto tmp = getFieldTmp(out);
    Dslash(tmp, in, other_parity);
    DslashXpay(out, tmp, this_parity, in, 4 * mass * mass);
  }

  void DiracImprovedStaggeredPC::MdagM(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &) const
  {
    errorQuda("MdagM is no longer defined for DiracImprovedStaggeredPC. Use M instead");
  }

  void DiracImprovedStaggeredPC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                                         cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                         const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      for (auto i = 0u; i < b.size(); i++) {
        // we desire solution to preconditioned system
        src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
        sol[i] = x[i].create_alias();
      }
      return;
    }

    for (auto i = 0u; i < b.size(); i++) {
      // we desire solution to full system.
      // With the convention given in DiracStaggered::M(),
      // the source is src = 2m b_e + D_eo b_o
      // But remember, DslashXpay actually applies
      // -D_eo. Flip the sign on 2m to compensate, and
      // then flip the overall sign.
      src[i] = x[i][other_parity].create_alias();
      DslashXpay(src[i], b[i][other_parity], this_parity, b[i][this_parity], -2.0 * mass);
      blas::ax(-1.0, src[i]);
      sol[i] = x[i][this_parity].create_alias();
    }
  }

  void DiracImprovedStaggeredPC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }

    for (auto i = 0u; i < b.size(); i++) {
      checkFullSpinor(x[i], b[i]);

      // create full solution
      // With the convention given in DiracStaggered::M(),
      // the reconstruct is x_o = 1/(2m) (b_o + D_oe x_e)
      // But remember: DslashXpay actually applies -D_oe,
      // so just like above we need to flip the sign
      // on b_o. We then correct this by applying an additional
      // minus sign when we rescale by 2m.
      DslashXpay(x[i][other_parity], x[i][this_parity], other_parity, b[i][other_parity], -1.0);
      blas::ax(-0.5 / mass, x[i][other_parity]);
    }
  }

  void DiracImprovedStaggeredPC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double, double mass,
                                                double, double, bool allow_truncation) const
  {
    if (T.getTransferType() == QUDA_TRANSFER_OPTIMIZED_KD || T.getTransferType() == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
      errorQuda("The optimized improved Kahler-Dirac operator is not built through createCoarseOp");

    StaggeredCoarseOp(Y, X, T, *fatGauge, *longGauge, *fatGauge, mass, allow_truncation, QUDA_ASQTADPC_DIRAC,
                      QUDA_MATPC_INVALID);
  }

} // namespace quda
