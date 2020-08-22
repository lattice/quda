#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <staggered_kd_build_xinv.h>

namespace quda {

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracParam &param) : DiracImprovedStaggered(param),
    Xinv(nullptr) {
      
    // for future reference
    const bool gpu = true;

    // Allocate the KD inverse block (inverse coarse clover)
    // Copied from `dirac_coarse.cpp`, `DiracCoarse::createY`
    const int ndim = 4;
    int xc[QUDA_MAX_DIM];
    for (int i = 0; i < ndim; i++) { xc[i] = gauge->X()[i]/2; }
    const int Nc_c = gauge->Ncolor() * 8; // 24
    const int Ns_c = 2; // staggered parity

    GaugeFieldParam gParam;
    memcpy(gParam.x, xc, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = gpu ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    auto precision = gauge->Precision();
    // right now the build Xinv routines only support single and double
    if (precision < QUDA_HALF_PRECISION) { 
      precision = QUDA_HALF_PRECISION;
    } else if (precision > QUDA_SINGLE_PRECISION) {
      precision = QUDA_SINGLE_PRECISION;
    }
    gParam.setPrecision( precision );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    Xinv = new cudaGaugeField(gParam);

    // Populate Xinv
    BuildStaggeredKahlerDiracInverse(*Xinv, *fatGauge, mass);
  }

  DiracImprovedStaggeredKD::DiracImprovedStaggeredKD(const DiracImprovedStaggeredKD &dirac) : DiracImprovedStaggered(dirac) {
    
    // deep copy Xinv
    GaugeFieldParam gParam(dirac.Xinv);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    Xinv = new cudaGaugeField(gParam);
    Xinv->copy(*dirac.Xinv);
  }

  DiracImprovedStaggeredKD::~DiracImprovedStaggeredKD() {
    if (Xinv) delete Xinv;
  }

  DiracImprovedStaggeredKD& DiracImprovedStaggeredKD::operator=(const DiracImprovedStaggeredKD &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
      
      // deep copy Xinv
      GaugeFieldParam gParam(dirac.Xinv);
      gParam.create = QUDA_NULL_FIELD_CREATE;
      Xinv = new cudaGaugeField(gParam);
      Xinv->copy(*dirac.Xinv);
    }
    return *this;
  }

  void DiracImprovedStaggeredKD::checkParitySpinor(const ColorSpinorField &in, const ColorSpinorField &out) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) {
      errorQuda("Staggered dslash requires 5-d fermion fields");
    }

    if (in.Precision() != out.Precision()) {
      errorQuda("Input and output spinor precisions don't match in dslash_quda");
    }

    if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("ColorSpinorFields are not full parity, in = %d, out = %d", 
                in.SiteSubset(), out.SiteSubset());
    }

    if (out.Volume()/out.X(4) != 2*gauge->VolumeCB() && out.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
      errorQuda("Spinor volume %lu doesn't match gauge volume %lu", out.Volume(), gauge->VolumeCB());
    }
  }

  void DiracImprovedStaggeredKD::Dslash(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity) const
  {
    errorQuda("The improved staggered Kahler-Dirac operator does not have a single parity form");
  }

  void DiracImprovedStaggeredKD::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity,
      const ColorSpinorField &x, const double &k) const
  {    
    errorQuda("The improved staggered Kahler-Dirac operator does not have a single parity form");
  }

  // Full staggered operator
  void DiracImprovedStaggeredKD::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    // Due to the staggered convention, the staggered part is applying
    // (  2m     -D_eo ) (x_e) = (b_e)
    // ( -D_oe   2m    ) (x_o) = (b_o)
    // ... but under the hood we need to catch the zero mass case.

    // TODO: add left vs right precond

    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp2, in);

    if (dagger == QUDA_DAG_NO) {
      // K-D op is right-block preconditioned
      ApplyStaggeredKahlerDiracInverse(*tmp2, in, *Xinv, false);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block
      if (mass == 0.) {
        ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim,
                               profile);
        flops += 1146ll * in.Volume();
      } else {
        ApplyImprovedStaggered(out, *tmp2, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim,
                             profile);
        flops += 1158ll * in.Volume();
      }
    } else { // QUDA_DAG_YES

      if (mass == 0.) {
        ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 0., in, QUDA_INVALID_PARITY, QUDA_DAG_NO, commDim,
                               profile);
        flops += 1146ll * in.Volume();
      } else {
        ApplyImprovedStaggered(*tmp2, in, *fatGauge, *longGauge, 2. * mass, in, QUDA_INVALID_PARITY, dagger, commDim,
                             profile);
        flops += 1158ll * in.Volume();
      }
      ApplyStaggeredKahlerDiracInverse(out, *tmp2, *Xinv, true);
      flops += (8ll * 48 - 2ll) * 48 * in.Volume() / 16; // for 2^4 block
    }

    deleteTmp(&tmp2, reset);

  }

  void DiracImprovedStaggeredKD::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    bool reset = newTmp(&tmp1, in);

    Mdag(*tmp1, in);
    M(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracImprovedStaggeredKD::KahlerDiracInv(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    ApplyStaggeredKahlerDiracInverse(out, in, *Xinv, dagger == QUDA_DAG_YES);
  }

  void DiracImprovedStaggeredKD::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			       ColorSpinorField &x, ColorSpinorField &b, 
			       const QudaSolutionType solType) const
  {
    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;  
  }

  void DiracImprovedStaggeredKD::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				   const QudaSolutionType solType) const
  {
    // do nothing

    // TODO: technically KD is a different type of preconditioning.
    // Should we support "preparing" and "reconstructing"?
  }

  void DiracImprovedStaggeredKD::updateFields(cudaGaugeField *gauge_in, cudaGaugeField *fat_gauge_in, cudaGaugeField *long_gauge_in,
                              cudaCloverField *clover_in)
  {
    Dirac::updateFields(fat_gauge_in, nullptr, nullptr, nullptr);
    fatGauge = fat_gauge_in;
    longGauge = long_gauge_in;

    // Recompute Xinv (I guess we should do that here?)
    BuildStaggeredKahlerDiracInverse(*Xinv, *fatGauge, mass);
  }

  void DiracImprovedStaggeredKD::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa,
                                              double mass, double mu, double mu_factor) const
  {
    errorQuda("DiracStaggeredKD does not support multigrid coarsening (yet)");
    //StaggeredCoarseOp(Y, X, T, *fatGauge, mass, QUDA_ASQTAD_DIRAC, QUDA_MATPC_INVALID);
  }

  void DiracImprovedStaggeredKD::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    DiracImprovedStaggered::prefetch(mem_space, stream);
    if (Xinv != nullptr) Xinv->prefetch(mem_space, stream);
  }

} // namespace quda
