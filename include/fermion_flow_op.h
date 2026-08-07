#pragma once

#include <memory>

#include <quda.h>
#include <enum_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <llfat_quda.h>
#include <unitarization_links.h>
#include <malloc_quda.h>

/**
   @file fermion_flow_op.h

   @brief Pluggable generator K_t for the fermion gradient flow driven by
   performGFlowQuda. The three-stage Runge-Kutta integrator only touches the
   flow operator at two points per sub-stage:

     1. update(thin_ext) -- refresh the operator's gauge-derived state from this
        sub-stage's flowed, extended thin links.
     2. apply(out, in)   -- out = K_t in, where K_t is a smoothing
        (negative-semidefinite) operator: the gauge Laplacian, or -DdagD.

   Everything else (RK coefficients, axpy accumulation, gauge stepping,
   measurement, host I/O) is operator-independent. The default operator is the
   4D gauge-covariant Laplacian, which reproduces the legacy performGFlowQuda
   behavior exactly.
*/

namespace quda
{

  class FermionFlowOp
  {
  public:
    virtual ~FermionFlowOp() = default;

    /**
       @brief Refresh the operator's gauge-derived state from this sub-stage's
       flowed, extended thin links.
       @param[in] thin_ext The flowed thin gauge field (extended/haloed)
    */
    virtual void update(const GaugeField &thin_ext) = 0;

    /**
       @brief Apply the smoothing generator: out = K_t in.
       @param[out] out Result field set
       @param[in] in Input field set
    */
    virtual void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) = 0;
  };

  /**
     @brief Gauge-covariant Laplacian flow generator. Wraps ApplyLaplace exactly
     as the legacy performGFlowQuda loop did: update() copies the flowed thin
     links into a helper field and exchanges ghosts; apply() forms
     a*(hopping) + b*in. With (dir=4, b=-8) this is the full 4D Laplacian -- the
     default, legacy operator. With (dir=3, b=-6) it is the spatial Laplacian.
  */
  class LaplaceFlowOp : public FermionFlowOp
  {
    GaugeField precise;  // helper gauge field fed to the Laplace operator
    const int *comm_dim; // which dimensions are partitioned (caller-owned)
    int parity;
    int dir;  // 4 = full 4D, 3 = spatial (t omitted)
    double a; // hopping coefficient
    double b; // diagonal coefficient
    TimeProfile &profile;

  public:
    /**
       @param[in] gauge_template Field whose params seed the helper gauge field
       @param[in] comm_dim Partitioned-dimension flags (must outlive this op)
       @param[in] parity Destination parity passed to ApplyLaplace
       @param[in] dir 4 for full 4D Laplacian, 3 for spatial
       @param[in] b Diagonal coefficient (-8 for 4D, -6 for 3D)
       @param[in] profile Time profile for the dslash
    */
    LaplaceFlowOp(const GaugeField &gauge_template, const int *comm_dim, int parity, int dir, double b,
                  TimeProfile &profile) :
      comm_dim(comm_dim), parity(parity), dir(dir), a(1.0), b(b), profile(profile)
    {
      // Mirror the legacy performGFlowQuda helper-field creation exactly: only
      // override create; inherit reconstruct (etc.) from the template so the
      // default-operator path reproduces the legacy results byte-for-byte.
      GaugeFieldParam gParam_helper(gauge_template);
      gParam_helper.create = QUDA_NULL_FIELD_CREATE;
      precise = GaugeField(gParam_helper);
    }

    void update(const GaugeField &thin_ext) override
    {
      copyExtendedGauge(precise, thin_ext, QUDA_CUDA_FIELD_LOCATION);
      precise.exchangeGhost();
    }

    void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) override
    { ApplyLaplace(out, in, precise, dir, a, b, in, parity, comm_dim, profile); }
  };

#ifdef GPU_STAGGERED_DIRAC
  /**
     @brief Naive (unimproved) staggered -DdagD flow generator, built on the
     flowed thin links. Unlike the Laplacian, the staggered operator carries KS
     phases and a temporal boundary condition; these are (re)applied to the
     helper links each sub-stage, using the caller-supplied KS phase convention
     (smear_param.staggered_phase_type, default MILC) and anti-periodic temporal
     BC. The generator returned is K_t = -1/4 MdagM (negative semi-definite for
     any mass, so the integrator's smoothing invariant holds).

     The 1/4 is a normalization, not a convention choice. QUDA's staggered Dirac
     operator uses the MILC "2m" convention, M = 2m + Dhop (so MdagM = 4m^2 - Dhop^2);
     that convention is correct for solvers but is twice the operator whose free-field
     limit is the covariant Laplacian. The fermion gradient flow (Luescher,
     arXiv:1302.5246) is d_t chi = Delta chi with Delta the covariant Laplacian,
     fixed by the heat-kernel smearing radius sqrt(8t): free-field eigenvalue ~ -p^2.
     -1/4 MdagM has free-field eigenvalue -sum_mu sin^2(p_mu) ~ -p^2, matching
     LaplaceFlowOp (and QEX's staggered -DdagD flow); the bare -MdagM would be 4x too
     strong (~ -4p^2), giving the wrong smearing radius. With mass 0 this reduces to
     1/4 Dhop^2, whose free-field limit is the gauge Laplacian on each parity.
  */
  class StaggeredFlowOp : public FermionFlowOp
  {
    GaugeField precise; // staggered-phased helper links fed to the Dirac op
    std::unique_ptr<Dirac> dirac;

  public:
    StaggeredFlowOp(const GaugeField &gauge_template, const QudaInvertParam &inv_param,
                    const QudaGaugeSmearParam &smear_param, const int *comm_dim, int /* parity */,
                    TimeProfile & /* profile */)
    {
      GaugeFieldParam gParam_helper(gauge_template);
      gParam_helper.create = QUDA_NULL_FIELD_CREATE;
      gParam_helper.reconstruct = QUDA_RECONSTRUCT_NO;                     // phased links: store explicitly
      // gParam_helper.t_boundary = QUDA_ANTI_PERIODIC_T;                     // standard staggered temporal BC
      gParam_helper.t_boundary = QUDA_PERIODIC_T;                     // standard staggered temporal BC
      gParam_helper.staggeredPhaseType = smear_param.staggered_phase_type; // caller-supplied KS convention
      precise = GaugeField(gParam_helper);

      DiracParam diracParam;
      diracParam.type = QUDA_STAGGERED_DIRAC;
      diracParam.mass = inv_param.mass; // 0 for a pure DdagD flow
      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = QUDA_MATPC_EVEN_EVEN; // full operator; MdagM is parity-independent,
                                                   // but the Dirac ctor requires a valid matpcType
      diracParam.gauge = &precise;
      for (int i = 0; i < 4; i++) diracParam.commDim[i] = comm_dim[i];
      dirac.reset(Dirac::create(diracParam));
    }

    void update(const GaugeField &thin_ext) override
    {
      // The helper holds phased links from the previous sub-stage; un-phase first
      // to reset the staggered-phase flag (the data is overwritten next anyway),
      // copy the freshly flowed thin links, then (re)apply the configured KS
      // phases + temporal BC. dirac reads precise through a stored pointer, so the
      // in-place refresh is seen on the next apply.
      if (precise.StaggeredPhaseApplied()) precise.removeStaggeredPhase();
      copyExtendedGauge(precise, thin_ext, QUDA_CUDA_FIELD_LOCATION);
      precise.exchangeGhost();
        
      precise.applyStaggeredPhase(); // uses staggeredPhaseType (caller-supplied) + t_boundary set above
      dirac->updateFields(&precise, nullptr, nullptr, nullptr);
    }

    void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) override
    {
      dirac->MdagM(out, in);  // MdagM = 4m^2 - D^2 >= 0 (QUDA staggered M = 2m + Dhop)
      blas::ax(-0.25, out);   // K_t = -1/4 MdagM = -m^2 + 1/4 Dhop^2; see class note
    }
  };
#endif

#ifdef GPU_WILSON_DIRAC
  /**
     @brief Wilson -DdagD flow generator, built on the flowed thin links. The
     Wilson operator (four-component spinors) uses kappa (inv_param.kappa) and the
     standard anti-periodic temporal boundary condition; the latter is applied by
     the gauge accessor at read time from the helper field's t_boundary, so -- unlike
     staggered -- no phase needs to be baked into the link data. The generator is
     K_t = -DdagD = -MdagM (negative semi-definite for any kappa).
  */
  class WilsonFlowOp : public FermionFlowOp
  {
    GaugeField precise; // helper links fed to the Wilson Dirac op
    std::unique_ptr<Dirac> dirac;

  public:
    WilsonFlowOp(const GaugeField &gauge_template, const QudaInvertParam &inv_param, const int *comm_dim,
                 int /* parity */, TimeProfile & /* profile */)
    {
      GaugeFieldParam gParam_helper(gauge_template);
      gParam_helper.create = QUDA_NULL_FIELD_CREATE;
      gParam_helper.reconstruct = QUDA_RECONSTRUCT_NO;
      gParam_helper.t_boundary = QUDA_ANTI_PERIODIC_T; // standard Wilson temporal BC (applied by the gauge accessor)
      precise = GaugeField(gParam_helper);

      DiracParam diracParam;
      diracParam.type = QUDA_WILSON_DIRAC;
      diracParam.kappa = inv_param.kappa;
      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = QUDA_MATPC_EVEN_EVEN; // full operator; MdagM is parity-independent
      diracParam.gauge = &precise;
      for (int i = 0; i < 4; i++) diracParam.commDim[i] = comm_dim[i];
      dirac.reset(Dirac::create(diracParam));
    }

    void update(const GaugeField &thin_ext) override
    {
      copyExtendedGauge(precise, thin_ext, QUDA_CUDA_FIELD_LOCATION);
      precise.exchangeGhost();
      dirac->updateFields(&precise, nullptr, nullptr, nullptr);
    }

    void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) override
    {
      dirac->MdagM(out, in); // K_t = -DdagD = -MdagM
      blas::ax(-1.0, out);
      // NOTE: unlike the staggered/HISQ ops, the Wilson generator's normalization has
      // not been reconciled to the covariant-Laplacian (sqrt(8t)) flow convention. The
      // kappa normalization differs from the staggered "2m" case, so the staggered 1/4
      // does NOT carry over -- resolve separately before trusting Wilson flow times.
    }
  };
#endif

#ifdef GPU_STAGGERED_DIRAC
  /**
     @brief HISQ -DdagD flow generator (full and truncated). Each sub-stage the
     two-level fat one-link field X is rebuilt from the flowed thin links:
       phase(thin) -> extend -> fat7 -> unitarize -> extend -> asqtad-fat = X,
     and (for full HISQ) the Naik three-link field L = longKSLink(W). KS phases +
     anti-periodic T are baked into the thin links before fattening (using the
     caller-supplied smear_param.staggered_phase_type), so they propagate into X
     and L. K_t = -1/4 MdagM (see StaggeredFlowOp for why the 1/4 is required):
       - with_long = true  : full HISQ -- DiracImprovedStaggered(fat = X, long = L).
       - with_long = false : truncated HISQ (Option A) -- the Naik term dropped,
                             applied with the cheap one-link DiracStaggered(gauge = X).
     The level-1 fat7 and level-2 asqtad path coefficients are caller-supplied
     (smear_param.hisq_fat7_coeff / hisq_asqtad_coeff), final: tadpole-scaled and,
     for the asqtad set, with the Naik correction eps_N already folded in (QUDA
     does not own the HISQ action). They are required when a HISQ flow is selected.
  */
  class HisqFlowOp : public FermionFlowOp
  {
    bool with_long;             // true = full HISQ (build + use Naik long links)
    GaugeFieldParam thin_param; // non-extended phased-thin template (MILC + anti-periodic)
    GaugeFieldParam raw_param;  // fattening output template (GENERAL_LINKS, no halo)
    GaugeField fat;             // X, dslash-ready (ASQTAD_FAT_LINKS, PAD)
    GaugeField lng;             // L, dslash-ready (ASQTAD_LONG_LINKS, PAD) -- only if with_long
    std::unique_ptr<Dirac> dirac;
    double c_fat7[6];   // level-1 fat7 path coefficients
    double c_asqtad[6]; // level-2 asqtad path coefficients

  public:
    HisqFlowOp(const GaugeField &gauge_template, const QudaInvertParam &inv_param, const QudaGaugeSmearParam &smear_param,
               const int *comm_dim, int /* parity */, TimeProfile & /* profile */, bool with_long) :
      with_long(with_long)
    {
      // HISQ path coefficients are caller-supplied (final: tadpole-scaled and, for
      // the level-2 asqtad set, eps_N-folded). The application owns the action, so
      // we do not compute them here. They are required for a HISQ flow.
      bool coeff_set = false;
      for (int i = 0; i < 6; i++) {
        c_fat7[i] = smear_param.hisq_fat7_coeff[i];
        c_asqtad[i] = smear_param.hisq_asqtad_coeff[i];
        if (c_fat7[i] != 0.0 || c_asqtad[i] != 0.0) coeff_set = true;
      }
      if (!coeff_set) errorQuda("HISQ fermion flow requires smear_param.hisq_fat7_coeff / hisq_asqtad_coeff to be set");

      // Phased-thin template: KS phases + anti-periodic T baked into the thin links.
      thin_param = GaugeFieldParam(gauge_template);
      thin_param.create = QUDA_NULL_FIELD_CREATE;
      thin_param.reconstruct = QUDA_RECONSTRUCT_NO;
      thin_param.t_boundary = QUDA_ANTI_PERIODIC_T;
      thin_param.staggeredPhaseType = smear_param.staggered_phase_type;

      // Fattening-output template: general links, explicit storage, no halo.
      raw_param = GaugeFieldParam(gauge_template);
      raw_param.create = QUDA_ZERO_FIELD_CREATE;
      raw_param.reconstruct = QUDA_RECONSTRUCT_NO;
      raw_param.link_type = QUDA_GENERAL_LINKS;
      raw_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

      // Dslash-ready fat field X (rebuilt in place each sub-stage). The anti-periodic
      // T and KS phases are already baked into the data, so use periodic T here.
      GaugeFieldParam fatParam(gauge_template);
      fatParam.create = QUDA_NULL_FIELD_CREATE;
      fatParam.reconstruct = QUDA_RECONSTRUCT_NO;
      fatParam.link_type = QUDA_ASQTAD_FAT_LINKS;
      fatParam.t_boundary = QUDA_PERIODIC_T;
      fatParam.staggeredPhaseType = smear_param.staggered_phase_type;
      fat = GaugeField(fatParam);

      if (with_long) {
        // Dslash-ready Naik long field L. nFace must be set to 3 explicitly: the
        // GaugeFieldParam(GaugeField&) copy-ctor inherits nFace from the (Wilson)
        // template rather than deriving it from link_type.
        GaugeFieldParam lngParam(gauge_template);
        lngParam.create = QUDA_NULL_FIELD_CREATE;
        lngParam.reconstruct = QUDA_RECONSTRUCT_NO;
        lngParam.link_type = QUDA_ASQTAD_LONG_LINKS;
        lngParam.nFace = 3;
        lngParam.t_boundary = QUDA_PERIODIC_T;
        lngParam.staggeredPhaseType = smear_param.staggered_phase_type;
        lng = GaugeField(lngParam);
      }

      setUnitarizeLinksConstants(1e-14, 1e-10, true, false, 1e-6, 1e-6);

      DiracParam diracParam;
      diracParam.mass = inv_param.mass;
      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = QUDA_MATPC_EVEN_EVEN;
      for (int i = 0; i < 4; i++) diracParam.commDim[i] = comm_dim[i];
      if (with_long) {
        diracParam.type = QUDA_ASQTAD_DIRAC; // full HISQ: fat one-link X + Naik long L
        diracParam.fatGauge = &fat;
        diracParam.longGauge = &lng;
      } else {
        diracParam.type = QUDA_STAGGERED_DIRAC; // truncated HISQ: one-link operator on X
        diracParam.gauge = &fat;
      }
      dirac.reset(Dirac::create(diracParam));
    }

    void update(const GaugeField &thin_ext) override
    {
      const lat_dim_t &R = thin_ext.R();
      // Phase the (trimmed) thin links, then re-extend so the phases reach the halo.
      GaugeField thin(thin_param);
      copyExtendedGauge(thin, thin_ext, QUDA_CUDA_FIELD_LOCATION);
      thin.exchangeGhost();
      thin.applyStaggeredPhase();
      std::unique_ptr<GaugeField> thinEx(createExtendedGauge(thin, R));
      // Level 1: V = fat7(thin); W = unitarize(V).
      GaugeField V(raw_param), W(raw_param);
      fatKSLink(V, *thinEx, c_fat7);

      int *fails_h = static_cast<int *>(host_pinned_malloc(sizeof(int)));
      int *fails_d = static_cast<int *>(get_mapped_device_pointer(fails_h));
      *fails_h = 0;
      unitarizeLinks(W, V, fails_d);
      if (*fails_h > 0) errorQuda("HISQ flow unitarization: %d failures", *fails_h);
      host_free(fails_h);
      // Level 2: X = asqtad-fat(W) [+ L = Naik long(W) for full HISQ].
      std::unique_ptr<GaugeField> WEx(createExtendedGauge(W, R));
      GaugeField Xraw(raw_param);
      fatKSLink(Xraw, *WEx, c_asqtad);
      fat.copy(Xraw);
      fat.exchangeGhost();

      if (with_long) {
        // longKSLink output must match lng's link_type/nFace (THREE_LINKS, nFace 3)
        // so the subsequent copy passes checkField -- unlike the fat path, where
        // ASQTAD_FAT_LINKS aliases GENERAL_LINKS (link_type 1, nFace 1).
        GaugeFieldParam lngRawParam(raw_param);
        lngRawParam.link_type = QUDA_ASQTAD_LONG_LINKS;
        lngRawParam.nFace = 3;
        GaugeField Lraw(lngRawParam);
        longKSLink(Lraw, *WEx, c_asqtad);
        lng.copy(Lraw);
        lng.exchangeGhost();
        dirac->updateFields(nullptr, &fat, &lng, nullptr); // DiracImprovedStaggered: (gauge ignored, fat, long)
      } else {
        dirac->updateFields(&fat, nullptr, nullptr, nullptr); // DiracStaggered: gauge = fat
      }
    }

    void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) override
    {
      dirac->MdagM(out, in);  // MdagM = 4m^2 - D^2 (QUDA staggered/HISQ M = 2m + Dhop)
      blas::ax(-0.25, out);   // K_t = -1/4 MdagM; see StaggeredFlowOp note on the 1/4
    }
  };
#endif

  /**
     @brief Factory: build the fermion-flow generator selected by type.
     @param[in] type The selected generator (default QUDA_FERMION_FLOW_LAPLACE_4D)
     @param[in] inv_param Spinor/operator parameters (mass, kappa) for Dirac-based operators
     @param[in] smear_param Flow parameters (staggered phase convention, HISQ path coefficients)
     @param[in] gauge_template Field whose params seed any helper gauge field
     @param[in] comm_dim Partitioned-dimension flags (must outlive the op)
     @param[in] parity Destination parity
     @param[in] profile Time profile for the dslash
     @return Owning pointer to the constructed operator
  */
  inline FermionFlowOp *createFermionFlowOp(QudaFermionFlowType type, const QudaInvertParam &inv_param,
                                            const QudaGaugeSmearParam &smear_param, const GaugeField &gauge_template,
                                            const int *comm_dim, int parity, TimeProfile &profile)
  {
    switch (type) {
    case QUDA_FERMION_FLOW_LAPLACE_4D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 4, -8.0, profile);
    case QUDA_FERMION_FLOW_LAPLACE_3D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 3, -6.0, profile);
    case QUDA_FERMION_FLOW_STAGGERED:
#ifdef GPU_STAGGERED_DIRAC
      return new StaggeredFlowOp(gauge_template, inv_param, smear_param, comm_dim, parity, profile);
#else
      errorQuda("Staggered fermion flow requires QUDA_DIRAC_STAGGERED to be enabled");
#endif
    case QUDA_FERMION_FLOW_WILSON:
#ifdef GPU_WILSON_DIRAC
      return new WilsonFlowOp(gauge_template, inv_param, comm_dim, parity, profile);
#else
      errorQuda("Wilson fermion flow requires QUDA_DIRAC_WILSON to be enabled");
#endif
    case QUDA_FERMION_FLOW_HISQ:
#ifdef GPU_STAGGERED_DIRAC
      return new HisqFlowOp(gauge_template, inv_param, smear_param, comm_dim, parity, profile, /* with_long = */ true);
#else
      errorQuda("HISQ fermion flow requires QUDA_DIRAC_STAGGERED to be enabled");
#endif
    case QUDA_FERMION_FLOW_HISQ_TRUNCATED:
#ifdef GPU_STAGGERED_DIRAC
      return new HisqFlowOp(gauge_template, inv_param, smear_param, comm_dim, parity, profile, /* with_long = */ false);
#else
      errorQuda("Truncated-HISQ fermion flow requires QUDA_DIRAC_STAGGERED to be enabled");
#endif
    default: errorQuda("Unknown fermion flow type %d", type);
    }
    return nullptr;
  }

} // namespace quda
