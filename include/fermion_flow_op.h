#pragma once

#include <memory>

#include <quda.h>
#include <enum_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <dirac_quda.h>
#include <blas_quda.h>

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
     helper links each sub-stage, using QUDA's standard staggered convention
     (MILC phases + anti-periodic temporal BC). With mass 0 this is the pure
     DdagD, whose free-field limit reduces to the gauge Laplacian on each parity.
     The generator returned is K_t = -DdagD = -MdagM (negative semi-definite for
     any mass, so the integrator's smoothing invariant holds).
  */
  class StaggeredFlowOp : public FermionFlowOp
  {
    GaugeField precise; // staggered-phased helper links fed to the Dirac op
    std::unique_ptr<Dirac> dirac;

  public:
    StaggeredFlowOp(const GaugeField &gauge_template, const QudaInvertParam &inv_param, const int *comm_dim,
                    int /* parity */, TimeProfile & /* profile */)
    {
      GaugeFieldParam gParam_helper(gauge_template);
      gParam_helper.create = QUDA_NULL_FIELD_CREATE;
      gParam_helper.reconstruct = QUDA_RECONSTRUCT_NO; // phased links: store explicitly
      gParam_helper.t_boundary = QUDA_ANTI_PERIODIC_T; // standard staggered temporal BC
      gParam_helper.staggeredPhaseType = QUDA_STAGGERED_PHASE_MILC;
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
      // copy the freshly flowed thin links, then (re)apply the MILC phases +
      // temporal BC. dirac reads precise through a stored pointer, so the
      // in-place refresh is seen on the next apply.
      if (precise.StaggeredPhaseApplied()) precise.removeStaggeredPhase();
      copyExtendedGauge(precise, thin_ext, QUDA_CUDA_FIELD_LOCATION);
      precise.exchangeGhost();
      precise.applyStaggeredPhase(); // uses staggeredPhaseType (MILC) + t_boundary set above
      dirac->updateFields(&precise, nullptr, nullptr, nullptr);
    }

    void apply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) override
    {
      dirac->MdagM(out, in); // MdagM = m^2 - D^2 >= 0; K_t = -DdagD = -MdagM
      blas::ax(-1.0, out);
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
    }
  };
#endif

  /**
     @brief Factory: build the fermion-flow generator selected by type.
     @param[in] type The selected generator (default QUDA_FERMION_FLOW_LAPLACE_4D)
     @param[in] inv_param Spinor/operator parameters (mass etc.) for Dirac-based operators
     @param[in] gauge_template Field whose params seed any helper gauge field
     @param[in] comm_dim Partitioned-dimension flags (must outlive the op)
     @param[in] parity Destination parity
     @param[in] profile Time profile for the dslash
     @return Owning pointer to the constructed operator
  */
  inline FermionFlowOp *createFermionFlowOp(QudaFermionFlowType type, const QudaInvertParam &inv_param,
                                            const GaugeField &gauge_template, const int *comm_dim, int parity,
                                            TimeProfile &profile)
  {
    switch (type) {
    case QUDA_FERMION_FLOW_LAPLACE_4D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 4, -8.0, profile);
    case QUDA_FERMION_FLOW_LAPLACE_3D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 3, -6.0, profile);
    case QUDA_FERMION_FLOW_STAGGERED:
#ifdef GPU_STAGGERED_DIRAC
      return new StaggeredFlowOp(gauge_template, inv_param, comm_dim, parity, profile);
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
    case QUDA_FERMION_FLOW_HISQ_TRUNCATED: errorQuda("Fermion flow type %d is not yet implemented", type);
    default: errorQuda("Unknown fermion flow type %d", type);
    }
    return nullptr;
  }

} // namespace quda
