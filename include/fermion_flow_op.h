#pragma once

#include <enum_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>

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

  /**
     @brief Factory: build the fermion-flow generator selected by type.
     @param[in] type The selected generator (default QUDA_FERMION_FLOW_LAPLACE_4D)
     @param[in] gauge_template Field whose params seed any helper gauge field
     @param[in] comm_dim Partitioned-dimension flags (must outlive the op)
     @param[in] parity Destination parity
     @param[in] profile Time profile for the dslash
     @return Owning pointer to the constructed operator
  */
  inline FermionFlowOp *createFermionFlowOp(QudaFermionFlowType type, const GaugeField &gauge_template,
                                            const int *comm_dim, int parity, TimeProfile &profile)
  {
    switch (type) {
    case QUDA_FERMION_FLOW_LAPLACE_4D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 4, -8.0, profile);
    case QUDA_FERMION_FLOW_LAPLACE_3D: return new LaplaceFlowOp(gauge_template, comm_dim, parity, 3, -6.0, profile);
    case QUDA_FERMION_FLOW_WILSON:
    case QUDA_FERMION_FLOW_STAGGERED:
    case QUDA_FERMION_FLOW_HISQ:
    case QUDA_FERMION_FLOW_HISQ_TRUNCATED: errorQuda("Fermion flow type %d is not yet implemented", type);
    default: errorQuda("Unknown fermion flow type %d", type);
    }
    return nullptr;
  }

} // namespace quda
