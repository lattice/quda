#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <tune_quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <worker.h>

namespace quda {

  /**
    @param pack Sets whether to use a kernel to pack the T dimension
    */
  void setKernelPackT(bool pack);

  /**
    @return Whether the T dimension is kernel packed or not
    */
  bool getKernelPackT();

  void pushKernelPackT(bool pack);
  void popKernelPackT();

  /**
     @brief Helper function that sets which dimensions the packing
     kernel should be packing for.
     @param[in] dim_pack Array that specifies which dimenstions need
     to be packed.
  */
  void setPackComms(const int *dim_pack);

  bool getDslashLaunch();

  void createDslashEvents();
  void destroyDslashEvents();

  /**
     @brief Driver for applying the Wilson stencil

     out = D * in

     where D is the gauged Wilson linear operator.

     If kappa is non-zero, the operation is given by out = x + kappa * D in.
     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] kappa Scale factor applied
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double kappa,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the Wilson-clover stencil

     out = A * x + kappa * D * in

     where D is the gauged Wilson linear operator.

     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in Input field that D is applied to
     @param[in] x Input field that A is applied to
     @param[in] U The gauge field used for the operator
     @param[in] A The clover field used for the operator
     @param[in] kappa Scale factor applied
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
      double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
       @brief Driver for applying the Wilson-clover stencil

       out = A * x + kappa * D * in

       where D is the gauged Wilson linear operator.

       This operator can be applied to both single parity
       (checker-boarded) fields, or to full fields.

       @param[out] out The output result field
       @param[in] in Input field that D is applied to
       @param[in] x Input field that A is applied to
       @param[in] U The gauge field used for the operator
       @param[in] A The clover field used for the operator
       @param[in] kappa Scale factor applied
       @param[in] mu Twist factor
       @param[in] x Vector field we accumulate onto to
       @param[in] parity Destination parity
       @param[in] dagger Whether this is for the dagger operator
       @param[in] comm_override Override for which dimensions are partitioned
       @param[in] profile The TimeProfile used for profiling the dslash
    */
  void ApplyWilsonCloverHasenbuschTwist(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                        const CloverField &A, double kappa, double mu, const ColorSpinorField &x,
                                        int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the preconditioned Wilson-clover stencil

     out = A^{-1} * D * in + x

     where D is the gauged Wilson linear operator and A is the clover
     field.  This operator can (at present) be applied to only single
     parity (checker-boarded) fields.  When the dagger operator is
     requested, we do not transpose the order of operations, e.g.

     out = A^{-\dagger} D^\dagger  (no xpay term)

     Although not a conjugate transpose of the regular operator, this
     variant is used to enable kernel fusion between the application
     of D and the subsequent application of A, e.g., in the symmetric
     dagger operator we need to apply

     M = (1 - kappa^2 D^{\dagger} A^{-1} D{^\dagger} A^{-1} )

     and since cannot fuse D{^\dagger} A^{-\dagger}, we instead fused
     A^{-\dagger} D{^\dagger}.

     If kappa is non-zero, the operation is given by out = x + kappa * A^{-1} D in.
     This operator can (at present) be applied to only single parity
     (checker-boarded) fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] A The clover field used for the operator
     @param[in] kappa Scale factor applied
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyWilsonCloverPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      const CloverField &A, double kappa, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
      TimeProfile &profile);

  /**
     @brief Driver for applying the twisted-mass stencil

     out = a * D * in + (1 + i*b*gamma_5) * x

     where D is the gauged Wilson linear operator.

     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied to Wilson term (typically -kappa)
     @param[in] b Twist factor applied (typically 2*mu*kappa)
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */

  /**
        @brief Driver for applying the Wilson-clover with twist for Hasenbusch

        out = (1 +/- ig5 b A) * x + kappa * A^{-1}D * in

        where D is the gauged Wilson linear operator.

        This operator can be applied to both single parity
        (checker-boarded) fields, or to full fields.

        @param[out] out The output result field
        @param[in] in Input field that D is applied to
        @param[in] x Input field that A is applied to
        @param[in] U The gauge field used for the operator
        @param[in] A The clover field used for the operator
        @param[in] kappa Scale factor applied
        @param[in] b Twist factor applied
        @param[in] x Vector field we accumulate onto to
        @param[in] parity Destination parity
        @param[in] dagger Whether this is for the dagger operator
        @param[in] comm_override Override for which dimensions are partitioned
        @param[in] profile The TimeProfile used for profiling the dslash
     */
  void ApplyWilsonCloverHasenbuschTwistPCClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                                 const CloverField &A, double kappa, double mu,
                                                 const ColorSpinorField &x, int parity, bool dagger,
                                                 const int *comm_override, TimeProfile &profile);

  /**
        @brief Driver for applying the Wilson-clover stencil with thist for Hasenbusch

        out = (1 +/- ig5 b A) * x + kappa * D * in

        where D is the gauged Wilson linear operator.

        This operator can be applied to both single parity
        (checker-boarded) fields, or to full fields.

        @param[out] out The output result field
        @param[in] in Input field that D is applied to
        @param[in] x Input field that A is applied to
        @param[in] U The gauge field used for the operator
        @param[in] A The clover field used for the operator
        @param[in] kappa Scale factor applied
        @param[in] b Twist factor applied
        @param[in] x Vector field we accumulate onto to
        @param[in] parity Destination parity
        @param[in] dagger Whether this is for the dagger operator
        @param[in] comm_override Override for which dimensions are partitioned
        @param[in] profile The TimeProfile used for profiling the dslash
     */
  void ApplyWilsonCloverHasenbuschTwistPCNoClovInv(ColorSpinorField &out, const ColorSpinorField &in,
                                                   const GaugeField &U, const CloverField &A, double kappa, double mu,
                                                   const ColorSpinorField &x, int parity, bool dagger,
                                                   const int *comm_override, TimeProfile &profile);

  // old
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
                        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile);

  /**
     @brief Driver for applying the preconditioned twisted-mass stencil

     out = a*(1 + i*b*gamma_5) * D * in + x

     where D is the gauged Wilson linear operator.  This operator can
     (at present) be applied to only single parity (checker-boarded)
     fields.  For the dagger operator, we generally apply the
     conjugate transpose operator

     out = x + D^\dagger A^{-\dagger}

     with the additional asymmetric special case, where we apply do not
     transpose the order of operations

     out = A^{-\dagger} D^\dagger  (no xpay term)

     This variant is required when have the asymmetric preconditioned
     operator and require the preconditioned twist term to remain in
     between the applications of D.  This would be combined with a
     subsequent non-preconditioned dagger operator, A*x - kappa^2 D, to
     form the full operator.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied to Wilson term ( typically kappa^2 / (1 + b*b) )
     @param[in] b Twist factor applied (typically -2*kappa*mu)
     @param[in] xpay Whether to do xpay or not
     @param[in] x Vector field we accumulate onto to when xpay is true
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] asymmetric Whether this is for the asymmetric preconditioned dagger operator (a*(1 - i*b*gamma_5) * D^dagger * in)
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
      double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
      const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the non-degenerate twisted-mass
     stencil

     out = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1) * x

     where D is the gauged Wilson linear operator.  The quark fields
     out, in and x are five dimensional, with the fifth dimension
     corresponding to the flavor dimension.  The convention is that
     the first 4-d slice (s=0) corresponds to the positive twist and
     the second slice (s=1) corresponds to the negative twist.

     This operator can be applied to both single parity
     (4d checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied to Wilson term (typically -kappa)
     @param[in] b Chiral twist factor applied (typically 2*mu*kappa)
     @param[in] c Flavor twist factor applied (typically -2*epsilon*kappa)
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyNdegTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
      double c, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the preconditioned non-degenerate
     twisted-mass stencil

     out = a * (1 + i*b*gamma_5*tau_3 + c*tau_1) * D * in + x

     where D is the gauged Wilson linear operator.  The quark fields
     out, in and x are five dimensional, with the fifth dimension
     corresponding to the flavor dimension.  The convention is that
     the first 4-d slice (s=0) corresponds to the positive twist and
     the second slice (s=1) corresponds to the negative twist.

     This operator can (at present) be applied to only single parity
     (checker-boarded) fields.

     For the dagger operator, we generally apply the
     conjugate transpose operator

     out = x + D^\dagger A^{-\dagger}

     with the additional asymmetric special case, where we apply do not
     transpose the order of operations

     out = A^{-\dagger} D^\dagger  (no xpay term)

     This variant is required when have the asymmetric preconditioned
     operator and require the preconditioned twist term to remain in
     between the applications of D.  This would be combined with a
     subsequent non-preconditioned dagger operator, A*x - kappa^2 D, to
     form the full operator.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied to Wilson term (typically -kappa^2/(1 + b*b -c*c) )
     @param[in] b Chiral twist factor applied (typically -2*mu*kappa)
     @param[in] c Flavor twist factor applied (typically 2*epsilon*kappa)
     @param[in] xpay Whether to do xpay or not
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] asymmetric Whether this is for the asymmetric preconditioned dagger operator (a*(1 - i*b*gamma_5) * D^dagger * in)
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyNdegTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      double a, double b, double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
      const int *comm_override, TimeProfile &profile);

  /**
       @brief Driver for applying the twisted-clover stencil

       out = a * D * in + (C + i*b*gamma_5) * x

       where D is the gauged Wilson linear operator, and C is the clover
       field.

       This operator can be applied to both single parity
       (4d checker-boarded) fields, or to full fields.

       @param[out] out The output result field
       @param[in] in The input field
       @param[in] U The gauge field used for the operator
       @param[in] C The clover field used for the operator
       @param[in] a Scale factor applied to Wilson term (typically -kappa)
       @param[in] b Chiral twist factor applied (typically 2*mu*kappa)
       @param[in] x Vector field we accumulate onto to
       @param[in] parity Destination parity
       @param[in] dagger Whether this is for the dagger operator
       @param[in] comm_override Override for which dimensions are partitioned
       @param[in] profile The TimeProfile used for profiling the dslash
    */
  void ApplyTwistedClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &C,
      double a, double b, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
      TimeProfile &profile);

  /**
     @brief Driver for applying the preconditioned twisted-clover stencil

     out = a * (C + i*b*gamma_5)^{-1} * D * in + x
         = a * C^{-2} (C - i*b*gamma_5) * D * in + x
         = A^{-1} * D * in + x

     where D is the gauged Wilson linear operator and C is the clover
     field.  This operator can (at present) be applied to only single
     parity (checker-boarded) fields.  When the dagger operator is
     requested, we do not transpose the order of operations, e.g.

     out = A^{-\dagger} D^\dagger  (no xpay term)

     Although not a conjugate transpose of the regular operator, this
     variant is used to enable kernel fusion between the application
     of D and the subsequent application of A, e.g., in the symmetric
     dagger operator we need to apply

     M = (1 - kappa^2 D^{\dagger} A^{-\dagger} D{^\dagger} A^{-\dagger} )

     and since cannot fuse D{^\dagger} A^{-\dagger}, we instead fused
     A^{-\dagger} D{^\dagger}.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] C The clover field used for the operator
     @param[in] a Scale factor applied to Wilson term ( typically 1 / (1 + b*b) or kappa^2 / (1 + b*b) )
     @param[in] b Twist factor applied (typically -2*kappa*mu)
     @param[in] xpay Whether to do xpay or not
     @param[in] x Vector field we accumulate onto to when xpay is true
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyTwistedCloverPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      const CloverField &C, double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
      const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the Domain-wall 5-d stencil to a
     5-d vector with 5-d preconditioned data order

     out = D_5 * in

     where D_5 is the 5-d wilson linear operator with fifth dimension
     boundary condition set by the fermion mass.

     If a is non-zero, the operation is given by out = x + a * D_5 in.
     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied (typically -kappa_5)
     @param[in] m_f Fermion mass parameter
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyDomainWall5D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_f,
      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the batched Wilson 4-d stencil to a
     5-d vector with 4-d preconditioned data order

     out = D * in

     where D is the gauged Wilson linear operator.

     If a is non-zero, the operation is given by out = x + a * D in.
     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the operator
     @param[in] a Scale factor applied
     @param[in] m_5 Wilson mass shift
     @param[in] b_5 Mobius coefficient array (length Ls)
     @param[in] c_5 Mobius coefficient array (length Ls)
     @param[in] x Vector field we accumulate onto to
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */

  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                         const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile);

  enum Dslash5Type {
    DSLASH5_DWF,
    DSLASH5_MOBIUS_PRE,
    DSLASH5_MOBIUS,
    M5_INV_DWF,
    M5_INV_MOBIUS,
    M5_INV_ZMOBIUS,
    M5_EOFA,
    M5INV_EOFA
  };

  /**
    Applying the following five kernels in the order of 4-0-1-2-3 is equivalent to applying
    the full even-odd preconditioned symmetric MdagM operator:
    op = (1 - M5inv * D4 * D5pre * M5inv * D4 * D5pre)^dag
        * (1 - M5inv * D4 * D5pre * M5inv * D4 * D5pre)
  */
  enum class MdwfFusedDslashType {
    D4_D5INV_D5PRE,
    D4_D5INV_D5INVDAG,
    D4DAG_D5PREDAG_D5INVDAG,
    D4DAG_D5PREDAG,
    D5PRE,
  };

  /**
     @brief Apply either the domain-wall / mobius Dslash5 operator or
     the M5 inverse operator.  In the current implementation, it is
     expected that the color-spinor fields are 4-d preconditioned.
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] x Auxilary input color-spinor field
     @param[in] m_f Fermion mass parameter
     @param[in] m_5 Wilson mass shift
     @param[in] b_5 Mobius coefficient array (length Ls)
     @param[in] c_5 Mobius coefficient array (length Ls)
     @param[in] a Scale factor use in xpay operator
     @param[in] dagger Whether this is for the dagger operator
     @param[in] type Type of dslash we are applying
  */
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
                    double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type);

  // Tensor core functions for Mobius DWF
  namespace mobius_tensor_core
  {
    void apply_fused_dslash(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, ColorSpinorField &y,
                            const ColorSpinorField &x, double m_f, double m_5, const Complex *b_5, const Complex *c_5,
                            bool dagger, int parity, int shift[4], int halo_shift[4], MdwfFusedDslashType type);
  }

  // The EOFA stuff
  namespace mobius_eofa
  {
    void apply_dslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
                       double m_5, const Complex *b_5, const Complex *c_5, double a, int eofa_pm, double inv,
                       double kappa, const double *eofa_u, const double *eofa_x, const double *eofa_y,
                       double sherman_morrison, bool dagger, Dslash5Type type);
  }

  /**
     @brief Driver for applying the Laplace stencil

     out = - kappa * A * in

     where A is the gauge laplace linear operator.

     If x is defined, the operation is given by out = x - kappa * A in.
     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the gauge Laplace
     @param[in] dir Direction of the derivative 0,1,2,3 to omit (-1 is full 4D)
     @param[in] a Scale factor applied to derivative
     @param[in] b Scale factor applied to aux field
     @param[in] x Vector field we accumulate onto to
  */
  void ApplyLaplace(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int dir, double a, double b,
                    const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Driver for applying the covariant derivative

     out = U * in

     where U is the gauge field in a particular direction.

     This operator can be applied to both single parity
     (checker-boarded) fields, or to full fields.

     @param[out] out The output result field
     @param[in] in The input field
     @param[in] U The gauge field used for the covariant derivative
     @param[in] mu Direction of the derivative. For mu > 3 it goes backwards
     @param[in] parity Destination parity
     @param[in] dagger Whether this is for the dagger operator
     @param[in] comm_override Override for which dimensions are partitioned
     @param[in] profile The TimeProfile used for profiling the dslash
  */
  void ApplyCovDev(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int mu, int parity,
                   bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Apply clover-matrix field to a color-spinor field
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] clover Clover-matrix field
     @param[in] inverse Whether we are applying the inverse or not
     @param[in] Field parity (if color-spinor field is single parity)
  */
  void ApplyClover(
      ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover, bool inverse, int parity);

  /**
     @brief Apply the staggered dslash operator to a color-spinor field.
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] U Gauge-Link (1-link or fat-link)
     @param[in] a xpay parameter (set to 0.0 for non-xpay version)
     @param[in] x Vector field we accumulate onto to
     @param[in] parity parity parameter
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] improved whether to apply the standard-staggered (false) or asqtad (true) operator
  */
  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile);

  /**
     @brief Apply the improved staggered dslash operator to a color-spinor field.
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] U Gauge-Link (1-link or fat-link)
     @param[in] L Long-Links for asqtad
     @param[in] a xpay parameter (set to 0.0 for non-xpay version)
     @param[in] x Vector field we accumulate onto to
     @param[in] parity parity parameter
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] improved whether to apply the standard-staggered (false) or asqtad (true) operator
  */
  void ApplyImprovedStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                              const GaugeField &L, double a, const ColorSpinorField &x, int parity, bool dagger,
                              const int *comm_override, TimeProfile &profile);

  /**
     @brief Apply the twisted-mass gamma operator to a color-spinor field.
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] d Which gamma matrix we are applying (C counting, so gamma_5 has d=4)
     @param[in] kappa kappa parameter
     @param[in] mu mu parameter
     @param[in] epsilon epsilon parameter
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] twist The type of kernel we are doing
  */
  void ApplyTwistGamma(ColorSpinorField &out, const ColorSpinorField &in, int d, double kappa, double mu,
		       double epsilon, int dagger, QudaTwistGamma5Type type);

  /**
     @brief Apply twisted clover-matrix field to a color-spinor field
     @param[out] out Result color-spinor field
     @param[in] in Input color-spinor field
     @param[in] clover Clover-matrix field
     @param[in] kappa kappa parameter
     @param[in] mu mu parameter
     @param[in] epsilon epsilon parameter
     @param[in] Field parity (if color-spinor field is single parity)
     @param[in] dagger Whether we are applying the dagger or not
     @param[in] twist The type of kernel we are doing
       if (twist == QUDA_TWIST_GAMMA5_DIRECT) apply (Clover + i*a*gamma_5) to the input spinor
       else if (twist == QUDA_TWIST_GAMMA5_INVERSE) apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  */
  void ApplyTwistClover(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
			double kappa, double mu, double epsilon, int parity, int dagger, QudaTwistGamma5Type twist);

  /**
     @brief Dslash face packing routine
     @param[out] ghost_buf Array of packed halos, order is [2*dim+dir]
     @param[in] field ColorSpinorField to be packed
     @param[in] location Locations where the packed fields are (Device, Host and/or Remote)
     @param[in] nFace Depth of halo
     @param[in] dagger Whether this is for the dagger operator
     @param[in] parity Field parity
     @param[in] spin_project Whether to spin_project when packing
     @param[in] a Twisted mass scale factor (for preconditioned twisted-mass dagger operator)
     @param[in] b Twisted mass chiral twist factor (for preconditioned twisted-mass dagger operator)
     @param[in] c Twisted mass flavor twist factor (for preconditioned non degenerate twisted-mass dagger operator)
     @param[in] stream Which stream are we executing in
  */
  void PackGhost(void *ghost[2 * QUDA_MAX_DIM], const ColorSpinorField &field, MemoryLocation location, int nFace,
                 bool dagger, int parity, bool spin_project, double a, double b, double c, const qudaStream_t &stream);

  /**
     @brief Applies a gamma5 matrix to a spinor (wrapper to ApplyGamma)
     @param[out] out Output field
     @param[in] in Input field
  */
  void gamma5(ColorSpinorField &out, const ColorSpinorField &in);

}

#endif // _DSLASH_QUDA_H
