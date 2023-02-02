#include "coarsecoarse_op.hpp"

namespace quda {

  constexpr int fineColor = @QUDA_MULTIGRID_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr bool use_mma = true;

  template<>
  void CoarseCoarseOp<fineColor, coarseColor, use_mma>(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                                                       const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass,
                                                       double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional)
  {
    QudaFieldLocation location = checkLocation(X, Y, gauge, clover, cloverInv);
    if (location == QUDA_CPU_FIELD_LOCATION) errorQuda("use_mma = true does not go with QUDA_CPU_FIELD_LOCATION.");

    //Create a field UV which holds U*V.  Has the same similar
    //structure to V but double the number of spins so we can store
    //the four distinct block chiral multiplications in a single UV
    //computation.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.nSpin *= 2; // so nSpin == 4
    UVparam.setPrecision(T.Vectors(location).Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    // The MMA implementation requires fields to be in AoS order: we check each gauge field, and create a copy if not.
    // The UV field is an intermediate field, its order doesn't matter; The AoS copy of the V field will be created
    // later in the code path.
    constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

    auto create_gauge_copy = [](const GaugeField &X, QudaGaugeFieldOrder order, bool copy_content) -> auto
    {
      GaugeField *output = nullptr;
      if (X.Order() == order) {
        output = const_cast<GaugeField *>(&X);
      } else {
        GaugeFieldParam param(X);
        param.order = order;
        output = cudaGaugeField::Create(param);
        if (copy_content) output->copy(X);
      }
      return static_cast<cudaGaugeField *>(output);
    };

    auto Y_order = create_gauge_copy(Y, gOrder, false);
    auto X_order = create_gauge_copy(X, gOrder, false);
    auto G_order = create_gauge_copy(gauge, gOrder, true);
    auto C_order = create_gauge_copy(clover, gOrder, true);
    auto I_order = create_gauge_copy(cloverInv, gOrder, true);

    GaugeField *Yatomic = nullptr;
    GaugeField *Xatomic = nullptr;

    if (Y.Precision() < QUDA_SINGLE_PRECISION) {
      // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
      // else we can just coarsen directly into the original fields
      GaugeFieldParam param(*X_order); // use X since we want scalar geometry
      param.location = location;
      param.order = gOrder;
      param.setPrecision(QUDA_SINGLE_PRECISION);

      Yatomic = GaugeField::Create(param);
      Xatomic = GaugeField::Create(param);
    } else {
      Yatomic = Y_order;
      Xatomic = X_order;
    }

    bool constexpr use_mma = true;
    calculateYcoarse<use_mma, fineColor, coarseColor>
      (*Y_order, *X_order, *Yatomic, *Xatomic, *uv, T, *G_order, *C_order, *I_order, kappa, mass, mu, mu_factor, dirac, matpc, need_bidirectional);

    if (Yatomic != Y_order) delete Yatomic;
    if (Xatomic != X_order) delete Xatomic;

    if (&Y != Y_order) {
      Y.copy(*Y_order);
      delete Y_order;
    }

    if (&X != X_order) {
      X.copy(*X_order);
      delete X_order;
    }

    if (&gauge != G_order) { delete G_order; }
    if (&clover != C_order) { delete C_order; }
    if (&cloverInv != I_order) { delete I_order; }

    delete uv;
  }

}
