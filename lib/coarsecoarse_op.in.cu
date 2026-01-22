#include "coarsecoarse_op.hpp"

namespace quda {

  constexpr int fineColor = @QUDA_MULTIGRID_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr bool use_mma = false;

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  template <>
  void CoarseCoarseOp<fineColor, coarseColor, use_mma>(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                                                       const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass, double mu,
                                                       double mu_factor, QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional)
  {
    QudaFieldLocation location = checkLocation(X, Y, gauge, clover, cloverInv);

    //Create a field UV which holds U*V.  Has the same similar
    //structure to V but double the number of spins so we can store
    //the four distinct block chiral multiplications in a single UV
    //computation.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.nSpin *= 2; // so nSpin == 4
    UVparam.setPrecision(T.Vectors().Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    GaugeField *Yatomic = &Y;
    GaugeField *Xatomic = &X;
    if (Y.Precision() < QUDA_SINGLE_PRECISION) {
      // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
      // else we can just coarsen directly into the original fields
      GaugeFieldParam param(X); // use X since we want scalar geometry
      param.location = location;
      param.setPrecision(QUDA_SINGLE_PRECISION, location == QUDA_CUDA_FIELD_LOCATION ? true : false);

      Yatomic = GaugeField::Create(param);
      Xatomic = GaugeField::Create(param);
    }

    GaugeField *G_prec = const_cast<GaugeField*>(&gauge);
    if (Y.Precision() != gauge.Precision()) {
      //Create a copy of the gauge field with correct precision
      GaugeFieldParam param(gauge);
      param.setPrecision(Y.Precision(), true);
      G_prec = new GaugeField(param);
      G_prec->copy(gauge);
    }

    GaugeField *C_prec = const_cast<GaugeField*>(&clover);
    if (Y.Precision() != clover.Precision()) {
      //Create a copy of the clover field with correct precision
      GaugeFieldParam param(clover);
      param.setPrecision(Y.Precision(), true);
      C_prec = new GaugeField(param);
      C_prec->copy(clover);
    }

    GaugeField *I_prec = const_cast<GaugeField*>(&cloverInv);
    if (Y.Precision() != cloverInv.Precision()) {
      //Create a copy of the cloverInv field with correct precision
      GaugeFieldParam param(cloverInv);
      param.setPrecision(Y.Precision(), true);
      I_prec = new GaugeField(param);
      I_prec->copy(cloverInv);
    }

    bool constexpr use_mma = false;
    calculateYcoarse<use_mma, fineColor, coarseColor>(Y, X, *Yatomic, *Xatomic, *uv, T, *G_prec, *C_prec, *I_prec, kappa, mass, mu, mu_factor, dirac, matpc,
                              need_bidirectional);

    if (Yatomic != &Y) delete Yatomic;
    if (Xatomic != &X) delete Xatomic;
    if (I_prec != &cloverInv) delete I_prec;
    if (C_prec != &clover) delete C_prec;
    if (G_prec != &gauge) delete G_prec;

    delete uv;
  }

}
