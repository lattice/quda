#include <string.h>
#include <multigrid.h>
#include <algorithm>
#include <blas_magma.h>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param, bool enable_gpu)
    : Dirac(param), transfer(param.transfer), dirac(param.dirac),
      Y_h(0), X_h(0), Y_d(0), X_d(0), enable_gpu(enable_gpu)
  { initializeCoarse(); }
      
  DiracCoarse::~DiracCoarse() {
    if (Y_h) delete Y_h;
    if (X_h) delete X_h;
    if (Xinv_h) delete Xinv_h;
    if (Y_d) delete Y_d;
    if (X_d) delete X_d;
    if (Xinv_d) delete Xinv_d;
  }

  void DiracCoarse::Dslash(ColorSpinorField &out, const ColorSpinorField &in,
			   const QudaParity parity) const {
    errorQuda("Not implemented");
  }

  void DiracCoarse::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in,
			       const QudaParity parity, const ColorSpinorField &x,
			       const double &k) const {
    if (Location(out,in) == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, x, *Y_d, *X_d, kappa, parity);
    } else if ( Location(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity);
    }
  }

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( Location(out, in) == QUDA_CUDA_FIELD_LOCATION ) {
      if (!enable_gpu) errorQuda("Cannot apply %s on GPU since enable_gpu has not been set", __func__);
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa);
    } else if ( Location(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa);
    }
  }

  void DiracCoarse::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const {
    errorQuda("Not implemented");
  }

  void DiracCoarse::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
			    ColorSpinorField &x, ColorSpinorField &b,
			    const QudaSolutionType) const { /* do nothing */  }

  void DiracCoarse::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				const QudaSolutionType) const { /* do nothing */ }

  //Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarse::createCoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X) const {
    CoarseCoarseOp(T, Y, X, *(this->Y_h), *(this->X_h), kappa);
  }

  void DiracCoarse::initializeCoarse() {
    QudaPrecision prec = transfer->Vectors().Precision();
    int ndim = transfer->Vectors().Ndim();
    int x[QUDA_MAX_DIM];
    //Number of coarse sites.
    const int *geo_bs = transfer->Geo_bs();
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];

    //Coarse Color
    int Nc_c = transfer->nvec();

    //Coarse Spin
    int Ns_c = transfer->Vectors().Nspin()/transfer->Spin_bs();

    GaugeFieldParam gParam;
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.precision = prec;
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;

    gParam.geometry = QUDA_VECTOR_GEOMETRY;
    Y_h = new cpuGaugeField(gParam);

    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    X_h = new cpuGaugeField(gParam);
    Xinv_h = new cpuGaugeField(gParam);

    dirac->createCoarseOp(*transfer,*Y_h,*X_h);

    {
      // invert the clover matrix field
      const int n = X_h->Ncolor();
      BlasMagmaArgs magma(X_h->Precision());
      magma.BatchInvertMatrix(((float**)Xinv_h->Gauge_p())[0], ((float**)X_h->Gauge_p())[0], n, X_h->Volume());
    }

    if (enable_gpu) {
      gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
      gParam.geometry = QUDA_VECTOR_GEOMETRY;
      int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
      gParam.pad = gParam.nFace * pad;
      Y_d = new cudaGaugeField(gParam);
      Y_d->copy(*Y_h);

      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      X_d = new cudaGaugeField(gParam);
      Xinv_d = new cudaGaugeField(gParam);
      X_d->copy(*X_h);
      Xinv_d->copy(*Xinv_h);
    }
  }

}
