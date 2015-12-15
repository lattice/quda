#include <string.h>
#include <multigrid.h>
#include <algorithm>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param, bool enable_gpu)
    : Dirac(param), transfer(param.transfer), dirac(param.dirac),
      Y_h(0), X_h(0), Y_d(0), X_d(0), enable_gpu(enable_gpu)
  { initializeCoarse(); }
      
  DiracCoarse::~DiracCoarse() {
    if (Y_h) delete Y_h;
    if (X_h) delete X_h;
    if (Y_d) delete Y_d;
    if (X_d) delete X_d;
  }	

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if ( Location(out, in) == QUDA_CUDA_FIELD_LOCATION ) {
      if (!enable_gpu)
	errorQuda("Cannot apply coarse grid operator on GPU since enable_gpu has not been set");
      ApplyCoarse(out, in, in, *Y_d, *X_d, kappa);
      //ApplyCoarse(out.Even(), in.Odd(), in.Even(), *Y_d, *X_d, kappa, QUDA_EVEN_PARITY);
      //ApplyCoarse(out.Odd(), in.Even(), in.Odd(), *Y_d, *X_d, kappa, QUDA_ODD_PARITY);
    } else if ( Location(out, in) == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa);
      //ApplyCoarse(out.Even(), in.Odd(), in.Even(), *Y_h, *X_h, kappa, QUDA_EVEN_PARITY);
      //ApplyCoarse(out.Odd(), in.Even(), in.Odd(), *Y_h, *X_h, kappa, QUDA_ODD_PARITY);
    }
  }

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
    
    dirac->createCoarseOp(*transfer,*Y_h,*X_h);

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
      X_d->copy(*X_h);
    }
  }

}
