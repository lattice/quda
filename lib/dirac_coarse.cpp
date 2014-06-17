#include <string.h>
#include <multigrid.h>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param) 
    : Dirac(param), transfer(param.transfer), dirac(param.dirac), Y(0), X(0) 
  { initializeCoarse(); }
      
  DiracCoarse::~DiracCoarse() {
    if (Y) delete Y;
    if (X) delete X;
  }	

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const 
  { ApplyCoarse(out,in,*Y,*X,kappa); }

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

    gParam.geometry = QUDA_VECTOR_GEOMETRY;
    Y = new cpuGaugeField(gParam);

    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    X = new cpuGaugeField(gParam);
    
    dirac->createCoarseOp(*transfer,*Y,*X);
  }

}
