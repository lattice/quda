#include <string.h>
#include <multigrid.h>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param) 
    : Dirac(param), transfer(param.transfer), dirac(param.dirac), Y(0), X(0), LY(0), FY(0), LX(0), FX(0) 
  { initializeCoarse(); }
      
  DiracCoarse::~DiracCoarse() {
    if (Y) delete Y;
    if (X) delete X;

    if (FY) delete FY;
    if (LY) delete LY;
    if (FX) delete FX;
    if (LX) delete LX;
  }	

  void DiracCoarse::M(ColorSpinorField &out, const ColorSpinorField &in) const 
  { ApplyCoarse(out,in,*Y,*X,kappa); }

  //Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarse::createCoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X) const {
    CoarseCoarseOp(T, Y, X, *(this->Y), *(this->X), kappa);
  }

  //improved staggered
  void DiracCoarse::createCoarseOp(const Transfer &T, GaugeField &FY, GaugeField &LY, GaugeField &FX, GaugeField &LX) const {
    errorQuda("\nImproved staggered is not supported yet.\n");    
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

    gParam.geometry = QUDA_VECTOR_GEOMETRY;
    Y = new cpuGaugeField(gParam);

    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    X = new cpuGaugeField(gParam);

    //if improved staggered:
    //Initialize LX, LY and FX, FY    

    dirac->createCoarseOp(*transfer,*Y,*X);//also for staggered links or improved staggered fat links

    if(gParam.link_type == QUDA_COARSE_LONG_LINKS) dirac->createCoarseOp(*transfer,*LY,*LX);
  }

}
