#include <string.h>
#include <multigrid.h>

namespace quda {

  DiracCoarse::DiracCoarse(const Dirac &d, const Transfer &t, ColorSpinorField &tmp, ColorSpinorField &tmp2) 
    : DiracMatrix(d), t(&t), tmp(tmp), tmp2(tmp2), Y(0), X(0) {
      initializeCoarse();
    }
      
  DiracCoarse::DiracCoarse(const Dirac *d, const Transfer *t, ColorSpinorField &tmp, ColorSpinorField &tmp2) 
    : DiracMatrix(d), t(t), tmp(tmp), tmp2(tmp2), Y(0), X(0) {
      initializeCoarse();
    }
  
  DiracCoarse::~DiracCoarse() {
    if (Y) delete Y;
    if (X) delete X;
  }	

  void DiracCoarse::operator()(ColorSpinorField &out, const ColorSpinorField &in) const {
#if 1
    ApplyCoarse(out,in,*Y,*X,dirac->kappa); 
#else
    t->P(tmp, in);
    dirac->M(tmp2, tmp);
    t->R(out, tmp2);
#endif
  }

  void DiracCoarse::initializeCoarse() {

    QudaPrecision prec = t->Vectors().Precision();
    int ndim = t->Vectors().Ndim();
    int x[QUDA_MAX_DIM];
    //Number of coarse sites.
    const int *geo_bs = t->Geo_bs();
    for(int i = 0; i < ndim; i++) {
      x[i] = t->Vectors().X(i)/geo_bs[i];
    }

    //Coarse Color
    int Nc_c = t->nvec();

    //Coarse Spin
    int Ns_c = t->Vectors().Nspin()/t->Spin_bs();

    GaugeFieldParam gParam = new GaugeFieldParam();
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.precision = prec;
    gParam.nDim = ndim;
    //gParam.siteDim= 2*ndim+1;
    //gParam.geometry = QUDA_COARSE_GEOMETRY;
    gParam.geometry = QUDA_VECTOR_GEOMETRY;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    Y = new cpuGaugeField(gParam);

   GaugeFieldParam gParam2 = new GaugeFieldParam();
    memcpy(gParam2.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam2.nColor = Nc_c*Ns_c;
    gParam2.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam2.order = QUDA_QDP_GAUGE_ORDER;
    gParam2.link_type = QUDA_COARSE_LINKS;
    gParam2.t_boundary = QUDA_PERIODIC_T;
    gParam2.create = QUDA_ZERO_FIELD_CREATE;
    gParam2.precision = prec;
    gParam2.nDim = ndim;
    gParam2.geometry = QUDA_SCALAR_GEOMETRY;
    gParam2.siteSubset = QUDA_FULL_SITE_SUBSET;
    X = new cpuGaugeField(gParam2);
    
    dirac->createCoarseOp(*t,*Y,*X);
  }

}
