#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <complex>

namespace quda {

  //Returns the index of the coarse color matrix Y.
  //sites is the total number of sites on the coarse lattice (both parities)
  //Ns_c is the number of blocked spin components
  //Nc_c is the number of coarse colors
  //x is the index of the site on the even-odd coarse lattice
  //parity determines whether the site is even or odd
  //s is the blocked spin index
  //c is the coarse color index
  int Yindex(int sites, int Ns_c, int Nc_c, int x, int parity, int s, int c) {
    return Ns_c*Nc_c*parity*sites/2 + Ns_c*Nc_c*x + Nc_c*s + c;
#if 0
    if((x >= sites/2) || (x < 0) || (parity < 0) || (parity > 1) || (s < 0) || (s >= Ns_c) || (c < 0) || (c > Nc_c)) {
      printfQuda("Bad Yindex: sites=%d Ns_c=%d Nc_c=%d,x=%d,parity=%d,s=%d,c=%d\n",sites,Ns_c,Nc_c,x,parity,s,c);
      return -1;
    }
#endif
  }

  template<typename Float, typename Vtype>
  void calculateY(std::complex<Float> *Y[], Vtype &V, const cudaGaugeField &gauge, int ndim, int *x, int *xc, int Nc, int Nc_c, int Ns, int Ns_c, int nvec, int *geo_bs, int spin_bs, int mass) {

    //Number of sites on the fine and coarse grids
    int fsize = 1;
    int csize = 1;
	
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y has been allocated.
  void CoarseOp(Transfer &T, void *Y[], QudaPrecision precision, const cudaGaugeField &gauge) {

    //First make a cpu gauge field from the cuda gauge field

    int pad = 0;
    GaugeFieldParam gf_param(gauge.X(), precision, gauge.Reconstruct(), pad, gauge.Geometry());
    gf_param.order = QUDA_QDP_GAUGE_ORDER;
    gf_param.fixed = gauge.GaugeFixed();
    gf_param.link_type = gauge.LinkType();
    gf_param.t_boundary = gauge.TBoundary();
    gf_param.anisotropy = gauge.Anisotropy();
    gf_param.gauge = NULL;
    gf_param.create = QUDA_NULL_FIELD_CREATE;

    cpuGaugeField g(gf_param);

    //Copy the cuda gauge field to the cpu
    gauge.saveCPUField(g, QUDA_CPU_FIELD_LOCATION);

    //Information about geometrical blocking, spin blocking
    // and number of nullvectors
    int ndim = g.Ndim();
    int geo_bs[QUDA_MAX_DIM];
    int spin_bs = T.Spin_bs();
    int nvec = T.nvec();

    //Fine grid size and coarse grid size
    int x[QUDA_MAX_DIM];
    int xc[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x[d] = g.X()[d];
      geo_bs[d] = T.Geo_bs()[d];
      xc[d] = x[d]/geo_bs[d];
    }

    //Fine and coarse colors and spins
    int Nc = T.Vectors().Ncolor();
    int Ns = T.Vectors().Nspin();
    int Nc_c = nvec;
    int Ns_c = Ns/spin_bs;

    if (precision == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(T.Vectors(),nvec);
      calculateY((std::complex<double> **)Y, *vOrder, gauge, ndim, x, xc, Nc, Nc_c, Ns, Ns_c, nvec, geo_bs, spin_bs, 0);
    }
    else {
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(T.Vectors(), nvec);
      calculateY((std::complex<float> **)Y, *vOrder, gauge, ndim, x, xc, Nc, Nc_c, Ns, Ns_c, nvec, geo_bs, spin_bs, 0);
    }

  }

} //namespace quda
