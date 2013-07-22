#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
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

//Does the heavy lifting of creating the coarse color matrices Y
template<typename Float>
void calculateY(std::complex<Float> *Y[], ColorSpinorFieldOrder<Float> &V, GaugeFieldOrder<Float> &G, int ndim, int *x_size, int *xc_size, int Nc, int Nc_c, int Ns, int Ns_c, int *geo_bs, int spin_bs, int mass) {

	//Number of sites on the fine and coarse grids
	int fsize = 1;
	int csize = 1;
	for (int d = 0; d<ndim; d++) {
	  fsize *= x_size[d];
	  csize *= xc_size[d];
	}
	
	//Create a field UV which holds U*V.  Has the same structure as V.
	ColorSpinorParam UVparam(V.Field());
	UVparam.create = QUDA_ZERO_FIELD_CREATE;
	cpuColorSpinorField UV(UVparam);
	ColorSpinorFieldOrder<Float> *UVorder = (ColorSpinorFieldOrder<Float> *) createOrder<Float>(UV,Nc_c);

        for(int d = 0; d < ndim; d++) {

	  //First calculate UV
	  UV(d,*UVorder, G, V, ndim, x_size, Nc, Nc_c, Ns);
	}
}

//Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{s,c}_mu(x) * V^{s,c}_mu(x+mu)
//Where:
//mu = dir
//s = fine spin
//c' = coarse color
//c = fine color
//FIXME: N.B. Only works if color-spin field and gauge field are parity ordered in the same way.  Need LatticeIndex function for generic ordering
template<typename Float>
void UV(int dir, ColorSpinorFieldOrder<Float> &UV, ColorSpinorFieldOrder<Float> &V, GaugeFieldOrder<Float> &G, int ndim, int *x_size, int Nc, int Nc_c, int Ns) {

	for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

          //U connects site x to site x+mu.  Thus, V lives at site x+mu if U_mu lives at site x.
          //FIXME: Uses LatticeIndex() for the color spinor field to determine gauge field index.
          //This only works if sites are ordered same way in both G and V.

	  int coord[QUDA_MAX_DIMENSION];
	  int coordV[QUDA_MAX_DIMENSION];
	  V.Field().LatticeIndex(coord, i);

          parity = 0;
          for(d = 0; d < ndim; d++);
            parity += coord[d];
          }
          parity = parity%2;

	  //Shift the V field w/respect to G
          coordV[dir] = (coord[dir]+1)%x_size[d];
	  int i_V;
          V.Field().OffsetIndex(i_V, coordV);

         for(int s = 0; s < Ns; s++) {  //Fine Spin
	   for(int ic_c = 0; ic_c < Nc_c; ic_c++) {  //Coarse Color
	     for(int ic = 0; ic < Nc; ic++) { //Fine Color rows of gauge field
	       for(int jc = 0; jc < Nc; jc++) {  //Fine Color columns of gauge field
	  	 UV(i, s, ic, ic_c) += G(dir, parity, i/2, ic, jc) * V(i_V, s, jc, ic_c);
	       }  //Fine color columns
             }  //Fine color rows
           }  //Coarse color
         }  //Fine Spin
       }  //Volume
}  //UV

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
    int x_size[QUDA_MAX_DIM];
    int xc_size[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x_size[d] = g.X()[d];
      geo_bs[d] = T.Geo_bs()[d];
      xc_size[d] = x_size[d]/geo_bs[d];
    }

    //Fine and coarse colors and spins
    int Nc = T.Vectors().Ncolor();
    int Ns = T.Vectors().Nspin();
    int Nc_c = nvec;
    int Ns_c = Ns/spin_bs;

    //Switch on precision.  Create the FieldOrder objects for gauge field and color rotation field V
    if (precision == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *vOrder = (ColorSpinorFieldOrder<double> *) colorspin::createOrder<double>(T.Vectors(),nvec);
      GaugeFieldOrder<double> *gOrder = (GaugeFieldOrder<double> *) gauge::createOrder<float>(g);
      //GaugeFieldOrder gOrder(g);
      calculateY((std::complex<double> **)Y, *vOrder, gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs, 0);
    }
    else {
      ColorSpinorFieldOrder<float> * vOrder = (ColorSpinorFieldOrder<float> *) colorspin::createOrder<float>(T.Vectors(), nvec);
      GaugeFieldOrder<float> *gOrder = (GaugeFieldOrder<float> *) gauge::createOrder<float>(g);
      calculateY((std::complex<float> **)Y, *vOrder, gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs, 0);
    }
  }  

} //namespace quda
