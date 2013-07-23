#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex>

namespace quda {

  //A simple Euclidean gamma matrix class for use with the Wilson projectors.
  class Gamma {
	private:
	int ndim;

	protected:

	//Which gamma matrix (dir = 0,4)
	//dir = 0: gamma^1, dir = 1: gamma^2, dir = 2: gamma^3, dir = 3: gamma^4, dir =4: gamma^5
	int dir;

	//The basis to be used.
	//QUDA_DEGRAND_ROSSI_GAMMA_BASIS is the chiral basis
	//QUDA_UKQCD_GAMMA_BASIS is the non-relativistic basis?
	QudaGammaBasis basis;

	//The column with the non-zero element for each row
	int coupling[4];
	//The value of the matrix element, for each row
	std::complex<int> elem[4];

	public:

	Gamma(int dir, QudaGammaBasis basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS) : ndim(4), dir(dir), basis(basis) {
	  std::complex<int> I(0,1);
	  if((dir==0) || (dir==1)) {
	    coupling[0] = 3;
	    coupling[1] = 2;
	    coupling[2] = 1;
	    coupling[3] = 0;
	  }
	  else if (dir == 2) {
	    coupling[0] = 2;
	    coupling[1] = 3;
	    coupling[2] = 0;
	    coupling[3] = 1;
	  }
	  else if ((dir == 3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    coupling[0] = 2;
	    coupling[1] = 3;
	    coupling[2] = 0;
	    coupling[3] = 1;
	  }
	  else if ((dir == 3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	    coupling[0] = 0;
	    coupling[1] = 1;
	    coupling[2] = 2;
	    coupling[3] = 3;
	  }
	  else if ((dir == 4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    coupling[0] = 0;
	    coupling[1] = 1;
	    coupling[2] = 2;
	    coupling[3] = 3;
	  }
	  else if ((dir == 4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	    coupling[0] = 2;
	    coupling[1] = 3;
	    coupling[2] = 0;
	    coupling[3] = 1;
	  }
	  else {
	    printfQuda("Warning: Gamma matrix not defined for dir = %d and basis = %d\n", dir, basis);
	    coupling[0] = 0;
	    coupling[1] = 0;
	    coupling[2] = 0;
	    coupling[3] = 0;
	  }


	  if((dir==0)) {
	    elem[0] = I;
	    elem[1] = I;
	    elem[2] = -I;
	    elem[3] = -I;
	  }
	  else if((dir==1) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = -1;
	    elem[1] = 1;
	    elem[2] = 1;
	    elem[3] = -1;
	  }
	  else if((dir==1) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = 1;
	    elem[1] = -1;
	    elem[2] = -1;
	    elem[3] = 1;
	  }
	  else if((dir==2)) {
	    elem[0] = I;
	    elem[1] = -I;
	    elem[2] = -I;
	    elem[3] = I;
	  }
	  else if((dir==3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = 1;
	    elem[1] = 1;
	    elem[2] = 1;
	    elem[3] = 1;
	  }
	  else if((dir==3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = 1;
	    elem[1] = 1;
	    elem[2] = -1;
	    elem[3] = -1;
	  }
	  else if((dir==4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = -1;
	    elem[1] = -1;
	    elem[2] = 1;
	    elem[3] = 1;
	  }
	  else if((dir==4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	    elem[0] = -1;
	    elem[1] = -1;
	    elem[2] = -1;
	    elem[3] = -1;
	  }
	  else {
	    elem[0] = 0;
	    elem[1] = 0;
	    elem[2] = 0;
	    elem[3] = 0;
	  }
	} 

	Gamma(const Gamma &g) : ndim(4), dir(g.dir), basis(g.basis) {
	  for(int i = 0; i < ndim+1; i++) {
	    coupling[i] = g.coupling[i];
	    elem[i] = g.elem[i];
	  }
	}

	~Gamma() {}

	//Returns the matrix element.
	std::complex<int> getelem(int row, int col) const {
	  if(coupling[row] == col) {
	    return elem[row];
	  }
	  else {
	    return 0;
	  }
	}

	//Like getelem, but one only needs to specify the row.
	//The column of the non-zero component is returned via the "col" reference
	std::complex<int> getrowelem(int row, int &col) const {
	  col = coupling[row];
	  return elem[row];
	}

        //Returns the type of Gamma matrix
        int Dir() const {
	  return dir;
        }
  };

  //Returns the index of the coarse color matrix Y.
  //This matrix is dense in both spin and color indices, in general
  //sites is the total number of sites on the coarse lattice (both parities)
  //Ns_c is the number of blocked spin components
  //Nc_c is the number of coarse colors
  //x is the index of the site on the even-odd coarse lattice
  //parity determines whether the site is even or odd
  //row_s = row index for spin
  //col_s = column index for spin
  //row_c = row index for coarse color
  //col_c = column index for coarse color
  int Yindex(int sites, int Ns_c, int Nc_c, int x, int parity, int row_s, int col_s, int row_c, int col_c) {
    int size_spin = Ns_c*Ns_c;
    int size_color = Nc_c*Nc_c;
    return size_spin*size_color*(parity*sites/2 + x) + size_color*(Ns_c*row_s + col_s) + Nc_c*row_c+col_c;
#if 0
    if((x >= sites/2) || (x < 0) || (parity < 0) || (parity > 1) || (s < 0) || (s >= Ns_c) || (c < 0) || (c > Nc_c)) {
      printfQuda("Bad Yindex: sites=%d Ns_c=%d Nc_c=%d,x=%d,parity=%d,s=%d,c=%d\n",sites,Ns_c,Nc_c,x,parity,s,c);
      return -1;
    }
#endif
  }

  //Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
  //Where:
  //mu = dir
  //s = fine spin
  //c' = coarse color
  //c = fine color
  //FIXME: N.B. Only works if color-spin field and gauge field are parity ordered in the same way.  Need LatticeIndex function for generic ordering
  template<typename Float>
  void UV(colorspinor::FieldOrder<Float> &UV, const colorspinor::FieldOrder<Float> &V, 
	  const gauge::FieldOrder<Float> &G, int dir, int ndim, const int *x_size, int Nc, int Nc_c, int Ns) {

    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      //U connects site x to site x+mu.  Thus, V lives at site x+mu if U_mu lives at site x.
      //FIXME: Uses LatticeIndex() for the color spinor field to determine gauge field index.
      //This only works if sites are ordered same way in both G and V.

      int coord[QUDA_MAX_DIM];
      int coordV[QUDA_MAX_DIM];
      V.Field().LatticeIndex(coord, i);

      int parity = 0;
      for(int d = 0; d < ndim; d++) {
	parity += coord[d];
      }
      parity = parity%2;

      //Shift the V field w/respect to G
      coordV[dir] = (coord[dir]+1)%x_size[dir];
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

  template<typename Float>
  void VUV(int dir, std::complex<Float> *Y, std::complex<Float> *X, const colorspinor::FieldOrder<Float> &UV, const colorspinor::FieldOrder<Float> &V, const Gamma &gamma, int ndim, const int *x_size, const int *xc_size, int Nc, int Nc_c, int Ns, int Ns_c, const int *geo_bs, int spin_bs) {

    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      Float half = 1.0/2.0;
      int coord[QUDA_MAX_DIM];
      int coord_coarse[QUDA_MAX_DIM];
      int coarse_size = 1;
      int coarse_parity = 0;
      int coarse_index = 0;
      V.Field().LatticeIndex(coord, i);
      for(int d = ndim-1; d >= 0; d++) {
	coord_coarse[d] = coord[d]/geo_bs[d];
	coarse_size *= xc_size[d];
	coarse_parity += coord_coarse[d];
	coarse_index *= xc_size[d];
	coarse_index += coord_coarse[d];
      }
      coarse_parity = coarse_parity%2;

      int coarse_site_offset = coarse_parity*coarse_size/2 + coarse_index/2;
      coarse_site_offset *= Nc_c*Nc_c*Ns_c*Ns_c;

      //Check to see if we are on the edge of a block, i.e.
      //if this color matrix connects adjacent blocks.
     std::complex<Float> *M;

     //If adjacent site is in same block, M = X
     if((coord[dir]+1)/geo_bs[dir] == coord_coarse[dir]) {
	M = X;
     }
     //If adjacent site is in different block, M = Y
     else {
	M = Y;
     }

     for(int s = 0; s < Ns; s++) { //Loop over fine spin

	//Spin part of the color matrix.  Will always
	//consist of two terms - diagonal and off-diagonal part
	//of P_mu = (1+\gamma_mu)

	//Coarse spin row index
	int s_c_row = s/spin_bs;

	//Use Gamma to calculate off-diagonal coupling and
	//column index.  Diagonal coupling is always 1.
	int s_col;
	std::complex<int> coupling = gamma.getrowelem(s, s_col);
	int s_c_col = s_col/spin_bs;

	//Precompute spin offsets
	int spin_diag_offset = Nc_c*Nc_c*(Ns_c*s_c_row + s_c_row);
	int spin_offdiag_offset = Nc_c*Nc_c*(Ns_c*s_c_row + s_c_col);
	
  
       for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Coarse Color row
	 for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Coarse Color column

          int coarse_color_offset = Nc_c*ic_c + jc_c;
	  int total_diag_offset = coarse_site_offset + spin_diag_offset + coarse_color_offset;
	  int total_offdiag_offset = coarse_site_offset + spin_offdiag_offset + coarse_color_offset;

           for(int ic = 0; ic < Nc; ic++) { //Sum over fine color

             //Diagonal Spin
	     M[total_diag_offset] += half*std::conj(V(i, s, ic, ic_c))*UV(i, s, ic, jc_c);
	     //Off-diagonal Spin
	     M[total_offdiag_offset] += half*std::conj(V(i, s, ic, ic_c))*UV(i,s_col, ic, jc_c); 
	   } //Fine color
	 } //Coarse Color column
       } //Coarse Color row
     } //Fine spin
   } //Volume
}

  //Calculate the reverse link, Y_{-\mu}(x+mu).
  //The reverse link is almost the complex conjugate of the forward link,
  //but with negative sign for the spin-off diagonal parts to account
  //for the forward/backward spin proejctors.
  //Note: No shifting in site index meaning Y_{-\mu}(x+mu) lives at site x.
  template<typename Float>
  void reverseY(int dir, const std::complex<Float> *Y_p, std::complex<Float> *Y_m, int ndim, const int *xc_size, int Nc_c, int Ns_c)  {
	int csize = 1;
	for(int d = 0; d < ndim; d++) {
	  csize *= xc_size[d];
	}
	
	for(int i = 0; i < csize; i++) { //Volume
	  int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
	  for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin row
	    for(int s_col = 0; s_col < Ns_c; s_col++) { //Spin column
	      int spin_offset = Nc_c*Nc_c*(Ns_c*s_row+s_col);
	      Float sign = 1.0;
	      if (s_row != s_col) {
		sign = -1.0;
	      }
	      for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color row
	        for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Color column
		  int offset = coarse_site_offset + spin_offset + Nc_c*ic_c + jc_c;
		  Y_m[offset] = sign*std::conj(Y_p[offset]);
	        } //Color column
	      } //Color row
	    } //Spin column
	  } //Spin row  
	} //Volume
}

  //Adds the reverse links to the coarse diagonal term, which is just
  //the conjugate of the existing coarse diagonal term but with
  //plus/minus signs for off-diagonal spin components
  //Also add the diagonal mass term from the original fine wilson operator:
  template<typename Float>
  void coarseDiagonal(std::complex<Float> *X, int ndim, const int *xc_size, int Nc_c, int Ns_c)  {
	int csize = 1;
	for(int d = 0; d < ndim; d++) {
	  csize *= xc_size[d];
	}

	int local = Nc_c*Nc_c;
	std::complex<Float> *Xlocal = new std::complex<Float>[local];
	
	for(int i = 0; i < csize; i++) { //Volume
	  int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
	  for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin row
	    for(int s_col = 0; s_col < Ns_c; s_col++) { //Spin column
	      int spin_offset = Nc_c*Nc_c*(Ns_c*s_row+s_col);
	      
	      for(int k = 0; k < local; k++) {
	        Xlocal[k] = X[coarse_site_offset+spin_offset+k];
	      }

	      Float sign = 1.0;	
	      if(s_row != s_col) {
		sign = -1.0;
	      }

	      for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color row
	        for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Color column
		  int offset = coarse_site_offset + spin_offset + Nc_c*ic_c + jc_c;
		  X[offset] += sign*std::conj(Xlocal[Nc_c*ic_c + jc_c]);
	        } //Color column
	      } //Color row
	    } //Spin column
	  } //Spin row  
	} //Volume
	delete [] Xlocal;
}

  template<typename Float>
  void addMass(std::complex<Float> *X, int ndim, const int *xc_size, int Nc_c, int Ns_c, double mass)  {
	int csize = 1;
	for(int d = 0; d < ndim; d++) {
	  csize *= xc_size[d];
	}

	for(int i = 0; i < csize; i++) { //Volume
	  int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
	  for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin 
	    int spin_offset = Nc_c*Nc_c*(Ns_c*s_row+s_row);	      
            for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color
              int offset = coarse_site_offset + spin_offset + Nc_c*ic_c + ic_c;
	      X[offset] += mass;
            } //Color
	  } //Spin
        } //Volume
}


  //Does the heavy lifting of creating the coarse color matrices Y
  template<typename Float>
  void calculateY(std::complex<Float> *Y[], const colorspinor::FieldOrder<Float> &V, const gauge::FieldOrder<Float> &G, int ndim, const int *x_size, const int *xc_size, int Nc, int Nc_c, int Ns, int Ns_c,const int *geo_bs, int spin_bs, double mass) {

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(V.Field());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField uv(UVparam);
    colorspinor::FieldOrder<Float> *UVorder = colorspinor::createOrder<Float>(uv,Nc_c);

    for(int d = 0; d < ndim; d++) {
      //First calculate UV
      UV<Float>(*UVorder, V, G, d, ndim, x_size, Nc, Nc_c, Ns);

      //Gamma matrix for this direction
      Gamma gamma(d, UVorder->Field().GammaBasis());

      //Calculate VUV for this direction, accumulate in the appropriate place
      VUV<Float>(d, Y[2*d], Y[2*ndim], *UVorder, V, gamma, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs);

      reverseY<Float>(d, Y[2*d], Y[2*d+1], ndim, xc_size, Nc_c, Ns_c);
    }

    coarseDiagonal<Float>(Y[2*ndim], ndim, xc_size, Nc_c, Ns_c);
    addMass<Float>(Y[2*ndim], ndim, xc_size, Nc_c, Ns_c, mass);

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
      colorspinor::FieldOrder<double> *vOrder = colorspinor::createOrder<double>(T.Vectors(),nvec);
      gauge::FieldOrder<double> *gOrder = gauge::createOrder<double>(g);
      calculateY<double>((std::complex<double>**)Y, *vOrder, *gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs, 0);
    }
    else {
      colorspinor::FieldOrder<float> * vOrder = colorspinor::createOrder<float>(T.Vectors(), nvec);
      gauge::FieldOrder<float> *gOrder = gauge::createOrder<float>(g);
      calculateY<float>((std::complex<float>**)Y, *vOrder, *gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs, 0);
    }
  }  

} //namespace quda
