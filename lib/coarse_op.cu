#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>

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
	//QUDA_UKQCD_GAMMA_BASIS is the non-relativistic basis.
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
	  else if((dir==1) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
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
	  else if((dir==3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
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
	  else if((dir==4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	    elem[0] = 1;
	    elem[1] = 1;
	    elem[2] = 1;
	    elem[3] = 1;
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
	  } else {
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
  void VUV(int dir, quda::complex<Float> *Y, quda::complex<Float> *X, const colorspinor::FieldOrder<Float> &UV, const colorspinor::FieldOrder<Float> &V, const Gamma &gamma, int ndim, const int *x_size, const int *xc_size, int Nc, int Nc_c, int Ns, int Ns_c, const int *geo_bs, int spin_bs) {

    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      Float half = 1.0/2.0;
      int coord[QUDA_MAX_DIM];
      int coord_coarse[QUDA_MAX_DIM];
      int coarse_size = 1;
      int coarse_parity = 0;
      int coarse_index = 0;
      V.Field().LatticeIndex(coord, i);
      for(int d = ndim-1; d >= 0; d--) {
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
     quda::complex<Float> *M;

     //If adjacent site is in same block, M = X
     if(((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) {
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
	     M[total_diag_offset] += half*quda::conj(V(i, s, ic, ic_c))*UV(i, s, ic, jc_c);
	     //Off-diagonal Spin
	     M[total_offdiag_offset] += half*quda::conj(V(i, s, ic, ic_c))*UV(i,s_col, ic, jc_c); 
	   } //Fine color
	 } //Coarse Color column
       } //Coarse Color row
     } //Fine spin
   } //Volume
}

  //Calculate the reverse link, Y_{-\mu}(x+mu).
  //The reverse link is almost the Hermitian conjugate of the forward link,
  //but with negative sign for the spin off-diagonal parts to account
  //for the forward/backward spin proejctors.
  //Note: No shifting in site index, so that site x holds the matrices Y_{\pm mu}(x, x+mu)
  template<typename Float>
  void reverseY(int dir, const quda::complex<Float> *Y_p, quda::complex<Float> *Y_m, int ndim, const int *xc_size, int Nc_c, int Ns_c)  {
	int csize = 1;
	for(int d = 0; d < ndim; d++) {
	  csize *= xc_size[d];
	}
	
	for(int i = 0; i < csize; i++) { //Volume
	  int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
	  for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin row
	    for(int s_col = 0; s_col < Ns_c; s_col++) { //Spin column
	      Float sign = 1.0;
	      if (s_row != s_col) {
		sign = -1.0;
	      }
	      for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color row
	        for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Color column
	          //Flip s_row, s_col and ic_c, jc_c indices on the rhs because of Hermitian conjugation.
		  Y_m[coarse_site_offset+Nc_c*Nc_c*(Ns_c*s_row+s_col)+Nc_c*ic_c+jc_c] = sign*quda::conj(Y_p[coarse_site_offset+Nc_c*Nc_c*(Ns_c*s_col+s_row)+Nc_c*jc_c+ic_c]);
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
  void coarseDiagonal(quda::complex<Float> *X, int ndim, const int *xc_size, int Nc_c, int Ns_c)  {
	int csize = 1;
	for(int d = 0; d < ndim; d++) {
	  csize *= xc_size[d];
	}

	int local = Nc_c*Nc_c;
	quda::complex<Float> *Xlocal = new quda::complex<Float>[local];
	
	for(int i = 0; i < csize; i++) { //Volume
	  int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
	  for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin row
	    for(int s_col = 0; s_col < Ns_c; s_col++) { //Spin column
	     
	      //Copy the Hermitian conjugate term to temp location 
	      for(int k = 0; k < local; k++) {
	        //Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
	        Xlocal[k] = X[coarse_site_offset+Nc_c*Nc_c*(Ns_c*s_col+s_row)+k];
	      }

	      Float sign = 1.0;	
	      if(s_row != s_col) {
		sign = -1.0;
	      }

	      for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color row
	        for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Color column
	          //Flip  color indices because of Hermitian conjugation.
	          //Spin indices already flipped while filling Xlocal.
		  X[coarse_site_offset+Nc_c*Nc_c*(Ns_c*s_row+s_col)+Nc_c*ic_c+jc_c] += sign*quda::conj(Xlocal[Nc_c*jc_c + ic_c]);
	        } //Color column
	      } //Color row
	    } //Spin column
	  } //Spin row  
	} //Volume
	delete [] Xlocal;
}

  //Currently unusued.  Combining mass and coarse clover term moved to the application of the operator.
  template<typename Float>
  void addMass(quda::complex<Float> *X, int ndim, const int *xc_size, int Nc_c, int Ns_c, double mass)  {
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
  void calculateY(quda::complex<Float> *Y[], const colorspinor::FieldOrder<Float> &V, const gauge::FieldOrder<Float> &G, int ndim, const int *x_size, const int *xc_size, int Nc, int Nc_c, int Ns, int Ns_c,const int *geo_bs, int spin_bs) {

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
    //addMass<Float>(Y[2*ndim], ndim, xc_size, Nc_c, Ns_c, mass);

  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y has been allocated.
  void CoarseOp(const Transfer &T, void *Y[], QudaPrecision precision, const cudaGaugeField &gauge) {

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
      calculateY<double>((quda::complex<double>**)Y, *vOrder, *gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs);
    }
    else {
      colorspinor::FieldOrder<float> * vOrder = colorspinor::createOrder<float>(T.Vectors(), nvec);
      gauge::FieldOrder<float> *gOrder = gauge::createOrder<float>(g);
      calculateY<float>((quda::complex<float>**)Y, *vOrder, *gOrder, ndim, x_size, xc_size, Nc, Nc_c, Ns, Ns_c, geo_bs, spin_bs);
    }
  }  

  //Apply the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float>
  void coarseDslash(colorspinor::FieldOrder<Float> &out, colorspinor::FieldOrder<Float> &in, quda::complex<Float> *Y[]) {
    int Nc = in.Ncolor();
    int Ns = in.Nspin();
    int ndim = in.Ndim();
    int sites = in.Volume();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x_size[d] = in.Field().X()[d];
    }

    for(int i = 0; i < sites; i++) { //Volume
      int coord[QUDA_MAX_DIM];
      int parity = 0;
      int gauge_index = 0;
      in.Field().LatticeIndex(coord,i);
      for(int dim = ndim-1; dim >= 0; dim--) {
        parity += coord[dim];
        gauge_index *= x_size[dim];
        gauge_index += coord[dim];
      }
      parity = parity%2;
      int gauge_site_offset = parity*sites/2 + gauge_index/2;
      gauge_site_offset *= Nc*Nc*Ns*Ns;


      for(int d = 0; d < ndim; d++) { //Ndim

	int forward[QUDA_MAX_DIM];
	int forward_spinor_index;
	int backward[QUDA_MAX_DIM];
        int backward_spinor_index;
	int backward_gauge_index = 0;
        int backward_parity = 0;

        //We need the coordinates of the forward site to index into the "out" field.
        //We need the coordinates of the backward site to index into the "out" field, \
	//as well as to retrieve the gauge field there for parallel transport.
	in.Field().LatticeIndex(forward,i);
        in.Field().LatticeIndex(backward,i);
	forward[d] = (forward[d] + 1)%x_size[d];
	backward[d] = (backward[d] - 1 + x_size[d])%x_size[d];
	out.Field().OffsetIndex(forward_spinor_index, forward);
	out.Field().OffsetIndex(backward_spinor_index, backward);

        for(int dim = ndim-1; dim >= 0; dim--) {
          backward_parity += backward[dim];
          backward_gauge_index *= x_size[dim];
          backward_gauge_index += backward[dim];
        }
	backward_parity = backward_parity%2;
        int backward_gauge_site_offset = backward_parity*sites/2 + backward_gauge_index/2;
        backward_gauge_site_offset *= Nc*Nc*Ns*Ns;


        for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
            for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
              for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	        //Forward link  
	        out(forward_spinor_index, s_row, c_row) += quda::conj(Y[2*d][gauge_site_offset+Nc*Nc*(Ns*s_col+s_row) + Nc*c_col+c_row])*in(i, s_col, c_col);
	        //Backward link
		out(backward_spinor_index, s_row, c_row) += Y[2*d+1][backward_gauge_site_offset+Nc*Nc*(Ns*s_row+s_col) + Nc*c_row+c_col]*in(i, s_col, c_col);
	      } //Color column
	    } //Spin column
	  } //Color row
        } //Spin row 
      } //Ndim
    }//Volume

}

  //out(x) = (mass-X)in(x), where X is the local color-spin matrix on the coarse grid.
  //For kappa normalization, mass = 1.0
  template<typename Float>
  void coarseClover(colorspinor::FieldOrder<Float> &out, const colorspinor::FieldOrder<Float> &in, const quda::complex<Float> *X) {

    int Nc = out.Ncolor();
    int Ns = out.Nspin();
    int ndim = out.Ndim();
    int sites = out.Volume();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x_size[d] = out.Field().X()[d];
    }


    for(int i = 0; i < out.Volume(); i++) { //Volume
      int coord[QUDA_MAX_DIM];
      int parity = 0;
      int gauge_index = 0;
      out.Field().LatticeIndex(coord,i);
      for(int dim = ndim-1; dim >= 0; dim--) {
        parity += coord[dim];
        gauge_index *= x_size[dim];
        gauge_index += coord[dim];
      }
      parity = parity%2;
      int gauge_site_offset = parity*sites/2 + gauge_index/2;
      gauge_site_offset *= Nc*Nc*Ns*Ns;

      for(int s = 0; s < Ns; s++) { //Spin out
        for(int c = 0; c < Nc; c++) { //Color out
	  out(i,s,c) += in(i,s,c);
	  for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	    for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	      out(i,s,c) -= X[gauge_site_offset+Nc*Nc*(Ns*s+s_col)+Nc*c+c_col]*in(i,s_col,c_col);
	    } //Color in
          } //Spin in
        } //Color out
     } //Spin out
   } //Volume
}

  //Zero out a field, using the accessor.
  template<typename Float>
  void setZero(colorspinor::FieldOrder<Float> &f) {
    for(int i = 0; i < f.Volume(); i++) {
      for(int s = 0; s < f.Nspin(); s++) {
	for(int c = 0; c < f.Ncolor(); c++) {
	  f(i,s,c) = (Float) 0.0;
        }
      }
    }
   }

  //Multiply a field by a real constant
  template<typename Float>
  void F_eq_rF(colorspinor::FieldOrder<Float> &f, Float r) {
    for(int i = 0; i < f.Volume(); i++) {
      for(int s = 0; s < f.Nspin(); s++) {
        for(int c = 0; c < f.Ncolor(); c++) {
          f(i,s,c) *= r;
        }
      }
    }
 }

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = (1-X)*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, void *Y[], QudaPrecision precision, double kappa) {

    int ndim = in.Ndim();	

    if (precision == QUDA_DOUBLE_PRECISION) {
      colorspinor::FieldOrder<double> *inOrder = colorspinor::createOrder<double>(in);
      colorspinor::FieldOrder<double> *outOrder = colorspinor::createOrder<double>(out);
      setZero(*outOrder);
      coarseDslash(*outOrder, *inOrder, (quda::complex<double>**)Y);
      F_eq_rF(*outOrder, -2*kappa);
      coarseClover(*outOrder, *inOrder, (quda::complex<double> *)Y[2*ndim]);
    }
    else {
      colorspinor::FieldOrder<float> *inOrder = colorspinor::createOrder<float>(in);
      colorspinor::FieldOrder<float> *outOrder = colorspinor::createOrder<float>(out);
      setZero(*outOrder);
      coarseDslash(*outOrder, *inOrder, (quda::complex<float>**)Y);
      F_eq_rF(*outOrder, (float) (-2*kappa));
      coarseClover(*outOrder, *inOrder, (quda::complex<float> *)Y[2*ndim]);
    }  

  }//ApplyCoarse

} //namespace quda
