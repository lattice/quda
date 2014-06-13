#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>

namespace quda {

  //A simple Euclidean gamma matrix class for use with the Wilson projectors.
  template <typename ValueType>
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
    complex<ValueType> elem[4];

  public:

    Gamma(int dir, QudaGammaBasis basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS) : ndim(4), dir(dir), basis(basis) {
      complex<ValueType> I(0,1);
      if((dir==0) || (dir==1)) {
	coupling[0] = 3;
	coupling[1] = 2;
	coupling[2] = 1;
	coupling[3] = 0;
      } else if (dir == 2) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else if ((dir == 3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else if ((dir == 3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	coupling[0] = 0;
	coupling[1] = 1;
	coupling[2] = 2;
	coupling[3] = 3;
      } else if ((dir == 4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	coupling[0] = 0;
	coupling[1] = 1;
	coupling[2] = 2;
	coupling[3] = 3;
      } else if ((dir == 4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	coupling[0] = 2;
	coupling[1] = 3;
	coupling[2] = 0;
	coupling[3] = 1;
      } else {
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
      } else if((dir==1) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = -1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = -1;
      } else if((dir==1) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = -1;
	elem[2] = -1;
	elem[3] = 1;
      } else if((dir==2)) {
	elem[0] = I;
	elem[1] = -I;
	elem[2] = -I;
	elem[3] = I;
      } else if((dir==3) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = 1;
      } else if((dir==3) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = -1;
	elem[3] = -1;
      } else if((dir==4) && (basis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS)) {
	elem[0] = -1;
	elem[1] = -1;
	elem[2] = 1;
	elem[3] = 1;
      } else if((dir==4) && (basis == QUDA_UKQCD_GAMMA_BASIS)) {
	elem[0] = 1;
	elem[1] = 1;
	elem[2] = 1;
	elem[3] = 1;
      } else {
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
    __device__ __host__ inline complex<ValueType> getelem(int row, int col) const {
      if(coupling[row] == col) {
	return elem[row];
      } else {
	return 0;
      }
    }

    //Like getelem, but one only needs to specify the row.
    //The column of the non-zero component is returned via the "col" reference
    __host__ __device__ inline complex<ValueType> getrowelem(int row, int &col) const {
      col = coupling[row];
      return elem[row];
    }

    //Returns the type of Gamma matrix
    inline int Dir() const {
      return dir;
    }
  };

  //Returns the non parity-blocked integer index for a lattice site.  Also calculates the parity index of a site.
  int gauge_offset_index(const int *x, const int *x_size, int ndim, int& parity) {
    parity = 0;
    int gauge_index = 0;
    for(int d = ndim-1; d >= 0; d--) {
      parity += x[d];
      gauge_index *= x_size[d];
      gauge_index += x[d];
    }
    parity = parity%2;
    return gauge_index;
  }


  //Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
  //Where:
  //mu = dir
  //s = fine spin
  //c' = coarse color
  //c = fine color
  //FIXME: N.B. Only works if color-spin field and gauge field are parity ordered in the same way.  Need LatticeIndex function for generic ordering
  template<typename Float, typename F, typename fineGauge>
  void computeUV(F &UV, const F &V, const fineGauge &G, int dir, int ndim, const int *x_size) {
	
    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      //U connects site x to site x+mu.  Thus, V lives at site x+mu if U_mu lives at site x.
      //FIXME: Uses LatticeIndex() for the color spinor field to determine gauge field index.
      //This only works if sites are ordered same way in both G and V.

      int coord[QUDA_MAX_DIM];
      int coordV[QUDA_MAX_DIM];
      V.LatticeIndex(coord, i);
      V.LatticeIndex(coordV, i);
      int parity = 0;
      int gauge_index = gauge_offset_index(coord, x_size, ndim, parity);

      //Shift the V field w/respect to G
      coordV[dir] = (coord[dir]+1)%x_size[dir];
      int i_V = 0;
      V.OffsetIndex(i_V, coordV);
      for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin
	for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
	  for(int ic = 0; ic < G.Ncolor(); ic++) { //Fine Color rows of gauge field
	    for(int jc = 0; jc < G.Ncolor(); jc++) {  //Fine Color columns of gauge field
	      UV(i, s, ic, ic_c) += G(dir, parity, gauge_index/2, ic, jc) * V(i_V, s, jc, ic_c);
	    }  //Fine color columns
	  }  //Fine color rows
	}  //Coarse color
      }  //Fine Spin
    }  //Volume
  }  //UV

  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void computeVUV(int dir, coarseGauge &Y, coarseGauge &X, const F &UV, const F &V, const Gamma<Float> &gamma, int ndim, const fineGauge &G, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs) {

    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      Float half = 1.0/2.0;
      int coord[QUDA_MAX_DIM];
      int coord_coarse[QUDA_MAX_DIM];
      int coarse_size = 1;
      int coarse_parity = 0;
      int coarse_index = 0;
      V.LatticeIndex(coord, i);
      for(int d = ndim-1; d >= 0; d--) {
	coord_coarse[d] = coord[d]/geo_bs[d];
	coarse_size *= xc_size[d];
      }
      coarse_index = gauge_offset_index(coord_coarse, xc_size, ndim, coarse_parity);

      //int coarse_site_offset = coarse_parity*coarse_size/2 + coarse_index/2;
      //coarse_site_offset *= Nc_c*Nc_c*Ns_c*Ns_c;

      //Check to see if we are on the edge of a block, i.e.
      //if this color matrix connects adjacent blocks.
      coarseGauge *M;
      //int dim_index = 2*dir+1;
      int dim_index = dir;
      //If adjacent site is in same block, M = X
      if(((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) {
	M = &X;
	dim_index = 0;
      }
      //If adjacent site is in different block, M = Y
      else {
     	M = &Y;
      }

      for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin

	//Spin part of the color matrix.  Will always
	//consist of two terms - diagonal and off-diagonal part
	//of P_mu = (1+\gamma_mu)

	//Coarse spin row index
	int s_c_row = s/spin_bs;

	//Use Gamma to calculate off-diagonal coupling and
	//column index.  Diagonal coupling is always 1.
	int s_col;
	complex<Float> coupling = gamma.getrowelem(s, s_col);
	int s_c_col = s_col/spin_bs;

	for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
	  for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column

	    for(int ic = 0; ic < G.Ncolor(); ic++) { //Sum over fine color
	      //Diagonal Spin
	      (*M)(dim_index,coarse_parity,coarse_index/2,s_c_row,s_c_row,ic_c,jc_c) += half * conj(V(i, s, ic, ic_c)) * UV(i, s, ic, jc_c); 

	      //Off-diagonal Spin
	      (*M)(dim_index,coarse_parity,coarse_index/2,s_c_row,s_c_col,ic_c,jc_c) += half * coupling * conj(V(i, s, ic, ic_c)) * UV(i, s_col, ic, jc_c);
	    } //Fine color
	  } //Coarse Color column
	} //Coarse Color row

      } //Fine spin

    } //Volume

  }

  //Calculate the reverse link, Y_{-\mu}(x+mu).
  //The reverse link is almost the same as the forward link,
  //but with negative sign for the spin off-diagonal parts to account
  //for the forward/backward spin proejctors.
  //Note: No shifting in site index, so that site x holds the matrices Y_{\pm mu}(x, x+mu)
  //M.C.: No longer used, as only forward link is stored.
  template<typename Float, typename Gauge>
  void reverseY(int dir,  Gauge &Y, int ndim, const int *xc_size, int Nc_c, int Ns_c)  {
    int csize = 1;
    for(int d = 0; d < ndim; d++) {
      csize *= xc_size[d];
    }
	
    for(int i = 0; i < csize; i++) { //Volume
      //int coarse_site_offset = i*Nc_c*Nc_c*Ns_c*Ns_c;
      for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin row
	for(int s_col = 0; s_col < Ns_c; s_col++) { //Spin column
	  Float sign = 1.0;
	  if (s_row != s_col) {
	    sign = -1.0;
	  }
	  for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color row
	    for(int jc_c = 0; jc_c < Nc_c; jc_c++) { //Color column
	      //Flip s_row, s_col and ic_c, jc_c indices on the rhs because of Hermitian conjugation.
	      Y(2*dir, i%2, i/2, s_row, s_col, ic_c, jc_c) = sign*Y(2*dir+1,i%2,i/2,s_row, s_col, ic_c, jc_c);
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
  template<typename Float, typename Gauge>
  void coarseDiagonal(Gauge &X, int ndim, const int *xc_size) {
    int csize = 1;
    for(int d = 0; d < ndim; d++) {
      csize *= xc_size[d];
    }

    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
    const int local = nSpin*nSpin*nColor*nColor;
    complex<Float> *Xlocal = new complex<Float>[local];
	
    for(int i = 0; i < csize; i++) { //Volume
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	     
	  //Copy the Hermitian conjugate term to temp location 
	  for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	    for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
	      //Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
	      Xlocal[nColor*nColor*(nSpin*s_col+s_row)+nColor*ic_c+jc_c] = X(0,i%2,i/2,s_row, s_col, ic_c, jc_c);
	    }	
	  }
	}
      }

      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column

	  Float sign = 1.0;	
	  if(s_row != s_col) {
	    sign = -1.0;
	  }

	  for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	    for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
	      //Transpose color part
	      X(0,i%2,i/2,s_row,s_col,ic_c,jc_c) =  sign*X(0,i%2,i/2,s_row,s_col,ic_c,jc_c)+conj(Xlocal[nColor*nColor*(nSpin*s_row+s_col)+nColor*jc_c+ic_c]);
	    } //Color column
	  } //Color row
	} //Spin column
      } //Spin row


    } //Volume
    delete [] Xlocal;
  }

  //Currently unusued.  Combining mass and coarse clover term moved to the application of the operator.
  template<typename Float, typename Gauge>
  void addMass(Gauge &Y, int ndim, const int *xc_size, int Nc_c, int Ns_c, double mass)  {
    //  void addMass(complex<Float> *X, int ndim, const int *xc_size, int Nc_c, int Ns_c, double mass)  {

    int csize = 1;
    for(int d = 0; d < ndim; d++) {
      csize *= xc_size[d];
    }

    for(int i = 0; i < csize; i++) { //Volume
      for(int s_row = 0; s_row < Ns_c; s_row++) { //Spin 
	for(int ic_c = 0; ic_c < Nc_c; ic_c++) { //Color
	  Y(2*ndim, i%2, i/2, s_row, s_row, ic_c, ic_c) += mass;
	} //Color
      } //Spin
    } //Volume

  }

  //Zero out a field, using the accessor.
  template<typename Float, typename F>
  void setZero(F &f) {
    for(int i = 0; i < f.Volume(); i++) {
      for(int s = 0; s < f.Nspin(); s++) {
	for(int c = 0; c < f.Ncolor(); c++) {
	  f(i,s,c) = (Float) 0.0;
        }
      }
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void calculateY(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge &G, const int *x_size) {
#if 1
    int ndim = G.Ndim();
    const int *xc_size = Y.Field().X();
    int geo_bs[QUDA_MAX_DIM]; 
    for(int d = 0; d < ndim; d++) {
      geo_bs[d] = x_size[d]/xc_size[d];
    }
    int spin_bs = V.Nspin()/Y.NspinCoarse();
#endif

    for(int d = 0; d < ndim; d++) {
      //First calculate UV
      setZero<Float,F>(UV);

      printfQuda("Computing %d UV\n", d);
      computeUV<Float>(UV, V, G, d, ndim, x_size);

      //Gamma matrix for this direction
      Gamma<Float> gamma(d, UV.GammaBasis());

      printfQuda("Computing %d VUV\n", d);
      //Calculate VUV for this direction, accumulate in the appropriate place
      computeVUV<Float>(d, Y, X, UV, V, gamma, ndim, G, x_size, xc_size, geo_bs, spin_bs);

      Float norm2 = 0;
      for (int x=0; x<UV.Volume(); x++) 
	for (int s=0; s<UV.Nspin(); s++) 
	  for (int c=0; c<UV.Ncolor(); c++) 
	    for (int v=0; v<UV.Nvec(); v++)
	      norm2 += norm(UV(x,s,c,v));

      printf("UV2[%d] = %e\n", d, norm2);

      norm2 = 0;
      for (int x=0; x<Y.Volume(); x++)
	for (int s=0; s<Y.Ncolor(); s++) 
	  for (int c=0; c<Y.Ncolor(); c++) 
	    norm2 += norm(Y(d,0,x,s,c));

      printf("Y2[%d] = %e\n", d, norm2);

      //MC - "forward" and "backward" coarse gauge fields need not be calculated and stored separately.  Only difference
      //is factor of -1 in off-diagonal spin, which can be applied directly by the Dslash operator.
      //reverseY<Float>(d, Y, ndim, xc_size,Y.NcolorCoarse(), Y.NspinCoarse());
#if 0
      for(int i = 0; i < Y.Volume(); i++) {
	for(int s = 0; s < Y.NspinCoarse(); s++) {
          for(int s_col = 0; s_col < Y.NspinCoarse(); s_col++) {
            for(int c = 0; c < Y.NcolorCoarse(); c++) {
              for(int c_col = 0; c_col < Y.NcolorCoarse(); c_col++) {
                printf("d=%d i=%d s=%d s_col=%d c=%d c_col=%d Y(2*d) = %e %e, Y(2*d+1) = %e %e\n",d,i,s,s_col,c,c_col,Y(2*d,i%2,i/2,s,s_col,c,c_col).real(),Y(2*d,i%2,i/2,s,s_col,c,c_col).imag(),Y(2*d+1,i%2,i/2,s,s_col,c,c_col).real(),Y(2*d+1,i%2,i/2,s,s_col,c,c_col).imag());
              }}}}}
#endif
    }

    printfQuda("Computing coarse diagonal\n");
    coarseDiagonal<Float>(X, ndim, xc_size);

    Float norm2 = 0;
    for (int x=0; x<X.Volume(); x++)
      for (int s=0; s<X.Ncolor(); s++) 
	for (int c=0; c<X.Ncolor(); c++) 
	  norm2 += norm(X(0,0,x,s,c));

    printf("X2 = %e\n", norm2);

#if 0
    for(int i = 0; i < Y.Volume(); i++) {
      for(int s = 0; s < Y.NspinCoarse(); s++) {
	for(int s_col = 0; s_col < Y.NspinCoarse(); s_col++) {
	  for(int c = 0; c < Y.NcolorCoarse(); c++) {
	    for(int c_col = 0; c_col < Y.NcolorCoarse(); c_col++) {
	      printf("d=%d i=%d s=%d s_col=%d c=%d c_col=%d Y(2*d) = %e %e\n",ndim,i,s,s_col,c,c_col,Y(2*ndim,i%2,i/2,s,s_col,c,c_col).real(),Y(2*ndim,i%2,i/2,s,s_col,c,c_col).imag());
	    }}}}}
#endif

    //addMass<Float>(Y[2*ndim], ndim, xc_size, Nc_c, Ns_c, mass);
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, 
	    int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    typedef typename colorspinor::FieldOrder<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;

    F vAccessor(const_cast<ColorSpinorField&>(T.Vectors()));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X)); 
    calculateY<Float>(yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, g.X());
  }


  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int fineSpin>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2) 
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) { 
      calculateY<Float,csOrder,gOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, uv, T, g);
    } else if (coarseColor == 24) {
      calculateY<Float,csOrder,gOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    if (uv.Nspin() == 4) {
      calculateY<Float,csOrder,gOrder,fineColor,4>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    if (g.Ncolor() == 3) {
      calculateY<Float,csOrder,gOrder,3>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

  template <typename Float>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateY<double>(Y, X, uv, T, g);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateY<float>(Y, X, uv, T, g);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X, const cudaGaugeField &gauge) {
    //  void CoarseOp(const Transfer &T, void *Y[], QudaPrecision precision, const cudaGaugeField &gauge) {

    QudaPrecision precision = Y.Precision();
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
    gf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    cpuGaugeField g(gf_param);

    //Copy the cuda gauge field to the cpu
    gauge.saveCPUField(g, QUDA_CPU_FIELD_LOCATION);

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField uv(UVparam);

    calculateY(Y, X, uv, T, g);
  }  

  //Apply the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float, typename F, typename Gauge>
  void coarseDslash(F &out, F &in, Gauge &Y) {
    // void coarseDslash(colorspinor::FieldOrder<Float> &out, colorspinor::FieldOrder<Float> &in, complex<Float> *Y[]) {
    int Nc = in.Ncolor();
    int Ns = in.Nspin();
    int ndim = in.Ndim();
    int sites = in.Volume();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x_size[d] = in.X(d);
    }

    for(int i = 0; i < sites; i++) { //Volume
      int coord[QUDA_MAX_DIM];
      int parity = 0;
      int gauge_index = 0;
      in.LatticeIndex(coord,i);
      gauge_index = gauge_offset_index(coord, x_size, ndim, parity);

      for(int d = 0; d < ndim; d++) { //Ndim

	int forward[QUDA_MAX_DIM];
	int forward_spinor_index;
	int backward[QUDA_MAX_DIM];
        int backward_spinor_index;
	int backward_gauge_index = 0;
        int backward_parity = 0;

        //We need the coordinates of the forward site to index into the "out" field.
        //We need the coordinates of the backward site to index into the "out" field, 
	//as well as to retrieve the gauge field there for parallel transport.
	in.LatticeIndex(forward,i);
        in.LatticeIndex(backward,i);
	forward[d] = (forward[d] + 1)%x_size[d];
	backward[d] = (backward[d] - 1 + x_size[d])%x_size[d];
	out.OffsetIndex(forward_spinor_index, forward);
	out.OffsetIndex(backward_spinor_index, backward);

        for(int dim = ndim-1; dim >= 0; dim--) {
          backward_parity += backward[dim];
          backward_gauge_index *= x_size[dim];
          backward_gauge_index += backward[dim];
        }
	backward_parity = backward_parity%2;

        for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	  for(int c_row = 0; c_row < Nc; c_row++) { //Color row
            for(int s_col = 0; s_col < Ns; s_col++) { //Spin column

	      Float sign = (s_row == s_col) ? 1.0 : -1.0;

              for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	        //Forward link  
		out(forward_spinor_index, s_row, c_row) += conj(Y(d,parity, gauge_index/2, s_col, s_row, c_col, c_row))*in(i, s_col, c_col);

	        //Backward link
		out(backward_spinor_index, s_row, c_row) += sign*Y(d, backward_parity, backward_gauge_index/2, s_row, s_col, c_row, c_col)*in(i, s_col, c_col);
	      } //Color column
	    } //Spin column
	  } //Color row
        } //Spin row 

      } //Ndim

    }//Volume

  }

  //out(x) = -X*in(x), where X is the local color-spin matrix on the coarse grid.
  template<typename Float, typename F, typename G>
  void coarseClover(F &out, const F &in, const G &X, Float kappa) {
    int Nc = out.Ncolor();
    int Ns = out.Nspin();
    int ndim = out.Ndim();
    int sites = out.Volume();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < ndim; d++) {
      x_size[d] = out.X(d);
    }


    for(int i = 0; i < out.Volume(); i++) { //Volume
      int coord[QUDA_MAX_DIM];
      int parity = 0;
      int gauge_index = 0;
      out.LatticeIndex(coord,i);
      gauge_index = gauge_offset_index(coord, x_size, ndim, parity);

      for(int s = 0; s < Ns; s++) { //Spin out
        for(int c = 0; c < Nc; c++) { //Color out
	  out(i,s,c) += in(i,s,c);
	  for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	    for(int c_col = 0; c_col < Nc; c_col++) { //Color in
              //printf("pre i = %d, s = %d, c = %d, s_col = %d, c_col = %d, parity = %d, gauge_index = %d, Y = %e, out(i,s,c) = %e, in(i,s_col,c_col) = %e\n",i,s,c,s_col,c_col,parity, gauge_index, Y(2*ndim,parity,gauge_index/2,s,s_col,c,c_col).real(),out(i,s,c).real(),in(i,s_col,c_col).real());
	      out(i,s,c) -= 2*kappa*X(0, parity, gauge_index/2, s, s_col, c, c_col)*in(i,s_col,c_col);
              //printf("post i = %d, s = %d, c = %d, s_col = %d, c_col = %d, out(i,s,c) = %e, in(i,s_col,c_col) = %e\n",i,s,c,s_col,c_col,out(i,s,c).real(),in(i,s_col,c_col).real());
	    } //Color in
          } //Spin in
        } //Color out
      } //Spin out
    } //Volume
  }

  //Multiply a field by a real constant
  template<typename Float, typename F>
  void F_eq_rF(F &f, Float r) {
    for(int i = 0; i < f.Volume(); i++) {
      for(int s = 0; s < f.Nspin(); s++) {
        for(int c = 0; c < f.Ncolor(); c++) {
          f(i,s,c) *= r;
        }
      }
    }
  }


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    int Ns_c = in.Nspin();
    int ndim = in.Ndim();
    
    typedef typename colorspinor::FieldOrder<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;

    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessor(const_cast<ColorSpinorField&>(in));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    setZero<Float, F>(outAccessor);
    coarseDslash<Float,F,G>(outAccessor, inAccessor, yAccessor);
    F_eq_rF(outAccessor, (-2.0*kappa));
    coarseClover(outAccessor, inAccessor, xAccessor, (Float)kappa);
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (in.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",in.Nspin());

    if (in.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, in, Y, X, kappa);
    } else if (in.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder fOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,fOrder,QUDA_QDP_GAUGE_ORDER>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", Y.FieldOrder());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", in.FieldOrder());
    }
  }


  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = (1-X)*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, in, Y, X, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
  }//ApplyCoarse

} //namespace quda
