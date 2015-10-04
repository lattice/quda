#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>

namespace quda {

  //A simple Euclidean gamma matrix class for use with the Wilson projectors.
  template <typename ValueType, QudaGammaBasis basis, int dir>
  class Gamma {
  private:
    const int ndim;

  protected:


    //Which gamma matrix (dir = 0,4)
    //dir = 0: gamma^1, dir = 1: gamma^2, dir = 2: gamma^3, dir = 3: gamma^4, dir =4: gamma^5
    //int dir;

    //The basis to be used.
    //QUDA_DEGRAND_ROSSI_GAMMA_BASIS is the chiral basis
    //QUDA_UKQCD_GAMMA_BASIS is the non-relativistic basis.
    //QudaGammaBasis basis;

    //The column with the non-zero element for each row
    int coupling[4];
    //The value of the matrix element, for each row
    complex<ValueType> elem[4];

  public:

    Gamma() : ndim(4) {
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

    Gamma(const Gamma &g) : ndim(4) {
      for(int i = 0; i < ndim+1; i++) {
	coupling[i] = g.coupling[i];
	elem[i] = g.elem[i];
      }
    }

    ~Gamma() {}

    //Returns the matrix element.
    __device__ __host__ inline complex<ValueType> getelem(int row, int col) const {
      return coupling[row] == col ? elem[row] : 0;
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
  template<typename Float, int dir, typename F, typename fineGauge>
  void computeUV(F &UV, const F &V, const fineGauge &G, int ndim, const int *x_size) {
	
    int coord[QUDA_MAX_DIM];
    for (int parity=0; parity<2; parity++) {
      int x_cb = 0;
      for (coord[3]=0; coord[3]<x_size[3]; coord[3]++) {
	for (coord[2]=0; coord[2]<x_size[2]; coord[2]++) {
	  for (coord[1]=0; coord[1]<x_size[1]; coord[1]++) {
	    for (coord[0]=0; coord[0]<x_size[0]/2; coord[0]++) {
	      int coord_tmp = coord[dir];

	      //Shift the V field w/respect to G (must be on full field coords)
	      int oddBit = (parity + coord[1] + coord[2] + coord[3])&1;
	      if (dir==0) coord[0] = 2*coord[0] + oddBit;
	      coord[dir] = (coord[dir]+1)%x_size[dir];
	      if (dir==0) coord[0] /= 2;
	      int y_cb = ((coord[3]*x_size[2]+coord[2])*x_size[1]+coord[1])*(x_size[0]/2) + coord[0];

              for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin
		for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
                  for(int ic = 0; ic < G.Ncolor(); ic++) { //Fine Color rows of gauge field
		    for(int jc = 0; jc < G.Ncolor(); jc++) {  //Fine Color columns of gauge field
		      UV(parity, x_cb, s, ic, ic_c) += G(dir, parity, x_cb, ic, jc) * V((parity+1)&1, y_cb, s, jc, ic_c);
		    }  //Fine color columns
		  }  //Fine color rows
		}  //Coarse color
	      }  //Fine Spin

	      coord[dir] = coord_tmp; //restore
	      x_cb++;
	    }
	  }
	}
      }
    } // parity

  }  //UV

  template<typename Float, int dir, typename F, typename coarseGauge, typename fineGauge, typename Gamma>
  void computeVUV(coarseGauge &Y, coarseGauge &X, const F &UV, const F &V, 
		  const Gamma &gamma, const fineGauge &G, const int *x_size, 
		  const int *xc_size, const int *geo_bs, int spin_bs) {

    const int nDim = 4;
    const Float half = 0.5;
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    // paralleling this requires care with respect to race conditions
    // on CPU, parallelize over dimension not parity

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      int x_cb = 0;
      for (coord[3]=0; coord[3]<x_size[3]; coord[3]++) {
	for (coord[2]=0; coord[2]<x_size[2]; coord[2]++) {
	  for (coord[1]=0; coord[1]<x_size[1]; coord[1]++) {
	    for (coord[0]=0; coord[0]<x_size[0]/2; coord[0]++) {

	      int oddBit = (parity + coord[1] + coord[2] + coord[3])&1;
	      coord[0] = 2*coord[0] + oddBit;
	      for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	      //Check to see if we are on the edge of a block, i.e.
	      //if this color matrix connects adjacent blocks.  If
	      //adjacent site is in same block, M = X, else M = Y
	      const bool isDiagonal = ((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir] ? true : false;
	      coarseGauge &M =  isDiagonal ? X : Y;
	      const int dim_index = isDiagonal ? 0 : dir;
	      
	      //printf("dir = %d (%d,%d,%d,%d)=(%d,%d) (%d,%d,%d,%d)=", dir, 
	      //   coord[0], coord[1], coord[2], coord[3], x_cb, parity,
	      //   coord_coarse[0], coord_coarse[1], coord_coarse[2], coord_coarse[3]);

	      int coarse_parity = 0;
	      for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	      coarse_parity &= 1;
	      coord_coarse[0] /= 2;
	      int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];
	      
	      //printf("(%d,%d)\n", coarse_x_cb, coarse_parity);

	      coord[0] /= 2;

  	        for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin
		  //Spin part of the color matrix.  Will always consist
		  //of two terms - diagonal and off-diagonal part of
		  //P_mu = (1+\gamma_mu)
		
		  int s_c_row = s/spin_bs; //Coarse spin row index
	
		  //Use Gamma to calculate off-diagonal coupling and
		  //column index.  Diagonal coupling is always 1.
		  int s_col;
		  complex<Float> coupling = gamma.getrowelem(s, s_col);
		  int s_c_col = s_col/spin_bs;

		  for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
		    for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column
		      for(int ic = 0; ic < G.Ncolor(); ic++) { //Sum over fine color
		        //Diagonal Spin
		        M(dim_index,coarse_parity,coarse_x_cb,s_c_row,s_c_row,ic_c,jc_c) += 
			  half * conj(V(parity, x_cb, s, ic, ic_c)) * UV(parity, x_cb, s, ic, jc_c); 
		      
		        //Off-diagonal Spin
		        M(dim_index,coarse_parity,coarse_x_cb,s_c_row,s_c_col,ic_c,jc_c) += 
			  half * coupling * conj(V(parity, x_cb, s, ic, ic_c)) * UV(parity, x_cb, s_col, ic, jc_c);
		      } //Fine color
		    } //Coarse Color column
		  } //Coarse Color row

	        } //Fine spin

	      x_cb++;
	    } // coord[0]
	  } // coord[1]
	} // coord[2]
      } // coord[3]
    } // parity

  }


  //Adds the identity matrix to the coarse local term.
  template<typename Float, typename Gauge>
  void addCoarseDiagonal(Gauge &X, int ndim, const int *xc_size) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
    complex<Float> *Xlocal = new complex<Float>[nSpin*nSpin*nColor*nColor];

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.Volume()/2; x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
         for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
            X(0,parity,x_cb,s,s,ic_c,ic_c) += 1.0;
         } //Color
        } //Spin
      } // x_cb
    } //parity
   }


  //Adds the reverse links to the coarse local term, which is just
  //the conjugate of the existing coarse local term but with
  //plus/minus signs for off-diagonal spin components
  //Also multiply by the appropriate factor of -2*kappa
  template<typename Float, typename Gauge>
  void createCoarseLocal(Gauge &X, int ndim, const int *xc_size, double kappa) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
    Float kap = (Float) kappa;
    complex<Float> *Xlocal = new complex<Float>[nSpin*nSpin*nColor*nColor];
	
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.Volume()/2; x_cb++) {

	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	    
	    //Copy the Hermitian conjugate term to temp location 
	    for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
		Xlocal[((nSpin*s_col+s_row)*nColor+ic_c)*nColor+jc_c] = X(0,parity,x_cb,s_row, s_col, ic_c, jc_c);
	      }	
	    }
	  }
	}
	      
	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	    
	    const Float sign = (s_row == s_col) ? 1.0 : -1.0;
		  
	    for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Transpose color part
		X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =  
		  -2*kap*(sign*X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)+conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	      } //Color column
	    } //Color row
	  } //Spin column
	} //Spin row

      } // x_cb
    } //parity

    delete []Xlocal;

  }

  //Zero out a field, using the accessor.
  template<typename Float, typename F>
  void setZero(F &f) {
    for(int parity = 0; parity < 2; parity++) {
      for(int x_cb = 0; x_cb < f.Volume()/2; x_cb++) {
	for(int s = 0; s < f.Nspin(); s++) {
	  for(int c = 0; c < f.Ncolor(); c++) {
	    for(int v = 0; v < f.Nvec(); v++) {
	      f(parity,x_cb,s,c,v) = (Float) 0.0;
	    }
	  }
	}
      }
    }
  }

  template<typename Float, typename coarseGauge, typename F, typename clover>
  void createCoarseClover(coarseGauge &X, F &V,  clover &C, int ndim, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs)  {

    const int nDim = 4;
    const Float half = 0.5;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];

    for (int parity=0; parity<2; parity++) {
      int x_cb = 0;
      for (coord[3]=0; coord[3]<x_size[3]; coord[3]++) {
        for (coord[2]=0; coord[2]<x_size[2]; coord[2]++) {
          for (coord[1]=0; coord[1]<x_size[1]; coord[1]++) {
            for (coord[0]=0; coord[0]<x_size[0]/2; coord[0]++) {

              int oddBit = (parity + coord[1] + coord[2] + coord[3])&1;
              coord[0] = 2*coord[0] + oddBit;
              for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];
              int coarse_parity = 0;
              for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
              coarse_parity &= 1;
              coord_coarse[0] /= 2;
              int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];

              coord[0] /= 2;

	      int s_c = 0;

              //If Nspin = 4, then the clover term has structure C_{\mu\nu} = \gamma_{\mu\nu}C^{\mu\nu}

                //printf("C.Ncolor() = %d C.NcolorCoarse() = %d\n",C.Ncolor(), C.NcolorCoarse());
                for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin row
		  s_c = s/spin_bs;
		  //On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
		  //in the same chiral block.
                  for(int s_col = s_c*spin_bs; s_col < (s_c+1)*spin_bs; s_col++) { //Loop over fine spin column
                    for(int ic_c = 0; ic_c < X.NcolorCoarse(); ic_c++) { //Coarse Color row
                      for(int jc_c = 0; jc_c < X.NcolorCoarse(); jc_c++) { //Coarse Color column

                        for(int ic = 0; ic < C.Ncolor(); ic++) { //Sum over fine color row
                          for(int jc = 0; jc < C.Ncolor(); jc++) {  //Sum over fine color column
			    X(0,coarse_parity,coarse_x_cb,s_c,s_c,ic_c,jc_c) += conj(V(parity, x_cb, s, ic, ic_c)) * C(0, parity, x_cb, s, s_col, ic, jc) * V(parity, x_cb, s_col, jc, jc_c);
                          } //Fine color column
                        }  //Fine color row
                      } //Coarse Color column
                    } //Coarse Color row
                  }  //Fine spin column
                } //Fine spin


              x_cb++;
            } // coord[0]
          } // coord[1]
        } // coord[2]
      } // coord[3]
    } // parity

  }

  //Calculates the coarse gauge field
  template<typename Float, typename F, typename coarseGauge, typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge &G, fineClover *C, const int *x_size, const int *xc_size, double kappa) {
    if (UV.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) errorQuda("Gamma basis not supported");
    const QudaGammaBasis basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int geo_bs[QUDA_MAX_DIM]; 
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse();

    for(int d = 0; d < nDim; d++) {
      //First calculate UV
      setZero<Float,F>(UV);

      printfQuda("Computing %d UV and VUV\n", d);
      //Calculate UV and then VUV for this direction, accumulating directly into the coarse gauge field Y
      if (d==0) {
        computeUV<Float,0>(UV, V, G, nDim, x_size);
        Gamma<Float, basis, 0> gamma;
        computeVUV<Float,0>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else if (d==1) {
        computeUV<Float,1>(UV, V, G, nDim, x_size);
        Gamma<Float, basis, 1> gamma;
        computeVUV<Float,1>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else if (d==2) {
        computeUV<Float,2>(UV, V, G, nDim, x_size);
        Gamma<Float, basis, 2> gamma;
        computeVUV<Float,2>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else {
        computeUV<Float,3>(UV, V, G, nDim, x_size);
        Gamma<Float, basis, 3> gamma;
        computeVUV<Float,3>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      }

      printf("UV2[%d] = %e\n", d, UV.norm2());
      printf("Y2[%d] = %e\n", d, Y.norm2(d));
    }
    printf("X2 = %e\n", X.norm2(0));
    printfQuda("Computing coarse diagonal\n");
    createCoarseLocal<Float>(X, nDim, xc_size, kappa);

    //If C!=NULL we have to coarsen the fine clover term and add it in.
    if (C != NULL) {
      printfQuda("Computing fine->coarse clover term\n");
      createCoarseClover<Float>(X, V, *C, nDim, x_size, xc_size, geo_bs, spin_bs);
      printf("X2 = %e\n", X.norm2(0));
    }
    //Otherwise, we have a fine Wilson operator.  The "clover" term for the Wilson operator
    //is just the identity matrix.
    else {
      addCoarseDiagonal<Float>(X, nDim, xc_size);
    }
    printf("X2 = %e\n", X.norm2(0));

}


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder,
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
    typedef typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder> cFine;

    F vAccessor(const_cast<ColorSpinorField&>(T.Vectors()));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));

    if(c != NULL) {
      cFine cAccessor(const_cast<CloverField&>(*c));

      calculateY<Float>(yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, &cAccessor, g.X(), Y.X(), kappa);
    }
    else {
      cFine *cAccessor = NULL;
      calculateY<Float>(yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, cAccessor, g.X(), Y.X(), kappa);
    }    
  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder, int fineColor, int fineSpin>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, uv, T, g, c, kappa);
    } else if (coarseColor == 24) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder, int fineColor>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (uv.Nspin() == 4) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,4>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv.Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (g.Ncolor() == 3) {
      calculateY<Float,csOrder,gOrder,clOrder,3>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    //If c == NULL, then this is standard Wilson.  csOrder is dummy and will not matter      
    if (c==NULL || c->Order() == QUDA_PACKED_CLOVER_ORDER) {
      calculateY<Float,csOrder,gOrder,QUDA_PACKED_CLOVER_ORDER>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", c->Order());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

 template <typename Float>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateY<double>(Y, X, uv, T, g, c, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateY<float>(Y, X, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X, const cudaGaugeField &gauge, const cudaCloverField *clover, double kappa) {
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

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField
    if(clover != NULL) {
      //Create a cpuCloverField from the cudaCloverField
      CloverFieldParam cf_param;
      cf_param.nDim = 4;
      cf_param.pad = pad;
      cf_param.precision = clover->Precision();
      for(int i = 0; i < cf_param.nDim; i++) {
        cf_param.x[i] = clover->X()[i];
      }

      cf_param.order = QUDA_PACKED_CLOVER_ORDER;
      cf_param.direct = true;
      cf_param.inverse = true;
      cf_param.clover = NULL;
      cf_param.norm = 0;
      cf_param.cloverInv = NULL;
      cf_param.invNorm = 0;
      cf_param.create = QUDA_NULL_FIELD_CREATE;
      cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

      cpuCloverField c(cf_param);
      clover->saveCPUField(c);

      calculateY(Y, X, uv, T, g, &c, kappa);
    }
    else {
      calculateY(Y, X, uv, T, g, NULL, kappa);
    }
  }

  //Adds the reverse links to the coarse local term, which is just
  //the conjugate of the existing coarse local term but with
  //plus/minus signs for off-diagonal spin components
  //Also multiply by the appropriate factor of -2*kappa
  template<typename Float, typename Gauge>
  void createKSCoarseLocal(Gauge &X, int ndim, const int *xc_size, double k) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
    if (nSpin != 2) errorQuda("\nWrong coarse spin degrees.\n");

    Float kap = (Float) k;//mass term
    complex<Float> *Xlocal = new complex<Float>[nSpin*nSpin*nColor*nColor];
	
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.Volume()/2; x_cb++) {

	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	    
	    //Copy the Hermitian conjugate term to temp location 
	    for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
		Xlocal[((nSpin*s_col+s_row)*nColor+ic_c)*nColor+jc_c] = X(0,parity,x_cb,s_row, s_col, ic_c, jc_c);
	      }	
	    }
	  }
	}
	      
	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
            for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
//!
	      if(s_row == s_col){
                X(0,parity,x_cb, parity, parity, ic_c,ic_c) += (parity == 0) ? +1.0 : -1.0;
                continue;
              }
//!
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Transpose color part
		X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) = 2*kap*(+X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)-conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));//always minus sign?
	      } //Color column
	    } //Color row
	  } //Spin column
	} //Spin row

      } // x_cb
    } //parity

    delete[] Xlocal;

    return;
  }

  //added HISQ links
  template<typename Float, int dir, typename F, typename fineGauge>
  void computeKSUV(F &UV, const F &V, const fineGauge *FL, const fineGauge *LL, int ndim, 
                   const int *x_size) {
	
    int coord[QUDA_MAX_DIM];
     
    for (int parity=0; parity<2; parity++) {
      int x_cb = 0;
      for (coord[3]=0; coord[3]<x_size[3]; coord[3]++) {
	for (coord[2]=0; coord[2]<x_size[2]; coord[2]++) {
	  for (coord[1]=0; coord[1]<x_size[1]; coord[1]++) {
	    for (coord[0]=0; coord[0]<x_size[0]/2; coord[0]++) {
	      int coord_tmp  = coord[dir];
              int coord_3[4] = {coord[0], coord[1], coord[2], coord[3]} ; 

	      //Shift the V field w/respect to G (must be on full field coords)
	      int oddBit = (parity + coord[1] + coord[2] + coord[3]) & 1;
	      if (dir==0) coord[0] = 2*coord[0] + oddBit;
              //!
 	      coord[dir]   = (coord[dir]+1)%x_size[dir];
              if(LL != NULL) coord_3[dir] = (coord_3[dir]+3)%x_size[dir];

	      if (dir==0) {coord[0] /= 2; coord_3[0] /= 2;}

	      int y_cb = ((coord[3]*x_size[2]+coord[2])*x_size[1]+coord[1])*(x_size[0]/2) + coord[0];
              int y3_cb = (LL != NULL) ? (((coord_3[3]*x_size[2]+coord_3[2])*x_size[1]+coord_3[1])*(x_size[0]/2) + coord_3[0]) : 0;

	      for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
                for(int ic = 0; ic < FL->Ncolor(); ic++) { //Fine Color rows of gauge field
		   for(int jc = 0; jc < FL->Ncolor(); jc++) {  //Fine Color columns of gauge field
		      UV(parity, x_cb, 0, ic, ic_c) += (*FL)(dir, parity, x_cb, ic, jc) * V((parity+1)&1, y_cb, 0, jc, ic_c);//mind transformation to the opposite parity field: in UVU operation.
                      if(LL != NULL) UV(parity, x_cb, 0, ic, ic_c) += (*LL)(dir, parity, x_cb, ic, jc) * V((parity+1)&1, y3_cb, 0, jc, ic_c);
		   }  //Fine color columns
		}  //Fine color rows
	      }  //Coarse color

	      coord[dir] = coord_tmp; //restore
	      x_cb++;
	    }
	  }
	}
      }
    } // parity

  }  //UV

//KS (also HISQ) operator:
  template<typename Float, int dir, typename F, typename coarseGauge>
  void computeKSVUV(coarseGauge &Y, coarseGauge &X, const F &UV, const F &V, const int nfinecolors,
		  const int *x_size, const int *xc_size, const int *geo_bs) {

    const int nDim = 4;
    Float half = 0.5;
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    // paralleling this requires care with respect to race conditions
    // on CPU, parallelize over dimension not parity
    Float eta = 1.0;

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      int x_cb = 0;
      for (coord[3]=0; coord[3]<x_size[3]; coord[3]++) {
        if(dir == 3) eta *= -1.0;
	for (coord[2]=0; coord[2]<x_size[2]; coord[2]++) {
          if(dir >= 2) eta *= -1.0;
	  for (coord[1]=0; coord[1]<x_size[1]; coord[1]++) {
            if(dir >= 1) eta *= -1.0;
	    for (coord[0]=0; coord[0]<x_size[0]/2; coord[0]++) {

	      int oddBit = (parity + coord[1] + coord[2] + coord[3])&1;
	      coord[0] = 2*coord[0] + oddBit;
	      for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	      //Check to see if we are on the edge of a block, i.e.
	      //if this color matrix connects adjacent blocks.  If
	      //adjacent site is in same block, M = X, else M = Y
	      bool isDiagonal = (((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) || (((coord[dir]+3)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) ? true : false;

	      coarseGauge &M =  isDiagonal ? X : Y;
	      const int dim_index = isDiagonal ? 0 : dir;

	      int coarse_parity = 0;
	      for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	      coarse_parity &= 1;
	      coord_coarse[0] /= 2;
	      int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];
	      
	      //printf("(%d,%d)\n", coarse_x_cb, coarse_parity);

	      coord[0] /= 2;

              int coarse_spin_row = parity == 0 ? 0 : 1  ;
              int coarse_spin_col = (1 - coarse_spin_row); 

              half *= eta; //multiply by sing factor 

              for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
		for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column
		  for(int ic = 0; ic < nfinecolors; ic++) { //Sum over fine color
		      M(dir,coarse_parity,coarse_x_cb,coarse_spin_row, coarse_spin_col,ic_c,jc_c) += half*conj(V(parity, x_cb, 0, ic, ic_c)) * UV(parity, x_cb, 0, ic, jc_c);
		  } //Fine color
		} //Coarse Color column
	      } //Coarse Color row
	      x_cb++;
	    } // coord[0]
	  } // coord[1]
	} // coord[2]
      } // coord[3]
    } // parity
    
    return;
  }

 //Calculates the coarse gauge field: separated from coarseSpin = 2 computations:
  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void calculateKSY(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge *FL, fineGauge *LL, const int *x_size, double k) {

    if (FL->Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    const int *xc_size = Y.Field().X();
    int geo_bs[QUDA_MAX_DIM]; 
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];

    for(int d = 0; d < nDim; d++) 
    {
      //First calculate UV
      setZero<Float,F>(UV);

      printfQuda("Computing %d UV and VUV\n", d);
      //Calculate UV and then VUV for this direction, accumulating directly into the coarse gauge field Y
      if (d==0) {
        computeKSUV<Float,0>(UV, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,0>(Y, X, UV, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else if (d==1) {
        computeKSUV<Float,1>(UV, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,1>(Y, X, UV, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else if (d==2) {
        computeKSUV<Float,2>(UV, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,2>(Y, X, UV, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else {
        computeKSUV<Float,3>(UV, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,3>(Y, X, UV, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      }

      printf("UV2[%d] = %e\n", d, UV.norm2());
      printf("Y2[%d] = %e\n", d, Y.norm2(d));
    }

    printf("X2 = %e\n", X.norm2(0));
    printfQuda("Computing coarse diagonal\n");
    createKSCoarseLocal<Float>(X, nDim, xc_size, k);

    printf("X2 = %e\n", X.norm2(0));

  }



  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int coarseColor, int coarseSpin>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {

    const int fineSpin = 1;

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseSpin*coarseColor,1,gOrder> gCoarse;

    F vAccessor(const_cast<ColorSpinorField&>(T.Vectors()));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine fAccessor(const_cast<GaugeField&>(*f));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));

    if(l != NULL) {
      gFine lAccessor(const_cast<GaugeField&>(*l));
      calculateKSY<Float>(yAccessor, xAccessor, uvAccessor, vAccessor, &fAccessor, &lAccessor, f->X(), k);
    }
    else {
      gFine *lAccessor = NULL;
      calculateKSY<Float>(yAccessor, xAccessor, uvAccessor, vAccessor, &fAccessor, lAccessor, f->X(), k);
    }    
  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {

    if ((T.Vectors().Nspin() != 1) && (T.Vectors().Nspin()/T.Spin_bs() != 2))  errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateKSY<Float,csOrder,gOrder,fineColor,2, coarseSpin>(Y, X, uv, T, f, l, k);
    } else if (coarseColor == 24) {
      calculateKSY<Float,csOrder,gOrder,fineColor,24, coarseSpin>(Y, X, uv, T, f, l, k);
    } else if (coarseColor == 48) {
      calculateKSY<Float,csOrder,gOrder,fineColor,48, coarseSpin>(Y, X, uv, T, f, l, k);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }


  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (f->Ncolor() == 3) {
      if( !l ) if( f->Ncolor() != l->Ncolor() ) errorQuda("Unsupported number of colors %d\n", l->Ncolor());
      calculateKSY<Float,csOrder,gOrder, 3>(Y, X, uv, T, f, l, k);
    } else {
      errorQuda("Unsupported number of colors %d\n", f->Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (f->FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      if( !l ) if( l->FieldOrder() != QUDA_QDP_GAUGE_ORDER ) errorQuda("Unsupported field order for long links %d\n", l->FieldOrder());
      calculateKSY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, uv, T, f, l, k);
    } else {
      errorQuda("Unsupported field order %d\n", f->FieldOrder());
    }
  }

 template <typename Float>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateKSY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, uv, T, f, l, k);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != f->Precision())
    {
      errorQuda("Unsupported precision mix");
      if(l != NULL) if(Y.Precision() != l->Precision()) errorQuda("Unsupported precision mix for long links.");
    }

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateKSY<double>(Y, X, uv, T, f, l, k);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateKSY<float>(Y, X, uv, T, f, l, k);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if(l)
     printfQuda("....done computing Y field for improved staggered operator\n");  
    else 
     printfQuda("....done computing Y field for naive staggered operator\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseKSOp(const Transfer &T, GaugeField &Y, GaugeField &X, const cudaGaugeField *fat_links, const cudaGaugeField *long_links,  double k) {
    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

    int pad = 0;
    GaugeFieldParam fat_param(fat_links->X(), precision, fat_links->Reconstruct(), pad, fat_links->Geometry());
    fat_param.order = QUDA_QDP_GAUGE_ORDER;
    fat_param.fixed = fat_links->GaugeFixed();
    fat_param.link_type = fat_links->LinkType();
    fat_param.t_boundary = fat_links->TBoundary();
    fat_param.anisotropy = fat_links->Anisotropy();
    fat_param.gauge = NULL;
    fat_param.create = QUDA_NULL_FIELD_CREATE;
    fat_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    cpuGaugeField *f = new cpuGaugeField(fat_param);
    cpuGaugeField *l = NULL;

    //Copy the cuda gauge field to the cpu
    fat_links->saveCPUField(*f, QUDA_CPU_FIELD_LOCATION);

    if(long_links)
    {
      GaugeFieldParam long_param(long_links->X(), precision, long_links->Reconstruct(), pad, long_links->Geometry());
      long_param.order = QUDA_QDP_GAUGE_ORDER;
      long_param.fixed = long_links->GaugeFixed();
      long_param.link_type = long_links->LinkType();
      long_param.t_boundary = long_links->TBoundary();
      long_param.anisotropy = long_links->Anisotropy();
      long_param.gauge = NULL;
      long_param.create = QUDA_NULL_FIELD_CREATE;
      long_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      //Copy the cuda gauge field to the cpu
      long_links->saveCPUField(*l, QUDA_CPU_FIELD_LOCATION);
    }



    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField uv(UVparam);

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField
    calculateKSY(Y, X, uv, T, f, l, k);

    delete f;
    if(long_links) delete l;

  }



  //Apply the coarse KS Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float, int nDim, typename F, typename Gauge>
  void coarseKSDslash(F &out, F &in, Gauge &Y, Gauge &X, Float k) {
    const int Nc = in.Ncolor();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) x_size[d] = in.X(d);

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      for(int x_cb = 0; x_cb < in.Volume()/2; x_cb++) { //Volume
	int coord[QUDA_MAX_DIM];
	in.LatticeIndex(coord,parity*in.Volume()/2+x_cb);

	for(int c = 0; c < Nc; c++) out(parity, x_cb, 0, c) = (Float)0.0; 

	for(int d = 0; d < nDim; d++) { //Ndim
	  //Forward link - compute fwd offset for spinor fetch
	  int coordTmp = coord[d];
	  coord[d] = (coord[d] + 1)%x_size[d];
	  int fwd_idx = 0;
	  for(int dim = nDim-1; dim >= 0; dim--) fwd_idx = x_size[dim] * fwd_idx + coord[dim];
	  coord[d] = coordTmp;

          for(int c_row = 0; c_row < Nc; c_row++) { //Color row
  	    for(int c_col = 0; c_col < Nc; c_col++) { //Color column
	        out(parity, x_cb, 0, c_row) += Y(d, parity, x_cb, 0, 1, c_row, c_col) * in((parity+1)&1, fwd_idx/2, 1, c_col);
	        out(parity, x_cb, 1, c_row) += Y(d, parity, x_cb, 1, 0, c_row, c_col) * in((parity+1)&1, fwd_idx/2, 0, c_col);
	    } //Color column
	  } //Color row

	  //Backward link - compute back offset for spinor and gauge fetch
	  int back_idx = 0;
	  coord[d] = (coordTmp - 1 + x_size[d])%x_size[d];
	  for(int dim = nDim-1; dim >= 0; dim--) back_idx = x_size[dim] * back_idx + coord[dim];
	  coord[d] = coordTmp;

          for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	     for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		  out(parity, x_cb, 0, c_row) += - conj(Y(d,(parity+1)&1, back_idx/2, 0, 1, c_col, c_row))* in((parity+1)&1, back_idx/2, 1, c_col);//(Remark: note the minus sign.)
		  out(parity, x_cb, 1, c_row) += - conj(Y(d,(parity+1)&1, back_idx/2, 1, 0, c_col, c_row))* in((parity+1)&1, back_idx/2, 0, c_col);//(Remark: note the minus sign.)
	     } //Color column
	  } //Color row
	} //nDim

	// apply mass term
	for (int c=0; c<Nc; c++) out(parity, x_cb, 0, c) *= -(Float)2.0*k;

	// apply clover term
        for(int c = 0; c < Nc; c++) { //Color out
           for(int c_col = 0; c_col < Nc; c_col++) { //Color in
                out(parity,x_cb,0,c) += X(0, parity, x_cb, 0, 0, c, c_col)*in(parity,x_cb,0,c_col);
                out(parity,x_cb,1,c) += X(0, parity, x_cb, 1, 0, c, c_col)*in(parity,x_cb,0,c_col);
                out(parity,x_cb,0,c) += X(0, parity, x_cb, 0, 1, c, c_col)*in(parity,x_cb,1,c_col);
                out(parity,x_cb,1,c) += X(0, parity, x_cb, 1, 1, c, c_col)*in(parity,x_cb,1,c_col);
	   } //Color in
        } //Color out
      }//VolumeCB
    } // parity

    return;    
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

 //Apply the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float, int nDim, typename F, typename Gauge>
  void coarseDslash(F &out, F &in, Gauge &Y, Gauge &X, Float kappa) {
    const int Nc = in.Ncolor();
    const int Ns = in.Nspin();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) x_size[d] = in.X(d);

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      for(int x_cb = 0; x_cb < in.Volume()/2; x_cb++) { //Volume
	int coord[QUDA_MAX_DIM];
	in.LatticeIndex(coord,parity*in.Volume()/2+x_cb);

	for(int s = 0; s < Ns; s++) for(int c = 0; c < Nc; c++) out(parity, x_cb, s, c) = (Float)0.0; 

	for(int d = 0; d < nDim; d++) { //Ndim
	  //Forward link - compute fwd offset for spinor fetch
	  int coordTmp = coord[d];
	  coord[d] = (coord[d] + 1)%x_size[d];
	  int fwd_idx = 0;
	  for(int dim = nDim-1; dim >= 0; dim--) fwd_idx = x_size[dim] * fwd_idx + coord[dim];
	  coord[d] = coordTmp;

	  for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	    for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	      for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
		Float sign = (s_row == s_col) ? 1.0 : -1.0;    
		for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		  out(parity, x_cb, s_row, c_row) += sign*Y(d, parity, x_cb, s_row, s_col, c_row, c_col)
		    * in((parity+1)&1, fwd_idx/2, s_col, c_col);
		} //Color column
	      } //Spin column
	    } //Color row
	  } //Spin row 

	  //Backward link - compute back offset for spinor and gauge fetch
	  int back_idx = 0;
	  coord[d] = (coordTmp - 1 + x_size[d])%x_size[d];
	  for(int dim = nDim-1; dim >= 0; dim--) back_idx = x_size[dim] * back_idx + coord[dim];
	  coord[d] = coordTmp;

	  for(int s_row = 0; s_row < Ns; s_row++) { //Spin row
	    for(int c_row = 0; c_row < Nc; c_row++) { //Color row
	      for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
		for(int c_col = 0; c_col < Nc; c_col++) { //Color column
		  out(parity, x_cb, s_row, c_row) += conj(Y(d,(parity+1)&1, back_idx/2, s_col, s_row, c_col, c_row))
		    * in((parity+1)&1, back_idx/2, s_col, c_col);
		} //Color column
	      } //Spin column
	    } //Color row
	  } //Spin row 
	} //nDim

	// apply kappa
	for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out(parity, x_cb, s, c) *= -(Float)2.0*kappa;

	// apply clover term
	for(int s = 0; s < Ns; s++) { //Spin out
	  for(int c = 0; c < Nc; c++) { //Color out
	    //This term is now incorporated into the matrix X.
	    //out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color in
	        //Factor of 2*kappa now incorporated in X
		//out(parity,x_cb,s,c) -= 2*kappa*X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
                out(parity,x_cb,s,c) += X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
	      } //Color in
	    } //Spin in
	  } //Color out
	} //Spin out

      }//VolumeCB
    } // parity
    
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;
    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessor(const_cast<ColorSpinorField&>(in));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    if(coarseSpin  == 2)
      coarseDslash<Float,4,F,G>(outAccessor, inAccessor, yAccessor, xAccessor, (Float)kappa);
    else if(coarseSpin  == 1)
      coarseKSDslash<Float,4,F,G>(outAccessor, inAccessor, yAccessor, xAccessor, (Float)kappa);
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {

    if (in.Ncolor() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2,coarseSpin>(out, in, Y, X, kappa);
    } else if (in.Ncolor() == 24) { 
      ApplyCoarse<Float,csOrder,gOrder,24,coarseSpin>(out, in, Y, X, kappa);
    } else if (in.Ncolor() == 48) { 
      ApplyCoarse<Float,csOrder,gOrder,48,coarseSpin>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }


  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (in.Nspin() > 2) errorQuda("Unsupported number of coarse spins %d\n",in.Nspin());

    if (in.Nspin() == 2) { 
      ApplyCoarse<Float,csOrder,gOrder,2>(out, in, Y, X, kappa);
    } else if (in.Ncolor() == 1) { 
      ApplyCoarse<Float,csOrder,gOrder,1>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", in.Nspin());
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
  //out(x) = M*in = X*in - 2*kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  //Note factor of 2*kappa compensates for the factor of 1/2 already
  //absorbed into the Y matrices.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, const GaugeField &X, double kappa) {
    if (Y.Precision() != in.Precision() || X.Precision() != Y.Precision() || Y.Precision() != out.Precision())
      errorQuda("Unsupported precision mix");

    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (out.Precision() != in.Precision() ||
	Y.Precision() != in.Precision() ||
	X.Precision() != in.Precision()) 
      errorQuda("Precision mismatch out=%d in=%d Y=%d X=%d", 
		out.Precision(), in.Precision(), Y.Precision(), X.Precision());

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, in, Y, X, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
  }//ApplyCoarse

} //namespace quda
