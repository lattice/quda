#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
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


  template<typename Float, typename F, typename fineGauge>
  void computeUVcoarse(F &UV, const F &V, const fineGauge &G, int dir, int ndim, const int *x_size) {
        
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
      for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin row
        for(int s_col; s_col < V.Nspin(); s_col++) { //Fine spin column
          for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
            for(int ic = 0; ic < G.Ncolor(); ic++) { //Fine Color rows of gauge field
              for(int jc = 0; jc < G.Ncolor(); jc++) {  //Fine Color columns of gauge field
                UV(i, s, ic, ic_c) += G(dir, parity, gauge_index/2, ic, jc, s, s_col) * V(i_V, s_col, jc, ic_c);
              }  //Fine color columns
            }  //Fine color rows
          }  //Coarse color
        } //Fine spin column
      }  //Fine Spin row
    }  //Volume
  }  //UV



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

  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void computeVUVcoarse(int dir, coarseGauge &Y, coarseGauge &X, const F &UV, const F &V, int ndim, const fineGauge &G, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs) {

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

      //Check to see if we are on the edge of a block, i.e.
      //if this color matrix connects adjacent blocks.
      coarseGauge *M;
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

      for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin row

        int s_c_row = s/spin_bs;

	for(int s_col = 0; s_col < V.Nspin(); s_col++) { //Fine spin column
          int s_c_col = s_col/spin_bs;
	
          for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
            for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column

              for(int ic = 0; ic < G.Ncolor(); ic++) { //Sum over fine color
                (*M)(dim_index,coarse_parity,coarse_index/2,s_c_row,s_c_col,ic_c,jc_c) += conj(V(i, s, ic, ic_c)) * UV(i, s_col, ic, jc_c);
              } //Fine color
            } //Coarse Color column
          } //Coarse Color row
	}  //Fine spin row
      } //Fine spin column

    } //Volume

  }


  template<typename Float, int dir, typename F, typename coarseGauge, typename fineGauge, typename Gamma>
  void computeVUV(coarseGauge &Y, coarseGauge &X, const F &UV, const F &V, 
		  const Gamma &gamma, const fineGauge &G, const int *x_size, 
		  const int *xc_size, const int *geo_bs, int spin_bs) {

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

  //Adds the reverse links to the coarse diagonal term, which is just
  //the conjugate of the existing coarse diagonal term but with
  //plus/minus signs for off-diagonal spin components
  template<typename Float, typename Gauge>
  void coarseDiagonal(Gauge &X, int ndim, const int *xc_size) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
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
		  sign*X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)+conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]);
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

  //Computes the coarse clover term, given the fine clover term.
  template<typename Float, typename coarseGauge, typename F, typename fineGauge>
  void createCoarseClover(coarseGauge &X, F &V, fineGauge &C, int ndim, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs)  {
    for(int i = 0; i < V.Volume(); i++) {  //Loop over entire fine lattice volume i.e. both parities

      //FIXME: Uses LatticeIndex() for the color spinor field to determine gauge field index.
      //This only works if sites are ordered same way in both G and V.

      int coord[QUDA_MAX_DIM];
      int coord_coarse[QUDA_MAX_DIM];
      int coarse_size = 1;
      int coarse_parity = 0;
      V.LatticeIndex(coord, i);
      for(int d = ndim-1; d >=0; d--) {
        coord_coarse[d] = coord[d]/geo_bs[d];
	coarse_size *= xc_size[d];
      }
      int parity = 0;
      int gauge_index = gauge_offset_index(coord, x_size, ndim, parity);
      int coarse_index = gauge_offset_index(coord_coarse, xc_size, ndim, coarse_parity);

      for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin row
        int s_c_row = s/spin_bs;
        for(int s_col; s_col < V.Nspin(); s_col++) { //Fine spin column
	  int s_c_col = s_col/spin_bs;
          for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color row
	    for(int jc_c = 0; jc_c < V.Nvec(); jc_c++) { //Coarse Color col
              for(int ic = 0; ic < C.Ncolor(); ic++) { //Fine Color rows of gauge field
                for(int jc = 0; jc < C.Ncolor(); jc++) {  //Fine Color columns of gauge field
	        X(0,coarse_parity, coarse_index/2, s_c_row, s_c_col, ic_c, jc_c) += conj(V(i,s, ic, ic_c))*C(0,parity, gauge_index/2,ic,jc)*V(i,s_col,jc,jc_c);

                }  //Fine color columns
              }  //Fine color rows
            }  //Coarse color col
          }  //Coarse color row
        } //Fine spin column
      }  //Fine Spin row
    }  //Volume
    
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void calculateY(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge &G, const int *x_size) {
    if (UV.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) errorQuda("Gamma basis not supported");
    const QudaGammaBasis basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    const int *xc_size = Y.Field().X();
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
    coarseDiagonal<Float>(X, nDim, xc_size);
    printf("X2 = %e\n", X.norm2(0));

#if 0
      for(int i = 0; i < Y.Volume(); i++) {
	for(int s = 0; s < Y.NspinCoarse(); s++) {
          for(int s_col = 0; s_col < Y.NspinCoarse(); s_col++) {
            for(int c = 0; c < Y.NcolorCoarse(); c++) {
              for(int c_col = 0; c_col < Y.NcolorCoarse(); c_col++) {
                printf("d=%d i=%d s=%d s_col=%d c=%d c_col=%d Y(2*d) = %e %e, Y(2*d+1) = %e %e\n",d,i,s,s_col,c,c_col,Y(2*d,i%2,i/2,s,s_col,c,c_col).real(),Y(2*d,i%2,i/2,s,s_col,c,c_col).imag(),Y(2*d+1,i%2,i/2,s,s_col,c,c_col).real(),Y(2*d+1,i%2,i/2,s,s_col,c,c_col).imag());
              }}}}}
    for(int i = 0; i < Y.Volume(); i++) {
      for(int s = 0; s < Y.NspinCoarse(); s++) {
	for(int s_col = 0; s_col < Y.NspinCoarse(); s_col++) {
	  for(int c = 0; c < Y.NcolorCoarse(); c++) {
	    for(int c_col = 0; c_col < Y.NcolorCoarse(); c_col++) {
	      printf("d=%d i=%d s=%d s_col=%d c=%d c_col=%d Y(2*d) = %e %e\n",nDim,i,s,s_col,c,c_col,
		     X(0,i%2,i/2,s,s_col,c,c_col).real(),
		     X(0,i%2,i/2,s,s_col,c,c_col).imag());
	    }
	  }
	}
      }
    }
#endif
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, 
	    int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T, GaugeField &g) {
    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
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
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() || 
	Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

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


  //out(x) = (1-2*kappa*X)*in(x), where X is the local color-spin matrix on the coarse grid.
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

 //Apply the coarse Dslash to a vector:
  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template<typename Float, int nDim, typename F, typename Gauge>
  void coarseDslash(F &out, F &in, Gauge &Y, Gauge &X, Float kappa) {
    int Nc = in.Ncolor();
    int Ns = in.Nspin();
    int x_size[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) x_size[d] = in.X(d);

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
	    out(parity,x_cb,s,c) += in(parity,x_cb,s,c);
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin in
	      for(int c_col = 0; c_col < Nc; c_col++) { //Color in
		out(parity,x_cb,s,c) -= 2*kappa*X(0, parity, x_cb, s, s_col, c, c_col)*in(parity,x_cb,s_col,c_col);
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
    coarseDslash<Float,4,F,G>(outAccessor, inAccessor, yAccessor, xAccessor, (Float)kappa);
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
    if (Y.Precision() != in.Precision() || X.Precision() != Y.Precision() || Y.Precision() != out.Precision())
      errorQuda("Unsupported precision mix");

    if (in.V() == out.V()) errorQuda("Aliasing pointers");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyCoarse<double>(out, in, Y, X, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, in, Y, X, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
  }//ApplyCoarse

} //namespace quda
