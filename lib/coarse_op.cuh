namespace quda {

  //Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
  //Where:
  //mu = dir
  //s = fine spin
  //c' = coarse color
  //c = fine color
  template<bool from_coarse, typename Float, int dim, typename F, typename fineGauge>
  void computeUV(F &UV, const F &V, const fineGauge &G, int ndim, const int *x_size, const int *comm_dim, int s_col=0) {

    int coord[5];
    coord[4] = 0;
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<V.VolumeCB(); x_cb++) {
	getCoords(coord, x_cb, x_size, parity);

	if ( comm_dim[dim] && (coord[dim] + 1 >= x_size[dim]) ) {
	  int nFace = 1;
	  int ghost_idx = ghostFaceIndex<1>(coord, x_size, dim, nFace);

	  for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin
	    for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
	      for(int ic = 0; ic < G.NcolorCoarse(); ic++) { //Fine Color rows of gauge field
		for(int jc = 0; jc < G.NcolorCoarse(); jc++) {  //Fine Color columns of gauge field
		  if (!from_coarse)
		    UV(parity, x_cb, s, ic, ic_c) +=
		      G(dim, parity, x_cb, ic, jc) * V.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c);
		  else
		    UV(parity, x_cb, s, ic, ic_c) += G(dim, parity, x_cb, s, s_col, ic, jc) *
		      V.Ghost(dim, 1, (parity+1)&1, ghost_idx, s_col, jc, ic_c);

		}  //Fine color columns
	      }  //Fine color rows
	    }  //Coarse color
	  }  //Fine Spin

	} else {
	  int y_cb = linkIndexP1(coord, x_size, dim);

	  for(int s = 0; s < V.Nspin(); s++) {  //Fine Spin
	    for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
	      for(int ic = 0; ic < G.NcolorCoarse(); ic++) { //Fine Color rows of gauge field
		for(int jc = 0; jc < G.NcolorCoarse(); jc++) {  //Fine Color columns of gauge field
		  if (!from_coarse)
		    UV(parity, x_cb, s, ic, ic_c) += G(dim, parity, x_cb, ic, jc) * V((parity+1)&1, y_cb, s, jc, ic_c);
		  else
		    UV(parity, x_cb, s, ic, ic_c) += G(dim, parity, x_cb, s, s_col, ic, jc) * V((parity+1)&1, y_cb, s_col, jc, ic_c);
		}  //Fine color columns
	      }  //Fine color rows
	    }  //Coarse color
	  }  //Fine Spin

	}

      } // c/b volume
    } // parity

  }  //UV


  template<bool from_coarse, typename Float, int dir, typename F, typename coarseGauge, typename fineGauge, typename Gamma>
  void computeVUV(coarseGauge &Y, coarseGauge &X, const F &UV, const F &V,
		  const Gamma &gamma, const fineGauge &G, const int *x_size,
		  const int *xc_size, const int *geo_bs, int spin_bs, int s_col=0) {

    const int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<UV.VolumeCB(); x_cb++) {
	getCoords(coord, x_cb, x_size, parity);
	for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	//Check to see if we are on the edge of a block, i.e.
	//if this color matrix connects adjacent blocks.  If
	//adjacent site is in same block, M = X, else M = Y
	const bool isDiagonal = ((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir] ? true : false;
	coarseGauge &M =  isDiagonal ? X : Y;
	const int dim_index = isDiagonal ? 0 : dir;

	int coarse_parity = 0;
	for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	coarse_parity &= 1;
	coord_coarse[0] /= 2;
	int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];

	coord[0] /= 2;

	for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin

	  if (!from_coarse) { // fine grid is top level

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
		    static_cast<Float>(0.5) * conj(V(parity, x_cb, s, ic, ic_c)) * UV(parity, x_cb, s, ic, jc_c);

		  //Off-diagonal Spin (backward link / positive projector applied)
		  M(dim_index,coarse_parity,coarse_x_cb,s_c_row,s_c_col,ic_c,jc_c) +=
		    static_cast<Float>(0.5) * coupling * conj(V(parity, x_cb, s, ic, ic_c)) * UV(parity, x_cb, s_col, ic, jc_c);
		} //Fine color
	      } //Coarse Color column
	    } //Coarse Color row

	  } else { // fine grid operator is a coarse operator

	    for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
	      for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column
		for(int ic = 0; ic < G.NcolorCoarse(); ic++) { //Sum over fine color
		  M(dim_index,coarse_parity,coarse_x_cb,s,s_col,ic_c,jc_c) +=
		    conj(V(parity, x_cb, s, ic, ic_c)) * UV(parity, x_cb, s, ic, jc_c);
		} //Fine color
	      } //Coarse Color column
	    } //Coarse Color row

	  }

	} //Fine spin
      } // c/b volume
    } // parity

  }


  //Adds the reverse links to the coarse local term, which is just
  //the conjugate of the existing coarse local term but with
  //plus/minus signs for off-diagonal spin components
  //Also multiply by the appropriate factor of -2*kappa
  template<typename Float, int nSpin, int nColor, typename Gauge>
  void createYreverse(Gauge &Y) {

    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
	for (int x_cb=0; x_cb<Y.VolumeCB(); x_cb++) {

	  for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	    for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column

	      const Float sign = (s_row == s_col) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

	      for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
		for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		  Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = sign*Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
		} //Color column
	      } //Color row
	    } //Spin column
	  } //Spin row

	} // x_cb
      } //parity
    } // dimension

  }


  //Adds the reverse links to the coarse local term, which is just
  //the conjugate of the existing coarse local term but with
  //plus/minus signs for off-diagonal spin components
  //Also multiply by the appropriate factor of -2*kappa
  template<typename Float, int nSpin, int nColor, typename Gauge>
  void createCoarseLocal(Gauge &X, double kappa) {
    Float kap = (Float) kappa;
    complex<Float> Xlocal[nSpin*nSpin*nColor*nColor];

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.VolumeCB(); x_cb++) {

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

	    const Float sign = (s_row == s_col) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

	    for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Transpose color part
		X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =
		  -2*kap*(sign*X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)
			  +conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	      } //Color column
	    } //Color row
	  } //Spin column
	} //Spin row

      } // x_cb
    } //parity

  }


  template<typename Float, int nDim, typename coarseGauge, typename F, typename clover>
  void createCoarseCloverFromFine(coarseGauge &X, F &V, clover &C, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs)  {

    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<C.VolumeCB(); x_cb++) {
	getCoords(coord, x_cb, x_size, parity);
	for (int d=0; d<nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	int coarse_parity = 0;
	for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	coarse_parity &= 1;
	coord_coarse[0] /= 2;
	int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];

	coord[0] /= 2;

	//If Nspin = 4, then the clover term has structure C_{\mu\nu} = \gamma_{\mu\nu}C^{\mu\nu}
	for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin row
	  int s_c = s/spin_bs;
	  //On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
	  //in the same chiral block.
	  for(int s_col = s_c*spin_bs; s_col < (s_c+1)*spin_bs; s_col++) { //Loop over fine spin column
	    for(int ic_c = 0; ic_c < X.NcolorCoarse(); ic_c++) { //Coarse Color row
	      for(int jc_c = 0; jc_c < X.NcolorCoarse(); jc_c++) { //Coarse Color column
		for(int ic = 0; ic < C.Ncolor(); ic++) { //Sum over fine color row
		  for(int jc = 0; jc < C.Ncolor(); jc++) {  //Sum over fine color column
		    X(0,coarse_parity,coarse_x_cb,s_c,s_c,ic_c,jc_c) +=
		      conj(V(parity, x_cb, s, ic, ic_c)) * C(parity, x_cb, s, s_col, ic, jc) * V(parity, x_cb, s_col, jc, jc_c);
		  } //Fine color column
		}  //Fine color row
	      } //Coarse Color column
	    } //Coarse Color row
	  }  //Fine spin column
	} //Fine spin

      } // c/b volume
    } // parity
  }

  template<typename Float, int nDim, typename coarseGauge, typename F, typename fineGauge>
  void createCoarseCloverFromCoarse(coarseGauge &X, F &V, fineGauge &C, const int *x_size, const int *xc_size, const int *geo_bs, int spin_bs)  {

    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<C.VolumeCB(); x_cb++) {
	getCoords(coord, x_cb, x_size, parity);
	for (int d=0; d<nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	int coarse_parity = 0;
	for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	coarse_parity &= 1;
	coord_coarse[0] /= 2;
	int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];

	coord[0] /= 2;

	//If Nspin != 4, then spin structure is a dense matrix
	//N.B. assumes that no further spin blocking is done in this case.
	for(int s = 0; s < V.Nspin(); s++) { //Loop over fine spin row
	  for(int s_col = 0; s_col < V.Nspin(); s_col++) { //Loop over fine spin column
	    for(int ic_c = 0; ic_c < X.NcolorCoarse(); ic_c++) { //Coarse Color row
	      for(int jc_c = 0; jc_c < X.NcolorCoarse(); jc_c++) { //Coarse Color column
		for(int ic = 0; ic < C.NcolorCoarse(); ic++) { //Sum over fine color row
		  for(int jc = 0; jc < C.NcolorCoarse(); jc++) {  //Sum over fine color column
		    X(0,coarse_parity,coarse_x_cb,s,s_col,ic_c,jc_c) +=
		      conj(V(parity, x_cb, s, ic, ic_c)) * C(0, parity, x_cb, s, s_col, ic, jc) * V(parity, x_cb, s_col, jc, jc_c);
		  } //Fine color column
		}  //Fine color row
	      } //Coarse Color column
	    } //Coarse Color row
	  }  //Fine spin column
	} //Fine spin

      } // c/b volume
    } // parity

  }

  //Zero out a field, using the accessor.
  template<typename Float, typename F>
  void setZero(F &f) {
    for(int parity = 0; parity < 2; parity++) {
      for(int x_cb = 0; x_cb < f.VolumeCB(); x_cb++) {
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

  //Adds the identity matrix to the coarse local term.
  template<typename Float, typename Gauge>
  void addCoarseDiagonal(Gauge &X) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.VolumeCB(); x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
         for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
            X(0,parity,x_cb,s,s,ic_c,ic_c) += 1.0;
         } //Color
        } //Spin
      } // x_cb
    } //parity
   }



} // namespace quda
