namespace quda {

  template <typename Float, typename coarseGauge, typename fineGauge, typename fineSpinor,
	    typename fineSpinorTmp, typename fineClover>
  struct CalculateYArg {

    coarseGauge Y;        /** Computed coarse link field */
    coarseGauge X;        /** Computed coarse clover field */

    fineSpinorTmp UV;        /** Temporary that stores the fine-link * spinor field product */

    const fineGauge U;    /** Fine grid link field */
    const fineSpinor V;   /** Fine grid spinor field */
    const fineClover C;   /** Fine grid clover field */

    int x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float kappa;                /** kappa value */

    CalculateYArg(coarseGauge &Y, coarseGauge &X, fineSpinorTmp &UV, const fineGauge &U, const fineSpinor &V,
		  const fineClover &C, double kappa, const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_)
      : Y(Y), X(X), UV(UV), U(U), V(V), C(C), kappa(static_cast<Float>(kappa)), spin_bs(spin_bs_)
    {
      if (V.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
	errorQuda("Gamma basis %d not supported", V.GammaBasis());

      for (int i=0; i<QUDA_MAX_DIM; i++) {
	x_size[i] = x_size_[i];
	xc_size[i] = xc_size_[i];
	geo_bs[i] = geo_bs_[i];
	comm_dim[i] = comm_dim_partitioned(i);
      }
    }
  };

  /**
     Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
     Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
  */
  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void computeUV(Arg &arg, int parity, int x_cb) {

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);

    for(int s = 0; s < fineSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	for(int v = 0; v < coarseColor; v++) {
	  arg.UV(parity,x_cb,s,c,v) = static_cast<Float>(0.0);
	}
      }
    }

    if ( arg.comm_dim[dim] && (coord[dim] + 1 >= arg.x_size[dim]) ) {
      int nFace = 1;
      int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	  for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	    for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	      if (!from_coarse)
		arg.UV(parity, x_cb, s, ic, ic_c) +=
		  arg.U(dim, parity, x_cb, ic, jc) * arg.V.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c);
	      else
		for (int s_col=0; s_col<fineSpin; s_col++) {
		  arg.UV(parity, x_cb, s_col*2+s, ic, ic_c) += arg.U(dim, parity, x_cb, s, s_col, ic, jc) *
		    arg.V.Ghost(dim, 1, (parity+1)&1, ghost_idx, s_col, jc, ic_c);
		} // which chiral block
	    }  //Fine color columns
	  }  //Fine color rows
	}  //Coarse color
      }  //Fine Spin

    } else {
      int y_cb = linkIndexP1(coord, arg.x_size, dim);

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	  for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	    for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	      if (!from_coarse)
		arg.UV(parity, x_cb, s, ic, ic_c) += arg.U(dim, parity, x_cb, ic, jc) * arg.V((parity+1)&1, y_cb, s, jc, ic_c);
	      else
		for (int s_col=0; s_col<fineSpin; s_col++) {
		  arg.UV(parity, x_cb, s_col*2+s, ic, ic_c) += arg.U(dim, parity, x_cb, s, s_col, ic, jc) * arg.V((parity+1)&1, y_cb, s_col, jc, ic_c);
		} // which chiral block
	    }  //Fine color columns
	  }  //Fine color rows
	}  //Coarse color
      }  //Fine Spin

    }

  } // computeUV

  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeUVCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.V.VolumeCB(); x_cb++) {
	computeUV<from_coarse,Float,dim,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    }   // parity
  }

  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.V.VolumeCB()) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeUV<from_coarse,Float,dim,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
  }

  /**
     Do a single V^\dagger * UV product
   */
  template <bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void multiplyVUV(complex<Float> vuv[], Arg &arg, int parity, int x_cb) {

    Gamma<Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim> gamma;

    constexpr Float half = static_cast<Float>(0.5);

    if (!from_coarse) { // fine grid is top level

      for(int s = 0; s < fineSpin; s++) { //Loop over fine spin

	//Spin part of the color matrix.  Will always consist
	//of two terms - diagonal and off-diagonal part of
	//P_mu = (1+\gamma_mu)

	int s_c_row = s/arg.spin_bs; //Coarse spin row index

	//Use Gamma to calculate off-diagonal coupling and
	//column index.  Diagonal coupling is always 1.
	int s_col;
	complex<Float> coupling = gamma.getrowelem(s, s_col);
	int s_c_col = s_col/arg.spin_bs;

	for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	  for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	    for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
	      //Diagonal Spin
	      vuv[((s_c_row*coarseSpin+s_c_row)*coarseColor+ic_c)*coarseColor+jc_c] +=
		half * conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s, ic, jc_c);

	      //Off-diagonal Spin (backward link / positive projector applied)
	      vuv[((s_c_row*coarseSpin+s_c_col)*coarseColor+ic_c)*coarseColor+jc_c] +=
		half * coupling * conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s_col, ic, jc_c);
	    } //Fine color
	  } //Coarse Color column
	} //Coarse Color row
      }

    } else { // fine grid operator is a coarse operator

      for (int s_col=0; s_col<fineSpin; s_col++) { // which chiral block
	for (int s = 0; s < fineSpin; s++) {
	  for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
		vuv[((s*coarseSpin+s_col)*coarseColor+ic_c)*coarseColor+jc_c] +=
		  conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, 2*s_col+s, ic, jc_c);
	      } //Fine color
	    } //Coarse Color column
	  } //Coarse Color row
	} //Fine spin
      }

    } // from_coarse

  }

  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void computeVUV(Arg &arg, int parity, int x_cb) {

    const int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= arg.xc_size[d];

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block, i.e.
    //if this color matrix connects adjacent blocks.  If
    //adjacent site is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;
    auto &M =  isDiagonal ? arg.X : arg.Y;
    const int dim_index = isDiagonal ? 0 : dim;

    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

    coord[0] /= 2;

    complex<Float> vuv[coarseSpin*coarseSpin*coarseColor*coarseColor];
    multiplyVUV<from_coarse, Float, dim, fineSpin, fineColor, coarseSpin, coarseColor, Arg>(vuv, arg, parity, x_cb);

    if (!from_coarse) {

      for (int s_row = 0; s_row < coarseSpin; s_row++) {
	for (int s_col = 0; s_col < coarseSpin; s_col++) {
	  for(int c_row = 0; c_row < coarseColor; c_row++) { //Coarse Color row
	    for(int c_col = 0; c_col < coarseColor; c_col++) { //Coarse Color column
	      //Diagonal Spin
	      M(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += vuv[((s_row*coarseSpin+s_col)*coarseColor+c_row)*coarseColor+c_col];
	    } //Coarse Color column
	  } //Coarse Color row
	}
      }
    } else { // fine grid operator is a coarse operator

      for (int s = 0; s < coarseSpin; s++) {
	for (int s_col; s < coarseSpin; s_col++) { // chiral block
	  for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	      M(dim_index,coarse_parity,coarse_x_cb,s,s_col,ic_c,jc_c) += vuv[s,s_col,ic_c,jc_c];
	    } //Coarse Color column
	  } //Coarse Color row
	} // chiral block
      }

    } // from_coarse

  }

  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeVUVCPU(Arg arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.UV.VolumeCB(); x_cb++) {
	computeVUV<from_coarse,Float,dim,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<bool from_coarse, typename Float, int dim, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeVUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.V.VolumeCB()) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeVUV<from_coarse,Float,dim,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
  }

  //Adds the reverse links to the coarse local term, which is just
  //the conjugate of the existing coarse local term but with
  //plus/minus signs for off-diagonal spin components
  //Also multiply by the appropriate factor of -2*kappa
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeYreverse(Arg &arg, int parity, int x_cb) {
    auto &Y = arg.Y;

    for (int d=0; d<4; d++) {
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

    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.UV.VolumeCB(); x_cb++) {
	computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.V.VolumeCB()) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
  }

  /**
   * Adds the reverse links to the coarse local term, which is just
   * the conjugate of the existing coarse local term but with
   * plus/minus signs for off-diagonal spin components so multiply by
   * the appropriate factor of -2*kappa
  */
  template<typename Float, int nSpin, int nColor, typename Arg>
  void computeCoarseLocal(Arg &arg, int parity, int x_cb)
  {
    auto X = arg.X;
    Float kap = arg.kappa;
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

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeCoarseLocalCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.UV.VolumeCB(); x_cb++) {
	computeCoarseLocal<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeCoarseLocalGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.V.VolumeCB()) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeCoarseLocal<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
  }


  template<bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void computeCoarseClover(Arg &arg, int parity, int x_cb) {

    const int nDim = 4;

    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= arg.xc_size[d];

    getCoords(coord, x_cb, arg.x_size, parity);
    for (int d=0; d<nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

    coord[0] /= 2;

    if (!from_coarse) {
      //If Nspin = 4, then the clover term has structure C_{\mu\nu} = \gamma_{\mu\nu}C^{\mu\nu}
      for(int s = 0; s < fineSpin; s++) { //Loop over fine spin row
	int s_c = s/arg.spin_bs;
	//On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
	//in the same chiral block.
	for(int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column
	  for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
		for(int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
		  arg.X(0,coarse_parity,coarse_x_cb,s_c,s_c,ic_c,jc_c) +=
		    conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
		} //Fine color column
	      }  //Fine color row
	    } //Coarse Color column
	  } //Coarse Color row
	}  //Fine spin column
      } //Fine spin
    } else {
      //If Nspin != 4, then spin structure is a dense matrix and there is now spin aggregation
      //N.B. assumes that no further spin blocking is done in this case.
      for(int s = 0; s < fineSpin; s++) { //Loop over spin row
	for(int s_col = 0; s_col < fineSpin; s_col++) { //Loop over spin column
	  for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c <coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
		for(int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
		  arg.X(0,coarse_parity,coarse_x_cb,s,s_col,ic_c,jc_c) +=
		    conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
		} //Fine color column
	      }  //Fine color row
	    } //Coarse Color column
	  } //Coarse Color row
	}  //Fine spin column
      } //Fine spin
    }

  }

  template <bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeCoarseCloverCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.C.VolumeCB(); x_cb++) {
	computeCoarseClover<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template <bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeCoarseCloverGPU(Arg &arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.C.VolumeCB()) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeCoarseClover<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg, parity, x_cb);
  }



  //Adds the identity matrix to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseDiagonalCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.X.VolumeCB(); x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
         for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
	   arg.X(0,parity,x_cb,s,s,ic_c,ic_c) += static_cast<Float>(1.0);
         } //Color
        } //Spin
      } // x_cb
    } //parity
   }


  //Adds the identity matrix to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void AddCoarseDiagonalGPU(Arg &arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.X.VolumeCB()) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    for(int s = 0; s < nSpin; s++) { //Spin
      for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
	arg.X(0,parity,x_cb,s,s,ic_c,ic_c) += static_cast<Float>(1.0);
      } //Color
    } //Spin
   }


  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_VUV,
    COMPUTE_REVERSE_Y,
    COMPUTE_COARSE_LOCAL,
    COMPUTE_COARSE_CLOVER,
    COMPUTE_DIAGONAL
  };

  template <bool from_coarse, typename Float, int fineSpin,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateY : public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    int dim;
    ComputeType type;

    long long flops() const
    {
      // 2 from parity, 8 from complex
      return !from_coarse ? 2*arg.V.VolumeCB()*8*arg.V.Nspin()*arg.V.Nvec()*arg.U.Ncolor() :
	2*arg.V.VolumeCB()*8*arg.V.Nspin()*arg.V.Nvec()*arg.U.NcolorCoarse();
    }
    long long bytes() const
    {
      return arg.UV.Bytes() + arg.V.Bytes() + 2*arg.U.Bytes();
    }
    unsigned int minThreads() const { return arg.V.VolumeCB(); }

  public:
    CalculateY(Arg &arg, ComputeType type, const ColorSpinorField &meta)
      : TunableVectorY(2), arg(arg), type(type), meta(meta)
    {
      strcpy(aux, meta.AuxString());
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (arg.comm_dim[0] ? '1' : '0');
      comm[1] = (arg.comm_dim[1] ? '1' : '0');
      comm[2] = (arg.comm_dim[2] ? '1' : '0');
      comm[3] = (arg.comm_dim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux,",comm=");
      strcat(aux,comm);
#endif
    }
    virtual ~CalculateY() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

	if (type == COMPUTE_UV) {

	  if      (dim==0) ComputeUVCPU<from_coarse,Float,0,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==1) ComputeUVCPU<from_coarse,Float,1,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==2) ComputeUVCPU<from_coarse,Float,2,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==3) ComputeUVCPU<from_coarse,Float,3,fineSpin,fineColor,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_VUV) {

	  if      (dim==0) ComputeVUVCPU<from_coarse,Float,0,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==1) ComputeVUVCPU<from_coarse,Float,1,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==2) ComputeVUVCPU<from_coarse,Float,2,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  else if (dim==3) ComputeVUVCPU<from_coarse,Float,3,fineSpin,fineColor,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_REVERSE_Y) {

	  ComputeYReverseCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_COARSE_LOCAL) {

	  ComputeCoarseLocalCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_COARSE_CLOVER) {

	  ComputeCoarseCloverCPU<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg);

	} else if (type == COMPUTE_DIAGONAL) {

	  AddCoarseDiagonalCPU<Float,coarseSpin,coarseColor>(arg);

	} else {
	  errorQuda("Undefined compute type %d", type);
	}
      } else {

      }
    }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which computation we are doing
     */
    void setComputeType(ComputeType type_) { type = type_; }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == COMPUTE_UV)            strcat(Aux,",computeUV");
      else if (type == COMPUTE_VUV)           strcat(Aux,",computeVUV");
      else if (type == COMPUTE_REVERSE_Y)     strcat(Aux,",computeYreverse");
      else if (type == COMPUTE_COARSE_LOCAL)  strcat(Aux,",computeCoarseLocal");
      else if (type == COMPUTE_COARSE_CLOVER) strcat(Aux,",computeCoarseClover");
      else if (type == COMPUTE_DIAGONAL)      strcat(Aux,",computeCoarseLocal");
      else errorQuda("Unknown type=%e\n", type);

      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
	if      (dim == 0) strcat(Aux,",dim=0");
	else if (dim == 1) strcat(Aux,",dim=1");
	else if (dim == 2) strcat(Aux,",dim=2");
	else if (dim == 3) strcat(Aux,",dim=3");
      }

      return TuneKey(meta.VolString(), typeid(*this).name(), Aux);
    }

  };


  template<typename Float, int n, typename Gauge>
  void createYpreconditioned(Gauge &Yhat, Gauge &Xinv, const Gauge &Y, const int *dim, int nFace, const int *commDim) {

    complex<Float> Ylocal[n*n];

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
	for (int x_cb=0; x_cb<Y.VolumeCB(); x_cb++) {

	  int coord[5];
	  getCoords(coord, x_cb, dim, parity);
	  coord[4] = 0;

	  const int ghost_idx = ghostFaceIndex<0>(coord, dim, d, nFace);

	  if ( commDim[d] && (coord[d] - nFace < 0) ) {

	    for(int i = 0; i<n; i++) {
	      for(int j = 0; j<n; j++) {
		Yhat.Ghost(d,1-parity,ghost_idx,i,j) = 0.0;
		for(int k = 0; k<n; k++) {
		  Yhat.Ghost(d,1-parity,ghost_idx,i,j) += Y.Ghost(d,1-parity,ghost_idx,i,k) * conj(Xinv(0,parity,x_cb,j,k));
		}

	      }
	    }

	  } else {
	    const int back_idx = linkIndexM1(coord, dim, d);

	    for(int i = 0; i<n; i++) {
	      for(int j = 0; j<n; j++) {
		Yhat(d,1-parity,back_idx,i,j) = 0.0;
		for(int k = 0; k<n; k++) {
		  Yhat(d,1-parity,back_idx,i,j) += Y(d,1-parity,back_idx,i,k) * conj(Xinv(0,parity,x_cb,j,k));
		}
	      }
	    }

	  }
	} // x_cb
      } //parity
    } // dimension

    // now do the forwards links X^{-1} * Y^{-\mu}
    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
	for (int x_cb=0; x_cb<Y.VolumeCB(); x_cb++) {

	  for(int i = 0; i<n; i++) {
	    for(int j = 0; j<n; j++) {
	      Yhat(d+4,parity,x_cb,i,j) = 0.0;
	      for(int k = 0; k<n; k++) {
		Yhat(d+4,parity,x_cb,i,j) += Xinv(0,parity,x_cb,i,k) * Y(d+4,parity,x_cb,k,j);
	      }
	    }
	  }

	} // x_cb
      } //parity
    } // dimension

  }



  //Calculates the coarse gauge field
  template<bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   QudaGaugeFieldOrder gOrder, typename F, typename Ftmp, typename coarseGauge, typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X, Ftmp &UV, F &V, fineGauge &G, fineClover &C,
		  GaugeField &Y_, GaugeField &X_, GaugeField &Xinv_, GaugeField &Yhat_, const ColorSpinorField &v, double kappa) {

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    int x_size[5];
    for (int i=0; i<4; i++) x_size[i] = v.X(i);
    x_size[4] = 1;

    int xc_size[5];
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int comm_dim[nDim];
    for (int i=0; i<nDim; i++) comm_dim[i] = comm_dim_partitioned(i);

    int geo_bs[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse();

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateYArg<Float,coarseGauge,fineGauge,F,Ftmp,fineClover> Arg;
    Arg arg(Y, X, UV, G, V, C, kappa, x_size, xc_size, geo_bs, spin_bs);
    CalculateY<from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> y(arg, COMPUTE_UV, v);

    for(int d = 0; d < nDim; d++) {
      printfQuda("Computing %d UV and VUV\n", d);

      y.setDimension(d);
      y.setComputeType(COMPUTE_UV);  // compute U*V product
      y.apply(0);

      printfQuda("UV2[%d] = %e\n", d, UV.norm2());

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);

      printfQuda("Y2[%d] = %e\n", d, Y.norm2(d));
    }

    y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the backwards direction
    y.apply(0);

    printfQuda("Computing coarse diagonal\n");
    y.setComputeType(COMPUTE_COARSE_LOCAL);
    y.apply(0);

    if (C.Volume() > 0) { // If C!=NULL we have to coarsen the fine clover term and add it in.
      printfQuda("Computing fine->coarse clover term\n");
      y.setComputeType(COMPUTE_COARSE_CLOVER);
      y.apply(0);
    } else {  //Otherwise, we just have to add the identity matrix
      y.setComputeType(COMPUTE_DIAGONAL);
      y.apply(0);

    }
    printfQuda("X2 = %e\n", X.norm2(0));


    {
      cpuGaugeField *X_h = static_cast<cpuGaugeField*>(&X_);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv_);

      // invert the clover matrix field
      const int n = X_h->Ncolor();
      BlasMagmaArgs magma(X_h->Precision());
      magma.BatchInvertMatrix(((float**)Xinv_h->Gauge_p())[0], ((float**)X_h->Gauge_p())[0], n, X_h->Volume());
    }

    // now exchange Y halos for multi-process dslash
    Y_.exchangeGhost();

    // compute the preconditioned links
    // Yhat_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
    // Yhat_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)

    {
      // use spin-ignorant accessor to make multiplication simpler
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,1,gOrder> gCoarse;
      gCoarse yAccessor(const_cast<GaugeField&>(Y_));
      gCoarse yHatAccessor(const_cast<GaugeField&>(Yhat_));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv_));
      int comm_dim[4];
      for (int i=0; i<4; i++) comm_dim[i] = comm_dim_partitioned(i);
      createYpreconditioned<Float,coarseSpin*coarseColor>(yHatAccessor, xInvAccessor, yAccessor, x_size, 1, comm_dim);
    }

    // fill back in the bulk of Yhat so that the backward link is updated on the previous node
    Yhat_.injectGhost();
  }



} // namespace quda
