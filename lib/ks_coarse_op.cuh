namespace quda {
  static bool bidirectional_debug = false;

  template <typename Float, typename coarseGauge, typename fineGauge, typename fineSpinor,
	    typename fineSpinorTmp, typename fineClover>
  struct CalculateKSYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */
    coarseGauge Xinv;        /** Computed coarse clover field */

    fineSpinorTmp *UV;        /** Temporary that stores the fine-link * spinor field product */
    fineSpinorTmp *UVL;        /** Temporary that stores the fine-link * spinor field product */

    const fineGauge *FL;       /** Fine grid link field */
    const fineGauge *LL;       /** Fine grid link field */
    const fineSpinor V;      /** Fine grid spinor field */
    const fineClover C;      /** Fine grid clover field (not top level) */
    const fineClover Cinv;   /** Fine grid clover field (not top level) */

    int x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float mass;                /** mass value */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    int nFace;

    CalculateKSYArg(coarseGauge &Y, coarseGauge &X, coarseGauge &Xinv, fineSpinorTmp *UV, fineSpinorTmp *UVL, const fineGauge *FL, const fineGauge *LL, const fineSpinor &V, const fineClover &C, const fineClover &Cinv,
		  double mass, const int *x_size_, const int *xc_size_, int *geo_bs_)
      : Y(Y), X(X), Xinv(Xinv), UV(UV), UVL(UVL), FL(FL), LL(LL),V(V), C(C), Cinv(Cinv), mass(static_cast<Float>(mass)),
	fineVolumeCB(V.VolumeCB()), coarseVolumeCB(X.VolumeCB()), nFace(3)
    {
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
  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void computeKSUV(Arg &arg, int parity, int x_cb) {

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);

    const int uvSpin = from_coarse ? fineSpin * 2 : 1;

    const int stag_sp = 0;

    for(int s = 0; s < uvSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	for(int v = 0; v < coarseColor; v++) {
	  (*arg.UV)(parity,x_cb,s,c,v) = static_cast<Float>(0.0);
          if( arg.UVL ) (*arg.UVL)(parity,x_cb,s,c,v) = static_cast<Float>(0.0);
	}
      }
    }

    if ( arg.comm_dim[dim] && (coord[dim] + 1 >= arg.x_size[dim]) ) {
#if 0
      int ghost_idx    = ghostFaceIndex<1>(coord, arg.x_size, dim, 1);//must be 1
      int ghost_idx_3d = (arg.LL && !from_coarse) ghostFaceIndex<3>(coord, arg.x_size, dim, arg.nFace) : 0;

      for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if(!from_coarse){
	      (*arg.UV)(parity, x_cb, stag_sp, ic, ic_c) +=
	      (*arg.FL)(dim, parity, x_cb, ic, jc) * arg.V.Ghost(dim, 1, (parity+1)&1, ghost_idx, jc, ic_c);
              if( arg.LL ) (*arg.UVL)(parity, x_cb, stag_sp, ic, ic_c) += (*arg.LL)(dim, parity, x_cb, ic, jc) * arg.V.Ghost(dim, 3, (parity+1)&1, ghost_idx_3d, stag_sp, jc, ic_c);
            } else {
              for(int s = 0; s < fineSpin; s++) {  //Fine Spin
                for (int s_col=0; s_col<fineSpin; s_col++) {
	            // on the coarse lattice if forwards then use the forwards links
		    (*arg.UV)(parity, x_cb, s_col*fineSpin+s, ic, ic_c) += (*arg.FL)(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, ic, jc) * arg.V.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c);
                } //which chiral block
              } //Fine Spin
            } // from coarse? 
	  }  //Fine color columns
	}  //Fine color rows
      }  //Coarse color
#endif
    } else {
      int y_cb = linkIndexP1(coord, arg.x_size, dim);
      int y3_cb = (arg.LL && !from_coarse) ? linkIndexP3(coord, arg.x_size, dim) : 0;

      for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if(!from_coarse){
	      (*arg.UV)(parity, x_cb, stag_sp, ic, ic_c) +=  (*arg.FL)(dim , parity, x_cb, ic, jc) *  arg.V((parity+1)&1, y_cb, stag_sp, jc, ic_c);
              if( arg.LL ) (*arg.UVL)(parity, x_cb, stag_sp, ic, ic_c) += (*arg.LL)(dim , parity, x_cb, ic, jc) * arg.V((parity+1)&1, y3_cb, stag_sp, jc, ic_c);
	    } else {
              for(int s = 0; s < fineSpin; s++) {  //Fine Spin
                 for (int s_col=0; s_col<fineSpin; s_col++) {
	           // on the coarse lattice if forwards then use the forwards links
		   (*arg.UV)(parity, x_cb, s_col*fineSpin+s, ic, ic_c) += (*arg.FL)(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, ic, jc) * arg.V((parity+1)&1, y_cb, s, jc, ic_c);
                 } // which chiral block
              } //Fine spin rows 
	    } // from coarse?
	  }  //Fine color columns
	}  //Fine color rows
      }  //Coarse color
    }
    
    return;
  } // computeUV

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeKSUVCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	computeKSUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    }   // parity
  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeKSUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
  }

  /**
     @brief Do a single (AV)^\dagger * UV product, where for preconditioned
     clover, AV correspond to the clover inverse multiplied by the
     packed null space vectors, else AV is simply the packed null
     space vectors.

     @param[out] vuv Result array
     @param[in,out] arg Arg storing the fields and parameters
     @param[in] Fine grid parity we're working on
     @param[in] x_cb Checkboarded x dimension
   */
  template <bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void multiplyKSVUV(complex<Float> vuv[], Arg &arg, int parity, int x_cb, const int factor) {

    for (int i = 0; i < factor*coarseColor*coarseColor; i++) vuv[i] = 0.0;

    if (!from_coarse) { // fine grid is top level

      const int stag_sp = 0;

      for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
        for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	  for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
             //if dir == QUDA_BACKWARDS or dir == QUDA_FORWARDS

	     vuv[ic_c*coarseColor+jc_c] +=  conj(arg.V(parity, x_cb, stag_sp, ic, ic_c)) * (*arg.UV)(parity, x_cb, stag_sp, ic, jc_c);
             if(arg.UVL != nullptr) vuv[coarseColor*coarseColor + ic_c*coarseColor+jc_c] +=  conj(arg.V(parity, x_cb, stag_sp, ic, ic_c)) * (*arg.UVL)(parity, x_cb, stag_sp, ic, jc_c);

	  } //Fine color
        } //Coarse Color column
      } //Coarse Color row

    } else { // fine grid operator is a coarse operator
      for (int s_col=0; s_col<fineSpin; s_col++) { // which chiral block
	for (int s = 0; s < fineSpin; s++) {
	  for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
		vuv[((s*coarseSpin+s_col)*coarseColor+ic_c)*coarseColor+jc_c] +=
		  conj(arg.V(parity, x_cb, s, ic, ic_c)) * (*arg.UV)(parity, x_cb, s_col*fineSpin+s, ic, jc_c);
	      } //Fine color
	    } //Coarse Color column
	  } //Coarse Color row
	} //Fine spin
      }
    } // from_coarse
    
    return;
  }
//TROUBLE HERE!
  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void computeKSVUV(Arg &arg, int parity, int x_cb) {

    const int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= arg.xc_size[d];

    const int stag_sp = 0;

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];


    bool isDiagonal = (((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim]) ? true : false;
    bool isDiagonal_long = (arg.UVL == nullptr) ? false : (((coord[dim]+3)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim]) ? true : false;
    
    auto *M =   isDiagonal ? (dir == QUDA_BACKWARDS ? &arg.X : &arg.Xinv) : &arg.Y;
    auto *M_L = (arg.UVL == nullptr) ? nullptr : (isDiagonal_long ? (dir == QUDA_BACKWARDS ? &arg.X : &arg.Xinv) : &arg.Y);
	      
    const int dim_index      = isDiagonal ? 0 : (dir == QUDA_BACKWARDS ? dim : dim + 4);
    const int dim_index_long = isDiagonal_long ? 0 : (dir == QUDA_BACKWARDS ? dim : dim + 4);

    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];
	      
    //printf("(%d,%d)\n", coarse_x_cb, coarse_parity);
    coord[0] /= 2;

    //coarse spin components:
    int s_row = parity == 0 ? 0 : 1  ;//coarse spin components
    int s_col = (1 - s_row);

    const int factor = (arg.UVL == nullptr) ? 1 : 2;

    complex<Float> vuv[2*coarseColor*coarseColor];//[factor*coarseColor*coarseColor]
    multiplyKSVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, parity, x_cb, factor);

    for(int c_row = 0; c_row < coarseColor; c_row++) { // Coarse Color row
       for(int c_col = 0; c_col < coarseColor; c_col++) { // Coarse Color column
         (*M)(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += vuv[c_row*coarseColor+c_col];
         if(M_L != nullptr) (*M_L)(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += vuv[coarseColor*coarseColor+c_row*coarseColor+c_col]; 
      }
    }

  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeKSVUVCPU(Arg arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
	computeKSVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeKSVUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(arg, parity, x_cb);
  }


  /**
   * Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeKSYreverse(Arg &arg, int parity, int x_cb) {
    auto &Y = arg.Y;

    for (int d=0; d<4; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
        const int s_col = 1 - s_row; 

	for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
	    Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = - Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
	  } //Color column
	} //Color row
      } //Spin row
    } // dimension

    return;
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeKSYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	computeKSYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeKSYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB()) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb);
  }

  /**
   * Adds the reverse links to the coarse local term, which is just
   * the conjugate of the existing coarse local term but with
   * plus/minus signs for off-diagonal spin components so multiply by
   * the appropriate factor of -kappa.
   *
  */
  template<bool bidirectional, typename Float, int nSpin, int nColor, typename Arg>
  void computeKSCoarseLocal(Arg &arg, int parity, int x_cb)
  {
    complex<Float> Xlocal[nSpin*nSpin*nColor*nColor];

    for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
      for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column

	//Copy the Hermitian conjugate term to temp location
	for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
	    //Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
	    Xlocal[((nSpin*s_col+s_row)*nColor+ic_c)*nColor+jc_c] = arg.X(0,parity,x_cb,s_row, s_col, ic_c, jc_c);
	  }
	}
      }
    }

    for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
      for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
             if(s_row == s_col){
                arg.X(0,parity,x_cb, s_row, s_col, ic_c,ic_c) = arg.mass;//in fact, 2*mass
             }else{
	       if (bidirectional) {
	         // here we have forwards links in Xinv and backwards links in X
	         arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =
		   (+arg.Xinv(0,parity,x_cb,s_row,s_col,ic_c,jc_c)
			    +conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	       } else {
	         // here we have just backwards links
	         arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =
		  (-arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)
			    +conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	       }
            }
	  } //Color column
	} //Color row
      } //Spin column
    } //Spin row
    
    return; 
  }

  template<bool bidirectional, typename Float, int nSpin, int nColor, typename Arg>
  void ComputeKSCoarseLocalCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	computeKSCoarseLocal<bidirectional,Float,nSpin,nColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template<bool bidirectional, typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeKSCoarseLocalGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSCoarseLocal<bidirectional,Float,nSpin,nColor,Arg>(arg, parity, x_cb);
  }

  template<bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void computeKSCoarseClover(Arg &arg, int parity, int x_cb) {

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

    if (from_coarse) {
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
    else
    {
       errorQuda("\nOperator not applicable to top level staggered!\n");
    }
  }

  template <bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeKSCoarseCloverCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	computeKSCoarseClover<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg, parity, x_cb);
      } // c/b volume
    } // parity
  }

  template <bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeKSCoarseCloverGPU(Arg &arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSCoarseClover<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg, parity, x_cb);
  }



  enum KS_ComputeType {
    KS_COMPUTE_UV,
    KS_COMPUTE_VUV,
    KS_COMPUTE_COARSE_CLOVER,
    KS_COMPUTE_REVERSE_Y,
    KS_COMPUTE_COARSE_LOCAL,
    KS_COMPUTE_INVALID
  };

  template <bool from_coarse, typename Float, int fineSpin,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateKSY : 
   public TunableVectorY {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    int dim;
    QudaDirection dir;
    KS_ComputeType type;
    bool bidirectional;

    long long flops() const
    {

      long long flops_ = 0;
      switch (type) {
      case KS_COMPUTE_UV:
	// when fine operator is coarse take into account that the link matrix has spin dependence
	flops_ = 2*arg.fineVolumeCB * 8 * fineSpin * coarseColor * fineColor * fineColor * !from_coarse ? 1 : fineSpin;
	break;
      case KS_COMPUTE_VUV:
	// when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
	flops_ = 2*arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case KS_COMPUTE_REVERSE_Y:
	// no floating point operations
	flops_ = 0;
	break;
      case KS_COMPUTE_COARSE_CLOVER:
	// when the fine operator is truly fine the clover multiplication is block sparse which halves the number of operations
	flops_ = 2*arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case KS_COMPUTE_COARSE_LOCAL:
	// complex addition over all components
	flops_ = 2*arg.coarseVolumeCB*coarseSpin*coarseSpin*coarseColor*coarseColor*2;
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      // 2 from parity, 8 from complex
      return flops_;
    }
//FIXME
    long long bytes() const
    {
      long long bytes_ = 0;
      switch (type) {
      case KS_COMPUTE_UV:
	bytes_ = (*arg.UV).Bytes() + arg.V.Bytes();
	break;
      case KS_COMPUTE_VUV:
	bytes_ = (*arg.UV).Bytes() + arg.V.Bytes();
	break;
      case KS_COMPUTE_COARSE_CLOVER:
	bytes_ = 2*arg.X.Bytes() + 2*arg.C.Bytes() + arg.V.Bytes(); // 2 from parity
	break;
      case KS_COMPUTE_REVERSE_Y:
	bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
      case KS_COMPUTE_COARSE_LOCAL:
      default:
	errorQuda("Undefined compute type %d", type);
      }
      return bytes_;
    }

    unsigned int minThreads() const {
      unsigned int threads = 0;
      switch (type) {
      case KS_COMPUTE_UV:
      case KS_COMPUTE_VUV:
      case KS_COMPUTE_COARSE_CLOVER:
	threads = arg.fineVolumeCB;
	break;
      case KS_COMPUTE_REVERSE_Y:
      case KS_COMPUTE_COARSE_LOCAL:
	threads = arg.coarseVolumeCB;
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      return threads;
    }

  public:
    CalculateKSY(Arg &arg, QudaDiracType dirac, const ColorSpinorField &meta)
      : TunableVectorY(2), arg(arg), type(type),
	bidirectional(dirac==QUDA_COARSEPC_DIRAC || bidirectional_debug),
	meta(meta), dim(0), dir(QUDA_BACKWARDS)
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
    virtual ~CalculateKSY() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

	if (type == KS_COMPUTE_UV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == KS_COMPUTE_VUV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSVUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSVUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSVUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSVUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSVUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSVUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSVUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSVUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }
        } else if (type == KS_COMPUTE_COARSE_CLOVER) {

	  ComputeKSCoarseCloverCPU<from_coarse,Float,fineSpin,fineColor,coarseColor>(arg);

	} else if (type == KS_COMPUTE_REVERSE_Y) {

	  ComputeKSYReverseCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == KS_COMPUTE_COARSE_LOCAL) {

	  if (bidirectional) ComputeKSCoarseLocalCPU<true,Float,coarseSpin,coarseColor>(arg);
	  else ComputeKSCoarseLocalCPU<false,Float,coarseSpin,coarseColor>(arg);

	} else {
	  errorQuda("Undefined compute type %d", type);
	}
      } else {
	errorQuda("GPU variant not yet implemented");
      }
    }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDirection(QudaDirection dir_) { dir = dir_; }

    /**
       Set which computation we are doing
     */
    void setComputeType(KS_ComputeType type_) { type = type_; }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == KS_COMPUTE_UV)            strcat(Aux,",computeKSUV");
      else if (type == KS_COMPUTE_VUV)           strcat(Aux,",computeKSVUV");
      else if (type == KS_COMPUTE_REVERSE_Y)     strcat(Aux,",computeKSYreverse");
      else if (type == KS_COMPUTE_COARSE_CLOVER) strcat(Aux,",computeCoarseClover");
      else if (type == KS_COMPUTE_COARSE_LOCAL)  strcat(Aux,",computeKSCoarseLocal");
      else errorQuda("Unknown type=%d\n", type);

      if (type == KS_COMPUTE_UV || type == KS_COMPUTE_VUV) {
	if      (dim == 0) strcat(Aux,",dim=0");
	else if (dim == 1) strcat(Aux,",dim=1");
	else if (dim == 2) strcat(Aux,",dim=2");
	else if (dim == 3) strcat(Aux,",dim=3");

	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");
      }

      return TuneKey(meta.VolString(), typeid(*this).name(), Aux);
    }

  };


  template<typename Float, int n, typename Gauge>
  void createKSYpreconditioned(Gauge &Yhat, Gauge &Xinv, const Gauge &Y, const int *dim, int nFace, const int *commDim) {

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

  /**
     @brief Calculate the coarse-link field, include the clover field,
     and its inverse, and finally also compute the preconditioned
     coarse link field.

     @param Y[out] Coarse link field accessor
     @param X[out] Coarse clover field accessor
     @param Xinv[out] Coarse clover inverse field accessor
     @param UV[out] Temporary accessor used to store fine link field * null space vectors
     @param UVL[out] Temporary accessor used to store fine link field * null space vectors
     space vectors (only applicable when fine-grid operator is the
     preconditioned clover operator else in general this just aliases V
     @param V[in] Packed null-space vector accessor
     @param FL[in] Fine grid link / gauge field accessor
     @param LL[in] Fine grid link / gauge field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param Xinv_[out] Coarse clover field
     @param Yhat_[out] Preconditioned coarse link field
     @param v[in] Packed null-space vectors
     @param mass[in] mass parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
//Note: fineSpin might be needed for coarsecoarse operator, with from_coarse = true

  template<bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   QudaGaugeFieldOrder gOrder, typename F, typename Ftmp, typename coarseGauge, typename fineGauge, typename fineClover>
  void calculateKSY(coarseGauge &Y, coarseGauge &X, coarseGauge &Xinv, Ftmp *UV, Ftmp *UVL, F &V, fineGauge *FL, fineGauge *LL, fineClover &C, fineClover &Cinv,
		  GaugeField &Y_, GaugeField &X_, GaugeField &Xinv_, GaugeField &Yhat_, const ColorSpinorField &v,
		  double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)//??
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    if( FL->Ndim() != 4 ) errorQuda("Number of dimensions not supported");
    if( LL ) if ( LL->Ndim() != 4 ) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[5];
    for (int i=0; i<4; i++) x_size[i] = v.X(i);
    x_size[4] = 1;

    int xc_size[5];
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM];
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateKSYArg<Float,coarseGauge,fineGauge,F,Ftmp,fineClover> Arg;

    if( (UVL == nullptr && LL != nullptr) || (UVL != nullptr && LL == nullptr) )   errorQuda("\nBad pointers detected.\n");

    Arg arg(Y, X, Xinv, UV, UVL, FL, LL, V, C, Cinv, mass, x_size, xc_size, geo_bs);
    CalculateKSY<from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> y(arg, dirac, v);

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = (dirac == QUDA_COARSEPC_DIRAC || bidirectional_debug);
    if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
    else printfQuda("Doing uni-directional link coarsening\n");


    // Now compute the coarse links
    for(int d = 0; d < nDim; d++) {
      y.setDimension(d);
      printfQuda("Computing %d UV and VUV\n", d);

      if (bidirectional_links) {
	y.setDirection(QUDA_FORWARDS); // what does this mean for the preconditioned coarse operator?
	y.setComputeType(KS_COMPUTE_UV);  // compute U*V product
	y.apply(0);
	printfQuda("UV2[%d] = %e\n", d, UV->norm2());

	y.setComputeType(KS_COMPUTE_VUV); // compute Y += VUV
	y.apply(0);
	printfQuda("Y2[%d] = %e\n", d, Y.norm2(4+d));
      }

      y.setDirection(QUDA_BACKWARDS);
      y.setComputeType(KS_COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      printfQuda("UAV2[%d] = %e\n", d, UV->norm2());

      y.setComputeType(KS_COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      printfQuda("Y2[%d] = %e\n", d, Y.norm2(d));
    }
    printfQuda("X2 = %e\n", X.norm2(0));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      printfQuda("Reversing links\n");
      y.setComputeType(KS_COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(0);
    }

    printfQuda("Computing coarse local\n");
    y.setComputeType(KS_COMPUTE_COARSE_LOCAL);
    y.apply(0);
    printfQuda("X2 = %e\n", X.norm2(0));

    if (dirac == QUDA_COARSE_DIRAC) {
      printfQuda("Computing fine->coarse 'clover' term\n");
      y.setComputeType(KS_COMPUTE_COARSE_CLOVER);
      y.apply(0);
      printfQuda("X2 = %e\n", X.norm2(0));
    }

    {
      cpuGaugeField *X_h = static_cast<cpuGaugeField*>(&X_);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv_);

      // invert the clover matrix field
      const int n = X_h->Ncolor();
      BlasMagmaArgs magma(X_h->Precision());
      magma.BatchInvertMatrix(((void**)Xinv_h->Gauge_p())[0], ((void**)X_h->Gauge_p())[0], n, X_h->Volume());
    }

    // now exchange Y halos for multi-process dslash
    Y_.exchangeGhost();

    // compute the preconditioned links
    // Yhat_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
    // Yhat_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)

    {
      // use spin-ignorant accessor to make multiplication simpler
      // alse with new accessor we ensure we're accessing the same ghost buffer in Y_ as was just exchanged
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,1,gOrder> gCoarse;
      gCoarse yAccessor(const_cast<GaugeField&>(Y_));
      gCoarse yHatAccessor(const_cast<GaugeField&>(Yhat_));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv_));
      int comm_dim[4];
      for (int i=0; i<4; i++) comm_dim[i] = comm_dim_partitioned(i);
      createKSYpreconditioned<Float,coarseSpin*coarseColor>(yHatAccessor, xInvAccessor, yAccessor, xc_size, 1, comm_dim);
    }

    // fill back in the bulk of Yhat so that the backward link is updated on the previous node
    Yhat_.injectGhost();
  }
} // namespace quda
