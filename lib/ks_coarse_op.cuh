#include <multigrid_helper.cuh>

namespace quda {

  // For coarsening un-preconditioned operators we use uni-directional
  // coarsening to reduce the set up code.  For debugging we can force
  // bi-directional coarsening.
  static bool bidirectional_debug = false;

  template <typename Float, int fineSpin, int coarseSpin, typename coarseGauge, typename fineGauge, typename fineSpinorTmp,
	    typename fineSpinorV, typename fineClover>
  struct CalculateKSYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */
    coarseGauge Xinv;        /** Computed coarse clover field */

    fineSpinorTmp *UV;        /** Temporary that stores the fine-fat-link * spinor field product */
    fineSpinorTmp *UVL;       /** Temporary that stores the fine-long-link * spinor field product */

    const fineGauge *FL;     /** Fine grid fat-link field */
    const fineGauge *LL;     /** Fine grid long-link field */
    const fineSpinorV V;     /** Fine grid spinor field */
    const fineClover C;      /** Fine grid clover field */
    const fineClover Cinv;   /** Fine grid clover field */

    int x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float mass;                /** kappa value */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    CalculateKSYArg(coarseGauge &Y, coarseGauge &X, coarseGauge &Xinv, fineSpinorTmp *UV, fineSpinorTmp *UVL, const fineGauge *FL, const fineGauge *LL, 
                  const fineSpinorV &V, const fineClover &C, const fineClover &Cinv, double mass, const int *x_size_, const int *xc_size_, int *geo_bs_) 
                  : Y(Y), X(X), Xinv(Xinv), UV(UV), UVL(UVL), FL(FL), LL(LL), V(V), C(C), Cinv(Cinv), mass(static_cast<Float>(mass)), fineVolumeCB(V.VolumeCB()), coarseVolumeCB(X.VolumeCB())
    {
      //if (V.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
	//errorQuda("Gamma basis %d not supported", V.GammaBasis());

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
  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor,
	   int coarseSpin, int coarseColor, typename Wtype, typename Arg>
  __device__ __host__ inline void computeKSUV(Arg &arg, const Wtype &W, int parity, int x_cb, int ic_c) {

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);

    constexpr int uvSpin = fineSpin * (from_coarse ? 2 : 1);

    complex<Float> UV[uvSpin][fineColor];

    constexpr int uvlSpin  = compute_3d_nbr ? uvSpin    : 1;
    constexpr int uvlColor = compute_3d_nbr ? fineColor : 1;

    complex<Float> UVL[uvlSpin][uvlColor];//used in fine-to-coarse constructors

    for(int s = 0; s < uvSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	UV[s][c] = static_cast<Float>(0.0);
        if( compute_3d_nbr ) UVL[s][c] = static_cast<Float>(0.0);
      }
    }

    if ( arg.comm_dim[dim] && (coord[dim] + 1 >= arg.x_size[dim]) ) {
      int ghost_idx       = ghostFaceIndex<1>(coord, arg.x_size, dim, 1);
      int ghost_long_idx  = ( compute_3d_nbr ) ? ghostFaceIndex<1>(coord, arg.x_size, dim, 3) : 0;

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if (!from_coarse) {
	      UV[s][ic] += (*arg.FL)(dim, parity, x_cb, ic, jc) * W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c);
              if( compute_3d_nbr ) UVL[s][ic] += (*arg.LL)(dim, parity, x_cb, ic, jc) * W.Ghost(dim, 1, (parity+1)&1, ghost_long_idx, s, jc, ic_c);
	    } else {
	      // on the coarse lattice if forwards then use the forwards links
              int s_col = 1 - s;// WARNING: we assume that only off-diagonal terms contribute
	      UV[s_col*fineSpin+s][ic] += (*arg.FL)(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc) * W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s_col, jc, ic_c);
	    }
	  }  //Fine color columns
	}  //Fine color rows
      }  //Fine Spin

    } else {
      int y_cb  = linkIndexP1(coord, arg.x_size, dim);
      int y3_cb = ( compute_3d_nbr ) ? linkIndexP3(coord, arg.x_size, dim) : 0;

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if (!from_coarse) {
	      UV[s][ic] += (*arg.FL)(dim, parity, x_cb, ic, jc) * W((parity+1)&1, y_cb, s, jc, ic_c);
              if( compute_3d_nbr ) UVL[s][ic] += (*arg.LL)(dim, parity, x_cb, ic, jc) * W((parity+1)&1, y3_cb, s, jc, ic_c);
	    } else {
              int s_col = 1 - s;
	      // on the coarse lattice if forwards then use the forwards links
	      UV[s_col*fineSpin+s][ic] +=  (*arg.FL)(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc) *  W((parity+1)&1, y_cb, s_col, jc, ic_c);
	    }
	  }  //Fine color columns
	}  //Fine color rows
      }  //Fine Spin

    }

    for(int s = 0; s < uvSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	(*arg.UV)(parity,x_cb,s,c,ic_c) = UV[s][c];
        if( compute_3d_nbr ) (*arg.UVL)(parity,x_cb,s,c,ic_c) = UVL[s][c];
      }
    }


  } // computeUV

  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeKSUVCPU(Arg &arg) {

    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	for (int ic_c=0; ic_c < coarseColor; ic_c++) // coarse color
          computeKSUV<from_coarse,compute_3d_nbr,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
      } // c/b volume
    }   // parity
  }

  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeKSUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    computeKSUV<from_coarse,compute_3d_nbr,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
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
  template <bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void multiplyKSVUV(complex<Float> vuv[], const Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {

    constexpr int elements = (from_coarse) ? coarseSpin*coarseSpin : ((compute_3d_nbr) ? 2 : 1);
#pragma unroll 
    for (int i=0; i<elements; i++) vuv[i] = 0.0;

    auto &UVref  = *arg.UV;
    auto &UVLref = *arg.UVL;

    if (!from_coarse) { // fine grid is top level

#pragma unroll
      for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color
        if (dir == QUDA_BACKWARDS) {
          //Off-diagonal:
	  vuv[0] += conj(arg.V(parity, x_cb, 0, ic, ic_c)) * UVref(parity, x_cb, 0, ic, jc_c);
          if(compute_3d_nbr) vuv[1] += conj(arg.V(parity, x_cb, 0, ic, ic_c)) * UVLref(parity, x_cb, 0, ic, jc_c);
	} else {
          //Off-diagonal Spin (forward link / negative projector applied)
	  vuv[0] -= conj(arg.V(parity, x_cb, 0, ic, ic_c)) * UVref(parity, x_cb, 0, ic, jc_c);
          if(compute_3d_nbr) vuv[1] -= conj(arg.V(parity, x_cb, 0, ic, ic_c)) * UVLref(parity, x_cb, 0, ic, jc_c);
	}
      } //Fine color

    } else { // fine grid operator is a coarse operator

#pragma unroll
      for (int s_col=0; s_col<fineSpin; s_col++) { // which chiral block
//#pragma unroll
//	for (int s = 0; s < fineSpin; s++) {
        {
          int s = 1 - s_col;
#pragma unroll
	  for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
	    vuv[s*coarseSpin+s_col] += conj(arg.V(parity, x_cb, s, ic, ic_c)) * UVref(parity, x_cb, s_col*fineSpin+s, ic, jc_c);
	  } //Fine color
	} //Fine spin
      }

    } // from_coarse

  }

  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void computeKSVUV(Arg &arg, int parity, int x_cb, int c_row, int c_col) {

    constexpr int nDim = 4;//5 ?
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];
    int coarse_size = 1;
    for(int d = 0; d<nDim; d++) coarse_size *= arg.xc_size[d];

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;
    const bool isDiagonal_long = (compute_3d_nbr) ? ((((coord[dim]+3)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim]) ? true : false) : false;


    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];
    coord[0] /= 2;

    constexpr int elems = (from_coarse) ? coarseSpin*coarseSpin : ((compute_3d_nbr) ? 2 : 1);
    constexpr int soffs = (from_coarse) ? coarseSpin : 1;
#ifdef __CUDA_ARCH__
    extern __shared__ complex<Float> s[];
    int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    complex<Float> *vuv = &s[tid*elems];
#else
    complex<Float> vuv[elems];
#endif

    multiplyKSVUV<from_coarse,compute_3d_nbr,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, parity, x_cb, c_row, c_col);

    if (!isDiagonal) {
      constexpr int dim_index = (dir == QUDA_BACKWARDS) ? dim : dim + 4;
#pragma unroll
      for (int s_row = 0; s_row < soffs; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < soffs; s_col++) { // Chiral column block

          int s_row_ = (!from_coarse) ? parity : s_row;
          int s_col_ = (!from_coarse) ? (1 - s_row_) : s_col;
 
	  arg.Y.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row_,s_col_,c_row,c_col,vuv[s_row*soffs+s_col]);
	}
      }
    } else if (dir == QUDA_BACKWARDS) { // store the forward and backward clover contributions separately for now since they can't be added coeherently easily
#pragma unroll
      for (int s_row = 0; s_row < soffs; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < soffs; s_col++) { // Chiral column block

          int s_row_ = (!from_coarse) ? parity : s_row;
          int s_col_ = (!from_coarse) ? (1 - s_row_) : s_col;

	  arg.X.atomicAdd(0,coarse_parity,coarse_x_cb,s_row_,s_col_,c_row,c_col,vuv[s_row*soffs+s_col]);
	  if(compute_3d_nbr) arg.X.atomicAdd(0,coarse_parity,coarse_x_cb,s_row_,s_col_,c_row,c_col,vuv[1]);//soffs*soffs+s_row*soffs+s_col=1
	}
      }
    } else {
#pragma unroll
      for (int s_row = 0; s_row < soffs; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < soffs; s_col++) { // Chiral column block

          int s_row_ = (!from_coarse) ? parity : s_row;
          int s_col_ = (!from_coarse) ? (1 - s_row_) : s_col;

	  arg.Xinv.atomicAdd(0,coarse_parity,coarse_x_cb,s_row_,s_col_,c_row,c_col,vuv[s_row*soffs+s_col]);
	  if(compute_3d_nbr) arg.Xinv.atomicAdd(0,coarse_parity,coarse_x_cb,s_row_,s_col_,c_row,c_col,vuv[s_row*soffs+s_col]);
	}
      }
    }

    if (compute_3d_nbr && !isDiagonal_long) {//this term is not valid if from_coarse = true 
      constexpr int dim_index_long = (dir == QUDA_BACKWARDS) ? dim : dim + 4;
      int s_row = parity;
      int s_col = (1 - s_row);
 
      arg.Y.atomicAdd(dim_index_long,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv[1]);
    } 

    return;
  }

  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeKSVUVCPU(Arg arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
	for (int c_row=0; c_row<coarseColor; c_row++)
	  for (int c_col=0; c_col<coarseColor; c_col++)
	    computeKSVUV<from_coarse,compute_3d_nbr,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col);
      } // c/b volume
    } // parity
  }

  template<bool from_coarse, bool compute_3d_nbr, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeKSVUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*coarseColor) return;

    int c_col = parity_c_col % coarseColor; // coarse color col index
    int parity = parity_c_col / coarseColor;

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    if (c_row >= coarseColor) return;
    computeKSVUV<from_coarse,compute_3d_nbr,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeKSYreverse(Arg &arg, int parity, int x_cb, int ic_c) {
    auto &Y = arg.Y;

    const Float sign = static_cast<Float>(-1.0);

    for (int d=0; d<4; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
        const int s_col = 1 - s_row;

        for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
          Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = sign * Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);//?
	} //Color column
      } //Spin row

    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeKSYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  computeKSYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
	}
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeKSYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeKSYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
  }

  /**
   * Adds the reverse links to the coarse local term, which is just
   * the conjugate of the existing coarse local term but with
   * plus/minus signs for off-diagonal spin components so multiply by
   * the appropriate factor of -kappa.
   *
  */
  template<bool bidirectional, typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeKSCoarseLocal(Arg &arg, int parity, int x_cb)
  {
    complex<Float> Xlocal[nSpin*nSpin*nColor*nColor];

    const Float sign = static_cast<Float>(-1.0);

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

            if(s_row == s_col) {
               arg.X(0,parity,x_cb, s_row, s_col, ic_c,ic_c) = arg.mass;//in fact, 2*mass
            } else {
	      if (bidirectional) {
	        // here we have forwards links in Xinv and backwards links in X
	        arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =
		  (arg.Xinv(0,parity,x_cb,s_row,s_col,ic_c,jc_c)
			    +conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	      } else {
	        // here we have just backwards links
	        arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) =
		  (sign*arg.X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)
			    +conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	      }
            }
	  } //Color column
	} //Color row
      } //Spin column
    } //Spin row

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


  template<bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ void computeKSCoarseClover(Arg &arg, int parity, int x_cb, int ic_c) {

    const int nDim = 4;//5

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

    complex<Float> X[coarseSpin*coarseSpin*coarseColor];
    for (int i=0; i<coarseSpin*coarseSpin*coarseColor; i++) X[i] = 0.0;

    if (!from_coarse) {
      return;//operation is undefined for the top-level staggered.
    } else {
      //If Nspin != 4, then spin structure is a dense matrix and there is now spin aggregation
      //N.B. assumes that no further spin blocking is done in this case.
      for(int s = 0; s < fineSpin; s++) { //Loop over spin row
	for(int s_col = 0; s_col < fineSpin; s_col++) { //Loop over spin column
	  //for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
	    for(int jc_c = 0; jc_c <coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
		for(int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
		  X[ (s*coarseSpin + s_col)*coarseColor + jc_c] +=
		    conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
		} //Fine color column
	      }  //Fine color row
	    } //Coarse Color column
	    //} //Coarse Color row
	}  //Fine spin column
      } //Fine spin
    }

    for (int si = 0; si < coarseSpin; si++) {
      for (int sj = 0; sj < coarseSpin; sj++) {
	//for (int ic = 0; ic < coarseColor; ic++) {
	  for (int jc = 0; jc < coarseColor; jc++) {
	    arg.X.atomicAdd(0,coarse_parity,coarse_x_cb,si,sj,ic_c,jc,X[(si*coarseSpin+sj)*coarseColor+jc]);
	  }
	  //}
      }
    }

  }

  template <bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeKSCoarseCloverCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	for (int ic_c=0; ic_c<coarseColor; ic_c++) {
	  computeKSCoarseClover<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg, parity, x_cb, ic_c);
	}
      } // c/b volume
    } // parity
  }

  template <bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeKSCoarseCloverGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    computeKSCoarseClover<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg, parity, x_cb, ic_c);
  }


  enum ComputeType {
    KS_COMPUTE_UV,
    KS_COMPUTE_VUV,
    KS_COMPUTE_COARSE_CLOVER,
    KS_COMPUTE_REVERSE_Y,
    KS_COMPUTE_COARSE_LOCAL,
    KS_COMPUTE_INVALID
  };

  template <bool from_coarse, bool compute_3d_nbr, typename Float, int fineSpin,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateKSY : public TunableVectorYZ {
  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    GaugeField &Y;
    GaugeField &X;
    GaugeField &Xinv;

    int dim;
    QudaDirection dir;
    ComputeType type;
    bool bidirectional;

    long long flops() const
    {
      long long flops_ = 0;
      switch (type) {
      case KS_COMPUTE_UV:
	// when fine operator is coarse take into account that the link matrix has spin dependence
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * coarseColor * fineColor * fineColor * (!from_coarse ? 1 : fineSpin);
	break;
      case KS_COMPUTE_VUV:
	// when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case KS_COMPUTE_COARSE_CLOVER:
	// when the fine operator is truly fine the clover multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case KS_COMPUTE_REVERSE_Y:
	// no floating point operations
	flops_ = 0;
	break;
      case KS_COMPUTE_COARSE_LOCAL:
	// complex addition over all components
	flops_ = 2l * arg.coarseVolumeCB*coarseSpin*coarseSpin*coarseColor*coarseColor*2;
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      // 2 from parity, 8 from complex
      return flops_;
    }
    long long bytes() const
    {
      long long bytes_ = 0;
      switch (type) {
      case KS_COMPUTE_UV:
	bytes_ = arg.UV->Bytes() + arg.V.Bytes() + 2*arg.FL->Bytes()*coarseColor;//no long link contribution!
	break;
      case KS_COMPUTE_VUV:
	bytes_ = 2*arg.Y.Bytes() + 2*arg.X.Bytes() + 2*arg.Xinv.Bytes() + arg.UV->Bytes() + arg.V.Bytes();
	break;
      case KS_COMPUTE_COARSE_CLOVER:
	bytes_ = 2*arg.X.Bytes() + 2*arg.C.Bytes() + arg.V.Bytes(); // 2 from parity
	break;
      case KS_COMPUTE_REVERSE_Y:
	bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
      case KS_COMPUTE_COARSE_LOCAL:
	bytes_ = 2*2*arg.X.Bytes(); // 2 from i/o, 2 from parity
	break;
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

    bool tuneGridDim() const { return false; } // don't tune the grid dimension

    unsigned int sharedBytesPerThread() const {
      return (type == KS_COMPUTE_VUV) ? coarseSpin*coarseSpin*sizeof(complex<Float>) : 0;
    }

  public:
    CalculateKSY(Arg &arg, QudaDiracType dirac, const ColorSpinorField &meta, GaugeField &Y, GaugeField &X, GaugeField &Xinv)
      : TunableVectorYZ(2,1), arg(arg), type(KS_COMPUTE_INVALID),
	bidirectional(dirac==QUDA_COARSEPC_DIRAC || bidirectional_debug),
	meta(meta), Y(Y), X(X), Xinv(Xinv), dim(0), dir(QUDA_BACKWARDS)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux,comm_dim_partitioned_string());
    }
    virtual ~CalculateKSY() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

	if (type == KS_COMPUTE_UV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSUVCPU<from_coarse,compute_3d_nbr,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == KS_COMPUTE_VUV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeKSVUVCPU<from_coarse,compute_3d_nbr,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == KS_COMPUTE_COARSE_CLOVER) {

	  ComputeKSCoarseCloverCPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg);

	} else if (type == KS_COMPUTE_REVERSE_Y) {

	  ComputeKSYReverseCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == KS_COMPUTE_COARSE_LOCAL) {

	  if (bidirectional) ComputeKSCoarseLocalCPU<true,Float,coarseSpin,coarseColor>(arg);
	  else ComputeKSCoarseLocalCPU<false,Float,coarseSpin,coarseColor>(arg);

	} else {
	  errorQuda("Undefined compute type %d", type);
	}
      } else {

	if (type == KS_COMPUTE_UV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeKSUVGPU<from_coarse,compute_3d_nbr,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == KS_COMPUTE_VUV) {
	  tp.grid.y = 2*coarseColor;
	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeKSVUVGPU<from_coarse,compute_3d_nbr,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == KS_COMPUTE_COARSE_CLOVER) {

	  ComputeKSCoarseCloverGPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>
	    <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == KS_COMPUTE_REVERSE_Y) {

	  ComputeKSYReverseGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == KS_COMPUTE_COARSE_LOCAL) {

	  if (bidirectional) ComputeKSCoarseLocalGPU<true,Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  else ComputeKSCoarseLocalGPU<false,Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else {
	  errorQuda("Undefined compute type %d", type);
	}
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
    void setComputeType(ComputeType type_) {
      type = type_;
      switch(type) {
      case KS_COMPUTE_VUV:
	resizeVector(1,coarseColor);
	break;
      case KS_COMPUTE_UV:
      case KS_COMPUTE_COARSE_CLOVER:
      case KS_COMPUTE_REVERSE_Y:
	resizeVector(2,coarseColor);
	break;
      default:
	resizeVector(2,1);
	break;
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == KS_COMPUTE_UV)            strcat(Aux,",computeUV");
      else if (type == KS_COMPUTE_VUV)           strcat(Aux,",computeVUV");
      else if (type == KS_COMPUTE_COARSE_CLOVER) strcat(Aux,",computeCoarseClover");
      else if (type == KS_COMPUTE_REVERSE_Y)     strcat(Aux,",computeYreverse");
      else if (type == KS_COMPUTE_COARSE_LOCAL)  strcat(Aux,",computeCoarseLocal");
      else errorQuda("Unknown type=%d\n", type);

      if (type == KS_COMPUTE_UV || type == KS_COMPUTE_VUV) {
	if      (dim == 0) strcat(Aux,",dim=0");
	else if (dim == 1) strcat(Aux,",dim=1");
	else if (dim == 2) strcat(Aux,",dim=2");
	else if (dim == 3) strcat(Aux,",dim=3");

	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");
      }

      const char *vol_str = (type == KS_COMPUTE_REVERSE_Y || type == KS_COMPUTE_COARSE_LOCAL) ? X.VolString () : meta.VolString();

      if (type == KS_COMPUTE_VUV || type == KS_COMPUTE_COARSE_CLOVER) {
	strcat(Aux,meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU," : ",CPU,");
	strcat(Aux,"coarse_vol=");
	strcat(Aux,X.VolString());
      } else {
	strcat(Aux,meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU" : ",CPU");
      }

      return TuneKey(vol_str, typeid(*this).name(), Aux);
    }

    void preTune() {
      switch (type) {
      case KS_COMPUTE_VUV:
	Y.backup();
	Xinv.backup();
      case KS_COMPUTE_COARSE_LOCAL:
      case KS_COMPUTE_COARSE_CLOVER:
	X.backup();
      case KS_COMPUTE_UV:
      case KS_COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }

    void postTune() {
      switch (type) {
      case KS_COMPUTE_VUV:
	Y.restore();
	Xinv.restore();
      case KS_COMPUTE_COARSE_LOCAL:
      case KS_COMPUTE_COARSE_CLOVER:
	X.restore();
      case KS_COMPUTE_UV:
      case KS_COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }
  };


  /**
     @brief Calculate the coarse-link field, including the coarse clover field.

     @param Y[out] Coarse link field accessor
     @param X[out] Coarse clover field accessor
     @param UV[out]  Temporary accessor used to store fine fat link field * null space vectors
     @param UVL[out] Temporary accessor used to store fine long link field * null space vectors
     @param V[in] Packed null-space vector accessor
     @param FL[in] Fine grid fat link / gauge field accessor
     @param LL[in] Fine grid fat link / gauge field accessor
     @param C[in] Fine grid clover field accessor
     @param Cinv[in] Fine grid clover inverse field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param mass[in] mass parameter
   */
  template<bool from_coarse, bool compute_3d_nbr, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   QudaGaugeFieldOrder gOrder, typename Ftmp, typename Vt, typename coarseGauge, typename fineGauge, typename fineClover>
  void calculateKSY(coarseGauge &Y, coarseGauge &X, coarseGauge &Xinv, Ftmp *UV, Ftmp *UVL, Vt &V, fineGauge *FL, fineGauge *LL, fineClover &C, fineClover &Cinv,
		  GaugeField &Y_, GaugeField &X_, GaugeField &Xinv_, ColorSpinorField *uv, ColorSpinorField *uvl, const ColorSpinorField &v,
		  double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    bool is_dirac_coarse = (dirac == QUDA_COARSEPC_DIRAC) ? true : false;
    if (is_dirac_coarse && fineSpin != 2)
      errorQuda("Input Dirac operator %d should have nSpin=2, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineSpin != 1)
      errorQuda("Input Dirac operator %d should have nSpin=1, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineColor != 3)
      errorQuda("Input Dirac operator %d should have nColor=3, not nColor=%d\n", dirac, fineColor);

    if (FL->Ndim() != 4) errorQuda("Number of dimensions not supported");
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
    typedef CalculateKSYArg<Float,fineSpin,coarseSpin,coarseGauge,fineGauge,Ftmp,Vt,fineClover> Arg;
    Arg arg(Y, X, Xinv, UV, UVL, FL, LL, V, C, Cinv, mass, x_size, xc_size, geo_bs);

    CalculateKSY<from_coarse, compute_3d_nbr, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> y(arg, dirac, v, Y_, X_, Xinv_);

    QudaFieldLocation location = Location(Y_, X_, Xinv_, v, v);
    printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = (dirac == QUDA_COARSEPC_DIRAC || bidirectional_debug);
    if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
    else printfQuda("Doing uni-directional link coarsening\n");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v.Ghost());  // point the accessor to the correct ghost buffer

    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    printfQuda("V2 = %e\n", arg.V.norm2());

    // First compute the coarse forward links if needed
    if (bidirectional_links) {
      for (int d = 0; d < nDim; d++) {
	y.setDimension(d);
	y.setDirection(QUDA_FORWARDS);
	printfQuda("Computing forward %d UV and VUV\n", d);

	if (uv->Precision() == QUDA_HALF_PRECISION) {
	  double U_max = 3.0*arg.FL->abs_max(from_coarse ? d+4 : d);
	  double uv_max = U_max * v.Scale();
	  uv->Scale(uv_max);
	  arg.UV->resetScale(uv_max);

	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
	}

        if(uvl && UVL) {
	  if (uvl->Precision() == QUDA_HALF_PRECISION) {
	    double U_max = 3.0*arg.LL->abs_max(from_coarse ? d+4 : d);
	    double uv_max = U_max * v.Scale();
	    uvl->Scale(uv_max);
	    arg.UVL->resetScale(uv_max);

	    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
	  }
        }

	y.setComputeType(KS_COMPUTE_UV);  // compute U*V product
	y.apply(0);
	printfQuda("UV2[%d] = %e\n", d, arg.UV->norm2());
	if ( UVL ) printfQuda("UVL2[%d] = %e\n", d, arg.UVL->norm2());

	y.setComputeType(KS_COMPUTE_VUV); // compute Y += VUV
	y.apply(0);
	printfQuda("Y2[%d] = %e\n", d, arg.Y.norm2(4+d));
      }
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      printfQuda("Computing backward %d UV and VUV\n", d);

      if (uv->Precision() == QUDA_HALF_PRECISION) {
	double U_max = 3.0*arg.FL->abs_max(d);
	double uv_max = U_max * v.Scale();
	uv->Scale(uv_max);
	arg.UV->resetScale(uv_max);

	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e av_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
      }

      if(uvl && UVL) {
        if (uvl->Precision() == QUDA_HALF_PRECISION) {
	  double U_max = 3.0*arg.LL->abs_max(from_coarse ? d+4 : d);
	  double uv_max = U_max * v.Scale();
	  uvl->Scale(uv_max);
	  arg.UVL->resetScale(uv_max);

	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
	}
      }

      y.setComputeType(KS_COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      printfQuda("UV2[%d] = %e\n", d, arg.UV->norm2());
      if ( UVL ) printfQuda("UVL2[%d] = %e\n", d, arg.UVL->norm2());

      y.setComputeType(KS_COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      printfQuda("Y2[%d] = %e\n", d, arg.Y.norm2(d));
    }
    printfQuda("X2 = %e\n", arg.X.norm2(0));

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
    printfQuda("X2 = %e\n", arg.X.norm2(0));

    // Check if we have a clover term that needs to be coarsened
    if (dirac == QUDA_COARSE_DIRAC) {
      printfQuda("Computing fine->coarse clover term\n");
      y.setComputeType(KS_COMPUTE_COARSE_CLOVER);
      y.apply(0);
    } 

    printfQuda("X2 = %e\n", arg.X.norm2(0));

  }

} // namespace quda
