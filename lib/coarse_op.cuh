#include <multigrid_helper.cuh>

// enable this for shared-memory atomics instead of global atomics.
// Doing so means that all of the coarsening for a coarse degree of
// freedom is handled by a single thread block.  This is presently
// slower than using global atomics (due to increased latency from
// having to run larger thread blocks)
//#define SHARED_ATOMIC

#ifdef SHARED_ATOMIC
// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
// if disabled then we pack multiple aggregates into a single block to improve coalescing
#ifdef SWIZZLE
#undef SWIZZLE
#endif
#endif

namespace quda {

  // For coarsening un-preconditioned operators we use uni-directional
  // coarsening to reduce the set up code.  For debugging we can force
  // bi-directional coarsening.
  // ESW hack for now: need to add logic so bi-directional coarsening
  // is triggered if ANY previous level is preconditioned, not just
  // the current level.
  static bool bidirectional_debug = true;

  template <typename Float, int fineSpin, int coarseSpin,
	    typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge, typename fineSpinor,
	    typename fineSpinorTmp, typename fineSpinorV, typename fineClover>
  struct CalculateYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    coarseGaugeAtomic Y_atomic;    /** Y atomic accessor used for computation before conversion to final format */
    coarseGaugeAtomic X_atomic;    /** X atomic accessor used for computation before conversion to final format */

    fineSpinorTmp UV;        /** Temporary that stores the fine-link * spinor field product */
    fineSpinor AV;           /** Temporary that stores the clover * spinor field product */

    const fineGauge U;       /** Fine grid link field */
    const fineSpinorV V;     /** Fine grid spinor field */
    const fineClover C;      /** Fine grid clover field */
    const fineClover Cinv;   /** Fine grid clover field */

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */
    const spin_mapper<fineSpin,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float kappa;                /** kappa value */
    Float mu;                   /** mu value */
    Float mu_factor;            /** multiplicative factor for mu applied when mu is added to the operator */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    const int *fine_to_coarse;
    const int *coarse_to_fine;

    const bool bidirectional;

    int_fastdiv aggregates_per_block; // number of aggregates per thread block
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

    Float max_h; // scalar that stores the maximum element of the dynamic clover inverse
    Float *max_d; // array that stores the maximum element per lattice site of the dynamic clover inverse

    CalculateYArg(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  fineSpinorTmp &UV, fineSpinor &AV, const fineGauge &U, const fineSpinorV &V,
		  const fineClover &C, const fineClover &Cinv, double kappa, double mu, double mu_factor,
		  const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_,
		  const int *fine_to_coarse, const int *coarse_to_fine, bool bidirectional)
      : Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic),
	UV(UV), AV(AV), U(U), V(V), C(C), Cinv(Cinv), spin_bs(spin_bs_), spin_map(),
	kappa(static_cast<Float>(kappa)), mu(static_cast<Float>(mu)), mu_factor(static_cast<Float>(mu_factor)),
        fineVolumeCB(V.VolumeCB()), coarseVolumeCB(X.VolumeCB()),
        fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
        bidirectional(bidirectional), aggregates_per_block(1), swizzle(1), max_d(nullptr)
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

    ~CalculateYArg() { }
  };

  /**
     Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
     Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
  */
  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor,
	   int coarseSpin, int coarseColor, typename Wtype, typename Arg>
  __device__ __host__ inline void computeUV(Arg &arg, const Wtype &W, int parity, int x_cb, int ic_c) {

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);

    constexpr int uvSpin = fineSpin * (from_coarse ? 2 : 1);

    complex<Float> UV[uvSpin][fineColor];

    for(int s = 0; s < uvSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	UV[s][c] = static_cast<Float>(0.0);
      }
    }

    if ( arg.comm_dim[dim] && (coord[dim] + 1 >= arg.x_size[dim]) ) {
      int nFace = 1;
      int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if (!from_coarse) {
	      UV[s][ic] += arg.U(dim, parity, x_cb, ic, jc) * W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c);
	    } else {
	      for (int s_col=0; s_col<fineSpin; s_col++) {
		// on the coarse lattice if forwards then use the forwards links
		UV[s_col*fineSpin+s][ic] += arg.U(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc) *
		  W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s_col, jc, ic_c);
	      } // which chiral block
	    }
	  }  //Fine color columns
	}  //Fine color rows
      }  //Fine Spin

    } else {
      int y_cb = linkIndexP1(coord, arg.x_size, dim);

      for(int s = 0; s < fineSpin; s++) {  //Fine Spin
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    if (!from_coarse) {
	      UV[s][ic] += arg.U(dim, parity, x_cb, ic, jc) * W((parity+1)&1, y_cb, s, jc, ic_c);
	    } else {
	      for (int s_col=0; s_col<fineSpin; s_col++) {
		// on the coarse lattice if forwards then use the forwards links
		UV[s_col*fineSpin+s][ic] +=
		  arg.U(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc) *
		  W((parity+1)&1, y_cb, s_col, jc, ic_c);
	      } // which chiral block
	    }
	  }  //Fine color columns
	}  //Fine color rows
      }  //Fine Spin

    }


    for(int s = 0; s < uvSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	arg.UV(parity,x_cb,s,c,ic_c) = UV[s][c];
      }
    }


  } // computeUV

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeUVCPU(Arg &arg) {

    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	for (int ic_c=0; ic_c < coarseColor; ic_c++) // coarse color
	  if (dir == QUDA_FORWARDS) // only for preconditioned clover is V != AV
	    computeUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
	  else
	    computeUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.AV, parity, x_cb, ic_c);
      } // c/b volume
    }   // parity
  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    if (dir == QUDA_FORWARDS) // only for preconditioned clover is V != AV
      computeUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
    else
      computeUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, arg.AV, parity, x_cb, ic_c);
  }

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x)
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void computeAV(Arg &arg, int parity, int x_cb, int ic_c) {

    complex<Float> AV[fineSpin][fineColor];

    for(int s = 0; s < fineSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	AV[s][c] = static_cast<Float>(0.0);
      }
    }

    for(int s = 0; s < fineSpin; s++) {  //Fine Spin
      const int s_c = arg.spin_map(s,parity); // Coarse spin

      //On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
      //in the same chiral block.
      for(int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column

	for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	  for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	    AV[s][ic] += arg.Cinv(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, ic_c);
	  }  //Fine color columns
	}  //Fine color rows
      }
    } //Fine Spin

    for (int s=0; s<fineSpin; s++) {
      for (int ic=0; ic<fineColor; ic++) {
	arg.AV(parity, x_cb, s, ic, ic_c) = AV[s][ic];
      }
    }

  } // computeAV

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeAVCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	for (int ic_c=0; ic_c < coarseColor; ic_c++) // coarse color
	  computeAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb, ic_c);
      } // c/b volume
    }   // parity
  }

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeAVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    computeAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb, ic_c);
  }

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x) for twisted-mass fermions
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void computeTMAV(Arg &arg, int parity, int x_cb, int v) {

    complex<Float> fp(1./(1.+arg.mu*arg.mu),-arg.mu/(1.+arg.mu*arg.mu));
    complex<Float> fm(1./(1.+arg.mu*arg.mu),+arg.mu/(1.+arg.mu*arg.mu));

    for(int s = 0; s < fineSpin/2; s++) {
      for(int c = 0; c < fineColor; c++) {
	arg.AV(parity,x_cb,s,c,v) = arg.V(parity,x_cb,s,c,v)*fp;
      }
    }

    for(int s = fineSpin/2; s < fineSpin; s++) {
      for(int c = 0; c < fineColor; c++) {
	arg.AV(parity,x_cb,s,c,v) = arg.V(parity,x_cb,s,c,v)*fm;
      }
    }

  } // computeTMAV

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeTMAVCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	for (int v=0; v<coarseColor; v++) // coarse color
	  computeTMAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb, v);
      } // c/b volume
    }   // parity
  }

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeTMAVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int v = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (v >= coarseColor) return;

    computeTMAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb, v);
  }

#ifdef DYNAMIC_CLOVER
#ifdef UGLY_DYNCLOV
#include<dyninv_clover_mg.cuh>
#else

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void applyCloverInv(Arg &arg, const complex<Float> UV[fineSpin][fineColor][coarseColor], int parity, int x_cb) {
    /* Applies the inverse of the clover term squared plus mu2 to the spinor */
    /* Compute (T^2 + mu2) first, then invert */
    /* We proceed by chiral blocks */

    for (int ch = 0; ch < 2; ch++) {	/* Loop over chiral blocks */
      Float diag[6], tmp[6];
      complex<Float> tri[15];	/* Off-diagonal components of the inverse clover term */

      /*	This macro avoid the infinitely long expansion of the tri products	*/
#define Cl(s1,c1,s2,c2) (arg.C(0, parity, x_cb, s1+2*ch, s2+2*ch, c1, c2))

      tri[0]  = Cl(0,1,0,0)*Cl(0,0,0,0).real() + Cl(0,1,0,1)*Cl(0,1,0,0) + Cl(0,1,0,2)*Cl(0,2,0,0) + Cl(0,1,1,0)*Cl(1,0,0,0) + Cl(0,1,1,1)*Cl(1,1,0,0) + Cl(0,1,1,2)*Cl(1,2,0,0);
      tri[1]  = Cl(0,2,0,0)*Cl(0,0,0,0).real() + Cl(0,2,0,2)*Cl(0,2,0,0) + Cl(0,2,0,1)*Cl(0,1,0,0) + Cl(0,2,1,0)*Cl(1,0,0,0) + Cl(0,2,1,1)*Cl(1,1,0,0) + Cl(0,2,1,2)*Cl(1,2,0,0);
      tri[3]  = Cl(1,0,0,0)*Cl(0,0,0,0).real() + Cl(1,0,1,0)*Cl(1,0,0,0) + Cl(1,0,0,1)*Cl(0,1,0,0) + Cl(1,0,0,2)*Cl(0,2,0,0) + Cl(1,0,1,1)*Cl(1,1,0,0) + Cl(1,0,1,2)*Cl(1,2,0,0);
      tri[6]  = Cl(1,1,0,0)*Cl(0,0,0,0).real() + Cl(1,1,1,1)*Cl(1,1,0,0) + Cl(1,1,0,1)*Cl(0,1,0,0) + Cl(1,1,0,2)*Cl(0,2,0,0) + Cl(1,1,1,0)*Cl(1,0,0,0) + Cl(1,1,1,2)*Cl(1,2,0,0);
      tri[10] = Cl(1,2,0,0)*Cl(0,0,0,0).real() + Cl(1,2,1,2)*Cl(1,2,0,0) + Cl(1,2,0,1)*Cl(0,1,0,0) + Cl(1,2,0,2)*Cl(0,2,0,0) + Cl(1,2,1,0)*Cl(1,0,0,0) + Cl(1,2,1,1)*Cl(1,1,0,0);

      tri[2]  = Cl(0,2,0,1)*Cl(0,1,0,1).real() + Cl(0,2,0,2)*Cl(0,2,0,1) + Cl(0,2,0,0)*Cl(0,0,0,1) + Cl(0,2,1,0)*Cl(1,0,0,1) + Cl(0,2,1,1)*Cl(1,1,0,1) + Cl(0,2,1,2)*Cl(1,2,0,1);
      tri[4]  = Cl(1,0,0,1)*Cl(0,1,0,1).real() + Cl(1,0,1,0)*Cl(1,0,0,1) + Cl(1,0,0,0)*Cl(0,0,0,1) + Cl(1,0,0,2)*Cl(0,2,0,1) + Cl(1,0,1,1)*Cl(1,1,0,1) + Cl(1,0,1,2)*Cl(1,2,0,1);
      tri[7]  = Cl(1,1,0,1)*Cl(0,1,0,1).real() + Cl(1,1,1,1)*Cl(1,1,0,1) + Cl(1,1,0,0)*Cl(0,0,0,1) + Cl(1,1,0,2)*Cl(0,2,0,1) + Cl(1,1,1,0)*Cl(1,0,0,1) + Cl(1,1,1,2)*Cl(1,2,0,1);
      tri[11] = Cl(1,2,0,1)*Cl(0,1,0,1).real() + Cl(1,2,1,2)*Cl(1,2,0,1) + Cl(1,2,0,0)*Cl(0,0,0,1) + Cl(1,2,0,2)*Cl(0,2,0,1) + Cl(1,2,1,0)*Cl(1,0,0,1) + Cl(1,2,1,1)*Cl(1,1,0,1);

      tri[5]  = Cl(1,0,0,2)*Cl(0,2,0,2).real() + Cl(1,0,1,0)*Cl(1,0,0,2) + Cl(1,0,0,0)*Cl(0,0,0,2) + Cl(1,0,0,1)*Cl(0,1,0,2) + Cl(1,0,1,1)*Cl(1,1,0,2) + Cl(1,0,1,2)*Cl(1,2,0,2);
      tri[8]  = Cl(1,1,0,2)*Cl(0,2,0,2).real() + Cl(1,1,1,1)*Cl(1,1,0,2) + Cl(1,1,0,0)*Cl(0,0,0,2) + Cl(1,1,0,1)*Cl(0,1,0,2) + Cl(1,1,1,0)*Cl(1,0,0,2) + Cl(1,1,1,2)*Cl(1,2,0,2);
      tri[12] = Cl(1,2,0,2)*Cl(0,2,0,2).real() + Cl(1,2,1,2)*Cl(1,2,0,2) + Cl(1,2,0,0)*Cl(0,0,0,2) + Cl(1,2,0,1)*Cl(0,1,0,2) + Cl(1,2,1,0)*Cl(1,0,0,2) + Cl(1,2,1,1)*Cl(1,1,0,2);

      tri[9]  = Cl(1,1,1,0)*Cl(1,0,1,0).real() + Cl(1,1,1,1)*Cl(1,1,1,0) + Cl(1,1,0,0)*Cl(0,0,1,0) + Cl(1,1,0,1)*Cl(0,1,1,0) + Cl(1,1,0,2)*Cl(0,2,1,0) + Cl(1,1,1,2)*Cl(1,2,1,0);
      tri[13] = Cl(1,2,1,0)*Cl(1,0,1,0).real() + Cl(1,2,1,2)*Cl(1,2,1,0) + Cl(1,2,0,0)*Cl(0,0,1,0) + Cl(1,2,0,1)*Cl(0,1,1,0) + Cl(1,2,0,2)*Cl(0,2,1,0) + Cl(1,2,1,1)*Cl(1,1,1,0);
      tri[14] = Cl(1,2,1,1)*Cl(1,1,1,1).real() + Cl(1,2,1,2)*Cl(1,2,1,1) + Cl(1,2,0,0)*Cl(0,0,1,1) + Cl(1,2,0,1)*Cl(0,1,1,1) + Cl(1,2,0,2)*Cl(0,2,1,1) + Cl(1,2,1,0)*Cl(1,0,1,1);

      diag[0] = arg.mu*arg.mu + Cl(0,0,0,0).real()*Cl(0,0,0,0).real() + norm(Cl(0,1,0,0)) + norm(Cl(0,2,0,0)) + norm(Cl(1,0,0,0)) + norm(Cl(1,1,0,0)) + norm(Cl(1,2,0,0));
      diag[1] = arg.mu*arg.mu + Cl(0,1,0,1).real()*Cl(0,1,0,1).real() + norm(Cl(0,0,0,1)) + norm(Cl(0,2,0,1)) + norm(Cl(1,0,0,1)) + norm(Cl(1,1,0,1)) + norm(Cl(1,2,0,1));
      diag[2] = arg.mu*arg.mu + Cl(0,2,0,2).real()*Cl(0,2,0,2).real() + norm(Cl(0,0,0,2)) + norm(Cl(0,1,0,2)) + norm(Cl(1,0,0,2)) + norm(Cl(1,1,0,2)) + norm(Cl(1,2,0,2));
      diag[3] = arg.mu*arg.mu + Cl(1,0,1,0).real()*Cl(1,0,1,0).real() + norm(Cl(0,0,1,0)) + norm(Cl(0,1,1,0)) + norm(Cl(0,2,1,0)) + norm(Cl(1,1,1,0)) + norm(Cl(1,2,1,0));
      diag[4] = arg.mu*arg.mu + Cl(1,1,1,1).real()*Cl(1,1,1,1).real() + norm(Cl(0,0,1,1)) + norm(Cl(0,1,1,1)) + norm(Cl(0,2,1,1)) + norm(Cl(1,0,1,1)) + norm(Cl(1,2,1,1));
      diag[5] = arg.mu*arg.mu + Cl(1,2,1,2).real()*Cl(1,2,1,2).real() + norm(Cl(0,0,1,2)) + norm(Cl(0,1,1,2)) + norm(Cl(0,2,1,2)) + norm(Cl(1,0,1,2)) + norm(Cl(1,1,1,2));

#undef Cl

      /*	INVERSION STARTS	*/

      for (int j=0; j<6; j++) {
        diag[j] = sqrt(diag[j]);
        tmp[j] = 1./diag[j];

        for (int k=j+1; k<6; k++) {
          int kj = k*(k-1)/2+j;
          tri[kj] *= tmp[j];
        }

        for(int k=j+1;k<6;k++){
          int kj=k*(k-1)/2+j;
          diag[k] -= (tri[kj] * conj(tri[kj])).real();
          for(int l=k+1;l<6;l++){
            int lj=l*(l-1)/2+j;
            int lk=l*(l-1)/2+k;
            tri[lk] -= tri[lj] * conj(tri[kj]);
          }
        }
      }

      /* Now use forward and backward substitution to construct inverse */
      complex<Float> v1[6];
      for (int k=0;k<6;k++) {
        for(int l=0;l<k;l++) v1[l] = complex<Float>(0.0, 0.0);

        /* Forward substitute */
        v1[k] = complex<Float>(tmp[k], 0.0);
        for(int l=k+1;l<6;l++){
          complex<Float> sum = complex<Float>(0.0, 0.0);
          for(int j=k;j<l;j++){
            int lj=l*(l-1)/2+j;
            sum -= tri[lj] * v1[j];
          }
          v1[l] = sum * tmp[l];
        }

        /* Backward substitute */
        v1[5] = v1[5] * tmp[5];
        for(int l=4;l>=k;l--){
          complex<Float> sum = v1[l];
          for(int j=l+1;j<6;j++){
            int jl=j*(j-1)/2+l;
            sum -= conj(tri[jl]) * v1[j];
          }
          v1[l] = sum * tmp[l];
        }

        /* Overwrite column k */
        diag[k] = v1[k].real();
        for(int l=k+1;l<6;l++){
          int lk=l*(l-1)/2+k;
          tri[lk] = v1[l];
        }
      }

      /*	Calculate the product for the current chiral block	*/

      //Then we calculate AV = Cinv UV, so  [AV = (C^2 + mu^2)^{-1} (Clover -/+ i mu)·Vector]
      //for in twisted-clover fermions, Cinv keeps (C^2 + mu^2)^{-1}

      for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  // Coarse Color
	for (int j=0; j<(fineSpin/2)*fineColor; j++) {	// This won't work for anything different than fineColor = 3, fineSpin = 4
	  int s = j / fineColor, ic = j % fineColor;

	  arg.AV(parity, x_cb, s+2*ch, ic, ic_c) += diag[j] * UV[s+2*ch][ic][ic_c];	// Diagonal clover

	  for (int k=0; k<j; k++) {
	    const int jk = j*(j-1)/2 + k;
	    const int s_col = k / fineColor, jc = k % fineColor;

	    arg.AV(parity, x_cb, s+2*ch, ic, ic_c) += tri[jk] * UV[s_col+2*ch][jc][ic_c]; // Off-diagonal
	  }

	  for (int k=j+1; k<(fineSpin/2)*fineColor; k++) {
	    int kj = k*(k-1)/2 + j;
	    int s_col = k / fineColor, jc = k % fineColor;

	    arg.AV(parity, x_cb, s+2*ch, ic, ic_c) += conj(tri[kj]) * UV[s_col+2*ch][jc][ic_c]; // Off-diagonal
	  }
	}
      }	// Coarse color
    } // Chirality
  }

#endif // UGLY_DYNCLOV

  template<typename Float, typename Arg>
  __device__ __host__ inline Float computeCloverInvMax(Arg &arg, int parity, int x_cb) {
    /* Applies the inverse of the clover term squared plus mu2 to the spinor */
    /* Compute (T^2 + mu2) first, then invert */
    /* We proceed by chiral blocks */

    Float max = 0.0;

    for (int ch = 0; ch < 2; ch++) {	/* Loop over chiral blocks */
      Float diag[6], tmp[6];
      complex<Float> tri[15];	/* Off-diagonal components of the inverse clover term */

      /*	This macro avoid the infinitely long expansion of the tri products	*/
#define Cl(s1,c1,s2,c2) (arg.C(0, parity, x_cb, s1+2*ch, s2+2*ch, c1, c2))

      tri[0]  = Cl(0,1,0,0)*Cl(0,0,0,0).real() + Cl(0,1,0,1)*Cl(0,1,0,0) + Cl(0,1,0,2)*Cl(0,2,0,0) + Cl(0,1,1,0)*Cl(1,0,0,0) + Cl(0,1,1,1)*Cl(1,1,0,0) + Cl(0,1,1,2)*Cl(1,2,0,0);
      tri[1]  = Cl(0,2,0,0)*Cl(0,0,0,0).real() + Cl(0,2,0,2)*Cl(0,2,0,0) + Cl(0,2,0,1)*Cl(0,1,0,0) + Cl(0,2,1,0)*Cl(1,0,0,0) + Cl(0,2,1,1)*Cl(1,1,0,0) + Cl(0,2,1,2)*Cl(1,2,0,0);
      tri[3]  = Cl(1,0,0,0)*Cl(0,0,0,0).real() + Cl(1,0,1,0)*Cl(1,0,0,0) + Cl(1,0,0,1)*Cl(0,1,0,0) + Cl(1,0,0,2)*Cl(0,2,0,0) + Cl(1,0,1,1)*Cl(1,1,0,0) + Cl(1,0,1,2)*Cl(1,2,0,0);
      tri[6]  = Cl(1,1,0,0)*Cl(0,0,0,0).real() + Cl(1,1,1,1)*Cl(1,1,0,0) + Cl(1,1,0,1)*Cl(0,1,0,0) + Cl(1,1,0,2)*Cl(0,2,0,0) + Cl(1,1,1,0)*Cl(1,0,0,0) + Cl(1,1,1,2)*Cl(1,2,0,0);
      tri[10] = Cl(1,2,0,0)*Cl(0,0,0,0).real() + Cl(1,2,1,2)*Cl(1,2,0,0) + Cl(1,2,0,1)*Cl(0,1,0,0) + Cl(1,2,0,2)*Cl(0,2,0,0) + Cl(1,2,1,0)*Cl(1,0,0,0) + Cl(1,2,1,1)*Cl(1,1,0,0);

      tri[2]  = Cl(0,2,0,1)*Cl(0,1,0,1).real() + Cl(0,2,0,2)*Cl(0,2,0,1) + Cl(0,2,0,0)*Cl(0,0,0,1) + Cl(0,2,1,0)*Cl(1,0,0,1) + Cl(0,2,1,1)*Cl(1,1,0,1) + Cl(0,2,1,2)*Cl(1,2,0,1);
      tri[4]  = Cl(1,0,0,1)*Cl(0,1,0,1).real() + Cl(1,0,1,0)*Cl(1,0,0,1) + Cl(1,0,0,0)*Cl(0,0,0,1) + Cl(1,0,0,2)*Cl(0,2,0,1) + Cl(1,0,1,1)*Cl(1,1,0,1) + Cl(1,0,1,2)*Cl(1,2,0,1);
      tri[7]  = Cl(1,1,0,1)*Cl(0,1,0,1).real() + Cl(1,1,1,1)*Cl(1,1,0,1) + Cl(1,1,0,0)*Cl(0,0,0,1) + Cl(1,1,0,2)*Cl(0,2,0,1) + Cl(1,1,1,0)*Cl(1,0,0,1) + Cl(1,1,1,2)*Cl(1,2,0,1);
      tri[11] = Cl(1,2,0,1)*Cl(0,1,0,1).real() + Cl(1,2,1,2)*Cl(1,2,0,1) + Cl(1,2,0,0)*Cl(0,0,0,1) + Cl(1,2,0,2)*Cl(0,2,0,1) + Cl(1,2,1,0)*Cl(1,0,0,1) + Cl(1,2,1,1)*Cl(1,1,0,1);

      tri[5]  = Cl(1,0,0,2)*Cl(0,2,0,2).real() + Cl(1,0,1,0)*Cl(1,0,0,2) + Cl(1,0,0,0)*Cl(0,0,0,2) + Cl(1,0,0,1)*Cl(0,1,0,2) + Cl(1,0,1,1)*Cl(1,1,0,2) + Cl(1,0,1,2)*Cl(1,2,0,2);
      tri[8]  = Cl(1,1,0,2)*Cl(0,2,0,2).real() + Cl(1,1,1,1)*Cl(1,1,0,2) + Cl(1,1,0,0)*Cl(0,0,0,2) + Cl(1,1,0,1)*Cl(0,1,0,2) + Cl(1,1,1,0)*Cl(1,0,0,2) + Cl(1,1,1,2)*Cl(1,2,0,2);
      tri[12] = Cl(1,2,0,2)*Cl(0,2,0,2).real() + Cl(1,2,1,2)*Cl(1,2,0,2) + Cl(1,2,0,0)*Cl(0,0,0,2) + Cl(1,2,0,1)*Cl(0,1,0,2) + Cl(1,2,1,0)*Cl(1,0,0,2) + Cl(1,2,1,1)*Cl(1,1,0,2);

      tri[9]  = Cl(1,1,1,0)*Cl(1,0,1,0).real() + Cl(1,1,1,1)*Cl(1,1,1,0) + Cl(1,1,0,0)*Cl(0,0,1,0) + Cl(1,1,0,1)*Cl(0,1,1,0) + Cl(1,1,0,2)*Cl(0,2,1,0) + Cl(1,1,1,2)*Cl(1,2,1,0);
      tri[13] = Cl(1,2,1,0)*Cl(1,0,1,0).real() + Cl(1,2,1,2)*Cl(1,2,1,0) + Cl(1,2,0,0)*Cl(0,0,1,0) + Cl(1,2,0,1)*Cl(0,1,1,0) + Cl(1,2,0,2)*Cl(0,2,1,0) + Cl(1,2,1,1)*Cl(1,1,1,0);
      tri[14] = Cl(1,2,1,1)*Cl(1,1,1,1).real() + Cl(1,2,1,2)*Cl(1,2,1,1) + Cl(1,2,0,0)*Cl(0,0,1,1) + Cl(1,2,0,1)*Cl(0,1,1,1) + Cl(1,2,0,2)*Cl(0,2,1,1) + Cl(1,2,1,0)*Cl(1,0,1,1);

      diag[0] = arg.mu*arg.mu + Cl(0,0,0,0).real()*Cl(0,0,0,0).real() + norm(Cl(0,1,0,0)) + norm(Cl(0,2,0,0)) + norm(Cl(1,0,0,0)) + norm(Cl(1,1,0,0)) + norm(Cl(1,2,0,0));
      diag[1] = arg.mu*arg.mu + Cl(0,1,0,1).real()*Cl(0,1,0,1).real() + norm(Cl(0,0,0,1)) + norm(Cl(0,2,0,1)) + norm(Cl(1,0,0,1)) + norm(Cl(1,1,0,1)) + norm(Cl(1,2,0,1));
      diag[2] = arg.mu*arg.mu + Cl(0,2,0,2).real()*Cl(0,2,0,2).real() + norm(Cl(0,0,0,2)) + norm(Cl(0,1,0,2)) + norm(Cl(1,0,0,2)) + norm(Cl(1,1,0,2)) + norm(Cl(1,2,0,2));
      diag[3] = arg.mu*arg.mu + Cl(1,0,1,0).real()*Cl(1,0,1,0).real() + norm(Cl(0,0,1,0)) + norm(Cl(0,1,1,0)) + norm(Cl(0,2,1,0)) + norm(Cl(1,1,1,0)) + norm(Cl(1,2,1,0));
      diag[4] = arg.mu*arg.mu + Cl(1,1,1,1).real()*Cl(1,1,1,1).real() + norm(Cl(0,0,1,1)) + norm(Cl(0,1,1,1)) + norm(Cl(0,2,1,1)) + norm(Cl(1,0,1,1)) + norm(Cl(1,2,1,1));
      diag[5] = arg.mu*arg.mu + Cl(1,2,1,2).real()*Cl(1,2,1,2).real() + norm(Cl(0,0,1,2)) + norm(Cl(0,1,1,2)) + norm(Cl(0,2,1,2)) + norm(Cl(1,0,1,2)) + norm(Cl(1,1,1,2));

#undef Cl

      /*	INVERSION STARTS	*/

      for (int j=0; j<6; j++) {
        diag[j] = sqrt(diag[j]);
        tmp[j] = 1./diag[j];

        for (int k=j+1; k<6; k++) {
          int kj = k*(k-1)/2+j;
          tri[kj] *= tmp[j];
        }

        for(int k=j+1;k<6;k++){
          int kj=k*(k-1)/2+j;
          diag[k] -= (tri[kj] * conj(tri[kj])).real();
          for(int l=k+1;l<6;l++){
            int lj=l*(l-1)/2+j;
            int lk=l*(l-1)/2+k;
            tri[lk] -= tri[lj] * conj(tri[kj]);
          }
        }
      }

      /* Now use forward and backward substitution to construct inverse */
      complex<Float> v1[6];
      for (int k=0;k<6;k++) {
        for(int l=0;l<k;l++) v1[l] = complex<Float>(0.0, 0.0);

        /* Forward substitute */
        v1[k] = complex<Float>(tmp[k], 0.0);
        for(int l=k+1;l<6;l++){
          complex<Float> sum = complex<Float>(0.0, 0.0);
          for(int j=k;j<l;j++){
            int lj=l*(l-1)/2+j;
            sum -= tri[lj] * v1[j];
          }
          v1[l] = sum * tmp[l];
        }

        /* Backward substitute */
        v1[5] = v1[5] * tmp[5];
        for(int l=4;l>=k;l--){
          complex<Float> sum = v1[l];
          for(int j=l+1;j<6;j++){
            int jl=j*(j-1)/2+l;
            sum -= conj(tri[jl]) * v1[j];
          }
          v1[l] = sum * tmp[l];
        }

        /* Overwrite column k */
        diag[k] = v1[k].real();
        for(int l=k+1;l<6;l++){
          int lk=l*(l-1)/2+k;
          tri[lk] = v1[l];
        }
      }

      for (int i=0; i<6; i++) max = max > abs(diag[i]) ? max : abs(diag[i]);
      for (int i=0; i<15; i++) max = max > abs(tri[i]) ? max : abs(tri[i]);
    } // Chirality

    return max;
  }

  template<typename Float, typename Arg>
  void ComputeCloverInvMaxCPU(Arg &arg) {
    Float max = 0.0;
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for reduction(max:max)
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	Float max_x = computeCloverInvMax<Float,Arg>(arg, parity, x_cb);
	max = max > max_x ? max : max_x;
      } // c/b volume
    }   // parity
    arg.max_h = max;
  }

  template<typename Float, typename Arg>
  __global__ void ComputeCloverInvMaxGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    arg.max_d[x_cb*2+parity] = computeCloverInvMax<Float,Arg>(arg, parity, x_cb);
  }
#endif // DYNAMIC_CLOVER

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x) for twisted-clover fermions
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void computeTMCAV(Arg &arg, int parity, int x_cb) {

    complex<Float> mu(0.,arg.mu);
    complex<Float> UV[fineSpin][fineColor][coarseColor];

    for (int s = 0; s < fineSpin; s++)
      for (int c = 0; c < fineColor; c++)
	for (int v = 0; v < coarseColor; v++)
	  UV[s][c][v] = static_cast<Float>(0.0);

    //First we store in UV the product [(Clover -/+ i mu)·Vector]
    for(int s = 0; s < fineSpin; s++) {  //Fine Spin
      const int s_c = arg.spin_map(s,parity); // Coarse Spin

      //On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
      //in the same chiral block.
      for(int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column

	for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	  for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	    for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	      UV[s][ic][ic_c] += arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, ic_c);
	    }  //Fine color columns
	  }  //Fine color rows
	} //Coarse color
      }
    } //Fine Spin

    for(int s = 0; s < fineSpin/2; s++) {  //Fine Spin
      for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color
	  UV[s][ic][ic_c] -= mu * arg.V(parity, x_cb, s, ic, ic_c);
	}  //Fine color
      } //Coarse color
    } //Fine Spin

    for(int s = fineSpin/2; s < fineSpin; s++) {  //Fine Spin
      for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	for(int ic = 0; ic < fineColor; ic++) { //Fine Color
	  UV[s][ic][ic_c] += mu * arg.V(parity, x_cb, s, ic, ic_c);
	}  //Fine color
      } //Coarse color
    } //Fine Spin

    for (int s = 0; s < fineSpin; s++)
      for (int c = 0; c < fineColor; c++)
	for (int v = 0; v < coarseColor; v++)
	  arg.AV(parity,x_cb,s,c,v) = static_cast<Float>(0.0);

#ifndef	DYNAMIC_CLOVER
    //Then we calculate AV = Cinv UV, so  [AV = (C^2 + mu^2)^{-1} (Clover -/+ i mu)·Vector]
    //for in twisted-clover fermions, Cinv keeps (C^2 + mu^2)^{-1}
    for(int s = 0; s < fineSpin; s++) {  //Fine Spin
      const int s_c = arg.spin_map(s,parity); // Coarse Spin

      //On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
      //in the same chiral block.
      for(int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column

	for(int ic_c = 0; ic_c < coarseColor; ic_c++) {  //Coarse Color
	  for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
	    for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
	      arg.AV(parity, x_cb, s, ic, ic_c) +=
		arg.Cinv(0, parity, x_cb, s, s_col, ic, jc) * UV[s_col][jc][ic_c];
	    }  //Fine color columns
	  }  //Fine color rows
	} //Coarse color
      }
    } //Fine Spin
#else
    applyCloverInv<Float,fineSpin,fineColor,coarseColor,Arg>(arg, UV, parity, x_cb);
#endif
  } // computeTMCAV

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeTMCAVCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
	computeTMCAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb);
      } // c/b volume
    }   // parity
  }

  template<typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeTMCAVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeTMCAV<Float,fineSpin,fineColor,coarseColor,Arg>(arg, parity, x_cb);
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
  template <bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg, typename Gamma>
  __device__ __host__ inline void multiplyVUV(complex<Float> vuv[], const Arg &arg, const Gamma &gamma, int parity, int x_cb, int ic_c, int jc_c) {

#pragma unroll
    for (int i=0; i<coarseSpin*coarseSpin; i++) vuv[i] = 0.0;

    if (!from_coarse) { // fine grid is top level

#pragma unroll
      for (int s = 0; s < fineSpin; s++) { //Loop over fine spin

	//Spin part of the color matrix.  Will always consist
	//of two terms - diagonal and off-diagonal part of
	//P_mu = (1+/-\gamma_mu)

	const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index

	//Use Gamma to calculate off-diagonal coupling and
	//column index.  Diagonal coupling is always 1.
	// If computing the backwards (forwards) direction link then
	// we desire the positive (negative) projector

	int s_col;
	const complex<Float> coupling = gamma.getrowelem(s, s_col);
	const int s_c_col = arg.spin_map(s_col,parity); // Coarse spin col index

#pragma unroll
	for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color
          if (dir == QUDA_BACKWARDS) {
            // here UV is really UAV
	    //Diagonal Spin
	    vuv[s_c_row*coarseSpin+s_c_row] += conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s, ic, jc_c);

	    //Off-diagonal Spin (backward link / positive projector applied)
	    vuv[s_c_row*coarseSpin+s_c_col] += coupling * conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s_col, ic, jc_c);
	  } else {
	    //Diagonal Spin
	    vuv[s_c_row*coarseSpin+s_c_row] += conj(arg.AV(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s, ic, jc_c);

	    //Off-diagonal Spin (forward link / negative projector applied)
	    vuv[s_c_row*coarseSpin+s_c_col] -= coupling * conj(arg.AV(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s_col, ic, jc_c);
	  }
	} //Fine color
      }

    } else { // fine grid operator is a coarse operator

#pragma unroll
      for (int s_col=0; s_col<fineSpin; s_col++) { // which chiral block
#pragma unroll
	for (int s = 0; s < fineSpin; s++) {
#pragma unroll
	  for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
	    vuv[s*coarseSpin+s_col] += conj(arg.AV(parity, x_cb, s, ic, ic_c)) * arg.UV(parity, x_cb, s_col*fineSpin+s, ic, jc_c);
	  } //Fine color
	} //Fine spin
      }

    } // from_coarse

  }

#ifndef SWIZZLE
  template<typename Arg>
  __device__ __host__ inline int virtualThreadIdx(const Arg &arg) {
    constexpr int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int warp_lane = threadIdx.x % warp_size;
    int tx = warp_id * (warp_size / arg.aggregates_per_block) + warp_lane / arg.aggregates_per_block;
    return tx;
  }

  template<typename Arg>
  __device__ __host__ inline int virtualBlockDim(const Arg &arg) {
    int block_dim_x = blockDim.x / arg.aggregates_per_block;
    return block_dim_x;
  }

  template<typename Arg>
  __device__ __host__ inline int coarseIndex(const Arg &arg) {
    constexpr int warp_size = 32;
    int warp_lane = threadIdx.x % warp_size;
    int x_coarse = blockIdx.x*arg.aggregates_per_block + warp_lane % arg.aggregates_per_block;
    return x_coarse;
  }

#else
  template<typename Arg>
  __device__ __host__ inline int virtualThreadIdx(const Arg &arg) { return threadIdx.x; }

  template<typename Arg>
  __device__ __host__ inline int virtualBlockDim(const Arg &arg) { return blockDim.x; }

  template<typename Arg>
  __device__ __host__ inline int coarseIndex(const Arg &arg) {
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    int x_coarse = blockIdx.x;
    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
    return x_coarse;
  }
#endif

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg, typename Gamma>
  __device__ __host__ void computeVUV(Arg &arg, const Gamma &gamma, int parity, int x_cb, int c_row, int c_col, int parity_coarse_, int coarse_x_cb_) {

    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

#if defined(SHARED_ATOMIC) && __CUDA_ARCH__
    int coarse_parity = parity_coarse_;
    int coarse_x_cb = coarse_x_cb_;
#else
    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];
#endif

    complex<Float> vuv[coarseSpin*coarseSpin];
    multiplyVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, gamma, parity, x_cb, c_row, c_col);

    constexpr int dim_index = (dir == QUDA_BACKWARDS) ? dim : dim + 4;
#if defined(SHARED_ATOMIC) && __CUDA_ARCH__
    __shared__ complex<storeType> X[4][coarseSpin][coarseSpin];
    __shared__ complex<storeType> Y[4][coarseSpin][coarseSpin];
    int x_ = coarse_x_cb%arg.aggregates_per_block;

    if (virtualThreadIdx(arg) == 0 && threadIdx.y == 0) {
      for (int s_row = 0; s_row<coarseSpin; s_row++) for (int s_col = 0; s_col<coarseSpin; s_col++)
	{ Y[x_][s_row][s_col] = 0; X[x_][s_row][s_col] = 0; }
    }

    __syncthreads();

    if (!isDiagonal) {
#pragma unroll
      for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	  if (gauge::fixed_point<Float,storeType>()) {
	    Float scale = arg.Y_atomic.accessor.scale;
	    complex<storeType> a(round(scale * vuv[s_row*coarseSpin+s_col].real()),
				 round(scale * vuv[s_row*coarseSpin+s_col].imag()));
	    atomicAdd(&Y[x_][s_row][s_col],a);
	  } else {
	    atomicAdd(&Y[x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
	  }
	}
      }
    } else {
#pragma unroll
      for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	  vuv[s_row*coarseSpin+s_col] *= -arg.kappa;
	  if (gauge::fixed_point<Float,storeType>()) {
	    Float scale = arg.X_atomic.accessor.scale;
	    complex<storeType> a(round(scale * vuv[s_row*coarseSpin+s_col].real()),
				 round(scale * vuv[s_row*coarseSpin+s_col].imag()));
	    atomicAdd(&X[x_][s_row][s_col],a);
	  } else {
	    atomicAdd(&X[x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
	  }
	}
      }
    }

    __syncthreads();

    if (virtualThreadIdx(arg)==0 && threadIdx.y==0) {

#pragma unroll
      for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	  arg.Y_atomic(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) = Y[x_][s_row][s_col];
	}
      }

      if (dir == QUDA_BACKWARDS) {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    arg.X_atomic(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row) += conj(X[x_][s_row][s_col]);
	  }
	}
      } else {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    arg.X_atomic(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += X[x_][s_row][s_col];
	  }
	}
      }

      if (!arg.bidirectional) {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    if (s_row == s_col) arg.X_atomic(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += X[x_][s_row][s_col];
	    else arg.X_atomic(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) -= X[x_][s_row][s_col];
	  }
	}
      }

    }

#else

    if (!isDiagonal) {
#pragma unroll
      for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	  arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv[s_row*coarseSpin+s_col]);
	}
      }
    } else {

      for (int s2=0; s2<coarseSpin*coarseSpin; s2++) vuv[s2] *= -arg.kappa;

      if (dir == QUDA_BACKWARDS) {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row,conj(vuv[s_row*coarseSpin+s_col]));
	  }
	}
      } else {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv[s_row*coarseSpin+s_col]);
	  }
	}
      }

      if (!arg.bidirectional) {
#pragma unroll
	for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
	  for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
	    const Float sign = (s_row == s_col) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);
	    arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,sign*vuv[s_row*coarseSpin+s_col]);
	  }
	}
      }

    }
#endif

  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg, typename Gamma>
  void ComputeVUVCPU(Arg arg, const Gamma &gamma) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
	for (int c_row=0; c_row<coarseColor; c_row++)
	  for (int c_col=0; c_col<coarseColor; c_col++)
	    computeVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, gamma, parity, x_cb, c_row, c_col, 0, 0);
      } // c/b volume
    } // parity
  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg, typename Gamma>
  __global__ void ComputeVUVGPU(Arg arg, const Gamma gamma) {

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*coarseColor) return;

#ifdef SHARED_ATOMIC
    int c_col = parity_c_col / 2; // coarse color col index
    int parity = parity_c_col % 2;

    int block_dim_x = virtualBlockDim(arg);
    int thread_idx_x = virtualThreadIdx(arg);
    int x_coarse = coarseIndex(arg);

    int parity_coarse = x_coarse >= arg.coarseVolumeCB ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

    // obtain fine index from this look up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on (optionally swizzled to improve cache efficiency)
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered

    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * block_dim_x + thread_idx_x];
    int x_cb = x_fine - parity*arg.fineVolumeCB;
#else
    int x_coarse_cb = 0;
    int parity_coarse = 0;

    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int c_col = parity_c_col % coarseColor; // coarse color col index
    int parity = parity_c_col / coarseColor;
#endif

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    if (c_row >= coarseColor) return;

    computeVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, gamma, parity, x_cb, c_row, c_col, parity_coarse, x_coarse_cb);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeYreverse(Arg &arg, int parity, int x_cb, int ic_c) {
    auto &Y = arg.Y_atomic;

    for (int d=0; d<4; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column

	  const Float sign = (s_row == s_col) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

	  for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
	    Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = sign*Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
	  } //Color column
	} //Spin column
      } //Spin row

    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
	}
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
  }

  template<bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ void computeCoarseClover(Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {

    const int nDim = 4;

    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    getCoords(coord, x_cb, arg.x_size, parity);
    for (int d=0; d<nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

    coord[0] /= 2;

    complex<Float> X[coarseSpin*coarseSpin];
    for (int i=0; i<coarseSpin*coarseSpin; i++) X[i] = 0.0;

    if (!from_coarse) {
      //If Nspin = 4, then the clover term has structure C_{\mu\nu} = \gamma_{\mu\nu}C^{\mu\nu}
      for(int s = 0; s < fineSpin; s++) { //Loop over fine spin row
	const int s_c = arg.spin_map(s,parity);
	//On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
	//in the same chiral block.
	for(int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column
	  //for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
          //for(int jc_c = 0; jc_c < coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
		for(int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
		  X[s_c*coarseSpin + s_c] +=
		    conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
		} //Fine color column
	      }  //Fine color row
              //} //Coarse Color column
	    //} //Coarse Color row
	}  //Fine spin column
      } //Fine spin
    } else {
      //If Nspin != 4, then spin structure is a dense matrix and there is now spin aggregation
      //N.B. assumes that no further spin blocking is done in this case.
      for(int s = 0; s < fineSpin; s++) { //Loop over spin row
	for(int s_col = 0; s_col < fineSpin; s_col++) { //Loop over spin column
	  //for(int ic_c = 0; ic_c < coarseColor; ic_c++) { //Coarse Color row
          //for(int jc_c = 0; jc_c <coarseColor; jc_c++) { //Coarse Color column
	      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
		for(int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
		  X[s*coarseSpin + s_col] +=
		    conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
		} //Fine color column
	      }  //Fine color row
              //} //Coarse Color column
	    //} //Coarse Color row
	}  //Fine spin column
      } //Fine spin
    }

    for (int si = 0; si < coarseSpin; si++) {
      for (int sj = 0; sj < coarseSpin; sj++) {
	//for (int ic = 0; ic < coarseColor; ic++) {
        //for (int jc = 0; jc < coarseColor; jc++) {
	    arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,si,sj,ic_c,jc_c,X[si*coarseSpin+sj]);
            //}
	  //}
      }
    }

  }

  template <bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  void ComputeCoarseCloverCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int jc_c=0; jc_c<coarseColor; jc_c++) {
          for (int ic_c=0; ic_c<coarseColor; ic_c++) {
            computeCoarseClover<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg, parity, x_cb, ic_c, jc_c);
          }
        }
      } // c/b volume
    } // parity
  }

  template <bool from_coarse, typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeCoarseCloverGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y; // parity and color column
    if (parity_c_col >= 2*coarseColor) return;
    int jc_c = parity_c_col % coarseColor; // coarse color col index
    int parity = parity_c_col / coarseColor;

    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    computeCoarseClover<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg, parity, x_cb, ic_c, jc_c);
  }



  //Adds the identity matrix to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseDiagonalCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
         for(int c = 0; c < nColor; c++) { //Color
	   arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(1.0,0.0);
         } //Color
        } //Spin
      } // x_cb
    } //parity
   }


  //Adds the identity matrix to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void AddCoarseDiagonalGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    for(int s = 0; s < nSpin; s++) { //Spin
      for(int c = 0; c < nColor; c++) { //Color
	arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(1.0,0.0);
      } //Color
    } //Spin
   }

  //Adds the twisted-mass term to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseTmDiagonalCPU(Arg &arg) {

    const complex<Float> mu(0., arg.mu*arg.mu_factor);

    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for(int s = 0; s < nSpin/2; s++) { //Spin
          for(int c = 0; c < nColor; c++) { //Color
            arg.X_atomic(0,parity,x_cb,s,s,c,c) += mu;
          } //Color
	} //Spin
	for(int s = nSpin/2; s < nSpin; s++) { //Spin
          for(int c = 0; c < nColor; c++) { //Color
            arg.X_atomic(0,parity,x_cb,s,s,c,c) -= mu;
          } //Color
	} //Spin
      } // x_cb
    } //parity
  }

  //Adds the twisted-mass term to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void AddCoarseTmDiagonalGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    const complex<Float> mu(0., arg.mu*arg.mu_factor);

    for(int s = 0; s < nSpin/2; s++) { //Spin
      for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
       arg.X_atomic(0,parity,x_cb,s,s,ic_c,ic_c) += mu;
      } //Color
    } //Spin
    for(int s = nSpin/2; s < nSpin; s++) { //Spin
      for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color
       arg.X_atomic(0,parity,x_cb,s,s,ic_c,ic_c) -= mu;
      } //Color
    } //Spin
   }

  /**
   * Convert the field from the atomic format to the required computation format, e.g. fixed point to floating point
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void convert(Arg &arg, int parity, int x_cb, int c_row, int c_col) {

    const auto &Yin = arg.Y_atomic;
    const auto &Xin = arg.X_atomic;

    for (int d=0; d<8; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	  complex<Float> Y = Yin(d,parity,x_cb,s_row,s_col,c_row,c_col);
	  arg.Y(d,parity,x_cb,s_row,s_col,c_row,c_col) = Y;
	} //Spin column
      } //Spin row
    } // dimension

    for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
      for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	arg.X(0,parity,x_cb,s_row,s_col,c_row,c_col) = Xin(0,parity,x_cb,s_row,s_col,c_row,c_col);
      }
    }

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ConvertCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for(int c_row = 0; c_row < nColor; c_row++) { //Color row
	  for(int c_col = 0; c_col < nColor; c_col++) { //Color column
	    convert<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
	  }
	}
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ConvertGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;

    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;

    convert<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_AV,
    COMPUTE_TMAV,
    COMPUTE_TMCAV,
    COMPUTE_CLOVER_INV_MAX,
    COMPUTE_VUV,
    COMPUTE_COARSE_CLOVER,
    COMPUTE_REVERSE_Y,
    COMPUTE_DIAGONAL,
    COMPUTE_TMDIAGONAL,
    COMPUTE_CONVERT,
    COMPUTE_INVALID
  };

  template <bool from_coarse, typename Float, int fineSpin,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateY : public TunableVectorYZ {
  public:

    template <int dim> using Gamma_ = Gamma<Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim>;

  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    GaugeField &Y;
    GaugeField &X;

    int dim;
    QudaDirection dir;
    ComputeType type;

    long long flops() const
    {
      long long flops_ = 0;
      switch (type) {
      case COMPUTE_UV:
	// when fine operator is coarse take into account that the link matrix has spin dependence
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * coarseColor * fineColor * fineColor * (!from_coarse ? 1 : fineSpin);
	break;
      case COMPUTE_AV:
      case COMPUTE_TMAV:
	// # chiral blocks * size of chiral block * number of null space vectors
	flops_ = 2l * arg.fineVolumeCB * 8 * (fineSpin/2) * (fineSpin/2) * (fineSpin/2) * fineColor * fineColor * coarseColor;
	break;
      case COMPUTE_TMCAV:
	// # Twice chiral blocks * size of chiral block * number of null space vectors
	flops_ = 4l * arg.fineVolumeCB * 8 * (fineSpin/2) * (fineSpin/2) * (fineSpin/2) * fineColor * fineColor * coarseColor;
	break;
      case COMPUTE_VUV:
	// when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case COMPUTE_COARSE_CLOVER:
	// when the fine operator is truly fine the clover multiplication is block sparse which halves the number of operations
	flops_ = 2l * arg.fineVolumeCB * 8 * fineSpin * fineSpin * coarseColor * coarseColor * fineColor * fineColor / (!from_coarse ? coarseSpin : 1);
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_CONVERT:
      case COMPUTE_CLOVER_INV_MAX: // FIXME
	// no floating point operations
	flops_ = 0;
	break;
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
	// read addition on the diagonal
	flops_ = 2l * arg.coarseVolumeCB*coarseSpin*coarseColor;
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
      case COMPUTE_UV:
	bytes_ = arg.UV.Bytes() + arg.V.Bytes() + 2*arg.U.Bytes()*coarseColor;
	break;
      case COMPUTE_AV:
	bytes_ = arg.AV.Bytes() + arg.V.Bytes() + 2*arg.C.Bytes();
	break;
      case COMPUTE_TMAV:
	bytes_ = arg.AV.Bytes() + arg.V.Bytes();
	break;
      case COMPUTE_TMCAV:
	bytes_ = arg.AV.Bytes() + arg.V.Bytes() + arg.UV.Bytes() + 4*arg.C.Bytes(); // Two clover terms and more temporary storage
	break;
      case COMPUTE_CLOVER_INV_MAX: // FIXME
	bytes_ = 2*arg.C.Bytes(); // read both parities of the clover field
	break;
      case COMPUTE_VUV:
	bytes_ = 2*arg.Y.Bytes() + (arg.bidirectional ? 1 : 2) * 2*arg.X.Bytes() + arg.UV.Bytes() + arg.V.Bytes();
	break;
      case COMPUTE_COARSE_CLOVER:
	bytes_ = 2*arg.X.Bytes() + 2*arg.C.Bytes() + arg.V.Bytes(); // 2 from parity
	break;
      case COMPUTE_REVERSE_Y:
	bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
	bytes_ = 2*2*arg.X.Bytes(); // 2 from i/o, 2 from parity
	break;
      case COMPUTE_CONVERT:
	bytes_ = 2*(arg.X.Bytes() + arg.X_atomic.Bytes() + 8*(arg.Y.Bytes() + arg.Y_atomic.Bytes()));
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      return bytes_;
    }

    unsigned int minThreads() const {
      unsigned int threads = 0;
      switch (type) {
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_VUV:
      case COMPUTE_COARSE_CLOVER:
	threads = arg.fineVolumeCB;
	break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
      case COMPUTE_CONVERT:
	threads = arg.coarseVolumeCB;
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
      return threads;
    }

    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    CalculateY(Arg &arg, const ColorSpinorField &meta, GaugeField &Y, GaugeField &X)
      : TunableVectorYZ(2,1), arg(arg), type(COMPUTE_INVALID),
	meta(meta), Y(Y), X(X), dim(0), dir(QUDA_BACKWARDS)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux,comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }
    virtual ~CalculateY() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

	if (type == COMPUTE_UV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==1) ComputeUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==2) ComputeUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	    else if (dim==3) ComputeUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == COMPUTE_AV) {

	  if (from_coarse) errorQuda("ComputeAV should only be called from the fine grid");
	  ComputeAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);

	} else if (type == COMPUTE_TMAV) {

	  if (from_coarse) errorQuda("ComputeTMAV should only be called from the fine grid");
	  ComputeTMAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);

	} else if (type == COMPUTE_TMCAV) {

	  if (from_coarse) errorQuda("ComputeTMCAV should only be called from the fine grid");
	  ComputeTMCAVCPU<Float,fineSpin,fineColor,coarseColor>(arg);

	} else if (type == COMPUTE_CLOVER_INV_MAX) {

	  if (from_coarse) errorQuda("ComputeInvCloverMax should only be called from the fine grid");
#ifdef DYNAMIC_CLOVER
	  ComputeCloverInvMaxCPU<Float>(arg);
#else
	  errorQuda("ComputeInvCloverMax only enabled with dynamic clover");
#endif

	} else if (type == COMPUTE_VUV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeVUVCPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<0>());
	    else if (dim==1) ComputeVUVCPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<1>());
	    else if (dim==2) ComputeVUVCPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<2>());
	    else if (dim==3) ComputeVUVCPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<3>());
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeVUVCPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<0>());
	    else if (dim==1) ComputeVUVCPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<1>());
	    else if (dim==2) ComputeVUVCPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<2>());
	    else if (dim==3) ComputeVUVCPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor>(arg, Gamma_<3>());
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == COMPUTE_COARSE_CLOVER) {

	  ComputeCoarseCloverCPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>(arg);

	} else if (type == COMPUTE_REVERSE_Y) {

	  ComputeYReverseCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_DIAGONAL) {

	  AddCoarseDiagonalCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_TMDIAGONAL) {

          AddCoarseTmDiagonalCPU<Float,coarseSpin,coarseColor>(arg);

	} else if (type == COMPUTE_CONVERT) {

	  ConvertCPU<Float,coarseSpin,coarseColor>(arg);

	} else {
	  errorQuda("Undefined compute type %d", type);
	}
      } else {

	if (type == COMPUTE_UV) {

	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeUVGPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeUVGPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeUVGPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeUVGPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeUVGPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==1) ComputeUVGPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==2) ComputeUVGPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	    else if (dim==3) ComputeUVGPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

	} else if (type == COMPUTE_AV) {

	  if (from_coarse) errorQuda("ComputeAV should only be called from the fine grid");
	  ComputeAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_TMAV) {

	  if (from_coarse) errorQuda("ComputeTMAV should only be called from the fine grid");
	  ComputeTMAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_TMCAV) {

	  if (from_coarse) errorQuda("ComputeTMCAV should only be called from the fine grid");
	  ComputeTMCAVGPU<Float,fineSpin,fineColor,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_CLOVER_INV_MAX) {

	  if (from_coarse) errorQuda("ComputeCloverInvMax should only be called from the fine grid");
#ifdef DYNAMIC_CLOVER
	  arg.max_d = static_cast<Float*>(pool_device_malloc(2 * arg.fineVolumeCB));
	  ComputeCloverInvMaxGPU<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  thrust_allocator alloc;
	  thrust::device_ptr<Float> ptr(arg.max_d);
	  arg.max_h = thrust::reduce(thrust::cuda::par(alloc), ptr, ptr+2*arg.fineVolumeCB, static_cast<Float>(0.0), thrust::maximum<Float>());
	  pool_device_free(arg.max_d);
#else
	  errorQuda("ComputeCloverInvMax only enabled with dynamic clover");
#endif

	} else if (type == COMPUTE_VUV) {
#ifndef SHARED_ATOMIC
	  //tp.grid.y = 2*coarseColor;
#else
	  tp.block.y = 2;
	  tp.grid.y = coarseColor;
	  tp.grid.z = coarseColor;
	  arg.swizzle = tp.aux.x;

	  arg.aggregates_per_block = tp.aux.y;
	  tp.block.x *= tp.aux.y;
	  tp.grid.x /= tp.aux.y;
#endif
	  if (dir == QUDA_BACKWARDS) {
	    if      (dim==0) ComputeVUVGPU<from_coarse,Float,0,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<0>());
	    else if (dim==1) ComputeVUVGPU<from_coarse,Float,1,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<1>());
	    else if (dim==2) ComputeVUVGPU<from_coarse,Float,2,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<2>());
	    else if (dim==3) ComputeVUVGPU<from_coarse,Float,3,QUDA_BACKWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<3>());
	  } else if (dir == QUDA_FORWARDS) {
	    if      (dim==0) ComputeVUVGPU<from_coarse,Float,0,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<0>());
	    else if (dim==1) ComputeVUVGPU<from_coarse,Float,1,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<1>());
	    else if (dim==2) ComputeVUVGPU<from_coarse,Float,2,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<2>());
	    else if (dim==3) ComputeVUVGPU<from_coarse,Float,3,QUDA_FORWARDS,fineSpin,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg, Gamma_<3>());
	  } else {
	    errorQuda("Undefined direction %d", dir);
	  }

#ifdef SHARED_ATOMIC
	  tp.block.x /= tp.aux.y;
	  tp.grid.x *= tp.aux.y;
#endif

	} else if (type == COMPUTE_COARSE_CLOVER) {

	  ComputeCoarseCloverGPU<from_coarse,Float,fineSpin,coarseSpin,fineColor,coarseColor>
	    <<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_REVERSE_Y) {

	  ComputeYReverseGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_DIAGONAL) {

	  AddCoarseDiagonalGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_TMDIAGONAL) {

          AddCoarseTmDiagonalGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

	} else if (type == COMPUTE_CONVERT) {

	  tp.grid.y = 2*coarseColor;
	  ConvertGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

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
      case COMPUTE_VUV:
#ifdef SHARED_ATOMIC
	resizeVector(1,1);
#else
	resizeVector(2*coarseColor,coarseColor);
#endif
	break;
      case COMPUTE_COARSE_CLOVER: // no shared atomic version so keep separate from above
	resizeVector(2*coarseColor,coarseColor);
        break;
      case COMPUTE_CONVERT:
	resizeVector(1,coarseColor);
	break;
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_REVERSE_Y:
	resizeVector(2,coarseColor);
	break;
      default:
	resizeVector(2,1);
	break;
      }
      // do not tune spatial block size for VUV or COARSE_CLOVER
      tune_block_x = (type == COMPUTE_VUV || type == COMPUTE_COARSE_CLOVER) ? false : true;
    }

    bool advanceAux(TuneParam &param) const
    {
      if (type != COMPUTE_VUV) return false;
#ifdef SHARED_ATOMIC
#ifdef SWIZZLE
      constexpr int max_swizzle = 4;
      if (param.aux.x < max_swizzle) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      if (param.aux.y < 4) {
        param.aux.y *= 2;
	return true;
      } else {
        param.aux.y = 1;
	return false;
      }
#endif
#else
      return false;
#endif
    }

    bool advanceSharedBytes(TuneParam &param) const {
      return (type == COMPUTE_VUV || type == COMPUTE_COARSE_CLOVER) ? false : Tunable::advanceSharedBytes(param);
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      if (type == COMPUTE_VUV) {
#ifdef SHARED_ATOMIC
	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
	param.grid.x = 2*arg.coarseVolumeCB;
	param.aux.x = 1; // swizzle factor
	param.aux.y = 1; // aggregates per block
#endif
      }
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      if (type == COMPUTE_VUV) {
#ifdef SHARED_ATOMIC
	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
	param.grid.x = 2*arg.coarseVolumeCB;
	param.aux.x = 1; // swizzle factor
	param.aux.y = 4; // aggregates per block
#endif
      }
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == COMPUTE_UV)                 strcat(Aux,",computeUV");
      else if (type == COMPUTE_AV)                 strcat(Aux,",computeAV");
      else if (type == COMPUTE_TMAV)               strcat(Aux,",computeTmAV");
      else if (type == COMPUTE_TMCAV)              strcat(Aux,",computeTmcAV");
      else if (type == COMPUTE_CLOVER_INV_MAX) strcat(Aux,",computeCloverInverseMax");
      else if (type == COMPUTE_VUV)                strcat(Aux,",computeVUV");
      else if (type == COMPUTE_COARSE_CLOVER)      strcat(Aux,",computeCoarseClover");
      else if (type == COMPUTE_REVERSE_Y)          strcat(Aux,",computeYreverse");
      else if (type == COMPUTE_DIAGONAL)           strcat(Aux,",computeCoarseDiagonal");
      else if (type == COMPUTE_TMDIAGONAL)         strcat(Aux,",computeCoarseTmDiagonal");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeConvert");
      else errorQuda("Unknown type=%d\n", type);

      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
	if      (dim == 0) strcat(Aux,",dim=0");
	else if (dim == 1) strcat(Aux,",dim=1");
	else if (dim == 2) strcat(Aux,",dim=2");
	else if (dim == 3) strcat(Aux,",dim=3");

	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");
      }

      const char *vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_DIAGONAL
			     || type == COMPUTE_TMDIAGONAL || type == COMPUTE_CONVERT) ? X.VolString () : meta.VolString();

      if (type == COMPUTE_VUV || type == COMPUTE_COARSE_CLOVER) {
	strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
               meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
	strcat(Aux,"coarse_vol=");
	strcat(Aux,X.VolString());
      } else {
	strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped" :
               meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device" : ",CPU");
      }

      return TuneKey(vol_str, typeid(*this).name(), Aux);
    }

    void preTune() {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_CONVERT:
	Y.backup();
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
      case COMPUTE_COARSE_CLOVER:
	X.backup();
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }

    void postTune() {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_CONVERT:
	Y.restore();
      case COMPUTE_DIAGONAL:
      case COMPUTE_TMDIAGONAL:
      case COMPUTE_COARSE_CLOVER:
	X.restore();
      case COMPUTE_UV:
      case COMPUTE_AV:
      case COMPUTE_TMAV:
      case COMPUTE_TMCAV:
      case COMPUTE_CLOVER_INV_MAX:
      case COMPUTE_REVERSE_Y:
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
     @param UV[out] Temporary accessor used to store fine link field * null space vectors
     @param AV[out] Temporary accessor use to store fine clover inverse * null
     space vectors (only applicable when fine-grid operator is the
     preconditioned clover operator else in general this just aliases V
     @param V[in] Packed null-space vector accessor
     @param G[in] Fine grid link / gauge field accessor
     @param C[in] Fine grid clover field accessor
     @param Cinv[in] Fine grid clover inverse field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param kappa[in] Kappa parameter
     @param mu[in] Twisted-mass parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
  template<bool from_coarse, typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename F,
	   typename Ftmp, typename Vt, typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  Ftmp &UV, F &AV, Vt &V, fineGauge &G, fineClover &C, fineClover &Cinv,
		  GaugeField &Y_, GaugeField &X_, ColorSpinorField &uv,
		  ColorSpinorField &av, const ColorSpinorField &v,
		  double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
		  const int *fine_to_coarse, const int *coarse_to_fine) {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    bool is_dirac_coarse = (dirac == QUDA_COARSE_DIRAC || dirac == QUDA_COARSEPC_DIRAC) ? true : false;
    if (is_dirac_coarse && fineSpin != 2)
      errorQuda("Input Dirac operator %d should have nSpin=2, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineSpin != 4)
      errorQuda("Input Dirac operator %d should have nSpin=4, not nSpin=%d\n", dirac, fineSpin);
    if (!is_dirac_coarse && fineColor != 3)
      errorQuda("Input Dirac operator %d should have nColor=3, not nColor=%d\n", dirac, fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) x_size[i] = v.X(i);
    x_size[4] = 1;

    int xc_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM] = { };
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse();

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_COARSEPC_DIRAC || bidirectional_debug ||
				dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC);

    if (getVerbosity() >= QUDA_VERBOSE) {
      if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
      else printfQuda("Doing uni-directional link coarsening\n");
    }

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateYArg<Float,fineSpin,coarseSpin,coarseGauge,coarseGaugeAtomic,fineGauge,F,Ftmp,Vt,fineClover> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, AV, G, V, C, Cinv, kappa,
	    mu, mu_factor, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    CalculateY<from_coarse, Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> y(arg, v, Y_, X_);

    QudaFieldLocation location = checkLocation(Y_, X_, av, v);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v, v.Ghost());  // point the accessor to the correct ghost buffer
    if (&v == &av) arg.AV.resetGhost(av, av.Ghost());
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("V2 = %e\n", arg.V.norm2());

    // If doing preconditioned clover then we first multiply the
    // null-space vectors by the clover inverse matrix, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing AV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
	double max = 6*arg.Cinv.abs_max(0);
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("clover max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_AV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // If doing preconditioned twisted-mass then we first multiply the
    // null-space vectors by the inverse twist, since this is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_MASSPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing TMAV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
	// this is just a trivial rescaling kernel, find the maximum
	complex<Float> fp(1./(1.+arg.mu*arg.mu),-arg.mu/(1.+arg.mu*arg.mu));
	complex<Float> fm(1./(1.+arg.mu*arg.mu),+arg.mu/(1.+arg.mu*arg.mu));
	double max = std::max(abs(fp), abs(fm));
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("tm max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_TMAV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // If doing preconditioned twisted-clover then we first multiply the
    // null-space vectors by the inverse of the squared clover matrix plus
    // mu^2, and then we multiply the result by the clover matrix. This is
    // needed for the coarse link computation
    if ( dirac == QUDA_TWISTED_CLOVERPC_DIRAC && (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing TMCAV\n");

      if (av.Precision() == QUDA_HALF_PRECISION) {
#ifdef DYNAMIC_CLOVER
	y.setComputeType(COMPUTE_CLOVER_INV_MAX);
	y.apply(0);
	double max = 6*sqrt(arg.max_h);
#else
	double max = 6*sqrt(arg.Cinv.abs_max(0));
#endif
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("tmc max %e\n", max);
	av.Scale(max);
	arg.AV.resetScale(max);
      }

      y.setComputeType(COMPUTE_TMCAV);
      y.apply(0);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("AV2 = %e\n", arg.AV.norm2());
    }

    // work out what to set the scales to
    if (coarseGaugeAtomic::fixedPoint()) {
      double max = 100.0; // FIXME - more accurate computation needed?
      arg.Y_atomic.resetScale(max);
      arg.X_atomic.resetScale(max);
    }

    // First compute the coarse forward links if needed
    if (bidirectional_links) {
      for (int d = 0; d < nDim; d++) {
	y.setDimension(d);
	y.setDirection(QUDA_FORWARDS);
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing forward %d UV and VUV\n", d);

	if (uv.Precision() == QUDA_HALF_PRECISION) {
	  double U_max = 3.0*arg.U.abs_max(from_coarse ? d+4 : d);
	  double uv_max = U_max * v.Scale();
	  uv.Scale(uv_max);
	  arg.UV.resetScale(uv_max);

	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
	}

	y.setComputeType(COMPUTE_UV);  // compute U*V product
	y.apply(0);
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("UV2[%d] = %e\n", d, arg.UV.norm2());

	y.setComputeType(COMPUTE_VUV); // compute Y += VUV
	y.apply(0);
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", d, arg.Y_atomic.norm2(4+d));
      }
    }

    if ( (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC) &&
	 (matpc == QUDA_MATPC_EVEN_EVEN || matpc == QUDA_MATPC_ODD_ODD) ) {
      av.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
      arg.AV.resetGhost(av, av.Ghost());  // make sure we point to the correct pointer in the accessor
      LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing backward %d UV and VUV\n", d);

      if (uv.Precision() == QUDA_HALF_PRECISION) {
	double U_max = 3.0*arg.U.abs_max(d);
	double uv_max = U_max * av.Scale();
	uv.Scale(uv_max);
	arg.UV.resetScale(uv_max);

	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e av_max = %e uv_max = %e\n", d, U_max, av.Scale(), uv_max);
      }

      y.setComputeType(COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("UAV2[%d] = %e\n", d, arg.UV.norm2());

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Y2[%d] = %e\n", d, arg.Y_atomic.norm2(d));

    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Reversing links\n");
      y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(0);
    }

    // Check if we have a clover term that needs to be coarsened
    if (dirac == QUDA_CLOVER_DIRAC || dirac == QUDA_COARSE_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing fine->coarse clover term\n");
      y.setComputeType(COMPUTE_COARSE_CLOVER);
      y.apply(0);
    } else {  //Otherwise, we just have to add the identity matrix
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Summing diagonal contribution to coarse clover\n");
      y.setComputeType(COMPUTE_DIAGONAL);
      y.apply(0);
    }

    if (arg.mu*arg.mu_factor!=0 || dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC) {
      if (dirac == QUDA_TWISTED_MASS_DIRAC || dirac == QUDA_TWISTED_CLOVER_DIRAC)
	arg.mu_factor += 1.;
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Adding mu = %e\n",arg.mu*arg.mu_factor);
      y.setComputeType(COMPUTE_TMDIAGONAL);
      y.apply(0);
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // now convert from atomic to application computation format if necesaary
    if (coarseGaugeAtomic::fixedPoint()) {
      y.setComputeType(COMPUTE_CONVERT);
      y.apply(0);
    }

  }


} // namespace quda
