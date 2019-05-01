#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <linalg.cuh>

#define max_color_per_block 8

namespace quda {

  // this is the storage type used when computing the coarse link variables
  // by using integers we have deterministic atomics
  typedef int storeType;

  template <typename Float, int fineSpin, int coarseSpin, int fineColor, int coarseColor,
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
    Float rescale;              /** rescaling factor used when rescaling the Y links if the maximum increases */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    const int *fine_to_coarse;
    const int *coarse_to_fine;

    const bool bidirectional;

    static constexpr int coarse_color = coarseColor;

    // To increase L2 locality we can schedule the geometry to grid.y and
    // the coarse colors to grid.x.  This will increase the potential for
    // L2 reuse since a given wave of thread blocks will be for different
    // coarse color but the same coarse grid point which will have common
    // loads.
    static constexpr bool coarse_color_wave = true;

    // Enable this for shared-memory atomics instead of global atomics.
    // Doing so means that all (modulo the parity) of the coarsening for a
    // coarse degree of freedom is handled by a single thread block.
    // For computeVUV only at present
    bool shared_atomic;

    // With parity_flip enabled we make parity the slowest running
    // dimension in the y-thread axis, and coarse color runs faster.  This
    // improves read locality at the expense of write locality
    bool parity_flip;

    int_fastdiv aggregates_per_block; // number of aggregates per thread block
    int_fastdiv grid_z; // this is the coarseColor grid that is wrapped into the x grid when coarse_color_wave is enabled
    int_fastdiv coarse_color_grid_z; // constant we ned to divide by

    Float max_h; // scalar that stores the maximum element of the dynamic clover inverse
    Float *max_d; // array that stores the maximum element per lattice site of the dynamic clover inverse

    int dim_index; // which direction / dimension we are working on

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
        bidirectional(bidirectional), shared_atomic(false), parity_flip(shared_atomic ? true : false),
        aggregates_per_block(1), max_d(nullptr)
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

  // complex multiply-add with optimal use of fma
  template<typename Float>
  inline __device__ __host__ void caxpy(const complex<Float> &a, const complex<Float> &x, complex<Float> &y) {
    y.x += a.x*x.x;
    y.x -= a.y*x.y;
    y.y += a.y*x.x;
    y.y += a.x*x.y;
  }

  /**
     Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
     Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
  */
  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor,
	   int coarseSpin, int coarseColor, typename Wtype, typename Arg>
  __device__ __host__ inline void computeUV(Arg &arg, const Wtype &W, int parity, int x_cb, int ic_c) {

    int coord[4];
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
	      caxpy(arg.U(dim, parity, x_cb, ic, jc), W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s, jc, ic_c), UV[s][ic]);
	    } else {
	      for (int s_col=0; s_col<fineSpin; s_col++) {
		// on the coarse lattice if forwards then use the forwards links
		caxpy(arg.U(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc),
                      W.Ghost(dim, 1, (parity+1)&1, ghost_idx, s_col, jc, ic_c), UV[s_col*fineSpin+s][ic]);
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
	      caxpy(arg.U(dim, parity, x_cb, ic, jc), W((parity+1)&1, y_cb, s, jc, ic_c), UV[s][ic]);
	    } else {
	      for (int s_col=0; s_col<fineSpin; s_col++) {
		// on the coarse lattice if forwards then use the forwards links
                caxpy(arg.U(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, ic, jc),
                      W((parity+1)&1, y_cb, s_col, jc, ic_c), UV[s_col*fineSpin+s][ic]);
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
  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void computeAV(Arg &arg, int parity, int x_cb, int ch, int ic_c)
  {
    constexpr int N = fineSpin * fineColor / 2;
    HMatrix<Float, N> A;

#pragma unroll
    for (int i = 0; i < N; i++) {
      int s_i = 2 * ch + i / fineColor;
      int c_i = i % fineColor;
#pragma unroll
      for (int j = 0; j <= i; j++) {
        int s_j = 2 * ch + j / fineColor;
        int c_j = j % fineColor;
#ifndef DYNAMIC_CLOVER
        A(i, j) = arg.Cinv(0, parity, x_cb, s_i, s_j, c_i, c_j);
#else
        A(i, j) = arg.C(0, parity, x_cb, s_i, s_j, c_i, c_j);
#endif
      }
    }

    ColorSpinor<Float, fineColor, fineSpin / 2> V;
    for (int s = 0; s < fineSpin / 2; s++) {
      for (int c = 0; c < fineColor; c++) { V(s, c) = arg.V(parity, x_cb, 2 * ch + s, c, ic_c); }
    }

#ifndef DYNAMIC_CLOVER
    auto AV = A * V;
#else
    // solve for the matrix
    linalg::Cholesky<HMatrix, Float, N> cholesky(A);
    auto AV = cholesky.backward(cholesky.forward(V));
#endif

#pragma unroll
    for (int s = 0; s < fineSpin / 2; s++) {
#pragma unroll
      for (int ic = 0; ic < fineColor; ic++) { arg.AV(parity, x_cb, 2 * ch + s, ic, ic_c) = AV(s, ic); }
    }

  } // computeAV

  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg> void ComputeAVCPU(Arg &arg)
  {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int ch = 0; ch < 2; ch++) { // Loop over chiral blocks

          for (int ic_c = 0; ic_c < coarseColor; ic_c++) { // coarse color
            computeAV<Float, fineSpin, fineColor, coarseColor>(arg, parity, x_cb, ch, ic_c);
          }
        }
      } // c/b volume
    }   // parity
  }

  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeAVGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int ch_parity = blockDim.y * blockIdx.y + threadIdx.y;
    if (ch_parity >= 4) return;
    int ch = ch_parity % 2;
    int parity = ch_parity / 2;

    int ic_c = blockDim.z * blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;

    if (ch == 0)
      computeAV<Float, fineSpin, fineColor, coarseColor>(arg, parity, x_cb, 0, ic_c);
    else
      computeAV<Float, fineSpin, fineColor, coarseColor>(arg, parity, x_cb, 1, ic_c);
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

  /**
     @brief Computes the clover field maximum, which is needed for
     setting the scale when using fixed point
   */
  template <typename Float, bool twist, typename Arg>
  __device__ __host__ inline Float computeCloverInvMax(Arg &arg, int parity, int x_cb)
  {

    Float max = 0.0;

    constexpr int nColor = 3;
    constexpr int nSpin = 4;
    constexpr int N = nColor * nSpin / 2;
    typedef HMatrix<Float, N> Mat;

#pragma unroll
    for (int ch = 0; ch < 2; ch++) { /* Loop over chiral blocks */
      Mat A;

#pragma unroll
      for (int i = 0; i < N; i++) {
#pragma unroll
        for (int j = 0; j <= i; j++) {
          A(i, j) = arg.C(0, parity, x_cb, 2 * ch + i / nColor, 2 * ch + j / nColor, i % nColor, j % nColor);
        }
      }

      if (twist) {
        A = A.square();
        A += arg.mu * arg.mu;
      }

      // compute the Colesky decomposition
      linalg::Cholesky<HMatrix, Float, N> cholesky(A);

      Mat Ainv = cholesky.invert();

      Float inv_max = Ainv.max();
      max = max > inv_max ? max : inv_max;

    } // chirality

    return max;
  }

  template <typename Float, bool twist, typename Arg> void ComputeCloverInvMaxCPU(Arg &arg)
  {
    Float max = 0.0;
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for reduction(max:max)
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        Float max_x = computeCloverInvMax<Float, twist, Arg>(arg, parity, x_cb);
        max = max > max_x ? max : max_x;
      } // c/b volume
    }   // parity
    arg.max_h = max;
  }

  template <typename Float, bool twist, typename Arg> __global__ void ComputeCloverInvMaxGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    arg.max_d[parity + 2 * x_cb] = computeCloverInvMax<Float, twist, Arg>(arg, parity, x_cb);
  }

#endif // DYNAMIC_CLOVER

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x) for twisted-clover fermions
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __device__ __host__ inline void computeTMCAV(Arg &arg, int parity, int x_cb, int ch, int ic_c)
  {
    constexpr int N = fineSpin * fineColor / 2;
    HMatrix<Float, N> A;

#pragma unroll
    for (int i = 0; i < N; i++) {
      int s_i = 2 * ch + i / fineColor;
      int c_i = i % fineColor;
#pragma unroll
      for (int j = 0; j <= i; j++) {
        int s_j = 2 * ch + j / fineColor;
        int c_j = j % fineColor;
        A(i, j) = arg.C(0, parity, x_cb, s_i, s_j, c_i, c_j);
      }
    }

    complex<Float> mu(0., arg.mu);
    if (ch == 0) mu *= static_cast<Float>(-1.0);

    ColorSpinor<Float, fineColor, fineSpin / 2> V;

    for (int s = 0; s < fineSpin / 2; s++) {
      for (int c = 0; c < fineColor; c++) { V(s, c) = arg.V(parity, x_cb, 2 * ch + s, c, ic_c); }
    }

    // first apply the clover matrix directly, including mu
    auto UV = A * V;
    UV += mu * V;

    // Then we calculate AV = Cinv UV, so  [AV = (C^2 + mu^2)^{-1} (Clover -/+ i mu)·Vector]
    // for in twisted-clover fermions, Cinv keeps (C^2 + mu^2)^{-1}

#ifndef DYNAMIC_CLOVER
    // load in the clover inverse matrix
    HMatrix<Float, N> Ainv;
#pragma unroll
    for (int i = 0; i < N; i++) {
      int s_i = 2 * ch + i / fineColor;
      int c_i = i % fineColor;
#pragma unroll
      for (int j = 0; j <= i; j++) {
        int s_j = 2 * ch + j / fineColor;
        int c_j = j % fineColor;
        Ainv(i, j) = arg.Cinv(0, parity, x_cb, s_i, s_j, c_i, c_j);
      }
    }
    auto AV = Ainv * UV;
#else
    // compute the clover inverse matrix with the already loaded clover matrix
    A = A.square();
    A += arg.mu * arg.mu;

    linalg::Cholesky<HMatrix, Float, N> cholesky(A);
    const auto AV = cholesky.backward(cholesky.forward(UV));
#endif

    for (int s = 0; s < fineSpin / 2; s++)
      for (int c = 0; c < fineColor; c++) arg.AV(parity, x_cb, 2 * ch + s, c, ic_c) = AV(s, c);
  } // computeTMCAV

  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg> void ComputeTMCAVCPU(Arg &arg)
  {
    for (int parity = 0; parity < 2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int ch = 0; ch < 2; ch++) {
          for (int ic_c = 0; ic_c < coarseColor; ic_c++) { // coarse color
            computeTMCAV<Float, fineSpin, fineColor, coarseColor, Arg>(arg, parity, x_cb, ch, ic_c);
          }
        }
      } // c/b volume
    }   // parity
  }

  template <typename Float, int fineSpin, int fineColor, int coarseColor, typename Arg>
  __global__ void ComputeTMCAVGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int ch_parity = blockDim.y * blockIdx.y + threadIdx.y;
    if (ch_parity >= 4) return;
    int ch = ch_parity % 2;
    int parity = ch_parity / 2;

    int ic_c = blockDim.z * blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;

    if (ch == 0)
      computeTMCAV<Float, fineSpin, fineColor, coarseColor, Arg>(arg, parity, x_cb, 0, ic_c);
    else
      computeTMCAV<Float, fineSpin, fineColor, coarseColor, Arg>(arg, parity, x_cb, 1, ic_c);
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

	const int s_col = gamma.getcol(s);
	const int s_c_col = arg.spin_map(s_col,parity); // Coarse spin col index

#pragma unroll
	for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color
          if (dir == QUDA_BACKWARDS) {
            complex<Float> V = arg.V(parity, x_cb, s, ic, ic_c);

            // here UV is really UAV
	    //Diagonal Spin
	    caxpy(conj(V), arg.UV(parity, x_cb, s, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_row]);

	    //Off-diagonal Spin (backward link / positive projector applied)
            caxpy( gamma.apply(s, conj(V)), arg.UV(parity, x_cb, s_col, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_col]);
	  } else {
            complex<Float> AV = arg.AV(parity, x_cb, s, ic, ic_c);

            //Diagonal Spin
	    caxpy(conj(AV), arg.UV(parity, x_cb, s, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_row]);

	    //Off-diagonal Spin (forward link / negative projector applied)
	    caxpy( -gamma.apply(s, conj(AV)), arg.UV(parity, x_cb, s_col, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_col]);
	  }
	} //Fine color
      }

    } else { // fine grid operator is a coarse operator

#pragma unroll
      for(int ic = 0; ic < fineColor; ic++) { //Sum over fine color
#pragma unroll
        for (int s = 0; s < fineSpin; s++) {
          complex<Float> AV = arg.AV(parity, x_cb, s, ic, ic_c);
#pragma unroll
          for (int s_col=0; s_col<fineSpin; s_col++) { // which chiral block
            complex<Float> UV = arg.UV(parity, x_cb, s_col*fineSpin+s, ic, jc_c);
            caxpy(conj(AV), UV, vuv[s*coarseSpin+s_col]);
          } //Fine color
        } //Fine spin
      }

    } // from_coarse

  }

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
    int x_coarse = (arg.coarse_color_wave ? blockIdx.y : blockIdx.x)*arg.aggregates_per_block + warp_lane % arg.aggregates_per_block;
    return x_coarse;
  }

  template<bool shared_atomic, bool parity_flip, bool from_coarse, typename Float, int dim, QudaDirection dir,
           int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg, typename Gamma>
  __device__ __host__ void computeVUV(Arg &arg, const Gamma &gamma, int parity, int x_cb, int c_row, int c_col, int parity_coarse_, int coarse_x_cb_) {

    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

    int coarse_parity = shared_atomic ? parity_coarse_ : 0;
    if (!shared_atomic) {
      for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
      coarse_parity &= 1;
      coord_coarse[0] /= 2;
    }
    int coarse_x_cb = shared_atomic ? coarse_x_cb_ : ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

    complex<Float> vuv[coarseSpin*coarseSpin];
    multiplyVUV<from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, gamma, parity, x_cb, c_row, c_col);

    const int dim_index = arg.dim_index % arg.Y_atomic.geometry;

    if (shared_atomic) {

#ifdef __CUDA_ARCH__
      __shared__ complex<storeType> X[max_color_per_block][max_color_per_block][4][coarseSpin][coarseSpin];
      __shared__ complex<storeType> Y[max_color_per_block][max_color_per_block][4][coarseSpin][coarseSpin];

      int x_ = coarse_x_cb%arg.aggregates_per_block;

      int tx = virtualThreadIdx(arg);
      int s_col = tx / coarseSpin;
      int s_row = tx % coarseSpin;

      int c_col_block = c_col % max_color_per_block;
      int c_row_block = c_row % max_color_per_block;

      if (tx < coarseSpin*coarseSpin) {
        Y[c_row_block][c_col_block][x_][s_row][s_col] = 0;
        X[c_row_block][c_col_block][x_][s_row][s_col] = 0;
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
              atomicAdd(&Y[c_row_block][c_col_block][x_][s_row][s_col],a);
            } else {
              atomicAdd(&Y[c_row_block][c_col_block][x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
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
              atomicAdd(&X[c_row_block][c_col_block][x_][s_row][s_col],a);
            } else {
              atomicAdd(&X[c_row_block][c_col_block][x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
            }
          }
        }
      }

      __syncthreads();

      if (tx < coarseSpin*coarseSpin && (parity == 0 || parity_flip == 1) ) {
        arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,Y[c_row_block][c_col_block][x_][s_row][s_col]);

        if (dir == QUDA_BACKWARDS) {
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row,conj(X[c_row_block][c_col_block][x_][s_row][s_col]));
        } else {
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,X[c_row_block][c_col_block][x_][s_row][s_col]);
        }

        if (!arg.bidirectional) {
          if (s_row == s_col) arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,X[c_row_block][c_col_block][x_][s_row][s_col]);
          else arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,-X[c_row_block][c_col_block][x_][s_row][s_col]);
        }
      }
#else
      errorQuda("Shared-memory atomic aggregation not supported on CPU");
#endif

    } else {

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

    }

  }

  template<bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeVUVCPU(Arg arg) {

    Gamma<Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim> gamma;
    constexpr bool shared_atomic = false; // not supported on CPU
    constexpr bool parity_flip = true;

    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
	for (int c_row=0; c_row<coarseColor; c_row++)
	  for (int c_col=0; c_col<coarseColor; c_col++)
	    computeVUV<shared_atomic,parity_flip,from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, gamma, parity, x_cb, c_row, c_col, 0, 0);
      } // c/b volume
    } // parity
  }

  // compute indices for shared-atomic kernel
  template <bool parity_flip, typename Arg>
  __device__ inline void getIndicesShared(const Arg &arg, int &parity, int &x_cb, int &parity_coarse, int &x_coarse_cb, int &c_col, int &c_row) {

    if (arg.coarse_color_wave) {
      int parity_c_col_block_z = blockDim.y*blockIdx.x + threadIdx.y;
      int c_col_block_z = parity_flip ? (parity_c_col_block_z % arg.coarse_color_grid_z ) : (parity_c_col_block_z / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col_block_z / arg.coarse_color_grid_z ) : (parity_c_col_block_z % 2);
      c_col = c_col_block_z % arg.coarse_color;
      c_row = blockDim.z*(c_col_block_z/arg.coarse_color) + threadIdx.z; // coarse color row index
    } else {
      int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
      c_col = parity_flip ? (parity_c_col % arg.coarse_color) : (parity_c_col / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col / arg.coarse_color) : (parity_c_col % 2);
      c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    }

    int block_dim_x = virtualBlockDim(arg);
    int thread_idx_x = virtualThreadIdx(arg);
    int x_coarse = coarseIndex(arg);

    parity_coarse = x_coarse >= arg.coarseVolumeCB ? 1 : 0;
    x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

    // obtain fine index from this look-up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkerboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered

    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * block_dim_x + thread_idx_x];
    x_cb = x_fine - parity*arg.fineVolumeCB;
  }

  // compute indices for global-atomic kernel
  template <bool parity_flip, typename Arg>
  __device__ inline void getIndicesGlobal(const Arg &arg, int &parity, int &x_cb, int &parity_coarse, int &x_coarse_cb, int &c_col, int &c_row) {

    x_cb = blockDim.x*(arg.coarse_color_wave ? blockIdx.y : blockIdx.x) + threadIdx.x;

    if (arg.coarse_color_wave) {
      int parity_c_col_block_z = blockDim.y*blockIdx.x + threadIdx.y;
      int c_col_block_z = parity_flip ? (parity_c_col_block_z % arg.coarse_color_grid_z ) : (parity_c_col_block_z / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col_block_z / arg.coarse_color_grid_z ) : (parity_c_col_block_z % 2);
      c_col = c_col_block_z % arg.coarse_color;
      c_row = blockDim.z*(c_col_block_z/arg.coarse_color) + threadIdx.z; // coarse color row index
    } else {
      int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
      c_col  = parity_flip ? (parity_c_col % arg.coarse_color) : (parity_c_col / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col / arg.coarse_color) : (parity_c_col % 2); // coarse color col index
      c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    }

    x_coarse_cb = 0;
    parity_coarse = 0;
  }

  template<bool shared_atomic, bool parity_flip, bool from_coarse, typename Float, int dim, QudaDirection dir,
           int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeVUVGPU(Arg arg) {

    Gamma<Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim> gamma;
    int parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row;
    if (shared_atomic) getIndicesShared<parity_flip>(arg, parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row);
    else getIndicesGlobal<parity_flip>(arg, parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row);

    if (parity > 1) return;
    if (c_col >= arg.coarse_color) return;
    if (c_row >= arg.coarse_color) return;
    if (!shared_atomic && x_cb >= arg.fineVolumeCB) return;

    computeVUV<shared_atomic,parity_flip,from_coarse,Float,dim,dir,fineSpin,fineColor,coarseSpin,coarseColor>(arg, gamma, parity, x_cb, c_row, c_col, parity_coarse, x_coarse_cb);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeYreverse(Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {
    auto &Y = arg.Y;

#pragma unroll
    for (int d=0; d<4; d++) {
#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
	for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	if (s_row == s_col)
	  Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
	else
	  Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = -Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
	} //Spin column
      } //Spin row

    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for (int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	  for (int jc_c = 0; jc_c < nColor; jc_c++) { //Color col
	    computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
	  }
	}
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_jc_c = blockDim.y*blockIdx.y + threadIdx.y; // parity and color col
    if (parity_jc_c >= 2*nColor) return;
    int parity = parity_jc_c / nColor;
    int jc_c = parity_jc_c % nColor;

    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    computeYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
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
      for (int s = 0; s < fineSpin; s++) { //Loop over fine spin row
	const int s_c = arg.spin_map(s,parity);
	//On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
	//in the same chiral block.
	for (int s_col = s_c*arg.spin_bs; s_col < (s_c+1)*arg.spin_bs; s_col++) { //Loop over fine spin column
          for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
            for (int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
              X[s_c*coarseSpin + s_c] +=
                conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
            } //Fine color column
          }  //Fine color row
	}  //Fine spin column
      } //Fine spin
    } else {
      //If Nspin != 4, then spin structure is a dense matrix and there is now spin aggregation
      //N.B. assumes that no further spin blocking is done in this case.
      for (int s = 0; s < fineSpin; s++) { //Loop over spin row
	for (int s_col = 0; s_col < fineSpin; s_col++) { //Loop over spin column
          for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color row
            for (int jc = 0; jc < fineColor; jc++) {  //Sum over fine color column
              X[s*coarseSpin + s_col] +=
                conj(arg.V(parity, x_cb, s, ic, ic_c)) * arg.C(0, parity, x_cb, s, s_col, ic, jc) * arg.V(parity, x_cb, s_col, jc, jc_c);
            } //Fine color column
          }  //Fine color row
	}  //Fine spin column
      } //Fine spin
    }

    for (int si = 0; si < coarseSpin; si++) {
      for (int sj = 0; sj < coarseSpin; sj++) {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,si,sj,ic_c,jc_c,X[si*coarseSpin+sj]);
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

    if (arg.dim_index < 8) {

      const auto &in = arg.Y_atomic;
      int d_in = arg.dim_index % in.geometry;
      int d_out = arg.dim_index % arg.Y.geometry;

#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
          complex<Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.Y(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
        } //Spin column
      } //Spin row

    } else {

      const auto &in = arg.X_atomic;
      int d_in = arg.dim_index % in.geometry;
      int d_out = arg.dim_index % arg.X.geometry;

#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
          complex<Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.X(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
        } //Spin column
      } //Spin row

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

  /**
   * Rescale the matrix elements by arg.rescale
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void rescaleY(Arg &arg, int parity, int x_cb, int c_row, int c_col) {

#pragma unroll
    for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
      for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
        complex<Float> M = arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col);
        arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col) = arg.rescale*M;
      } //Spin column
    } //Spin row

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void RescaleYCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
	for(int c_row = 0; c_row < nColor; c_row++) { //Color row
	  for(int c_col = 0; c_col < nColor; c_col++) { //Color column
	    rescaleY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
	  }
	}
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void RescaleYGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;

    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;

    rescaleY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

} // namespace quda
