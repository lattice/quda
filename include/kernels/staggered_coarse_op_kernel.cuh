#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>

#define max_color_per_block 8

namespace quda {

  // this is the storage type used when computing the coarse link variables
  // by using integers we have deterministic atomics
  typedef int storeType;

  template <typename Float, int coarseSpin, int fineColor, int coarseColor,
      typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge,
      typename fineSpinorTmp>
  struct CalculateStaggeredYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    coarseGaugeAtomic Y_atomic;    /** Y atomic accessor used for computation before conversion to final format */
    coarseGaugeAtomic X_atomic;    /** X atomic accessor used for computation before conversion to final format */

    fineSpinorTmp UV;        /** Temporary that stores the fine-link * spinor field product */

    const fineGauge U;       /** Fine grid (fat-)link field */
    // May have a long-link variable in the future.

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */
    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float mass;                 /** staggered mass value */
    Float rescale;              /** rescaling factor used when rescaling the Y links if the maximum increases */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    const int *fine_to_coarse;
    const int *coarse_to_fine;

    const bool bidirectional;

    static constexpr int coarse_color = coarseColor;

    int dim_index; // which direction / dimension we are working on

    CalculateStaggeredYArg(coarseGauge &Y, coarseGauge &X,
      coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
      fineSpinorTmp &UV, const fineGauge &U,
      double mass, const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_,
      const int *fine_to_coarse, const int *coarse_to_fine, bool bidirectional)
      : Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic),
        UV(UV), U(U), spin_bs(spin_bs_), spin_map(),
        mass(static_cast<Float>(mass)), 
        fineVolumeCB(U.VolumeCB()), coarseVolumeCB(X.VolumeCB()),
        fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
        bidirectional(bidirectional)
    {

      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
        geo_bs[i] = geo_bs_[i];
        comm_dim[i] = comm_dim_partitioned(i);
      }
    }

    ~CalculateStaggeredYArg() { }
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
     Calculates the matrix UV^{c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{c}_mu(x+mu)
     Where: mu = dir, c' = coarse color, c = fine color
     Staggered fermions don't carry spin.
  */
  template<typename Float, int dim, QudaDirection dir, int fineColor,
     int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void computeStaggeredUV(Arg &arg, int parity, int x_cb, int ic_f, int jc_f) {

    

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);
    coord[dim]++; // linkIndexP1, it's fine if this wraps because we're modding by 2.
    int hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);

    for (int h = 0; h < 8; h++)
      arg.UV(parity,x_cb,0,ic_f,8*jc_f+h) = (h == hyperCorner) ? arg.U(dim,parity,x_cb,ic_f,jc_f) : 0.0;

  } // computeStaggeredUV

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredUVCPU(Arg &arg) {

    for (int parity=0; parity<2; parity++) {
      #pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int ic_f=0; ic_f < fineColor; ic_f++) { // fine color
          for (int jc_f=0; jc_f < fineColor; jc_f++) { // fine color
            computeStaggeredUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, ic_f, jc_f);
          }
        } // coarse color
      } // c/b volume
    }   // parity
  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeStaggeredUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int c = blockDim.z*blockIdx.z + threadIdx.z; // fine color
    if (c >= fineColor*fineColor) return;
    int ic_f = c/fineColor;
    int jc_f = c%fineColor;
    computeStaggeredUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, ic_f, jc_f);
  }


  template<typename Float, int dim, QudaDirection dir,
           int fineColor, int coarseSpin, typename Arg>
  __device__ __host__ void ComputeStaggeredVUV(Arg &arg, int parity, int x_cb, int ic_f, int jc_f) {

    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    // Get coords.
    getCoords(coord, x_cb, arg.x_size, parity);

    // The coarse color row depends on my fine hypercube corner.
    int hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
    int c_row = 8*ic_f + hyperCorner;

    // The coarse color column depends on my fine hypercube+mu corner.
    coord[dim]++; // linkIndexP1, it's fine if this wraps because we're modding by 2.
    hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
    coord[dim]--;
    int c_col = 8*jc_f + hyperCorner;

    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    // Fine parity gives the coarse spin
    const int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index
    const int s_c_col = arg.spin_map(s,1-parity); // Coarse spin col index


    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

    int coarse_parity = 0;

    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;

    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];


    complex<Float> vuv;

    if (dir == QUDA_BACKWARDS) {
      vuv = arg.UV(parity, x_cb, 0, ic_f, c_col);
    } else {
      vuv = -arg.UV(parity, x_cb, 0, ic_f, c_col);
    }

    const int dim_index = arg.dim_index % arg.Y_atomic.geometry;


    if (!isDiagonal) {
      arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col,vuv);
    } else { // (isDiagonal)

      if (dir == QUDA_BACKWARDS) {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_c_col,s_c_row,c_col,c_row,conj(vuv));
      } else { // (dir == QUDA_FORWARDS)
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col,vuv);
      }

      if (!arg.bidirectional) {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col,-vuv);
      } // end (!arg.bidirectional)

    } // end (isDiagonal)

  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, typename Arg>
  void ComputeStaggeredVUVCPU(Arg arg) {

    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int ic_f=0; ic_f<fineColor; ic_f++) {
          for (int jc_f=0; jc_f<fineColor; jc_f++) {
            ComputeStaggeredVUV<Float,dim,dir,fineColor,coarseSpin>(arg, parity, x_cb, ic_f, jc_f);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }


  template<typename Float, int dim, QudaDirection dir,
           int fineColor, int coarseSpin, typename Arg>
  __global__ void ComputeStaggeredVUVGPU(Arg arg) {

    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int c = blockDim.z*blockIdx.z + threadIdx.z; // fine color
    if (c >= fineColor*fineColor) return;
    int ic_f = c/fineColor;
    int jc_f = c%fineColor;
    
    ComputeStaggeredVUV<Float,dim,dir,fineColor,coarseSpin>(arg, parity, x_cb, ic_f, jc_f);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeStaggeredYreverse(Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {
    auto &Y = arg.Y;

#pragma unroll
    for (int d=0; d<4; d++) {
#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { // chiral row block
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { // chiral col block
          Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = -Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
        } // chiral col
      } // chiral row
    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeStaggeredYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
          for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color col
            computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
          }
        }
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeStaggeredYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_jc_c = blockDim.y*blockIdx.y + threadIdx.y; // parity and color col
    if (parity_jc_c >= 2*nColor) return;
    int parity = parity_jc_c / nColor;
    int jc_c = parity_jc_c % nColor;


    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
  }

  //Adds the mass to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseStaggeredMassCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
          for(int c = 0; c < nColor; c++) { //Color
            arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(2.0*arg.mass,0.0); // staggered conventions
          } //Color
        } //Spin
      } // x_cb
    } //parity
  }


  // Adds the mass to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void AddCoarseStaggeredMassGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    for(int s = 0; s < nSpin; s++) { //Spin
      for(int c = 0; c < nColor; c++) { //Color
        arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(2.0*arg.mass,0.0); // staggered conventions
      } //Color
    } //Spin
   }

  /**
   * Convert the field from the atomic format to the required computation format, e.g. fixed point to floating point
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void convertStaggered(Arg &arg, int parity, int x_cb, int c_row, int c_col) {

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
      } // spin row
    } // arg.dim_index >= 8
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ConvertStaggeredCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int c_row = 0; c_row < nColor; c_row++) { //Color row
          for(int c_col = 0; c_col < nColor; c_col++) { //Color column
            convertStaggered<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
          }
        }
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ConvertStaggeredGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;

    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;

    convertStaggered<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

  /**
   * Rescale the matrix elements by arg.rescale
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void rescaleStaggeredY(Arg &arg, int parity, int x_cb, int c_row, int c_col) {
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
  void RescaleStaggeredYCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
  for(int c_row = 0; c_row < nColor; c_row++) { //Color row
    for(int c_col = 0; c_col < nColor; c_col++) { //Color column
      rescaleStaggeredY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
    }
  }
      } // c/b volume
    } // parity
  }
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void RescaleStaggeredYGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;
    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;
    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;
    rescaleStaggeredY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

} // namespace quda
