#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>

namespace quda {

  template <typename Float, int coarseSpin, int fineColor, int coarseColor,
            typename coarseGauge, typename fineGauge>
  struct CalculateStaggeredYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    const fineGauge U;       /** Fine grid (fat-)link field */
    // May have a long-link variable in the future.

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */
    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    Float mass;                 /** staggered mass value */
    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    static constexpr int coarse_color = coarseColor;

    CalculateStaggeredYArg(coarseGauge &Y, coarseGauge &X, const fineGauge &U, double mass,
                           const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_) :
      Y(Y),
      X(X),
      U(U),
      spin_bs(spin_bs_),
      spin_map(),
      mass(static_cast<Float>(mass)),
      fineVolumeCB(U.VolumeCB()),
      coarseVolumeCB(X.VolumeCB())
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
        geo_bs[i] = geo_bs_[i];
      }
    }

  };

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void ComputeStaggeredVUV(Arg &arg, int parity, int x_cb, int ic_f, int jc_f)
  {
    constexpr int nDim = 4;
    int coord[nDim];
    int coord_coarse[nDim];

    getCoords(coord, x_cb, arg.x_size, parity);

    // Compute coarse coordinates
    for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];
    int coarse_parity = 0;
    for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0]/2;

    // Fine parity gives the coarse spin
    constexpr int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index
    const int s_c_col = arg.spin_map(s,1-parity); // Coarse spin col index

    // The coarse color row depends on my fine hypercube corner.
    int hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
    int c_row = 8*ic_f + hyperCorner;

#pragma unroll
    for (int mu=0; mu < nDim; mu++) {
      // The coarse color column depends on my fine hypercube+mu corner.
      coord[mu]++; // linkIndexP1, it's fine if this wraps because we're modding by 2.
      int hyperCorner_mu = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
      coord[mu]--;

      int c_col = 8*jc_f + hyperCorner_mu;

      //Check to see if we are on the edge of a block.  If adjacent site
      //is in same block, M = X, else M = Y
      const bool isDiagonal = ((coord[mu]+1)%arg.x_size[mu])/arg.geo_bs[mu] == coord_coarse[mu] ? true : false;

      complex<Float> vuv = arg.U(mu,parity,x_cb,ic_f,jc_f);

      if (!isDiagonal) {
        // backwards
        arg.Y(mu,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = vuv;
        // forwards
        arg.Y(nDim + mu,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
      } else { // (isDiagonal)
        // backwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_col,s_c_row,c_col,c_row) = conj(vuv);
        // forwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
      } // end (isDiagonal)
    }

    // lastly, add staggered mass term to diagonal
    if (ic_f == 0 && jc_f == 0 && x_cb < arg.coarseVolumeCB) {
#pragma unroll
      for (int s = 0; s < coarseSpin; s++) {
#pragma unroll
        for (int c = 0; c < coarseColor; c++) {
          arg.X(0,parity,x_cb,s,s,c,c) = static_cast<Float>(2.0) * complex<Float>(arg.mass,0.0); // staggered conventions. No need to +=
        } //Color
      } //Spin
    }
  }

  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredVUVCPU(Arg arg)
  {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int ic_f=0; ic_f<fineColor; ic_f++) {
          for (int jc_f=0; jc_f<fineColor; jc_f++) {
            ComputeStaggeredVUV<Float,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, ic_f, jc_f);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }

  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeStaggeredVUVGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int c = blockDim.y*blockIdx.y + threadIdx.y; // fine color
    if (c >= fineColor*fineColor) return;
    int ic_f = c / fineColor;
    int jc_f = c % fineColor;
    
    int parity = blockDim.z*blockIdx.z + threadIdx.z;

    ComputeStaggeredVUV<Float,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, ic_f, jc_f);
  }

} // namespace quda
