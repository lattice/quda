#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float_, int coarseSpin_, int fineColor_, int coarseColor_,
            typename coarseGauge, typename fineGauge, bool kd_build_x_ = false>
  struct CalculateStaggeredYArg : kernel_param<> {

    using real = typename mapper<Float_>::type;
    static constexpr int coarseSpin = coarseSpin_;
    static_assert(coarseSpin == 2, "Only coarseSpin == 2 is supported");
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseColor = coarseColor_;
    static_assert(8 * fineColor == coarseColor, "requires 8 * fineColor == coarseColor");

    static constexpr bool kd_build_x = kd_build_x_; /** If true, then we only build X and not Y */

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    const fineGauge U;       /** Fine grid (fat-)link field */
    // May have a long-link variable in the future.

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    real mass;                /** staggered mass value */
    const int fineVolumeCB;   /** Fine grid volume */
    const int coarseVolumeCB; /** Coarse grid volume */

    static constexpr int coarse_color = coarseColor;

    CalculateStaggeredYArg(coarseGauge &Y, coarseGauge &X, const fineGauge &U, double mass,
                           const int *x_size_, const int *xc_size_, int *geo_bs_ = nullptr) :
      kernel_param(dim3(U.VolumeCB(), fineColor * fineColor, 2)),
      Y(Y),
      X(X),
      U(U),
      spin_map(),
      mass(static_cast<real>(mass)),
      fineVolumeCB(U.VolumeCB()),
      coarseVolumeCB(X.VolumeCB())
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
        geo_bs[i] = geo_bs_ ? geo_bs_[i] : 2;
      }
    }

  };

  template <typename Arg>
  struct ComputeStaggeredVUV {
    const Arg &arg;
    constexpr ComputeStaggeredVUV(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int c, int parity)
    {
      using real = typename Arg::real;
      constexpr int nDim = 4;

      int ic_f = c / Arg::fineColor;
      int jc_f = c % Arg::fineColor;

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
        const bool isDiagonal = Arg::kd_build_x ?
          ((coord[mu] + 1) % arg.x_size[mu]) / 2 == coord_coarse[mu] :
          ((coord[mu] + 1) % arg.x_size[mu]) / arg.geo_bs[mu] == coord_coarse[mu];

        complex<real> vuv = arg.U(mu,parity,x_cb,ic_f,jc_f);

        if (isDiagonal) {
          // backwards
          arg.X(0,coarse_parity,coarse_x_cb,s_c_col,s_c_row,c_col,c_row) = conj(vuv);
          // forwards
          arg.X(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
        } if (!isDiagonal && !Arg::kd_build_x) {
          // backwards
          arg.Y(mu,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = vuv;
          // forwards
          arg.Y(nDim + mu,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
        }
      }

      // lastly, add staggered mass term to diagonal
      if (ic_f == 0 && jc_f == 0 && x_cb < arg.coarseVolumeCB) {
#pragma unroll
        for (int s = 0; s < Arg::coarseSpin; s++) {
#pragma unroll
          for (int c = 0; c < Arg::coarseColor; c++) {
            arg.X(0,parity,x_cb,s,s,c,c) = static_cast<real>(2.0) * complex<real>(arg.mass, 0.0); // staggered conventions. No need to +=
          } //Color
        } //Spin
      }
    }
  };

} // namespace quda
