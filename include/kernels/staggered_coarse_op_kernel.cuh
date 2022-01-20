#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float, int fineColor_, QudaGaugeFieldOrder order, bool kd_build_x_ = false>
  struct CalculateStaggeredYArg : kernel_param<> {

    using real = typename mapper<Float>::type;

    static constexpr int nDim = 4;
    static constexpr int fineColor = fineColor_;
    static constexpr int coarseSpin = 2;
    static constexpr int coarseColor = 24;

    static constexpr bool kd_build_x = kd_build_x_; /** If true, then we only build X and not Y */

    using gFine = typename gauge::FieldOrder<real,fineColor,1,order>;
    using gCoarse = typename gauge::FieldOrder<real, coarseColor * coarseSpin, coarseSpin, order, true, Float>;

    gCoarse Y;           /** Computed coarse link field */
    gCoarse X;           /** Computed coarse clover field */

    const gFine U;       /** Fine grid (fat-)link field */

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    const int fineVolumeCB;   /** Fine grid volume */
    const int coarseVolumeCB; /** Coarse grid volume */
    real two_mass;            /** Two times the staggered mass value */

    CalculateStaggeredYArg(GaugeField &Y, GaugeField &X, const GaugeField &U, double mass) :
      kernel_param(dim3(U.VolumeCB(), fineColor * fineColor, 2)),
      Y(Y),
      X(X),
      U(U),
      spin_map(),
      fineVolumeCB(U.VolumeCB()),
      coarseVolumeCB(X.VolumeCB()),
      two_mass(static_cast<real>(2. * mass))
    {
      if (X.Ndim() != nDim || Y.Ndim() != nDim)
        errorQuda("Number of dimensions %d %d is not supported", X.Ndim(), Y.Ndim());
      if (X.Ncolor() != coarseColor * coarseSpin || Y.Ncolor() != coarseColor * coarseSpin)
        errorQuda("Unsupported coarse colors %d %d", coarseSpin * X.Ncolor(), coarseSpin * Y.Ncolor());
      for (int i=0; i<nDim; i++) {
        x_size[i] = U.X()[i];
        xc_size[i] = X.X()[i];
        // check that local volumes are consistent
        if (2 * xc_size[i] != x_size[i]) {
          errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", static_cast<int>(x_size[i]), xc_size[i]);
        }
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
      using complex = complex<real>;
      constexpr int nDim = 4;

      int ic_f = c / Arg::fineColor;
      int jc_f = c % Arg::fineColor;

      int coord[nDim];
      int coord_coarse[nDim];

      getCoords(coord, x_cb, arg.x_size, parity);

      // Compute coarse coordinates
      for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d] / 2;
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
        const bool isDiagonal = ((coord[mu] + 1) % arg.x_size[mu]) / 2 == coord_coarse[mu];

        complex vuv = arg.U(mu,parity,x_cb,ic_f,jc_f);

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
            arg.X(0,parity,x_cb,s,s,c,c) = complex(arg.two_mass, 0.0); // staggered conventions. No need to +=
          } //Color
        } //Spin
      }
    }
  };

} // namespace quda
