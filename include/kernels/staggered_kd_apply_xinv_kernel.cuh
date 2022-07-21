#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <register_traits.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, int dagger_>
  struct StaggeredKDBlockArg : kernel_param<> {
    using real = typename mapper<Float>::type;

    static constexpr int nDim = 4;

    static constexpr int nSpin = 1;
    static constexpr int nColor = nColor_;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // seems to be legacy, copied from dslash_staggered.cuh
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct = QUDA_RECONSTRUCT_NO;
    static constexpr bool gauge_direct_load = false; // seems to be legacy, copied from dslash_staggered.cuh
    using X = typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, QUDA_GHOST_EXCHANGE_PAD>::type;

    static constexpr bool dagger = dagger_;

    F out;               /** output vector field */
    const F in;          /** input vector field */
    const X xInv;        /** Kahler-Dirac inverse gauge field */
    int_fastdiv X0h;     /** One-half of X dimension length */
    int_fastdiv dim[4];  /** full lattice dimensions */
    const int volumeCB;  /** checkerboarded volume */

    StaggeredKDBlockArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &xInv) :
      kernel_param(dim3(in.VolumeCB(), 2, 1)),
      out(out),
      in(in),
      xInv(xInv),
      X0h(out.X()[0]/2),
      volumeCB(in.VolumeCB())
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in); // check all orders match
      checkPrecision(out, in, xInv); // check all precisions match
      checkLocation(out, in, xInv);
      if (xInv.Ndim() != nDim)
        errorQuda("Number of dimensions is not supported");

      for (int i=0; i<nDim; i++) {
        dim[i] = out.X()[i];
      }
    }
  };

  template<typename Arg> struct StaggeredKDBlockApply {

    using real = typename Arg::real;
    using Vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using Link = Matrix<complex<real>, Arg::nColor>;

    const Arg &arg;
    constexpr StaggeredKDBlockApply(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      // Get coordinates
      constexpr auto nDim = Arg::nDim;
      Coord<nDim> coord;
      coord.x_cb = x_cb;
      coord.X = getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

      // Get location of unit corner of hypercube
      int x_c[nDim];
#pragma unroll
      for (int d = 0; d < nDim; d++)
        x_c[d] = 2 * (coord[d] / 2);

      Vector out;

      // only needed for dagger
      // global parity == parity w/in the KD block
      int my_corner = 8 * parity + 4 * (coord[3] % 2) + 2 * (coord[2] % 2) + (coord[1] % 2);

      // Begin accumulating into the output vector
      int nbr_corner = 0;
#pragma unroll
      for (int nbr_parity = 0; nbr_parity < 2; nbr_parity++) {
#pragma unroll
        for (int nbr_t = 0; nbr_t < 2; nbr_t++) {
#pragma unroll
          for (int nbr_z = 0; nbr_z < 2; nbr_z++) {
#pragma unroll
            for (int nbr_y = 0; nbr_y < 2; nbr_y++) {
              const int offset[4] = { (nbr_parity + nbr_t + nbr_z + nbr_y) & 1, nbr_y, nbr_z, nbr_t };
              const int neighbor_idx = linkIndexShift(x_c, offset, arg.dim);
              const Link Xinv = Arg::dagger ? arg.xInv(my_corner, neighbor_idx, nbr_parity) : arg.xInv(nbr_corner, coord.x_cb, parity);
              const Vector in = arg.in(neighbor_idx, nbr_parity);
              out = mv_add(Arg::dagger ? conj(Xinv) : Xinv, in, out);
              nbr_corner++;
            }
          }
        }
      }

      // And we're done
      arg.out(coord.x_cb, parity) = out;
    }
  };

} // namespace quda
