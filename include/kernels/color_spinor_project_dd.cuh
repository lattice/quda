#include <color_spinor_field_order.h>
#include <dslash_helper.cuh>
#include <kernel.h>

namespace quda
{

  using namespace colorspinor;

  template <typename Float, typename DDArg, int nSpin_, int nColor_, typename Order>
  struct ProjectDDArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    Order out;
    const DDArg dd;
    const int parity;
    const int nParity; // number of parities we're working on
    const int_fastdiv X0h;
    const int_fastdiv dim[5]; // full lattice dimensions
    const int commCoord[5];

    ProjectDDArg(ColorSpinorField &out) :
      kernel_param(dim3(out.VolumeCB(), out.SiteSubset(), 1)),
      out(out),
      dd(out),
      parity(out.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0),
      nParity(out.SiteSubset()),
      X0h(nParity == 2 ? out.X(0) / 2 : out.X(0)),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1},
      commCoord {comm_coord(0) * dim[0], comm_coord(1) * dim[1], comm_coord(2) * dim[2], comm_coord(3) * dim[3],
                 comm_coord(4) * dim[4]}

    {
    }
  };

  template <typename Arg> struct ProjectDD_ {
    const Arg &arg;
    constexpr ProjectDD_(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      Coord<4> coord;
      coord.X = getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);
      for (int i = 0; i < coord.size(); i++) { coord.gx[i] = arg.commCoord[i] + coord.x[i]; }
      if (arg.dd.isZero(coord)) {
        ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin> zero;
        arg.out(x_cb, (parity + arg.parity) & 1) = zero;
      }
    }
  };

} // namespace quda
