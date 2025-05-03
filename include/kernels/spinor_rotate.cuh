#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{
  template <typename Float_, int nSpin_, int nColor_, QudaReconstructType recon_>
  struct RotateSpinorArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;
    typedef typename colorspinor_mapper<Float, nSpin, nColor>::type Spinor;

    Spinor src;
    const Gauge rot;

    int X[4]; // grid dimensions
    int border[4];

    RotateSpinorArg(ColorSpinorField &src, const GaugeField &rot) :
      kernel_param(dim3(src.LocalVolumeCB(), 2, 1)), src(src), rot(rot)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = rot.R()[dir];
        X[dir] = rot.X()[dir] - border[dir] * 2;
      }
    }
  };

  template <typename Arg> struct RotateSpinor {
    const Arg &arg;
    constexpr RotateSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;
      typedef ColorSpinor<real, Arg::nColor, Arg::nSpin> Fermion;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link g = arg.rot(0, linkIndex(x, X), parity);
      Fermion V = arg.src(x_cb, parity);
      V = g * V;

      arg.src(x_cb, parity) = V;
    }
  };
} // namespace quda
