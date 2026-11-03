#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{
  template <typename Float_, int nColor_, QudaReconstructType recon_> struct RotateGaugeArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    Gauge out;
    const Gauge in;
    const Gauge rot;

    int X[4]; // grid dimensions
    int border[4];

    RotateGaugeArg(GaugeField &out, const GaugeField &in, const GaugeField &rot) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, 4)), out(out), in(in), rot(rot)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = rot.R()[dir];
        X[dir] = rot.X()[dir] - border[dir] * 2;
      }
    }
  };

  template <typename Arg> struct RotateGauge {
    const Arg &arg;
    constexpr RotateGauge(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int x[4], X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) { X[dr] = arg.X[dr]; }
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link g, U;
      U = arg.in(dir, x_cb, parity);
      g = arg.rot(0, linkIndex(x, X), parity);
      U = g * U;
      g = arg.rot(0, linkIndexP1(x, X, dir), 1 - parity);
      U = U * conj(g);

      arg.out(dir, x_cb, parity) = U;
    }
  };
} // namespace quda
