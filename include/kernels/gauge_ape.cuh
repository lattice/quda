#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int apeDim_>
  struct GaugeAPEArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int apeDim = apeDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    const Gauge in;

    int X[4];    // grid dimensions
    int border[4];
    const Float alpha;
    const int dir_ignore;
    const Float tolerance;

    GaugeAPEArg(GaugeField &out, const GaugeField &in, double alpha, int dir_ignore) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, apeDim)),
      out(out),
      in(in),
      alpha(alpha),
      dir_ignore(dir_ignore),
      tolerance(in.toleranceSU3())
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
      }
    }
  };
  
  template <typename Arg> struct APE {
    const Arg &arg;
    constexpr APE(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      dir = dir + (dir >= arg.dir_ignore);

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap, TestU, I;
      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, X, parity, dir, Stap, arg.dir_ignore);

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);

      Stap = Stap * (arg.alpha / ((real)(2 * (Arg::apeDim - 1))));
      setIdentity(&I);

      TestU = I * (static_cast<real>(1.0) - arg.alpha) + Stap * conj(U);
      polarSu3<real>(TestU, arg.tolerance);
      U = TestU * U;
    
      arg.out(dir, linkIndexShift(x, dx, X), parity) = U;
    }
  };
} // namespace quda
