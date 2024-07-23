#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, int parity_> struct GaugeFixArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int parity = parity_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    const Gauge u;
    Gauge rot;

    int X[4]; // grid dimensions
    int border[4];
    const Float relax_boost;
    const int dir_ignore;
    const Float tolerance;

    GaugeFixArg(const GaugeField &u, GaugeField &rot, double relax_boost, int dir_ignore) :
      kernel_param(dim3(u.LocalVolumeCB())),
      u(u),
      rot(rot),
      relax_boost(relax_boost),
      dir_ignore(dir_ignore),
      tolerance(u.toleranceSU3())
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = u.R()[dir];
        X[dir] = u.X()[dir] - border[dir] * 2;
      }
    }
  };

  // g' = \frac{K^\dagger g^\dagger}{\sqrt{\det(K^\dagger g^\dagger)}} g = \frac{K^\dagger}{\sqrt{\det(K^\dagger)}}
  template <int su2_index, typename Link, typename Float>
  __device__ __host__ inline void minimize_gK(Link &g, Link &gK, Float versors[4])
  {
    int i1, i2;
    switch (su2_index) {
    case 0: i1 = 0, i2 = 1; break;
    case 1: i1 = 1, i2 = 2; break;
    case 2: i1 = 0, i2 = 2; break;
    default: break;
    }

    versors[0] = gK(i1, i1).real() + gK(i2, i2).real();
    versors[1] = gK(i1, i1).imag() - gK(i2, i2).imag();
    versors[2] = gK(i1, i2).real() - gK(i2, i1).real();
    versors[3] = gK(i1, i2).imag() + gK(i2, i1).imag();

    Float norm
      = sqrt(versors[0] * versors[0] + versors[1] * versors[1] + versors[2] * versors[2] + versors[3] * versors[3]);
    versors[0] /= norm;
#pragma unroll
    for (int i = 1; i < 4; ++i) { versors[i] /= -norm; }

    setIdentity(&gK);
    gK(i1, i1) = complex(versors[0], versors[1]);
    gK(i2, i2) = complex(versors[0], -versors[1]);
    gK(i1, i2) = complex(versors[2], versors[3]);
    gK(i2, i1) = complex(-versors[2], versors[3]);

    g = gK * g;
  }

  template <typename Arg> struct GaugeFix : computeStapleOps {
    const Arg &arg;
    template <typename... OpsArgs>
    constexpr GaugeFix(const Arg &arg, const OpsArgs &...ops) : KernelOpsT(ops...), arg(arg)
    {
    }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;
      constexpr int parity = Arg::parity;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0, 0, 0, 0};
      Link g, K, tmp, tmp2;
      g = arg.rot(0, linkIndexShift(x, dx, X), parity);
#pragma unroll
      for (int dir = 0; dir < 4; ++dir) {
        if (dir != arg.dir_ignore) {
          tmp = arg.u(dir, linkIndexShift(x, dx, X), parity);
          dx[dir]++;
          tmp2 = arg.rot(0, linkIndexShift(x, dx, X), 1 - parity);
          dx[dir]--;
          K += tmp * conj(tmp2);
          dx[dir]--;
          tmp = arg.u(dir, linkIndexShift(x, dx, X), 1 - parity);
          tmp2 = arg.rot(0, linkIndexShift(x, dx, X), 1 - parity);
          dx[dir]++;
          K += conj(tmp2 * tmp);
        }
      }

      real versors[4];
      // loop over SU(2) subgroup indices
      tmp = g * K;
      minimize_gK<0>(g, tmp, versors);
      tmp = g * K;
      minimize_gK<1>(g, tmp, versors);
      tmp = g * K;
      minimize_gK<2>(g, tmp, versors);
      polarSu3(g, arg.tolerance);

      arg.rot(0, linkIndexShift(x, dx, X), parity) = g;
    }
  };
} // namespace quda
