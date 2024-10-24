#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool compute_theta_>
  struct GaugeFixQualityArg : public ReduceArg<array<double, 2>> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool compute_theta = compute_theta_;
    typedef typename gauge_mapper<Float, recon>::type Gauge;

    const Gauge U;

    int X[4]; // grid dimensions
    int border[4];
    const int dir_ignore;

    GaugeFixQualityArg(const GaugeField &U, int dir_ignore) :
      ReduceArg<reduce_t>(dim3(U.LocalVolumeCB(), 2)), U(U), dir_ignore(dir_ignore)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = U.R()[dir];
        X[dir] = U.X()[dir] - border[dir] * 2;
      }
    }
  };

  template <typename Arg> struct GaugeFixQuality : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr GaugeFixQuality(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t quality {0, 0};

      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      Link U, V;
#pragma unroll
      for (int dir = 0; dir < 4; ++dir) {
        if (dir != arg.dir_ignore) {
          U = arg.U(dir, linkIndex(x, X), parity);
          V += U;
        }
      }
      quality[0] = getTrace(V).real();

      if constexpr (Arg::compute_theta) {
#pragma unroll
        for (int dir = 0; dir < 4; ++dir) {
          if (dir != arg.dir_ignore) {
            U = arg.U(dir, linkIndexM1(x, X, dir), 1 - parity);
            V -= U;
          }
        }
        V -= conj(V);
        SubTraceUnit(V);
        quality[1] = getRealTraceUVdagger(V, V);
      }

      return operator()(quality, value);
    }
  };
} // namespace quda
