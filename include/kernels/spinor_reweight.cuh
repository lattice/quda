#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>

namespace quda
{

  using namespace colorspinor;

  template <typename Float_, int nSpin_, int nColor_> struct SpinorDistanceReweightArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    using V = typename colorspinor_mapper<Float, nSpin, nColor>::type;

    int X[4];
    V v;
    Float alpha0;
    int t0;
    SpinorDistanceReweightArg(ColorSpinorField &v, Float alpha0, int t0) :
      kernel_param(dim3(v.VolumeCB(), v.SiteSubset(), 1)), v(v), alpha0(alpha0), t0(t0)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = v.X()[dir];
      X[0] *= (v.SiteSubset() == 1) ? 2 : 1; // need full lattice dims
    }
  };

  template <typename Float> __device__ __host__ inline Float distanceWeight(Float alpha0, int t0, int t, int nt)
  {
    if (alpha0 > 0) {
      return cosh(alpha0 * Float((t - t0 + nt) % nt - nt / 2));
    } else {
      return 1 / cosh(alpha0 * Float((t - t0 + nt) % nt - nt / 2));
    }
  }

  template <typename Arg> struct DistanceReweightSpinor {
    const Arg &arg;
    constexpr DistanceReweightSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using Vector = ColorSpinor<typename Arg::Float, Arg::nColor, Arg::nSpin>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      Vector tmp = arg.v(x_cb, parity);
      tmp *= distanceWeight(arg.alpha0, arg.comms_coord[3] * arg.X[3] + x[3], arg.t0, arg.comms_dim[3] * arg.X[3]);
      arg.v(x_cb, parity) = tmp;
    }
  };

} // namespace quda
