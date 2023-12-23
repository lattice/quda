#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <comm_quda.h>
#include <kernel.h>

namespace quda {

  using namespace colorspinor;

  template <typename Float_, int nSpin_, int nColor_, bool inverse_>
  struct SpinorDistanceReweightArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    using V = typename colorspinor_mapper<Float, nSpin, nColor>::type;

    static constexpr bool inverse = inverse_;

    int X[4];
    V v;
    Float alpha;
    int t0;
    int comm_dim_t;
    int comm_coord_t;
    SpinorDistanceReweightArg(ColorSpinorField &v, Float alpha, int t0) :
      kernel_param(dim3(v.VolumeCB(), v.SiteSubset(), 1)),
      v(v), alpha(alpha), t0(t0), comm_dim_t(comms_dim[3]), comm_coord_t(comms_coord[3])
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = v.X()[dir];
      X[0] *= (v.SiteSubset() == 1) ? 2 : 1; // need full lattice dims
    }
  };


  template<bool inverse, typename Float>
  __device__ __host__ inline Float genDistanceWeight(Float alpha, int t0, int t, int nt) {
    if constexpr (inverse) {
        return 1 / cosh(alpha * Float((t - t0 + nt) % nt - nt / 2));
    } else {
        return cosh(alpha * Float((t - t0 + nt) % nt - nt / 2));
    }
  }

  template <typename Arg> struct DistanceReweightSpinor {
    const Arg &arg;
    constexpr DistanceReweightSpinor(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using Vector = ColorSpinor<typename Arg::Float, Arg::nColor, Arg::nSpin>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      Vector tmp = arg.v(x_cb, parity);
      tmp *= genDistanceWeight<Arg::inverse>(arg.alpha, arg.comm_coord_t * arg.X[3] + x[3], arg.t0, arg.comm_dim_t * arg.X[3]);
      arg.v(x_cb, parity) = tmp;
    }
  };

}
