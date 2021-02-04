#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon_, int type_>
  struct KernelArg : public ReduceArg<double2> {
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int type = type_;
    using real = typename mapper<Float>::type;
    using Gauge = typename gauge_mapper<real, recon>::type;
    int X[4]; // grid dimensions
    int border[4];
    Gauge u;
    dim3 threads; // number of active threads required

    KernelArg(const GaugeField &u) :
      ReduceArg<double2>(),
      u(u),
      threads(u.LocalVolumeCB(), 2, 1)
    {
      for (int dir=0; dir<4; ++dir) {
        border[dir] = u.R()[dir];
        X[dir] = u.X()[dir] - border[dir]*2;
      }
    }

    __device__ __host__ auto init() const { return zero<double2>(); }
  };

  template <typename Arg> struct DetTrace : plus<double2> {
    using reduce_t = double2;
    using plus<reduce_t>::operator();
    Arg &arg;
    constexpr DetTrace(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the determinant or trace at site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      int X[4];
#pragma unroll
      for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for(int dr=0; dr<4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2*arg.border[dr];
      }
#pragma unroll
      for (int mu = 0; mu < 4; mu++) {
        Matrix<complex<typename Arg::real>, Arg::nColor> U = arg.u(mu, linkIndex(x, X), parity);
        complex<double> local = Arg::type == 0 ? getDeterminant(U) : getTrace(U);
        value = plus::operator()(value, static_cast<reduce_t&>(local));
      }

      return value;
    }
  };

} // namespace quda
