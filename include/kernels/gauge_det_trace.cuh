#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda {

  enum struct compute_type { determinant, trace };

  template <typename Float, int nColor_, QudaReconstructType recon_, compute_type type_>
  struct KernelArg : public ReduceArg<array<double, 2>> {
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    static constexpr compute_type type = type_;
    using real = typename mapper<Float>::type;
    using Gauge = typename gauge_mapper<real, recon>::type;
    int X[4]; // grid dimensions
    int border[4];
    Gauge u;

    KernelArg(const GaugeField &u) :
      ReduceArg<reduce_t>(dim3(u.LocalVolumeCB(), 2, 1)),
      u(u)
    {
      for (int dir=0; dir<4; ++dir) {
        border[dir] = u.R()[dir];
        X[dir] = u.X()[dir] - border[dir]*2;
      }
    }
  };

  template <typename Arg> struct DetTrace : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr DetTrace(const Arg &arg) : arg(arg) {}
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
        auto local = Arg::type == compute_type::determinant ? getDeterminant(U) : getTrace(U);
        value = operator()(value, reduce_t{local.real(), local.imag()});
      }

      return value;
    }
  };

} // namespace quda
