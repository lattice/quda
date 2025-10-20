#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <byte_array.h>
#include <kernel.h>

namespace quda
{

  template <typename store_t, int nColor, QudaReconstructType recon> struct GaugeShiftArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    using Link = Matrix<complex<real>, nColor>;
    using Gauge = typename gauge_mapper<store_t, recon>::type;

    int X[4]; // true grid dimensions
    Gauge out;
    const Gauge in;
    int shift;

    GaugeShiftArg(GaugeField &out, const GaugeField &in, int shift) :
      kernel_param(dim3(in.VolumeCB(), 2, 4)), out(out), in(in), shift(shift)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = in.X()[dir];
    }
  };

  template <typename Arg> struct GaugeShift {
    const Arg &arg;
    constexpr GaugeShift(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = typename Arg::Link;

      byte_array<int8_t, 4> x = {};
      getCoords(x, x_cb, arg.X, parity);

      if (x[dir] < arg.shift && arg.comms_dim[dir] > 1) { // on the boundary so we need to fetch from the ghost zone
        const int ghost_idx = ghostFaceIndex<0, 4>(x, arg.X, dir, arg.shift);
        Link U = arg.in.Ghost(dir, ghost_idx, 1 - parity);
        arg.out(dir, x_cb, parity) = U;
      } else { // simple shift
        byte_array<int8_t, 4> dx = {};
        dx[dir] = dx[dir] - arg.shift;
        int x_cb_back = linkIndexShift(x, dx, arg.X);
        Link U = arg.in(dir, x_cb_back, 1 - parity);
        arg.out(dir, x_cb, parity) = U;
      }
    }
  };

} // namespace quda
