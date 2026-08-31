#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <packed_array.h>
#include <kernel.h>

namespace quda
{

  template <typename store_t, int nColor, QudaReconstructType recon, bool verify_ = false>
  struct GaugeShiftArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    using Link = Matrix<complex<real>, nColor>;
    using RawLink = array<store_t, recon>;
    using Gauge = typename gauge_mapper<store_t, recon>::type;

    static constexpr bool gauge_direct_load = false;
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    template <bool shifted>
    using G = typename gauge_mapper<store_t, recon, 2 * nColor * nColor, QUDA_STAGGERED_PHASE_NO, ghost, false,
                                    QUDA_NATIVE_GAUGE_ORDER, shifted, QUDA_VECTOR_GEOMETRY>::type;
    static constexpr bool verify = verify_;

    int X[4]; // true grid dimensions
    G<true> out;
    const G<false> in;
    int shift;
    int volume_cb;
    // fuzz factor for verifying the shifted field - not guaranteed to be bitwise identical
    static constexpr real epsilon = std::is_same_v<store_t, double> ? 1e-14 : 3e-7;

    GaugeShiftArg(GaugeField &out, const GaugeField &in, int shift) :
      kernel_param(dim3(in.VolumeCB(), 2, 4)), out(out), in(in), shift(shift), volume_cb(in.VolumeCB())
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
      packed_array<uint16_t, 4> x = {};
      getCoords(x, x_cb, arg.X, parity);

      if constexpr (!Arg::verify) {
        typename Arg::RawLink link;
        if (x[dir] < arg.shift && arg.comms_dim_partitioned[dir]) { // on boundary so we fetch from ghost
          const int ghost_idx = ghostFaceIndexStaggered<0>(x, arg.X, dir, 1);
          arg.in.raw_load_ghost(link, ghost_idx, dir, 1 - parity);
          arg.out.raw_save(link, x_cb, dir, parity);
        } else { // simple shift
          packed_array<int8_t, 4> dx = {};
          dx[dir] = dx[dir] - arg.shift;
          int x_cb_back = linkIndexShift(x, dx, arg.X);
          arg.in.raw_load(link, x_cb_back, dir, 1 - parity);
          arg.out.raw_save(link, x_cb, dir, parity);

          if (x[dir] >= arg.X[dir] - arg.shift && arg.comms_dim_partitioned[dir]) { // write the ghost
            const int ghost_idx = ghostFaceIndexStaggered<1>(x, arg.X, dir, arg.shift);
            arg.in.raw_load(link, x_cb, dir, parity);
            arg.out.raw_save_ghost(link, ghost_idx, dir, 1 - parity);
          }
        }
      } else {
        // verify the shifting has worked
        using Link = typename Arg::Link;
        if (x[dir] < arg.shift && arg.comms_dim_partitioned[dir]) {
          const int ghost_idx = ghostFaceIndexStaggered<0>(x, arg.X, dir, 1);
          Link in = arg.in.Ghost(dir, ghost_idx, 1 - parity);
          Link out = arg.out(dir, x_cb, parity);
          assert((in - out).L1() < arg.epsilon);
        } else {
          packed_array<int8_t, 4> dx = {};
          dx[dir] = dx[dir] - arg.shift;
          int x_cb_back = linkIndexShift(x, dx, arg.X);
          Link in = arg.in(dir, x_cb_back, 1 - parity);
          Link out = arg.out(dir, x_cb, parity);
          assert((in - out).L1() < arg.epsilon);

          if (x[dir] >= arg.X[dir] - arg.shift && arg.comms_dim_partitioned[dir]) {
            const int ghost_idx = ghostFaceIndexStaggered<1>(x, arg.X, dir, arg.shift);
            Link in = arg.in(dir, x_cb, parity);
            Link out = arg.out.Ghost(dir, ghost_idx, 1 - parity);
            assert((in - out).L1() < arg.epsilon);
          }
        }
      }
    }
  };

} // namespace quda
