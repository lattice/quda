#pragma once

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.h>
#include <reduce_helper.h>
#include <kernels/madwf_transfer.cuh>
#include <fast_intdiv.h>
#include <madwf_ml.h>

/**
  @file This file contains the argument and kernel that forms a tensor outer product in the fifth
  dimension given two input vectors,
      T_{st} = conj(x_s * y_t),
  where s and t are fifth dimension indices. T_{st} is a spin matrix so T has shape Ls*4-by-Ls*4.
*/

namespace quda
{

  namespace madwf_ml
  {

    template <class storage_t, int nSpin_, int nColor_>
    struct Tensor5DReduceArg : ReduceArg<array<complex<typename mapper<storage_t>::type>, nSpin_ * nSpin_>> {

      static constexpr int nSpin = nSpin_;
      static constexpr int nColor = nColor_;

      using F = typename colorspinor_mapper<storage_t, nSpin, nColor>::type;
      using real = typename mapper<storage_t>::type;
      using reduce_t = array<complex<real>, nSpin * nSpin>;
      using Vector = ColorSpinor<real, nColor, nSpin>;

      const F x; // xput vector field
      const F y; // yput vector field

      const int_fastdiv Ls_y; // length of 5th dimension of the in field

      const int volume_4d_cb;

      const int nParity;

      static constexpr unsigned int max_n_batch_block = 1u;

      Tensor5DReduceArg(const ColorSpinorField &x, const ColorSpinorField &y) :
        ReduceArg<reduce_t>(dim3(x.VolumeCB() / x.X(4), x.SiteSubset(), x.X(4) * y.X(4)), x.X(4) * y.X(4)),
        x(x),
        y(y),
        Ls_y(y.X(4)),
        volume_4d_cb(y.VolumeCB() / y.X(4)),
        nParity(y.SiteSubset())
      {
        if (volume_4d_cb != static_cast<int>(x.VolumeCB() / x.X(4))) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
                    static_cast<int>(x.VolumeCB() / x.X(4)));
        }

        checkNative(y, x);
      }
    };

    template <class Arg> struct Tensor5DReduce : plus<typename Arg::reduce_t> {
      using reduce_t = typename Arg::reduce_t;
      using plus<reduce_t>::operator();
      static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y

      const Arg &arg;
      constexpr Tensor5DReduce(const Arg &arg) : arg(arg) { }
      static constexpr const char *filename() { return KERNEL_FILE; }

      // overload comm_reduce to prevent any global reduction
      static inline void comm_reduce(std::vector<reduce_t> &) { }

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int x_cb, int parity, int batch_idx)
      {
        int y_s = batch_idx % arg.Ls_y;
        int x_s = batch_idx / arg.Ls_y;

        using Vector = typename Arg::Vector;

        const Vector x = arg.x(x_s * arg.volume_4d_cb + x_cb, parity);
        const Vector y = arg.y(y_s * arg.volume_4d_cb + x_cb, parity);

#pragma unroll
        for (int y_spin = 0; y_spin < Arg::nSpin; y_spin++) {
#pragma unroll
          for (int x_spin = 0; x_spin < Arg::nSpin; x_spin++) {
            sum[x_spin * Arg::nSpin + y_spin] += innerProduct(y, x, y_spin, x_spin);
          }
        }

        return sum;
      }
    };

  } // namespace madwf_ml

} // namespace quda
