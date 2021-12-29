#pragma once

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.cuh>
#include <reduce_helper.h>
#include <kernels/madwf_transfer.cuh>

/**
  @file This file contains the argument and kernel that forms a tensor outer product in the fifth
  dimension given two input vectors,
      T_{st} = out_s * in_t,
  where s and t are fifth dimension indices. T_{st} is a spin matrix so T has shape Ls*4-by-Ls*4.
*/

namespace quda
{

  namespace madwf_ml
  {

    template <class storage_t> struct Tensor5DReduceArg : ReduceArg<complex<typename mapper<storage_t>::type>> {

      using F = typename colorspinor_mapper<storage_t, 4, 3>::type;
      using real = typename mapper<storage_t>::type;
      using reduce_t = complex<real>;
      using Vector = ColorSpinor<real, 3, 4>;

      const F out; // output vector field
      const F in;  // input vector field

      const int Ls_out; // length of 5th dimension of the out field
      const int Ls_in;  // length of 5th dimension of the in field

      const int volume_4d_cb;

      const int nParity;

      Tensor5DReduceArg(const ColorSpinorField &out, const ColorSpinorField &in) :
        ReduceArg<reduce_t>(dim3(out.VolumeCB() / out.X(4), out.X(4) * in.X(4) * spin_dim * spin_dim, out.SiteSubset()), out.X(4) * in.X(4) * spin_dim * spin_dim),
        out(out),
        in(in),
        Ls_out(out.X(4)),
        Ls_in(in.X(4)),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
        nParity(in.SiteSubset())
      {
        if (volume_4d_cb != static_cast<int>(out.VolumeCB() / Ls_out)) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
                    static_cast<int>(out.VolumeCB() / Ls_out));
        }

        if (in.Nspin() != 4) errorQuda("nSpin = %d not supported", in.Nspin());
        if (in.Ncolor() != 3) errorQuda("nColor = %d not supported", in.Ncolor());
        if (out.Nspin() != 4) errorQuda("nSpin = %d not supported", out.Nspin());
        if (out.Ncolor() != 3) errorQuda("nColor = %d not supported", out.Ncolor());

        checkNative(in, out);
      }

      __device__ __host__ reduce_t init() const
      {
        return {0, 0};
      }

    };

    template <class Arg> struct Tensor5DReduce {

      const Arg &arg;
      constexpr Tensor5DReduce(const Arg &arg) : arg(arg) { }
      static constexpr const char *filename() { return KERNEL_FILE; }

      using reduce_t = typename Arg::reduce_t;

      static constexpr bool do_sum = true;

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int x_cb, int y, int parity)
      {
        // y = (v_s, w_s, v_spin, w_spin)
        int w_spin = y % spin_dim; y /= spin_dim;
        int v_spin = y % spin_dim; y /= spin_dim;
        int w_s = y % arg.Ls_in; y /= arg.Ls_in;
        int v_s = y;

        using Vector = typename Arg::Vector;

        const Vector v = arg.out(v_s * arg.volume_4d_cb + x_cb, parity);
        const Vector w = arg.in(w_s * arg.volume_4d_cb + x_cb, parity);

        sum += conj(innerProduct(v, w, v_spin, w_spin));
        return sum;
      }

      __device__ __host__ inline reduce_t operator()(reduce_t a, reduce_t b) const
      {
        return a + b;
      }

    };

  } // namespace madwf_ml

} // namespace quda
