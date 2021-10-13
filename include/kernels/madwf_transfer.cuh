#pragma once

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.cuh>
#include <madwf_transfer.h>

namespace quda
{

  namespace madwf_ml
  {
    constexpr int spin_dim = 4;
    constexpr int color_dim = 3;
    constexpr int sm_dim = spin_dim * spin_dim;

    constexpr int color_spin_dim = spin_dim * color_dim;
    constexpr int wm_dim = color_spin_dim * color_spin_dim;

    template <class real> using WilsonVector = ColorSpinor<real, color_dim, spin_dim>;

    template <class real> using SpinMatrix = Matrix<complex<real>, spin_dim>;

    template <class real, transfer_5D_type transfer_type> struct transfer_5D_mapper {
    };

    template <class real> struct transfer_5D_mapper<real, transfer_5D_type::Spin> {
      using type = SpinMatrix<real>;
    };

    template <bool dagger, class real>
    __device__ __host__ inline WilsonVector<real> matrix_vector_multiply(const SpinMatrix<real> &m,
                                                                         const WilsonVector<real> &v)
    {
      WilsonVector<real> out; // out is initialized to zero
#pragma unroll
      for (int color = 0; color < color_dim; color++) {
#pragma unroll
        for (int column = 0; column < spin_dim; column++) {
          auto v_col = v(column, color);
#pragma unroll
          for (int row = 0; row < spin_dim; row++) {
            if (dagger) {
              out(row, color) += conj(m(column, row)) * v_col;
            } else {
              out(row, color) += m(row, column) * v_col;
            }
          }
        }
      }
      return out;
    }

    template <class storage_type, class matrix_type_, bool dagger_> struct Transfer5DArg : kernel_param<> {

      using F = typename colorspinor_mapper<storage_type, 4, 3>::type;
      using real = typename mapper<storage_type>::type;
      using Vector = ColorSpinor<real, 3, 4>;
      using matrix_type = matrix_type_;

      static constexpr bool dagger = dagger_;

      F out;      // output vector field
      const F in; // input vector field

      const int Ls_out; // length of 5th dimension of the out field
      const int Ls_in;  // length of 5th dimension of the in field

      const int volume_4d_cb;

      const matrix_type *wm_p;

      const int nParity;

      Transfer5DArg(ColorSpinorField &out, const ColorSpinorField &in, const matrix_type *wm_p) :
        kernel_param(dim3(out.VolumeCB() / out.X(4), out.X(4), out.SiteSubset())),
        out(out),
        in(in),
        Ls_out(out.X(4)),
        Ls_in(in.X(4)),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
        wm_p(wm_p),
        nParity(in.SiteSubset())
      {

        if (volume_4d_cb != static_cast<int>(out.VolumeCB() / Ls_out)) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
                    static_cast<int>(out.VolumeCB() / Ls_out));
        }

        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
        if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
        if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
      }
    };

    template <class Arg> struct Transfer5D {

      const Arg &arg;
      constexpr Transfer5D(const Arg &arg) : arg(arg) { }
      static constexpr const char *filename() { return KERNEL_FILE; }

      /**
        @brief Apply the Ls_out-by-Ls_in matrix to the input vector
        @param[in] parity Parity we are on
        @param[in] x_b Checkerboarded 4-d space-time index
        @param[in] s The output Ls dimension coordinate
       */
      __device__ __host__ inline void operator()(int x_cb, int s, int parity)
      {
        constexpr bool dagger = Arg::dagger;

        using real = typename Arg::real;
        using Vector = typename Arg::Vector;
        using matrix_type = typename Arg::matrix_type;

        const int Ls_in = arg.Ls_in;
        const int Ls_out = arg.Ls_out;
        const int volume_4d_cb = arg.volume_4d_cb;
        const matrix_type *wm_p = arg.wm_p;

        int thread_idx = target::thread_idx().y * target::block_dim().x + target::thread_idx().x;
        SharedMemoryCache<real> cache;
        while (thread_idx < static_cast<int>(Ls_out * Ls_in * sizeof(matrix_type) / sizeof(real))) {
          cache.data()[thread_idx] = reinterpret_cast<const real *>(wm_p)[thread_idx];
          thread_idx += target::block_dim().y * target::block_dim().x;
        }
        cache.sync();

        if (x_cb >= volume_4d_cb) return;
        if (s >= Ls_out) return;
        if (parity >= arg.nParity) return;

        Vector out;
        // t -> s_in, s-> s_out
        for (int t = 0; t < Ls_in; t++) {
          Vector in = arg.in(t * volume_4d_cb + x_cb, parity);
          int wm_index = dagger ? t * Ls_out + s : s * Ls_in + t;
          out += matrix_vector_multiply<dagger>(reinterpret_cast<const matrix_type *>(cache.data())[wm_index], in);
        }
        arg.out(s * volume_4d_cb + x_cb, parity) = out;
      }
    };
  } // namespace madwf_ml

} // namespace quda
