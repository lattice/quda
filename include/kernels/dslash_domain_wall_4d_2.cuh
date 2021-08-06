#pragma once

#include <shared_memory_cache_helper.cuh>
#include <kernels/dslash_domain_wall_4d.cuh>

namespace quda
{

  template <class Vector> __device__ __host__ inline Vector shuffle_colorspinor(const Vector &src)
  {
    Vector out;
#ifdef __CUDA_ARCH__
#pragma unroll
    for (int i = 0; i < Vector::size; i++) {
      out.data[i].real(__shfl_xor_sync(0xffffffff, src.data[i].real(), 1));
      out.data[i].imag(__shfl_xor_sync(0xffffffff, src.data[i].imag(), 1));
    }
#endif
    return out;
  }

  template <class Matrix, class Float, int Nc>
  __device__ __host__ inline ColorSpinor<Float, Nc, 2> stencil(const Matrix &u, const ColorSpinor<Float, Nc, 2> &vv,
                                                               int dim, int sign, const ColorSpinor<Float, Nc, 2> &ww,
                                                               int half_spinor_index)
  {
    ColorSpinor<Float, Nc, 2> out;
    using Coloror = ColorSpinor<Float, Nc, 1>;
    Coloror v;
    Coloror w;
    switch (dim) {
    case 0: // x dimension
      v = vv.get_coloror(0);
      w = shuffle_colorspinor(ww.get_coloror(1));
      switch (sign) {
      case 1: // positive projector
      {
        Coloror x = half_spinor_index ? v - i_(w) : v + i_(w);
        x = u * x;
        Coloror y = shuffle_colorspinor(half_spinor_index ? i_(x) : -i_(x));
        out = combine_half_spinors(x, y);
      } break;
      case -1: // negative projector
      {
        Coloror x = half_spinor_index ? v + i_(w) : v - i_(w);
        x = u * x;
        Coloror y = shuffle_colorspinor(half_spinor_index ? -i_(x) : i_(x));
        out = combine_half_spinors(x, y);
      } break;
      }
      break;
    case 1: // y dimension
      v = vv.get_coloror(0);
      w = shuffle_colorspinor(ww.get_coloror(1));
      switch (sign) {
      case 1: // positive projector
      {
        Coloror x = half_spinor_index ? v - w : v + w;
        x = u * x;
        Coloror y = shuffle_colorspinor(half_spinor_index ? -x : x);
        out = combine_half_spinors(x, y);
      } break;
      case -1: // negative projector
      {
        Coloror x = half_spinor_index ? v + w : v - w;
        x = u * x;
        Coloror y = shuffle_colorspinor(half_spinor_index ? x : -x);
        out = combine_half_spinors(x, y);
      } break;
      }
      break;
    case 2: // z dimension
      v = half_spinor_index ? vv.get_coloror(1) : vv.get_coloror(0);
      w = shuffle_colorspinor(half_spinor_index ? ww.get_coloror(0) : ww.get_coloror(1));
      switch (sign) {
      case 1: // positive projector
      {
        Coloror x = v + i_(w);
        x = u * x;
        Coloror y = shuffle_colorspinor(-i_(x));
        out = half_spinor_index ? combine_half_spinors(y, x) : combine_half_spinors(x, y);
      } break;
      case -1: // negative projector
      {
        Coloror x = v - i_(w);
        x = u * x;
        Coloror y = shuffle_colorspinor(i_(x));
        out = half_spinor_index ? combine_half_spinors(y, x) : combine_half_spinors(x, y);
      } break;
      }
      break;
    case 3: // t dimension
      v = vv.get_coloror(0);
      w = shuffle_colorspinor(ww.get_coloror(1));
      switch (sign) {
      case 1: // positive projector
      {
        Coloror x = half_spinor_index ? static_cast<Float>(2) * w : static_cast<Float>(2) * v;
        x = u * x;
        Coloror y = shuffle_colorspinor(x);
        ColorSpinor<Float, Nc, 2> zero;
        out = half_spinor_index ? zero : combine_half_spinors(x, y);
      } break;
      case -1: // negative projector
      {
        Coloror x = half_spinor_index ? static_cast<Float>(2) * v : static_cast<Float>(2) * w;
        x = u * x;
        Coloror y = shuffle_colorspinor(x);
        ColorSpinor<Float, Nc, 2> zero;
        out = half_spinor_index ? combine_half_spinors(x, y) : zero;
      } break;
      }
      break;
    }

    return out;
  }

  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg>
  __device__ __host__ inline void applyWilson2(ColorSpinor<typename Arg::real, Arg::nColor, 2> out[4], const Arg &arg, Coord &coord, int parity, int idx, int thread_dim,
                                               bool &active, int half_spinor_index)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    SharedMemoryCache<HalfVector> cache(target::block_dim());
    constexpr int s_batch = 4;

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
          = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // Do something
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
#pragma unroll
          for (int ss = 0; ss < s_batch; ss++) {
            // Load half spinor
            HalfVector in = arg.in.template operator()<2>(fwd_idx + (coord.s * s_batch + ss) * arg.dc.volume_4d_cb, their_spinor_parity,
                half_spinor_index); // 0 or 1

            out[ss] += stencil(U, in, d, proj_dir, in, half_spinor_index);
          }
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
#pragma unroll
          for (int ss = 0; ss < s_batch; ss++) {
            HalfVector in = arg.in.template operator()<2>(back_idx + (coord.s * s_batch + ss) * arg.dc.volume_4d_cb, their_spinor_parity,
                half_spinor_index);

            out[ss] += stencil(conj(U), in, d, proj_dir, in, half_spinor_index);
          }
        }
      }

    } // nDim
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4D2 : dslash_default {

    const Arg &arg;
    constexpr domainWall4D2(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      using HalfVector = ColorSpinor<real, Arg::nColor, 2>;

      int half_spinor_index = idx % 2;
      idx = idx / 2;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim; // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;

      constexpr int s_batch = 4;

      HalfVector stencil_out[s_batch];

      applyWilson2<nParity, dagger, mykernel_type>(stencil_out, arg, coord, parity, idx, thread_dim, active, half_spinor_index);

#pragma unroll
      for (int ss = 0; ss < s_batch; ss++) {
        int xs = coord.x_cb + (s * s_batch + ss) * arg.dc.volume_4d_cb;
#if 0
        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x = arg.x(xs, my_spinor_parity);
          out = x + arg.a_5[s] * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out(xs, my_spinor_parity);
          out = x + (xpay ? arg.a_5[s] * out : out);
        }
#endif
        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out.template operator()<2>(xs, my_spinor_parity, half_spinor_index) = stencil_out[ss];
      }
    }
  };

} // namespace quda
