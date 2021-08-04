#pragma once

#include <shared_memory_cache_helper.cuh>
#include <kernels/dslash_domain_wall_4d.cuh>

namespace quda
{

  template <class Matrix, class Vector>
  __device__ __host__ inline Vector stencil(const Matrix &u, const Vector &v, int d, int proj_dir, const Vector &w)
  {
    Vector out;
    return out;
  }

  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg>
  __device__ __host__ inline auto applyWilson2(const Arg &arg, Coord &coord, int parity, int idx, int thread_dim, bool &active, int half_spinor_index)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    HalfVector out;

    SharedMemoryCache<HalfVector> cache(target::block_dim());

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
          // Load half spinor
          // Vector in = arg.in(fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
          HalfVector in = arg.in.template operator()<2>(fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity, half_spinor_index); // 0 or 1
#ifdef __CUDA_ARCH__
          if (d > 0) { __syncwarp(); }
#endif
          cache.save(in);
#ifdef __CUDA_ARCH__
          __syncwarp();
#endif
          out += U * in.project(d, proj_dir, cache, 1 - half_spinor_index * 2);
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
          HalfVector in = arg.in.template operator()<2>(back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity, half_spinor_index);

#ifdef __CUDA_ARCH__
          __syncwarp();
#endif
          cache.save(in);

#ifdef __CUDA_ARCH__
          __syncwarp();
#endif
          out += conj(U) * in.project(d, proj_dir, cache, 1 - half_spinor_index * 2);
        }
      }

    } // nDim

    return out;
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4D2 : dslash_default {

    const Arg &arg;
    constexpr domainWall4D2(const Arg &arg) : arg(arg) {}
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
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;

      HalfVector stencil_out = applyWilson2<nParity, dagger, mykernel_type>(arg, coord, parity, idx, thread_dim, active, half_spinor_index);

      SharedMemoryCache<HalfVector> cache(target::block_dim());
#ifdef __CUDA_ARCH__ 
      __syncwarp();
#endif
      cache.save(stencil_out);
#ifdef __CUDA_ARCH__ 
      __syncwarp();
#endif

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
#if 0
      if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(xs, my_spinor_parity);
        out = x + arg.a_5[s] * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(xs, my_spinor_parity);
        out = x + (xpay ? arg.a_5[s] * out : out);
      }
#endif 
      if (half_spinor_index == 0) {
        auto tid = target::thread_idx();
        Vector out = combine_half_spinors(stencil_out, cache.load(tid.x + 1, tid.y, tid.z));
        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
      }
    }
  };

} // namespace quda
