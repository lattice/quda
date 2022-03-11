#pragma once

#include <constant_kernel_arg.h>
#include <kernels/dslash_domain_wall_4d.cuh>
#include <kernels/dslash_domain_wall_m5_mma.cuh>
#include <target_device.h>

namespace quda
{

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4DM5invMma : dslash_default {

    static constexpr Dslash5Type dslash5_type = Arg::type;

    const Arg &arg;
    constexpr domainWall4DM5invMma(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int, int s, int parity)
    {
#ifdef __CUDA_ARCH__
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool idle = false;
      int idx_base = target::block_idx().x * target::block_dim().x; // blockIdx.x * blockDim.x; // base.
      auto op_a = construct_m5inv<dagger>(arg);

      while (idx_base < arg.volume_4d_cb) {
        int idx = idx_base + target::thread_idx().x; // threadIdx.x;
        if (idx >= arg.volume_4d_cb) { idle = true; }

        bool active
          = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)

        const int my_spinor_parity = nParity == 2 ? parity : 0;
        int thread_dim; // which dimension is thread working on (fused kernel only)
        auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, 0, thread_dim);
        Vector stencil_out;
        if (!idle) {
          applyWilson<nParity, dagger, mykernel_type>(stencil_out, arg, coord, parity, idx, thread_dim, active);
        }

        Vector out;
        // In the following `x_cb` are all passed as `x_cb = 0`, since it will not be used if `shared = true`, and `shared = true`
        int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
        // Apply the m5inv.
        out = m5inv_mma(op_a, stencil_out, arg);

        if (!idle) {
          if (xpay && mykernel_type == INTERIOR_KERNEL) {
            Vector x = arg.x(xs, my_spinor_parity);
            out = x + arg.a_5[s] * out;
          } else if (mykernel_type != INTERIOR_KERNEL && active) {
            Vector x = arg.out(xs, my_spinor_parity);
            out = x + (xpay ? arg.a_5[s] * out : out);
          }
          if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
        }

        idx_base += target::grid_dim().x * target::block_dim().x; // gridDim.x * blockDim.x;
      }
#endif
    }
  };

} // namespace quda
