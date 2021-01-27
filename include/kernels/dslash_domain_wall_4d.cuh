#pragma once

#include <kernels/dslash_wilson.cuh>

#include <mma_tensor_op/gemm.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
  struct DomainWall4DArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    int Ls;                             /** fifth dimension length */
    complex<real> a_5[QUDA_MAX_DWF_LS]; /** xpay scale factor for each 4-d subvolume */

    DomainWall4DArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                    const Complex *b_5, const Complex *c_5, bool xpay, const ColorSpinorField &x, int parity,
                    bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? a : 0.0, x, parity, dagger, comm_override),
      Ls(in.X(4))
    {
      if (b_5 == nullptr || c_5 == nullptr)
        for (int s = 0; s < Ls; s++) a_5[s] = a; // 4-d Shamir
      else
        for (int s = 0; s < Ls; s++) a_5[s] = 0.5 * a / (b_5[s] * (m_5 + 4.0) + 1.0); // 4-d Mobius
    }
  };

  template <class Coord, class Arg> __device__ bool is_out_of_bound(bool fwd, int d, Coord &coord, const Arg &arg)
  {
    switch (d) {
    case 0: return fwd ? (coord[0] + arg.nFace >= arg.dim[0]) : (coord[0] - arg.nFace < 0); break;
    case 1: return fwd ? (coord[1] + arg.nFace >= arg.dim[1]) : (coord[1] - arg.nFace < 0); break;
    case 2: return fwd ? (coord[2] + arg.nFace >= arg.dim[2]) : (coord[2] - arg.nFace < 0); break;
    case 3: return fwd ? (coord[3] + arg.nFace >= arg.dim[3]) : (coord[3] - arg.nFace < 0); break;
    }
    return false;
  }

  template <int Ls, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void alternative_dslash_kernel(Arg arg)
  {

    static_assert(kernel_type == INTERIOR_KERNEL, "Currently only for interior kernel.");
    // XXX: `blockDim.x` has to be equal to 1.

    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;
    int x_cb_4d = blockIdx.x;

    using real = typename mapper<typename Arg::Float>::type;
    using fixed_point_t = typename Arg::Float;
    using Spinor = ColorSpinor<real, 1, 4>;
    using Link = Matrix<complex<real>, Arg::nColor>;
    int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr int Nc = 3;

    constexpr int M = Ls * 4 * 2; // Ls * spin * complex
    // constexpr int N = Nc * 2;     // Nc * complex
    constexpr int K = Nc * 8; // Nc * 8-way stencil

    constexpr int bM = M;
    constexpr int bN = 8; // 8 -> 6
    constexpr int bK = K; // TODO: Change this for MMA.

    extern __shared__ float smem_ptr[];

    mma::SharedMemoryObject<real, bM, bK, 1, bM> smem_obj_a(smem_ptr);
    mma::SharedMemoryObject<real, bN, bK, 1, bN> smem_obj_b(smem_obj_a.ptr + bK * bM);
    mma::SharedMemoryObject<real, bM, bN, bN, 1> smem_obj_c(smem_obj_b.ptr + bK * bN);

    while (thread_idx < arg.Ls * K) {

      int s = thread_idx % Ls;         // fifth dimension
      int color_dir = thread_idx / Ls; // Joint idx for color and dir
      int color = color_dir % Nc;
      int dir = color_dir / Nc;
      int d = dir % 4;
      bool fwd = dir < 4;
      int fwd_bwd = 1 - (dir / 4) * 2; // +1 for fwd, -1 for bwd
      int proj_dir = dagger ? +fwd_bwd : -fwd_bwd;
      int dim;
      bool active;
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, x_cb_4d, s, parity, dim);
      int cs_idx = getNeighborIndexCB(coord, d, fwd_bwd, arg.dc);
      bool ghost = is_out_of_bound(fwd, d, coord, arg) && isActive<kernel_type>(active, dim, d, coord, arg);
      // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
      int gauge_parity = parity; // TODO: Change for 5d domain wall

      int gauge_idx = fwd ? coord.x_cb : cs_idx; // TODO: Change for 5d domain wall

      if (s == 0 && color == 0) {
        if (!ghost) {
          Link U = arg.U(d, gauge_idx, fwd ? gauge_parity : 1 - gauge_parity);
#pragma unroll
          for (int ci = 0; ci < 3; ci++) {
#pragma unroll
            for (int cj = 0; cj < 3; cj++) { // ind_n, ind_k
              smem_obj_b(cj * 2 + 0, ci * 8 + dir) = fwd ? +U(cj, ci).real() : +U(ci, cj).real();
              smem_obj_b(cj * 2 + 1, ci * 8 + dir) = fwd ? +U(cj, ci).imag() : -U(ci, cj).imag();
            }
          }
        } else {
#pragma unroll
          for (int ci = 0; ci < 3; ci++) {
#pragma unroll
            for (int cj = 0; cj < 3; cj++) { // ind_n, ind_k
              smem_obj_b(cj * 2 + 0, ci * 8 + dir) = 0;
              smem_obj_b(cj * 2 + 1, ci * 8 + dir) = 0;
            }
          }
        }
        // smem_obj_b.store<real>(fwd ? U : conj(U), dir);
      }

      // Load one color component of the color-spinor: Vector = 4x2 real's, or a "spinor"
      Spinor in = arg.in(cs_idx + s * arg.dc.volume_4d_cb, their_spinor_parity, color);
      // Spin project the coloror, and store to smem
      Spinor projected_in = in.project(d, proj_dir).reconstruct(d, proj_dir);
      if (!ghost) {
#pragma unroll
        for (int spin = 0; spin < 4; spin++) {
          // m, k, color == 0
          smem_obj_a(s * 8 + spin * 2 + 0, color * 8 + dir) = projected_in(spin, 0).real(); // real
          smem_obj_a(s * 8 + spin * 2 + 1, color * 8 + dir) = projected_in(spin, 0).imag(); // imag
        }
      } else {
#pragma unroll
        for (int spin = 0; spin < 4; spin++) {
          smem_obj_a(s * 8 + spin * 2 + 0, color * 8 + dir) = 0; // real
          smem_obj_a(s * 8 + spin * 2 + 1, color * 8 + dir) = 0; // imag
        }
      }

      thread_idx += blockDim.x * blockDim.y;
    }

    __syncthreads();

#if 0
      constexpr int MMA_M = 16;
      constexpr int MMA_N = 8;
      constexpr int MMA_K = 4;

      constexpr int tile_row_dim = bM / MMA_M; // number of tiles in the column dimension
      constexpr int tile_col_dim = bN / MMA_N; // number of tiles in the row dimension
      constexpr int tile_acc_dim = bK / MMA_K; // number of tiles in the accumulate dimension

      constexpr int total_warp = Ls * 8 / 32; // TODO: Change this .Total number of warps in the CTA

      constexpr int total_tile = tile_row_dim * tile_col_dim; // Total number of tiles dividing operand C
      constexpr int warp_cycle = total_tile / total_warp;     // Number of tiles each warp needs to calculate
#endif

#if 1
    for (int m = threadIdx.x; m < bM; m += blockDim.x) {
      for (int n = threadIdx.y; n < bN; n += blockDim.y) {
        smem_obj_c(m, n) = 0;
        for (int k = 0; k < bK; k++) smem_obj_c(m, n) += smem_obj_a(m, k) * smem_obj_b(n, k);
      }
    }
#else
#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {
      // The logical warp assigned to each part of the matrix.
      int logical_warp_index = wrm.warp_id * warp_cycle + c;
      int warp_row = logical_warp_index / tile_col_dim;
      int warp_col = logical_warp_index - warp_row * tile_col_dim;

      MmaOperandC op_c;

#pragma unroll
      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
        gemm(smem_obj_a, smem_obj_b, op_c, warp_row, warp_col, tile_k, wrm);
      }

      op_c.template store(smem_obj_c, warp_row, warp_col, wrm);
    }
#endif

    __syncthreads();

    __shared__ real reduce_buffer[Ls];

    thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (thread_idx < Ls) { reduce_buffer[thread_idx] = 0; }
    __syncthreads();
    while (thread_idx < Ls * 12) {
      int s = thread_idx % Ls;
      int spin_color = thread_idx / Ls;
      int spin = spin_color / 3;
      int color = spin_color % 3;
      real p_real = smem_obj_c(s * 8 + spin * 2 + 0, color * 2 + 0) - smem_obj_c(s * 8 + spin * 2 + 1, color * 2 + 1);
      real p_imag = smem_obj_c(s * 8 + spin * 2 + 1, color * 2 + 0) + smem_obj_c(s * 8 + spin * 2 + 0, color * 2 + 1);

      atomicMax(reinterpret_cast<unsigned *>(&reduce_buffer[s]),
                __float_as_uint(fabsf(p_real) > fabsf(p_imag) ? fabsf(p_real) : fabsf(p_imag)));

      thread_idx += blockDim.x * blockDim.y;
    }

    __syncthreads();

    thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (thread_idx < Ls) { arg.out.norm[x_cb_4d + arg.dc.volume_4d_cb * thread_idx] = reduce_buffer[thread_idx]; }
    while (thread_idx < Ls * 12) {
      int s = thread_idx % Ls;
      int spin_color = thread_idx / Ls;
      int spin = spin_color / 3;
      int color = spin_color % 3;
      real fp_real = smem_obj_c(s * 8 + spin * 2 + 0, color * 2 + 0) - smem_obj_c(s * 8 + spin * 2 + 1, color * 2 + 1);
      real fp_imag = smem_obj_c(s * 8 + spin * 2 + 1, color * 2 + 0) + smem_obj_c(s * 8 + spin * 2 + 0, color * 2 + 1);
      fixed_point_t fixed_real
        = static_cast<fixed_point_t>(fp_real * fixedMaxValue<fixed_point_t>::value / reduce_buffer[s]);
      fixed_point_t fixed_imag
        = static_cast<fixed_point_t>(fp_imag * fixedMaxValue<fixed_point_t>::value / reduce_buffer[s]);

      arg.out(x_cb_4d + arg.dc.volume_4d_cb * s, parity, spin, color) = complex<fixed_point_t>(fixed_real, fixed_imag);

      thread_idx += blockDim.x * blockDim.y;
    }
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4D : dslash_default {

    Arg &arg;
    constexpr domainWall4D(Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      applyWilson<nParity, dagger, kernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
      if (xpay && kernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(xs, my_spinor_parity);
        out = x + arg.a_5[s] * out;
      } else if (kernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(xs, my_spinor_parity);
        out = x + (xpay ? arg.a_5[s] * out : out);
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
    }

    static constexpr bool has_alternative_kernel() { return sizeof(typename Arg::Float) == 2; }

    /** XXX: This is for testing only
       @brief This is a helper function to luanch the dslash kernel with strategy that is
       different from the "one-thread-per-site" one.
    */
    static void launch(TuneParam &tp, const qudaStream_t &stream, Arg &arg)
    {
      TuneParam tune_param(tp);

      tune_param.set_max_shared_bytes = true;

      // If Ls == 12, 12 * 8 =  96 = 3 warps.
      // If Ls == 16, 16 * 8 = 128 = 4 warps.
      // TODO: Tune block.x
      tune_param.block.x = 8;
      tune_param.block.y = arg.dc.Ls;
      tune_param.block.z = 1;

      tune_param.grid.x = arg.dc.volume_4d_cb;
      tune_param.grid.y = 1;
      tune_param.grid.z = nParity;

      tune_param.shared_bytes = (arg.dc.Ls * 8 * 24 + 8 * 24 + arg.dc.Ls * 8 * 8) * sizeof(float);

      qudaLaunchKernel(alternative_dslash_kernel<12, nParity, dagger, xpay, kernel_type, Arg>, tune_param, stream, arg);
    }
  };

} // namespace quda
