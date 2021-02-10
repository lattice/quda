#pragma once

#include <kernels/dslash_wilson.cuh>

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

  constexpr int warp_size = 32;

  template <class OpA>
    __device__ inline void mma(float op_c[], const OpA &op_a, float op_b) {
#pragma unroll
      for (int r = 0; r < 2; r++) {
        asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(op_c[r * 4 + 0 + 0]),
              "+f"(op_c[r * 4 + 0 + 1]),
              "+f"(op_c[r * 4 + 2 + 0]),
              "+f"(op_c[r * 4 + 2 + 1])
            : "r"(__float_as_uint(op_a(r * 2 + 0).real())),
              "r"(__float_as_uint(op_a(r * 2 + 1).real())),
              "r"(__float_as_uint(op_b)));
      }
      if ((threadIdx.x / 4) % 2 == 1) { op_b = -op_b; }
#pragma unroll
      for (int r = 0; r < 2; r++) {
        // Note the reverted order of real and imag for op_c
        asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(op_c[r * 4 + 0 + 1]),
              "+f"(op_c[r * 4 + 0 + 0]),
              "+f"(op_c[r * 4 + 2 + 1]),
              "+f"(op_c[r * 4 + 2 + 0])
            : "r"(__float_as_uint(op_a(r * 2 + 0).imag())),
              "r"(__float_as_uint(op_a(r * 2 + 1).imag())),
              "r"(__float_as_uint(op_b))); 
      }
    }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void specialized_dslash_kernel(Arg arg)
  {
    static_assert(kernel_type == INTERIOR_KERNEL, "Currently only for interior kernel.");

    constexpr int Ls = 16;
    constexpr int s_per_warp = 8;

    int thread_idx = threadIdx.x;
    int lane_id = thread_idx % warp_size;
    int warp_id = thread_idx / warp_size;

    int warp_k = lane_id % 4;
    int warp_m = lane_id / 4;
    int warp_n = warp_m;

    constexpr int reg_length_a = (s_per_warp * 8 / 16) * 2;
    constexpr int reg_length_c = (s_per_warp * 8 / 16) * 2;

    using float_type = typename mapper<typename Arg::Float>::type;
    using fixed_type = typename Arg::Float;
    using Spinor = ColorSpinor<float_type, 1, 4>;

    Spinor projected_in;
    float op_c[reg_length_c];
    float op_b;

#pragma unroll
    for (int c = 0; c < reg_length_c; c++) { op_c[c] = 0; }

    fixed_type op_b_fixed;

    int color = warp_k;
    int s = warp_m + warp_id * s_per_warp;

    int color_m = warp_k;
    int color_n = warp_n / 2;
    int comp = warp_n % 2;

    int x_cb = blockIdx.x;

    constexpr int Ns = Arg::nSpin;
    constexpr int Nc = Arg::nColor;

    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;
    int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    const int gauge_parity = parity;

    int thread_dim;
    bool active;
    auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, x_cb, s, parity, thread_dim);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = x_cb;
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
          = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (s < Ls  && warp_k < Nc && !ghost) {
          Spinor in = arg.in(fwd_idx + s * arg.dc.volume_4d_cb, their_spinor_parity, color);
          // Spinor in = arg.in(fwd_idx * Ls + s, their_spinor_parity, color);
          projected_in = in.project(d, proj_dir).reconstruct(d, proj_dir);
        } else {
#pragma unroll
          for (int r = 0; r < reg_length_a / 2; r++) { projected_in(r).real(0); projected_in(r).imag(0); }
        }

        if (warp_k < Nc && warp_n < Nc * 2 && !ghost) {
          op_b_fixed = arg.U(gauge_idx, d, gauge_parity, color_n, color_m, comp); // color i, j, complex
          op_b = static_cast<float_type>(op_b_fixed) * fixedInvMaxValue<fixed_type>::value;
        } else {
          op_b = 0;          
        }

        mma(op_c, projected_in, op_b);
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = back_idx;
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (s < Ls  && warp_k < Nc && !ghost) {
          Spinor in = arg.in(back_idx + s * arg.dc.volume_4d_cb, their_spinor_parity, color);
          // Spinor in = arg.in(back_idx * Ls + s, their_spinor_parity, color);
          projected_in = in.project(d, proj_dir).reconstruct(d, proj_dir);
        } else {
#pragma unroll
          for (int r = 0; r < reg_length_a / 2; r++) { projected_in(r).real(0); projected_in(r).imag(0); }
        }

        if (warp_k < Nc && warp_n < Nc * 2 && !ghost) {
          op_b_fixed = arg.U(gauge_idx, d, 1 - gauge_parity, color_m, color_n, comp); // color i, j, complex
          if (comp == 1) { op_b_fixed = -op_b_fixed; }
          op_b = static_cast<float_type>(op_b_fixed) * fixedInvMaxValue<fixed_type>::value;
        } else {
          op_b = 0;          
        }

        mma(op_c, projected_in, op_b);
      }
    } // nDim

    unsigned max = __float_as_uint(0);

#pragma unroll
    for (int c = 0; c < reg_length_c; c++) {
      unsigned x = __float_as_uint(fabs(op_c[c]));
      if (max < x) { max = x; }
    }

#pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
      unsigned fetch = __shfl_down_sync(0xffffffff, max, offset);
      if (max < fetch) { max = fetch; }
    }
    max = __shfl_sync(0xffffffff, max, (lane_id / 4) * 4);

    // TODO: Fix the parity.
    if (s < Ls && color == 0) arg.out.norm[x_cb + arg.dc.volume_4d_cb * s] = __uint_as_float(max);
    // if (s < Ls && color == 0) arg.out.norm[x_cb * Ls + s] = __uint_as_float(max);

    complex<fixed_type> output[Ns];
    float_type scale = fixedMaxValue<fixed_type>::value / __uint_as_float(max);
#pragma unroll
    for (int spin = 0; spin < 4; spin++) {
      fixed_type fixed_real = static_cast<fixed_type>(op_c[spin * 2 + 0] * scale);
      fixed_type fixed_imag = static_cast<fixed_type>(op_c[spin * 2 + 1] * scale);
      output[spin].real(fixed_real);
      output[spin].imag(fixed_imag);
    }
    if (s < Ls && color < 3) arg.out.save_spinor(output, x_cb + arg.dc.volume_4d_cb * s, parity, color);
    // if (s < Ls && color < 3) arg.out.save_spinor(output, x_cb * Ls + s, parity, color);
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct domainWall4D;

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4DImpl: public Dslash<domainWall4D, Arg> {

    using Dslash = Dslash<domainWall4D, Arg>;
    using Dslash::arg;
    using Dslash::in;
    using Dslash::out;

    domainWall4DImpl(const ColorSpinorField &out, const ColorSpinorField &in, Arg &arg): Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      tp.set_max_shared_bytes = true;

      qudaLaunchKernel(specialized_dslash_kernel<nParity, dagger, xpay, kernel_type, Arg>, tp, stream, arg);
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool advanceTuneParam(TuneParam &param) const {
      return Tunable::advanceSharedBytes(param);
    }

    virtual int blockStep() const { return 64; }
    virtual int blockMin() const { return 64; }
    virtual int blockMax() const { return 64; }

    void initTuneParam(TuneParam &param) const
    {
      param.block.x = blockMin();
      param.block.y = 1;
      param.block.z = 1;

      param.grid.x = arg.dc.volume_4d_cb;
      param.grid.y = 1;
      param.grid.z = nParity;

      param.shared_bytes = sharedBytesPerBlock(param);
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), ","); }

    void preTune() {}  // FIXME - use write to determine what needs to be saved
    void postTune() {} // FIXME - use write to determine what needs to be saved
 
  };

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

    static constexpr bool has_specialized_kernel() { return sizeof(typename Arg::Float) == 2 && kernel_type == INTERIOR_KERNEL; }

    static void specialized_launch(const qudaStream_t &stream, const ColorSpinorField &out, const ColorSpinorField &in, Arg &arg) {
      domainWall4DImpl<nParity, dagger, xpay, kernel_type, Arg> impl(out, in, arg);
      impl.apply(stream);
    }
  };

} // namespace quda
