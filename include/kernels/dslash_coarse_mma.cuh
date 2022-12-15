#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <shared_memory_cache_helper.h>
#include <kernel.h>
#include <warp_collective.h>
#include <dslash_quda.h>

namespace quda {

  enum DslashType {
    DSLASH_INTERIOR,
    DSLASH_EXTERIOR,
    DSLASH_FULL
  };

  // we use two colors per thread unless we have large dim_stride, when we're aiming for maximum parallelism
  constexpr int colors_per_thread(int nColor, int dim_stride) { return (nColor % 2 == 0 && nColor <= 32 && dim_stride <= 2) ? 2 : 1; }

  template <bool dslash_, bool clover_, bool dagger_, DslashType type_, int color_stride_, int dim_stride_, typename Float,
            typename yFloat, typename ghostFloat, int nSpin_, int nColor_, bool native>
  struct DslashCoarseArg : kernel_param<> {
    static constexpr bool dslash = dslash_;
    static constexpr bool clover = clover_;
    static constexpr bool dagger = dagger_;
    static constexpr DslashType type = type_;
    static constexpr int color_stride = color_stride_;
    static constexpr int dim_stride = dim_stride_;

    using real = typename mapper<Float>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nDim = 4;
    static constexpr int nFace = 1;

    static constexpr QudaFieldOrder csOrder = native ? colorspinor::getNative<real>(nSpin) : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    static constexpr QudaGaugeFieldOrder gOrder = native ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;

    using G = typename colorspinor::GhostOrder<real, nSpin, nColor, 1, csOrder, Float, ghostFloat>;
    // disable ghost to reduce arg size
    using F = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, csOrder, Float, ghostFloat, true>;
    using GY = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;

    static constexpr unsigned int max_n_src = 64;
    const int_fastdiv n_src;
    F out[max_n_src];
    F inA[max_n_src];
    F inB[max_n_src];
    G halo;
    const GY Y;
    const GY X;
    const real kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;
    int ghostFaceCB[4];

    DslashCoarseArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                    cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                    real kappa, int parity, const ColorSpinorField &halo) :
      kernel_param(dim3(color_stride * X.VolumeCB(), out[0].SiteSubset() * out.size(),
                        2 * dim_stride * 2 * (nColor / colors_per_thread(nColor, dim_stride)))),
      n_src(out.size()),
      halo(halo, nFace),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out[0].SiteSubset()),
      X0h(((3 - nParity) * out[0].X(0)) / 2),
      dim {(3 - nParity) * out[0].X(0), out[0].X(1), out[0].X(2), out[0].X(3), out[0].Ndim() == 5 ? out[0].X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out[0].VolumeCB() / dim[4])
    {
      if (out.size() > max_n_src) errorQuda("vector set size %lu greater than max size %d", out.size(), max_n_src);
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->inA[i] = inA[i];
        this->inB[i] = inB[i];
      }
      // ghostFaceCB does not include the batch (5th) dimension at present
      for (int i = 0; i < 4; i++) ghostFaceCB[i] = halo.getDslashConstant().ghostFaceCB[i];
    }
  };

  /**
     @brief Helper function to determine if should halo computation
  */
  template <DslashType type>
  static constexpr bool doHalo() {
    switch(type) {
    case DSLASH_EXTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     @brief Helper function to determine if should interior computation
  */
  template <DslashType type>
  static constexpr bool doBulk() {
    switch(type) {
    case DSLASH_INTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     Applies the coarse dslash on a given parity and checkerboard site index
     /out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)

     @param[in,out] out The result vector
     @param[in] thread_dir Direction
     @param[in] x_cb The checkerboarded site index
     @param[in] src_idx Which src are we working on
     @param[in] parity The site parity
     @param[in] s_row Which spin row are acting on
     @param[in] color_block Which color row are we acting on
     @param[in] color_off Which color column offset are we acting on
     @param[in] arg Arguments
   */
  template <typename V, typename Arg>
  __device__ __host__ inline void applyDslashMma(V &out, int x_cb, int parity, const Arg &arg)
  {
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    int coord[4];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    if (!thread_dir || target::is_host()) {

      //Forward gather - compute fwd offset for spinor fetch
#pragma unroll
      for(int d = 0; d < Arg::nDim; d++) // loop over dimension
      {
        const int fwd_idx = linkIndexP1(coord, arg.dim, d);

        if (arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
          if constexpr (doHalo<Arg::type>()) {
#if 0
            int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
#pragma unroll
            for(int color_local = 0; color_local < Mc; color_local++) { //Color row
              int c_row = color_block + color_local; // global color index
              int row = s_row * Arg::nColor + c_row;
#pragma unroll
              for(int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
                for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                  int col = s_col * Arg::nColor + c_col + color_offset;
                  if (!Arg::dagger)
                    out[color_local] = cmac(arg.Y(d+4, parity, x_cb, row, col),
                        arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                  else
                    out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                        arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                }
              }
            }
#endif
          }
        } else if constexpr (doBulk<Arg::type>()) {
#if 1
          constexpr int M = nSpin * nColor;
          constexpr int N = nVec;
          constexpr int K = nSpin * nColor;

          constexpr int lda = N;
          constexpr int ldb = N;
          constexpr int ldc = N;

          using mma_t = smma_t;
          using Config = MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

          auto a = arg.Y(Arg::dagger ? d : d + 4, parity, x_cb, 0, 0);
          auto b = arg.inA(their_spinor_parity, fwd_idx, 0, 0, 0);

          Config::template perform_mma<a_dagger, b_dagger, false>(a, b, c, 0, 0);
#else
#pragma unroll
          for(int color_local = 0; color_local < Mc; color_local++) { //Color row
            int c_row = color_block + color_local; // global color index
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for(int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
              for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local] = cmac(arg.Y(d+4, parity, x_cb, row, col),
                      arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col+color_offset), out[color_local]);
                else
                  out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                      arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col+color_offset), out[color_local]);
              }
            }
          }
#endif
        }
      } // nDim
    }

    if (thread_dir || target::is_host()) {

      //Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
      for(int d = thread_dim; d < Arg::nDim; d += Arg::dim_stride)
      {
        const int back_idx = linkIndexM1(coord, arg.dim, d);
        if (arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
          if constexpr (doHalo<Arg::type>()) {
#if 0
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
#pragma unroll
            for (int color_local=0; color_local<Mc; color_local++) {
              int c_row = color_block + color_local;
              int row = s_row * Arg::nColor + c_row;
#pragma unroll
              for (int s_col=0; s_col < Arg::nSpin; s_col++)
#pragma unroll
                for (int c_col=0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                  int col = s_col * Arg::nColor + c_col + color_offset;
                  if (!Arg::dagger)
                    out[color_local] = cmac(conj(arg.Y.Ghost(d, 1-parity, ghost_idx, col, row)),
                        arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                  else
                    out[color_local] = cmac(conj(arg.Y.Ghost(d+4, 1-parity, ghost_idx, col, row)),
                        arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                }
            }
#endif
          }
        } else if constexpr (doBulk<Arg::type>()) {
          const int gauge_idx = back_idx;
#if 1
          constexpr int M = nSpin * nColor;
          constexpr int N = nVec;
          constexpr int K = nSpin * nColor;

          constexpr int lda = N;
          constexpr int ldb = N;
          constexpr int ldc = N;

          using mma_t = smma_t;
          using Config = MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

          auto a = arg.Y(Arg::dagger ? d + 4: d, 1 - parity, gauge_idx, 0, 0);
          auto b = arg.inA(their_spinor_parity, fwd_idx, 0, 0, 0);

          constexpr bool a_dagger = true;
          constexpr bool b_dagger = false;
          Config::template perform_mma<a_dagger, b_dagger, false>(a, b, c, 0, 0);
#else
#pragma unroll
          for(int color_local = 0; color_local < Mc; color_local++) {
            int c_row = color_block + color_local;
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for(int s_col = 0; s_col < Arg::nSpin; s_col++)
#pragma unroll
              for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local] = cmac(conj(arg.Y(d, 1-parity, gauge_idx, col, row)),
                      arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col+color_offset), out[color_local]);
                else
                  out[color_local] = cmac(conj(arg.Y(d+4, 1-parity, gauge_idx, col, row)),
                      arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col+color_offset), out[color_local]);
              }
          }
#endif
        }

      } //nDim
    } // forwards / backwards thread split
  }

  /**
     Applies the coarse clover matrix on a given parity and
     checkerboard site index

     @param out The result out += X * in
     @param X The coarse clover field
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <int Mc, typename V, typename Arg>
    __device__ __host__ inline void applyClover(V &out, const Arg &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset)
    {
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;

      // M is number of colors per thread
#pragma unroll
      for(int color_local = 0; color_local < Mc; color_local++) {//Color out
        int c = color_block + color_local; // global color index
        int row = s * Arg::nColor + c;
#pragma unroll
        for (int s_col = 0; s_col < Arg::nSpin; s_col++) //Spin in
#pragma unroll
          for (int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color in
            //Factor of kappa and diagonal addition now incorporated in X
            int col = s_col * Arg::nColor + c_col + color_offset;
            if (!Arg::dagger) {
              out[color_local] = cmac(arg.X(0, parity, x_cb, row, col), arg.inB[src_idx](spinor_parity, x_cb, s_col, c_col+color_offset), out[color_local]);
            } else {
              out[color_local] = cmac(conj(arg.X(0, parity, x_cb, col, row)), arg.inB[src_idx](spinor_parity, x_cb, s_col, c_col+color_offset), out[color_local]);
            }
          }
      }
    }

  template <typename Arg> struct CoarseDslashMma {
    const Arg &arg;
    constexpr CoarseDslash(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int parity_x_cb, int thread_idx_y, int thread_idx_z)
    {
      int parity = parity_x_cb % 2;
      int x_cb = parity_x_cb / 2;

      using mma_t = smma_t<Arg::Float>;

      accumulator out;

      applyDslashMma ...;

      out *= -arg.kappa;

      applyClover;

      write out ;     
    }
  };

} // namespace quda
