#include <multigrid_helper.cuh>

#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <constant_kernel_arg.h> // allow for large parameter structs
#include <block_reduce_helper.h>
#include <fast_intdiv.h>
#include <array.h>
#include <block_reduction_kernel.h>

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
//#define SWIZZLE

namespace quda {

  // number of vectors to simultaneously orthogonalize
  template <int nColor, int nVec, int block_size> constexpr int tile_size()
  {
    return nColor == 3 && block_size < 1024 ? (nVec % 4 == 0 ? 4 : nVec % 3 == 0 ? 3 : 2) : 1;
  }

  template <int nColor, int nVec> constexpr int tile_size(int block_size)
  {
    return nColor == 3 && block_size < 1024 ? (nVec % 4 == 0 ? 4 : nVec % 3 == 0 ? 3 : 2) : 1;
  }

  /**
      Kernel argument struct
  */
  template <bool is_device_, typename vFloat, typename Rotator, typename Vector, int fineSpin_, int nColor_, int coarseSpin_, int nVec_>
  struct BlockOrthoArg : kernel_param<> {
    using sum_t = double;
    using real = typename mapper<vFloat>::type;
    static constexpr bool is_device = is_device_;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int chiral_blocks = fineSpin == 1 ? 2 : coarseSpin;
    static constexpr int nVec = nVec_;
    Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    const int nBlockOrtho; // number of times we Gram-Schmidt
    int coarseVolume;
    int fineVolumeCB;
    int aggregate_size_cb; // number of geometric elements in each checkerboarded block

    static constexpr bool swizzle = false;
    int_fastdiv swizzle_factor; // for transposing blockIdx.x mapping to coarse grid coordinate

    const Vector B[nVec];

    static constexpr bool launch_bounds = true;
    dim3 grid_dim;
    dim3 block_dim;

    template <typename... T>
    BlockOrthoArg(ColorSpinorField &V, const int *fine_to_coarse, const int *coarse_to_fine, int parity,
                  const int *geo_bs, const int n_block_ortho, const ColorSpinorField &meta, T... B) :
      kernel_param(dim3(meta.VolumeCB() * (fineSpin > 1 ? meta.SiteSubset() : 1), 1, chiral_blocks)),
      V(V),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(meta.SiteSubset()),
      nBlockOrtho(n_block_ortho),
      fineVolumeCB(meta.VolumeCB()),
      B{*B...},
      grid_dim(),
      block_dim()
    {
      int aggregate_size = 1;
      for (int d = 0; d < V.Ndim(); d++) aggregate_size *= geo_bs[d];
      aggregate_size_cb = aggregate_size / 2;
      coarseVolume = meta.Volume() / aggregate_size;
      if (nParity != 2) errorQuda("BlockOrtho only presently supports full fields");
    }
  };

  template <typename Arg> struct BlockOrtho_ {
    const Arg &arg;
    static constexpr unsigned block_size = Arg::block_size;
    static constexpr int fineSpin = Arg::fineSpin;
    static constexpr int spinBlock = (fineSpin == 1) ? 1 : fineSpin / Arg::coarseSpin; // size of spin block
    static constexpr int nColor = Arg::nColor;

    // on the device we rely on thread parallelism, where as on the host we assign a vector of block_size to each thread
    static constexpr int n_sites_per_thread = Arg::is_device ? 1 : block_size;

    // on the device we expect number of active threads equal to block_size, and on the host just a single thread
    static constexpr int n_threads_per_block = Arg::is_device ? block_size : 1;

    static_assert(n_sites_per_thread * n_threads_per_block == block_size,
                  "Product of n_sites_per_thread and n_threads_per_block must equal block_size");

    // mVec is the number of vectors to orthogonalize at once
    static constexpr int mVec = tile_size<nColor, Arg::nVec, block_size>();
    static_assert(Arg::nVec % mVec == 0, "mVec must be a factor of nVec");

    using sum_t = typename Arg::sum_t;
    using dot_t = array<complex<sum_t>, mVec>;
    using real = typename Arg::real;

    constexpr BlockOrtho_(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void load(ColorSpinor<real, nColor, spinBlock> &v, int parity, int x_cb, int chirality, int i)
    {
#pragma unroll
      for (int s = 0; s < spinBlock; s++)
#pragma unroll
        for (int c = 0; c < nColor; c++) v(s, c) = arg.V(parity, x_cb, chirality * spinBlock + s, c, i);
    }

    __device__ __host__ inline void save(int parity, int x_cb, int chirality, int i, const ColorSpinor<real, nColor, spinBlock> &v)
    {
#pragma unroll
      for (int s = 0; s < spinBlock; s++)
#pragma unroll
        for (int c = 0; c < nColor; c++) arg.V(parity, x_cb, chirality * spinBlock + s, c, i) = v(s, c);
    }

    __device__ __host__ inline void operator()(dim3 block, dim3 thread)
    {
      int x_coarse = block.x;
      int x_fine_offset = thread.x;
      int chirality = block.z;

      int parity[n_sites_per_thread];
      int x_offset_cb[n_sites_per_thread];
      int x_cb[n_sites_per_thread];

      for (int tx = 0; tx < n_sites_per_thread; tx++) {
        int x_fine_offset_tx = x_fine_offset * n_sites_per_thread + tx;
        // all threads with x_fine_offset greater than aggregate_size_cb are second parity
        int parity_offset = (x_fine_offset_tx >= arg.aggregate_size_cb && fineSpin != 1) ? 1 : 0;
        x_offset_cb[tx] = x_fine_offset_tx - parity_offset * arg.aggregate_size_cb;
        parity[tx] = fineSpin == 1 ? chirality : arg.nParity == 2 ? parity_offset : arg.parity;

        x_cb[tx] = x_offset_cb[tx] >= arg.aggregate_size_cb ? 0 :
          arg.coarse_to_fine[ (x_coarse*2 + parity[tx]) * arg.aggregate_size_cb + x_offset_cb[tx] ] - parity[tx]*arg.fineVolumeCB;
      }
      if (fineSpin == 1) chirality = 0; // when using staggered chirality is mapped to parity

      constexpr int block_dim = 1;
      BlockReduce<dot_t, block_dim> dot_reducer{0};
      BlockReduce<sum_t, block_dim> norm_reducer{0};

      // loop over number of block orthos
      for (int n = 0; n < arg.nBlockOrtho; n++) {
        for (int j = 0; j < Arg::nVec; j += mVec) {

          ColorSpinor<real, nColor, spinBlock> v[mVec][n_sites_per_thread];

          for (int tx = 0; tx < n_sites_per_thread; tx++) {
            if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
            if (n == 0) { // load from B on first Gram-Schmidt, otherwise V.
              if (chirality == 0) {
#pragma unroll
                for (int m = 0; m < mVec; m++) arg.B[j+m].template load<spinBlock>(v[m][tx].data, parity[tx], x_cb[tx], 0);
              } else {
#pragma unroll
                for (int m = 0; m < mVec; m++) arg.B[j+m].template load<spinBlock>(v[m][tx].data, parity[tx], x_cb[tx], 1);
              }
            } else {
#pragma unroll
              for (int m = 0; m < mVec; m++) load(v[m][tx], parity[tx], x_cb[tx], chirality, j + m);
            }
          }

          for (int i = 0; i < j; i++) { // compute (j,i) block inner products
            ColorSpinor<real, nColor, spinBlock> vi[n_sites_per_thread];

            dot_t dot{0};
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
              load(vi[tx], parity[tx], x_cb[tx], chirality, i);

#pragma unroll
              for (int m = 0; m < mVec; m++) dot[m] += innerProduct(vi[tx], v[m][tx]);
            }

            dot = dot_reducer.template AllSum<false>(dot);

            // subtract the blocks to orthogonalise
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
#pragma unroll
              for (int m = 0; m < mVec; m++) caxpy(-complex<real>(dot[m].real(), dot[m].imag()), vi[tx], v[m][tx]);
            }
          } // i

          // now orthogonalize over the block diagonal and normalize each entry
#pragma unroll
          for (int m = 0; m < mVec; m++) {

            dot_t dot{0};
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
#pragma unroll
              for (int i = 0; i < m; i++) dot[i] += innerProduct(v[i][tx], v[m][tx]);
            }
            
            dot = dot_reducer.template AllSum<false>(dot);
            
            sum_t nrm = 0.0;
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
#pragma unroll
              for (int i = 0; i < m; i++) caxpy(-complex<real>(dot[i].real(), dot[i].imag()), v[i][tx], v[m][tx]); // subtract the blocks to orthogonalise
              nrm += norm2(v[m][tx]);
            }

            nrm = norm_reducer.template AllSum<false>(nrm);
            auto nrm_inv = nrm > 0.0 ? quda::rsqrt(nrm) : 0.0;

            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
              v[m][tx] *= nrm_inv;
            }
          }

          for (int tx = 0; tx < n_sites_per_thread; tx++) {
            if (x_offset_cb[tx] >= arg.aggregate_size_cb) break;
#pragma unroll
            for (int m = 0; m < mVec; m++) save(parity[tx], x_cb[tx], chirality, j + m, v[m][tx]);
          }
        } // j
      }   // n
    }
  };

} // namespace quda
