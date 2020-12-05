#include <multigrid_helper.cuh>
#include <fast_intdiv.h>

// this removes ghost accessor reducing the parameter space needed
#define DISABLE_GHOST true // do not rename this (it is both a template parameter and a macro)

#include <color_spinor_field_order.h>
#include <cub_helper.cuh>

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
//#define SWIZZLE

namespace quda {

  // number of vectors to simultaneously orthogonalize
  template <int nColor, int nVec> constexpr int tile_size() { return nColor == 3 ? (nVec % 4 == 0 ? 4 : nVec % 3 == 0 ? 3 : 2) : 1; }

  /**
      Kernel argument struct
  */
  template <typename vFloat, typename Rotator, typename Vector, int fineSpin_, int nColor_, int coarseSpin_, int nVec_>
  struct BlockOrthoArg {
    using sum_t = double;
    using real = typename mapper<vFloat>::type;
    static constexpr int fineSpin = fineSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int nVec = nVec_;
    static constexpr int mVec = tile_size<nColor, nVec>();
    static_assert(nVec % mVec == 0, "mVec must be a factor of nVec");
    Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    const int nBlockOrtho; // number of times we Gram-Schmidt
    int coarseVolume;
    int fineVolumeCB;
    int geoBlockSizeCB; // number of geometric elements in each checkerboarded block
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate
    const Vector B[nVec];
    template <typename... T>
    BlockOrthoArg(ColorSpinorField &V, const int *fine_to_coarse, const int *coarse_to_fine, int parity,
                  const int *geo_bs, const int n_block_ortho, const ColorSpinorField &meta, T... B) :
      V(V),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      spin_map(),
      parity(parity),
      nParity(meta.SiteSubset()),
      nBlockOrtho(n_block_ortho),
      B{*B...}
    {
      int geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      geoBlockSizeCB = geoBlockSize/2;
      coarseVolume = meta.Volume() / geoBlockSize;
      fineVolumeCB = meta.VolumeCB();
      if (nParity != 2) errorQuda("BlockOrtho only presently supports full fields");
    }
  };

  template <int block_size, typename Arg> struct BlockOrtho_ {
    Arg &arg;
    static constexpr int fineSpin = Arg::fineSpin;
    static constexpr int spinBlock = (fineSpin == 1) ? 1 : fineSpin / Arg::coarseSpin; // size of spin block
    static constexpr int nColor = Arg::nColor;

    // on the device we rely on thread parallelism, where as on the host we assign a vector of block_size to each thread
    static constexpr int n_sites_per_thread = device::is_device() ? 1 : block_size;

    // on the device we expect number of active threads equal to block_size, and on the host just a signle thread 
    static constexpr int n_threads_per_block = device::is_device() ? block_size : 1;

    static_assert(n_sites_per_thread * n_threads_per_block == block_size, "Product of n_sites_per_thread and n_threads_per_block must equal block_size");

    using sum_t = typename Arg::sum_t;
    using dot_t = vector_type<complex<sum_t>, Arg::mVec>;
    using real = typename Arg::real;

    constexpr BlockOrtho_(Arg &arg) : arg(arg) {}
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

    __device__ inline void operator()(int x_coarse, int x_offset_cb_thread, int parity, int chirality)
    {
      using dotReduce = cub::BlockReduce<dot_t, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, fineSpin == 1 ? 1 : 2>;
      using normReduce = cub::BlockReduce<sum_t, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, fineSpin == 1 ? 1 : 2>;

      __shared__ typename dotReduce::TempStorage dot_storage;
      typename normReduce::TempStorage *norm_storage = (typename normReduce::TempStorage *)&dot_storage;
      dot_t *dot_ = (dot_t *)&dot_storage;
      sum_t *nrm_ = (sum_t *)&dot_storage;

      if (fineSpin == 1) { // when using staggered, parity is just chirality
        parity = chirality;
        chirality = 0;
      }

      int x_offset_cb[n_sites_per_thread];
      int x_cb[n_sites_per_thread];
#pragma unroll
      for (int tx = 0; tx < n_sites_per_thread; tx++) {
        x_offset_cb[tx] = x_offset_cb_thread * n_sites_per_thread + tx;
        x_cb[tx] = x_offset_cb[tx] < arg.geoBlockSizeCB ? arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + x_offset_cb[tx] ] - parity*arg.fineVolumeCB : 0;
      }

      // loop over number of block orthos
      for (int n = 0; n < arg.nBlockOrtho; n++) {
        for (int j = 0; j < Arg::nVec; j += Arg::mVec) {

          ColorSpinor<real, nColor, spinBlock> v[Arg::mVec][n_sites_per_thread];
#pragma unroll
          for (int tx = 0; tx < n_sites_per_thread; tx++) {
            if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
            if (n == 0) { // load from B on first Gram-Schmidt, otherwise V.
#pragma unroll
              for (int m = 0; m < Arg::mVec; m++) arg.B[j+m].template load<spinBlock>(v[m][tx].data, parity, x_cb[tx], chirality);
            } else {
#pragma unroll
              for (int m = 0; m < Arg::mVec; m++) load(v[tx][m], parity, x_cb[tx], chirality, j + m);
            }
          }

          for (int i = 0; i < j; i++) { // compute (j,i) block inner products
            ColorSpinor<real, nColor, spinBlock> vi[n_sites_per_thread];

            dot_t dot;
#pragma unroll
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
              load(vi[tx], parity, x_cb[tx], chirality, i);

#pragma unroll
              for (int m = 0; m < Arg::mVec; m++) dot[m] += innerProduct(vi[tx], v[m][tx]);
            }

            __syncthreads();
            dot = dotReduce(dot_storage).Sum(dot);
            if (threadIdx.x == 0 && threadIdx.y == 0) *dot_ = dot;
            __syncthreads();
            dot = *dot_;

            // subtract the blocks to orthogonalise
#pragma unroll
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
#pragma unroll
              for (int m = 0; m < Arg::mVec; m++) caxpy(-complex<real>(dot[m].real(), dot[m].imag()), vi[tx], v[m][tx]);
            }
          } // i

          // now orthogonalize over the block diagonal and normalize each entry
#pragma unroll
          for (int m = 0; m < Arg::mVec; m++) {

            dot_t dot;
#pragma unroll
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
#pragma unroll
              for (int i = 0; i < m; i++) dot[i] += innerProduct(v[i][tx], v[m][tx]);
            }
            
            __syncthreads();
            dot = dotReduce(dot_storage).Sum(dot);
            if (threadIdx.x == 0 && threadIdx.y == 0) *dot_ = dot;
            __syncthreads();
            dot = *dot_;
            
            sum_t nrm = 0.0;
#pragma unroll
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
#pragma unroll
              for (int i = 0; i < m; i++) caxpy(-complex<real>(dot[i].real(), dot[i].imag()), v[i][tx], v[m][tx]); // subtract the blocks to orthogonalise
              nrm += norm2(v[m][tx]);
            }

            __syncthreads();
            nrm = normReduce(*norm_storage).Sum(nrm);
            if (threadIdx.x == 0 && threadIdx.y == 0) *nrm_ = nrm;
            __syncthreads();
            nrm = *nrm_;
            auto nrm_inv = nrm > 0.0 ? rsqrt(nrm) : 0.0;

#pragma unroll
            for (int tx = 0; tx < n_sites_per_thread; tx++) {
              if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
              v[m][tx] *= nrm_inv;
            }
          }

#pragma unroll
          for (int tx = 0; tx < n_sites_per_thread; tx++) {
            if (x_offset_cb[tx] >= arg.geoBlockSizeCB) break;
#pragma unroll
            for (int m = 0; m < Arg::mVec; m++) save(parity, x_cb[tx], chirality, j + m, v[m][tx]);
          }
        } // j
      }   // n
    }
  };

  template <int block_size, template <int, typename> class Transformer, typename Arg>
  __launch_bounds__(2 * block_size) __global__ void blockOrthoGPU(Arg arg)
  {
    int x_coarse = blockIdx.x;
#ifdef SWIZZLE
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
#endif
    int parity = (arg.nParity == 2) ? threadIdx.y + blockIdx.y*blockDim.y : arg.parity;
    int x_fine_offset_cb = threadIdx.x;
    int chirality = blockIdx.z; // which chiral block we're working on (if chirality is present)

    Transformer<block_size, Arg> f(arg);
    f(x_coarse, x_fine_offset_cb, parity, chirality);
  }

} // namespace quda
