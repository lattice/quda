#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <register_traits.h>

namespace quda {

  template <typename vFloatSpinor_, typename vFloatGauge_, int coarseSpin_, int fineColor_,
            int coarseColor_, typename fineColorSpinor, typename xInvGauge, bool dagger_>
  struct ApplyStaggeredKDBlockArg {

    using vFloatSpinor = vFloatSpinor_;
    using vFloatGauge = vFloatGauge_;

    using Float = typename mapper<vFloatGauge>::type;

    static constexpr int fineColor = fineColor_;
    static constexpr int fineSpin = 1;
    static constexpr int coarseColor = coarseColor_;
    static constexpr int carseSpin = coarseSpin_;
    static constexpr bool dagger = dagger_;

    fineColorSpinor out;      /** Output staggered spinor field */
    const fineColorSpinor in; /** Input staggered spinor field */
    const xInvGauge xInv;     /** Kahler-Dirac inverse field */

    int x_size[QUDA_MAX_DIM];           /** Dimensions of fine grid */
    int_fastdiv xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    const int coarseVolumeCB;   /** Coarse grid volume */

    ApplyStaggeredKDBlockArg(fineColorSpinor &out, const fineColorSpinor &in, const xInvGauge &xInv,
                           const int *x_size_, const int *xc_size_) :
      out(out),
      in(in),
      xInv(xInv),
      coarseVolumeCB(xInv.VolumeCB())
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
      }
    }

  };

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void ComputeStaggeredKDBlock(Arg &arg, int parity, int x_cb, int ic_f, int jc_f)
  {
    constexpr int nDim = 4;
    int coord[nDim];
    int coord_coarse[nDim];

    getCoords(coord, x_cb, arg.x_size, parity);

    // Compute coarse coordinates
    for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/2;
    int coarse_parity = 0;
    for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0]/2;

    // Fine parity gives the coarse spin
    constexpr int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index
    const int s_c_col = arg.spin_map(s,1-parity); // Coarse spin col index

    // The coarse color row depends on my fine hypercube corner.
    int hyperCorner = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
    int c_row = 8*ic_f + hyperCorner;

#pragma unroll
    for (int mu=0; mu < nDim; mu++) {
      // The coarse color column depends on my fine hypercube+mu corner.
      coord[mu]++; // linkIndexP1, it's fine if this wraps because we're modding by 2.
      int hyperCorner_mu = 4*(coord[3]%2)+2*(coord[2]%2)+(coord[1]%2);
      coord[mu]--;

      int c_col = 8*jc_f + hyperCorner_mu;

      //Check to see if we are on the edge of a block.  If adjacent site
      //is in same block, M = X, else it's a hopping term and we ignore it
      const bool isDiagonal = ((coord[mu]+1)%arg.x_size[mu])/2 == coord_coarse[mu] ? true : false;

      complex<Float> vuv = arg.U(mu,parity,x_cb,ic_f,jc_f);

      if (isDiagonal) {
        // backwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_col,s_c_row,c_col,c_row) = conj(vuv);
        // forwards
        arg.X(0,coarse_parity,coarse_x_cb,s_c_row,s_c_col,c_row,c_col) = -vuv;
      } // end (isDiagonal)
    }

    // add staggered mass term to diagonal
    if (ic_f == 0 && jc_f == 0 && x_cb < arg.coarseVolumeCB) {
#pragma unroll
      for (int s = 0; s < coarseSpin; s++) {
#pragma unroll
        for (int c = 0; c < coarseColor; c++) {
          arg.X(0,parity,x_cb,s,s,c,c) = static_cast<Float>(2.0) * complex<Float>(arg.mass,0.0); // staggered conventions. No need to +=
        } //Color
      } //Spin
    }
  }

  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredKDBlockCPU(Arg arg)
  {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int ic_f=0; ic_f<fineColor; ic_f++) {
          for (int jc_f=0; jc_f<fineColor; jc_f++) {
            ComputeStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, ic_f, jc_f);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }

  template<typename Arg>
  __global__ void ApplyStaggeredKDBlock(Arg arg)
  {
    if (Arg::dagger) return; // FIXME

    using real = Arg::Float;
    using complex = complex<real>;
    extern __shared__ complex cs_buffer[];

    // Vector type for spinor
    using real_spinor = typename mapper<Arg::vFloatSpinor>::type;
    using Vector = ColorSpinor<real_spinor, Arg::fineColor, 1>;

    // For warp-wide reductions
    typedef cub::WarpReduce<real,32> WarpReduce32;
    __shared__ typename WarpReduce32::TempStorage temp_storage_32;
    typedef cub::WarpReduce<real,16> WarpReduce32;
    __shared__ typename WarpReduce16::TempStorage temp_storage_16;

    // eventually we'll do multiple KD blocks per CUDA block
    // for now, just one KD block per warp
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    const int warp_size = 32;

    const int x_coarse_cb = tid / warp_size;
    const int warp_id = tid % warp_size;
    complex* buffer = reinterpret_cast<complex*>(cs_buffer) + warp_id * 48; // each warp stores 48 complex nums to shared memory

    const int parity_coarse = blockDim.y*blockIdx.y + threadIdx.y;

    const int x_coarse = 2 * x_coarse_cb + 1;

    // This is going to get confusing real fast so buckle up!

    // Part 1. Figure out which site my thread is responsible for.
    //         There's probably a better way to do this.
    int coarseCoords[4];
    getCoords(coarseCoords, x_coarse_cb, xc_size, parity_coarse); // this has int div...

    // Grab one corner of the hypercube
    int tmp_warp_id = warp_id;
    int y_bit = tmp_warp_id & 1; tmp_warp_id >>= 1;
    int z_bit = tmp_warp_id & 1; tmp_warp_id >>= 1;
    int t_bit = tmp_warp_id & 1; tmp_warp_id >>= 1;
    int parity = tmp_warp_id & 1;

    // Last xc_size[0] is intentional
    int x_cb = ((((2 * coarseCoords[3] + t_bit) * x_size[2]) + 2 * coarseCoords[2] + z_bit) * x_size[1] + 2 * coarseCoords[1] + y_bit) * xc_size[0] + coarseCoords[0];

    // Alright, that was gross.

    // Part 2. Load my ColorSpinor, store it to shared memory
    // in a convenient order for the Xinv multiply
    // it's a shame we can't generically use LDGSTS?
    if (warp_id < 16) {
      const Vector in = arg.in(x_cb, parity);

      // ColorSpinor type may not equal compute type (if ColorSpinor type is double)
      // Will find out if this cast works at compile time 
      buffer[warp_id] = static_cast<complex>(in[0]);
      buffer[warp_id+16] = static_cast<complex>(in[1]);
      buffer[warp_id+32] = static_cast<complex>(in[2]);
    }
    __syncwarp();

    // Part 3. // Multiply by Xinv. 

    Vector out;

    #pragma unroll // eh, we'll see if this is a good idea.
    for (int coarse_spin_row = 0; coarse_spin_row < Arg::coarseSpin; coarse_spin_row++) {
      #pragma unroll
      for (int coarse_color_row = 0; coarse_color_row < Arg::coarseColor; coarse_color_row++) {
        // load a row, reduce, continue
        int color_idx = warp_id % 24;
        int spin_idx = warp_id / 24;
        complex xinv_elem = arg.Xinv(parity_coarse, x_coarse_cb, spin_idx, color_idx);
        complex prod = cmul(xinv_elem, buffer[warp_id]);
        __syncwarp();

        // Reduce
        real re_reduce, im_reduce;
        re_reduce = WarpReduce32(temp_storage_32).Sum(prod.real());
        im_reduce = WarpReduce32(temp_storage_32).Sum(prod.imag());

        // First 16 threads grab the remaining 16 elements
        if (warp_id < 16) {
          xinv_elem = arg.Xinv(parity_coarse, x_coarse_cb, 1, color_idx + 8);
          prod = cmul(xinv_elem, buffer[warp_id+32]);
        } else {
          prod = complex(0,0);
        }

        re_reduce += WarpReduce16(temp_storage_16).Sum(prod.real());
        im_reduce += WarpReduce16(temp_storage_16).Sum(prod.real());

        // broadcast
        Float val_re = __shfl_sync(0xffffffff, re_reduce, 0);
        Float val_im = __shfl_sync(0xffffffff, im_reduce, 0);

        int cs_index = coarse_spin_row * Arg::coarseColor + coarse_color_row;
        if (warp_id < 16 && warp_id == cs_index / 3) {
          out[cs_index % 3] = { val_re, val_im };
        }
      }
    }

    // Part 4. Store
    __syncwarp();
    if (warp_id < 16) {
      arg.out(x_cb, parity) = out;
    }
    
    // Who knows
  }

} // namespace quda
