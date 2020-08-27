#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <register_traits.h>
#include <cub_helper.cuh>

namespace quda {

  template <typename vFloatSpinor_, typename vFloatGauge_, int coarseSpin_, int fineColor_,
            int coarseColor_, bool dagger_, typename fineColorSpinor, typename xInvGauge>
  struct ApplyStaggeredKDBlockArg {

    using vFloatSpinor = vFloatSpinor_;
    using vFloatGauge = vFloatGauge_;

    using Float = typename mapper<vFloatGauge>::type;

    static constexpr int fineColor = fineColor_;
    static constexpr int fineSpin = 1;
    static constexpr int coarseColor = coarseColor_;
    static constexpr int coarseSpin = coarseSpin_;
    static constexpr int coarseDof = coarseColor*coarseSpin;
    static constexpr bool dagger = dagger_;

    static constexpr int blockSizeKD = 16; /** Elements per KD block */
    static constexpr int paddedSizeKD = blockSizeKD; // padding is currently broken + 1; /** Padded size */

    fineColorSpinor out;      /** Output staggered spinor field */
    const fineColorSpinor in; /** Input staggered spinor field */
    const xInvGauge xInv;     /** Kahler-Dirac inverse field */

    int_fastdiv x_size[QUDA_MAX_DIM];           /** Dimensions of fine grid */
    int_fastdiv xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    const int fineVolumeCB;
    const int_fastdiv coarseVolumeCB;   /** Coarse grid volume */

    ApplyStaggeredKDBlockArg(fineColorSpinor &out, const fineColorSpinor &in, const xInvGauge &xInv,
                           const int *x_size_, const int *xc_size_) :
      out(out),
      in(in),
      xInv(xInv),
      fineVolumeCB(in.volumeCB),
      coarseVolumeCB(xInv.VolumeCB())
    {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
      }
    }

  };

  template<typename Arg>
  __global__ void ApplyStaggeredKDBlockGPU(Arg arg)
  {
    if (Arg::dagger) return; // FIXME

    // Vector type for spinor
    using real_spinor = typename mapper<typename Arg::vFloatSpinor>::type;
    using complex_spinor = complex<real_spinor>;
    using Vector = ColorSpinor<real_spinor, Arg::fineColor, 1>;

    // Type for gauge/compute
    using real = typename Arg::Float;
    using complex = complex<real>;
    extern __shared__ complex cs_buffer[];

    // For each "dot product" in the mat-vec
    typedef cub::WarpReduce<real,16> WarpReduce16;
    __shared__ typename WarpReduce16::TempStorage temp_storage_16;

    /////////////////////////////////
    // Figure out some identifiers //
    /////////////////////////////////

    // What is my overall thread id?
    const unsigned int tid = ((blockIdx.y*gridDim.x + blockIdx.x)*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

    if (tid >= arg.fineVolumeCB*2) return;

    // Decompose into factors of 16.
    const unsigned int fast_idx = tid & 0b1111;
    const unsigned int mid_idx = (tid >> 4) & 0b1111;
    const unsigned int slow_idx = tid >> 8;

    // The fundamental unit of work is 256 threads = 16 KD blocks
    // What's the first KD block in my unit of work?
    const unsigned int x_coarse_first = 16 * slow_idx;

    unsigned int x_coarse_tmp, coarse_x, coarse_y, coarse_z, coarse_t;

    // What's my KD block for loading Xinv?
    // [0-15] loads consecutive elements of Xinv for the first block,
    // [16-31] loads consecutive elements of Xinv for the second block, etc
    const unsigned int x_coarse_xinv = x_coarse_first + mid_idx;
    const unsigned int x_coarse_xinv_cb = x_coarse_xinv >> 1;

    x_coarse_tmp = x_coarse_xinv;
    coarse_x = x_coarse_tmp % arg.xc_size[0];
    x_coarse_tmp = x_coarse_tmp / arg.xc_size[0];
    coarse_y = x_coarse_tmp % arg.xc_size[1];
    x_coarse_tmp = x_coarse_tmp / arg.xc_size[1];
    coarse_z = x_coarse_tmp % arg.xc_size[2];
    coarse_t = x_coarse_tmp / arg.xc_size[2];

    const unsigned int parity_coarse_xinv = (coarse_x + coarse_y + coarse_z + coarse_t) & 1;

    // What's my KD block for loading spinors?
    // [0-15] loads first corner of consecutive hypercubes,
    // [16-31] loads the second corner, etc
    const unsigned int x_coarse_spinor = x_coarse_first + fast_idx;
    const unsigned int x_coarse_spinor_cb = x_coarse_spinor >> 1;
    
    // There's got to be a better way to do this
      x_coarse_tmp = x_coarse_spinor;
      coarse_x = x_coarse_tmp % arg.xc_size[0];
      x_coarse_tmp = x_coarse_tmp / arg.xc_size[0];
      coarse_y = x_coarse_tmp % arg.xc_size[1];
      x_coarse_tmp = x_coarse_tmp / arg.xc_size[1];
      coarse_z = x_coarse_tmp % arg.xc_size[2];
      coarse_t = x_coarse_tmp / arg.xc_size[2];

    const unsigned int parity_coarse_spinor = (coarse_x + coarse_y + coarse_z + coarse_t) & 1;

    //printf("coarseIdx %d %d %d %d %d %d %d\n", x_coarse_first,
    //        x_coarse_spinor, x_coarse_spinor_cb, parity_coarse_spinor, 
    //        x_coarse_xinv, x_coarse_xinv_cb, parity_coarse_xinv);

    /////////////////////////////////////
    // Set up my shared memory buffers //
    /////////////////////////////////////

    // This is complicated b/c we need extra padding to avoid bank conflcits
    // Check `staggered_kd_apply_xinv.cu` for the dirty details
    constexpr int buffer_size = Arg::fineColor * 16 * Arg::paddedSizeKD; //256 * ((Arg::fineColor * 16 * Arg::paddedSizeKD * sizeof(complex)) / 256 + 1);

    // Which unit of work am I within this block?
    const int unit_of_work = (threadIdx.x + threadIdx.y * blockDim.x) / 256;
    complex* in_buffer = cs_buffer + (2 * unit_of_work) * buffer_size;
    complex* out_buffer = cs_buffer + (2 * unit_of_work + 1) * buffer_size;
    
    ////////////////////////////////////////////////////
    // Hey, real work! What ColorVector am I loading? //
    ////////////////////////////////////////////////////
    
    int coarseCoords[4];
    getCoords(coarseCoords, x_coarse_spinor_cb, arg.xc_size, parity_coarse_spinor);

    // What corner of the hypercube am I grabbing?
    int tmp_mid_idx = mid_idx;
    int y_bit = tmp_mid_idx & 1; tmp_mid_idx >>= 1;
    int z_bit = tmp_mid_idx & 1; tmp_mid_idx >>= 1;
    int t_bit = tmp_mid_idx & 1; tmp_mid_idx >>= 1;
    int spin_bit = tmp_mid_idx;

    int x_bit = (y_bit + z_bit + t_bit) & 1;
    if (spin_bit) x_bit = 1 - x_bit;

    int parity_spinor = (x_bit + y_bit + z_bit + t_bit) & 1;

    // Last xc_size[0] is intentional
    int x_spinor_cb = (((((2 * coarseCoords[3] + t_bit) * arg.x_size[2]) + 2 * coarseCoords[2] + z_bit) * arg.x_size[1] + 2 * coarseCoords[1] + y_bit) * arg.x_size[0] + 2 * coarseCoords[0] + x_bit) >> 1;

    //if (x_coarse_first == 0) {
    //  printf("tid %03d spinor_cb %02d spinor_parity %d xinv_cb %02d xinv_parity %d\n",
    //    tid & 0b11111111, x_spinor_cb, parity_spinor, x_coarse_xinv_cb, parity_coarse_xinv);
    //}

    // Load!
    int buffer_index = mid_idx & 0b111; // 0th component of fine color
    //buffer_index += 8 * c_f;
    buffer_index += 24 * spin_bit; // spin offset
    buffer_index += Arg::fineColor * Arg::paddedSizeKD * fast_idx;

    // arg.fineVolumeCB will always divide into 256 b/c the fine volume
    // needs to be some multiple of 4^4, which equals 256!
    //if (x_spinor_cb < arg.fineVolumeCB) {

      const Vector in = arg.in(x_spinor_cb, parity_spinor);

      // Legacy comment:
      // ♫ do you believe in bank conflicts ♫
      for (int c_f = 0; c_f < Arg::fineColor; c_f++) {
        // the 51, 17 is to avoid bank conflicts
        //auto in_spinor = in(0,c_f);
        in_buffer[buffer_index + 8 * c_f] = static_cast<complex>(in(0,c_f)); // { (real)(in_spinor.real()), (real)(in_spinor.imag()) };
      }

    //} else {
    //
    //  for (int c_f = 0; c_f < Arg::fineColor; c_f++) {
    //    // the 51, 17 is to avoid bank conflicts
    //    in_buffer[buffer_index + 8 * c_f] = { 0, 0 };
    //  }
    //}

    // in reality we only need to sync over my chunk of 256 threads
    __syncthreads();

    /////////////////////////////
    // Multiply by Xinv, store //
    /////////////////////////////

    //if (x_coarse_xinv_cb < arg.coarseVolumeCB) {
      #pragma unroll // eh, we'll see if this is a good idea.
      for (int coarse_spin_row = 0; coarse_spin_row < Arg::coarseSpin; coarse_spin_row++) {
        #pragma unroll
        for (int coarse_color_row = 0; coarse_color_row < Arg::coarseColor; coarse_color_row++) {
          
          real re_sum = 0, im_sum = 0;

          // load rows in three chunks
          #pragma unroll
          for (int elem = fast_idx; elem < Arg::coarseSpin*Arg::coarseColor; elem += Arg::blockSizeKD) {
            const int color_idx = elem % Arg::coarseColor;
            const int spin_idx = elem / Arg::coarseColor;

            int padded_elem = (elem / Arg::blockSizeKD) * Arg::paddedSizeKD + (elem % Arg::blockSizeKD);

            const complex xinv_elem = arg.xInv(0, parity_coarse_xinv, x_coarse_xinv_cb, coarse_spin_row, spin_idx, coarse_color_row, color_idx);
            const complex cs_component = in_buffer[Arg::paddedSizeKD * Arg::fineColor * mid_idx + padded_elem];
            complex prod = cmul(xinv_elem, cs_component);

            re_sum += WarpReduce16(temp_storage_16).Sum(prod.real());
            im_sum += WarpReduce16(temp_storage_16).Sum(prod.imag());
          }

          __syncwarp();

          if (fast_idx == 0)
          {
            out_buffer[Arg::paddedSizeKD * Arg::fineColor * mid_idx + coarse_spin_row * Arg::coarseColor + coarse_color_row] = { re_sum, im_sum };
          }
        }
      }
    //}

    __syncthreads();

    /////////////////////////////////////
    // Store: one whole thing is easy! //
    /////////////////////////////////////

    //if (x_spinor_cb < arg.fineVolumeCB) {
      Vector out;

      #pragma unroll
      for (int c_f = 0; c_f < Arg::fineColor; c_f++) {
        //printf("%d %e + I %e\n", x_spinor_cb, out_buffer[Arg::paddedSizeKD * Arg::fineColor * fast_idx + Arg::paddedSizeKD * c_f + mid_idx].real(),
        //                                      out_buffer[Arg::paddedSizeKD * Arg::fineColor * fast_idx + Arg::paddedSizeKD * c_f + mid_idx].imag());
        out(0,c_f) = static_cast<complex_spinor>(out_buffer[buffer_index + 8 * c_f]);
      }

      arg.out(x_spinor_cb, parity_spinor) = out;
    //}
    
    // Who knows
  }

} // namespace quda
