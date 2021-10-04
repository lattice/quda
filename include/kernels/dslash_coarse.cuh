#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <float_vector.h>
#include <shared_memory_cache_helper.cuh>
#include <kernel.h>
#include <warp_collective.h>

namespace quda {

  enum DslashType {
    DSLASH_INTERIOR,
    DSLASH_EXTERIOR,
    DSLASH_FULL
  };

  // we use two colors per thread unless we have large dim_stride, when we're aiming for maximum parallelism
  constexpr int colors_per_thread(int nColor, int dim_stride) { return (nColor % 2 == 0 && dim_stride <= 2) ? 2 : 1; }

  template <bool dslash_, bool clover_, bool dagger_, DslashType type_, int color_stride_, int dim_stride_, typename Float,
            typename yFloat, typename ghostFloat, int nSpin_, int nColor_, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
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

    using F = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, csOrder, Float, ghostFloat>;
    using G = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;
    using GY = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;

    F out;
    const F inA;
    const F inB;
    const GY Y;
    const GY X;
    const real kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int nFace;  // hard code to 1 for now
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;

    inline DslashCoarseArg(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
                           const GaugeField &Y, const GaugeField &X, real kappa, int parity) :
      kernel_param(dim3(color_stride * X.VolumeCB(), out.SiteSubset(), 2 * dim_stride * 2 * (nColor / colors_per_thread(nColor, dim_stride)))),
      out(const_cast<ColorSpinorField &>(out)),
      inA(const_cast<ColorSpinorField &>(inA)),
      inB(const_cast<ColorSpinorField &>(inB)),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      nFace(1),
      X0h(((3 - nParity) * out.X(0)) / 2),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out.VolumeCB() / dim[4])
    {  }
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

     @param out The result vector
     @param thread_dir Direction
     @param x_cb The checkerboarded site index
     @param src_idx dummy for now
     @param parity The site parity
     @param s_row Which spin row are acting on
     @param color_block Which color row are we acting on
     @param color_off Which color column offset are we acting on
     @param arg Arguments
   */
  template <int Mc, int thread_dim, typename V, typename Arg>
  __device__ __host__ inline void applyDslash(V &out, int thread_dir, int x_cb, int src_idx, int parity, int s_row, int color_block, int color_offset, const Arg &arg)
  {
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[5];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);
    coord[4] = src_idx;

    if (!thread_dir || target::is_host()) {

      //Forward gather - compute fwd offset for spinor fetch
#pragma unroll
      for(int d = thread_dim; d < Arg::nDim; d += Arg::dim_stride) // loop over dimension
      {
	const int fwd_idx = linkIndexP1(coord, arg.dim, d);

	if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	  if (doHalo<Arg::type>()) {
            int ghost_idx = ghostFaceIndex<1, 5>(coord, arg.dim, d, arg.nFace);

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
                                            arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		  else
		    out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                                            arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		}
	      }
	    }
	  }
	} else if (doBulk<Arg::type>()) {
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
                                          arg.inA(their_spinor_parity, fwd_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		else
		  out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                                          arg.inA(their_spinor_parity, fwd_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
	      }
	    }
	  }
	}

      } // nDim
    }

    if (thread_dir || target::is_host()) {

      //Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
      for(int d = thread_dim; d < Arg::nDim; d += Arg::dim_stride)
	{
	const int back_idx = linkIndexM1(coord, arg.dim, d);
	const int gauge_idx = back_idx;
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  if (doHalo<Arg::type>()) {
            const int ghost_idx = ghostFaceIndex<0, 5>(coord, arg.dim, d, arg.nFace);
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
                                            arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		  else
		    out[color_local] = cmac(conj(arg.Y.Ghost(d+4, 1-parity, ghost_idx, col, row)),
                                            arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		}
	    }
	  }
	} else if (doBulk<Arg::type>()) {
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
                                          arg.inA(their_spinor_parity, back_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
		else
		  out[color_local] = cmac(conj(arg.Y(d+4, 1-parity, gauge_idx, col, row)),
                                          arg.inA(their_spinor_parity, back_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
	      }
	  }
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
	    out[color_local] = cmac(arg.X(0, parity, x_cb, row, col), arg.inB(spinor_parity, x_cb+src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
	  } else {
	    out[color_local] = cmac(conj(arg.X(0, parity, x_cb, col, row)), arg.inB(spinor_parity, x_cb+src_idx*arg.volumeCB, s_col, c_col+color_offset), out[color_local]);
	  }
	}
    }
  }

  template <bool is_device> struct dim_collapse {
    template <typename T, typename Arg> void operator()(T &out, int, int, const Arg &arg)
    {
      out *= -arg.kappa;
    }
  };

  template <> struct dim_collapse<true> {
    template <typename T, typename Arg> __device__ __host__ inline void operator()(T &out, int dir, int dim, const Arg &arg)
    {
      SharedMemoryCache<T> cache(target::block_dim());
      // only need to write to shared memory if not master thread
      if (dim > 0 || dir) cache.save(out);

      cache.sync(); // recombine the foward and backward results

      if (dir == 0 && dim == 0) {
        // full split over dimension and direction
#pragma unroll
        for (int d=1; d < Arg::dim_stride; d++) { // get remaining forward gathers (if any)
          // 4-way 1,2,3  (stride = 4)
          // 2-way 1      (stride = 2)
          out += cache.load_z(target::thread_idx().z + d * 2 + 0);
        }

#pragma unroll
        for (int d=0; d < Arg::dim_stride; d++) { // get all backward gathers
          out += cache.load_z(target::thread_idx().z + d * 2 + 1);
        }

        out *= -arg.kappa;
      }
    }
  };

  template <typename Arg> struct CoarseDslash {
    const Arg &arg;
    constexpr CoarseDslash(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb_color_offset, int parity, int sMd)
    {
      int x_cb = x_cb_color_offset;
      int color_offset = 0;

      if (target::is_device() && Arg::color_stride > 1) { // on the device we support warp fission of the inner product
        const int lane_id = target::thread_idx().x % device::warp_size();
        const int warp_id = target::thread_idx().x / device::warp_size();
        const int vector_site_width = device::warp_size() / Arg::color_stride; // number of sites per warp

        x_cb = target::block_idx().x * (target::block_dim().x / Arg::color_stride) + warp_id * (device::warp_size() / Arg::color_stride) + lane_id % vector_site_width;
        color_offset = lane_id / vector_site_width;
      }

      parity = (arg.nParity == 2) ? parity : arg.parity;

      // z thread dimension is (( s*(Nc/Mc) + color_block )*dim_thread_split + dim)*2 + dir
      constexpr int Mc = colors_per_thread(Arg::nColor, Arg::dim_stride);
      int dir = sMd & 1;
      int sMdim = sMd >> 1;
      int dim = sMdim % Arg::dim_stride;
      int sM = sMdim / Arg::dim_stride;
      int s = sM / (Arg::nColor/Mc);
      int color_block = (sM % (Arg::nColor/Mc)) * Mc;

      constexpr int src_idx = 0;
      vector_type<complex <typename Arg::real>, Mc> out;

      if (Arg::dslash) {
        if (dim == 0)      applyDslash<Mc, 0>(out, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);
        else if (dim == 1) applyDslash<Mc, 1>(out, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);
        else if (dim == 2) applyDslash<Mc, 2>(out, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);
        else if (dim == 3) applyDslash<Mc, 3>(out, dir, x_cb, src_idx, parity, s, color_block, color_offset, arg);

        target::dispatch<dim_collapse>(out, dir, dim, arg);
      }

      if (doBulk<Arg::type>() && Arg::clover && dir==0 && dim==0) applyClover<Mc>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);

      if (dir==0 && dim==0) {
        const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;

        // reduce down to the first group of column-split threads
        out = warp_combine<Arg::color_stride>(out);

#pragma unroll
        for (int color_local=0; color_local<Mc; color_local++) {
          int c = color_block + color_local; // global color index
          if (color_offset == 0) {
            // if not halo we just store, else we accumulate
            if (doBulk<Arg::type>()) arg.out(my_spinor_parity, x_cb+src_idx*arg.volumeCB, s, c) = out[color_local];
            else arg.out(my_spinor_parity, x_cb+src_idx*arg.volumeCB, s, c) += out[color_local];
          }
        }
      }
    }
  };

} // namespace quda
