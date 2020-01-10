#include <color_spinor_field_order.h>
#include <cub_helper.cuh>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
#ifndef SWIZZLE
#define SWIZZLE
#endif

namespace quda {

  using namespace quda::colorspinor;

  /** 
      Kernel argument struct
  */
  template <typename Float, typename vFloat, int fineSpin, int fineColor,
	    int coarseSpin, int coarseColor, QudaFieldOrder order>
  struct RestrictArg {

    FieldOrderCB<Float,coarseSpin,coarseColor,1,order> out;
    const FieldOrderCB<Float,fineSpin,fineColor,1,order> in;
    const FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order,vFloat> V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

    RestrictArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
		const int *fine_to_coarse, const int *coarse_to_fine, int parity)
      : out(out), in(in), V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
	spin_map(), parity(parity), nParity(in.SiteSubset()), swizzle(1)
    { }

    RestrictArg(const RestrictArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,order> &arg) :
      out(arg.out), in(arg.in), V(arg.V), 
      fine_to_coarse(arg.fine_to_coarse), coarse_to_fine(arg.coarse_to_fine), spin_map(),
      parity(arg.parity), nParity(arg.nParity), swizzle(arg.swizzle)
    { }
  };

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, int coarse_colors_per_thread,
	    class FineColor, class Rotator>
  __device__ __host__ inline void rotateCoarseColor(complex<Float> out[fineSpin*coarse_colors_per_thread],
						    const FineColor &in, const Rotator &V,
						    int parity, int nParity, int x_cb, int coarse_color_block) {
    const int spinor_parity = (nParity == 2) ? parity : 0;
    const int v_parity = (V.Nparity() == 2) ? parity : 0;

#pragma unroll
    for (int s=0; s<fineSpin; s++)
#pragma unroll
      for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	out[s*coarse_colors_per_thread+coarse_color_local] = 0.0;
      }

#pragma unroll
    for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
      int i = coarse_color_block + coarse_color_local;
#pragma unroll
      for (int s=0; s<fineSpin; s++) {

	constexpr int color_unroll = fineColor == 3 ? 3 : 2;

	complex<Float> partial[color_unroll];
#pragma unroll
	for (int k=0; k<color_unroll; k++) partial[k] = 0.0;

#pragma unroll
	for (int j=0; j<fineColor; j+=color_unroll) {
#pragma unroll
	  for (int k=0; k<color_unroll; k++)
	    partial[k] += conj(V(v_parity, x_cb, s, j+k, i)) * in(spinor_parity, x_cb, s, j+k);
	}

#pragma unroll
	for (int k=0; k<color_unroll; k++) out[s*coarse_colors_per_thread + coarse_color_local] += partial[k];
      }
    }

  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int coarse_colors_per_thread, typename Arg>
  void Restrict(Arg arg) {
    for (int parity_coarse=0; parity_coarse<2; parity_coarse++) 
      for (int x_coarse_cb=0; x_coarse_cb<arg.out.VolumeCB(); x_coarse_cb++)
	for (int s=0; s<coarseSpin; s++) 
	  for (int c=0; c<coarseColor; c++)
	    arg.out(parity_coarse, x_coarse_cb, s, c) = 0.0;

    // loop over fine degrees of freedom
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.in.VolumeCB(); x_cb++) {

	int x = parity*arg.in.VolumeCB() + x_cb;
	int x_coarse = arg.fine_to_coarse[x];
	int parity_coarse = (x_coarse >= arg.out.VolumeCB()) ? 1 : 0;
	int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();
	
	for (int coarse_color_block=0; coarse_color_block<coarseColor; coarse_color_block+=coarse_colors_per_thread) {
	  complex<Float> tmp[fineSpin*coarse_colors_per_thread];
	  rotateCoarseColor<Float,fineSpin,fineColor,coarseColor,coarse_colors_per_thread>
	    (tmp, arg.in, arg.V, parity, arg.nParity, x_cb, coarse_color_block);

	  for (int s=0; s<fineSpin; s++) {
	    for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	      int c = coarse_color_block + coarse_color_local;
	      arg.out(parity_coarse,x_coarse_cb,arg.spin_map(s,parity),c) += tmp[s*coarse_colors_per_thread+coarse_color_local];
	    }
	  }

	}
      }
    }

  }

  /**
     Here, we ensure that each thread block maps exactly to a
     geometric block.  Each thread block corresponds to one geometric
     block, with number of threads equal to the number of fine grid
     points per aggregate, so each thread represents a fine-grid
     point.  The look up table coarse_to_fine is the mapping to
     each fine grid point.
  */
  template <int block_size, typename Float, int fineSpin, int fineColor, int coarseSpin,
            int coarseColor, int coarse_colors_per_thread, typename Arg>
  __global__ void RestrictKernel(Arg arg) {

#ifdef SWIZZLE
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    int x_coarse = blockIdx.x;
    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
#else
    int x_coarse = blockIdx.x;
#endif

    int parity_coarse = x_coarse >= arg.out.VolumeCB() ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();

    // obtain fine index from this look up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on (swizzled to improve cache efficiency)
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered
    int parity = arg.nParity == 2 ? threadIdx.y : arg.parity;
    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * blockDim.x + threadIdx.x];
    int x_fine_cb = x_fine - parity*arg.in.VolumeCB();

    int coarse_color_block = (blockDim.z*blockIdx.z + threadIdx.z) * coarse_colors_per_thread;
    if (coarse_color_block >= coarseColor) return;

    complex<Float> tmp[fineSpin*coarse_colors_per_thread];
    rotateCoarseColor<Float,fineSpin,fineColor,coarseColor,coarse_colors_per_thread>
      (tmp, arg.in, arg.V, parity, arg.nParity, x_fine_cb, coarse_color_block);

    typedef vector_type<complex<Float>, coarseSpin*coarse_colors_per_thread> vector;
    vector reduced;

    // first lets coarsen spin locally
    for (int s=0; s<fineSpin; s++) {
      for (int v=0; v<coarse_colors_per_thread; v++) {
	reduced[arg.spin_map(s,parity)*coarse_colors_per_thread+v] += tmp[s*coarse_colors_per_thread+v];
      }
    }

    // now lets coarsen geometry across threads
    if (arg.nParity == 2) {
      typedef cub::BlockReduce<vector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;

      // note this is not safe for blockDim.z > 1
      reduced = BlockReduce(temp_storage).Sum(reduced);
    } else {
      typedef cub::BlockReduce<vector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;

      // note this is not safe for blockDim.z > 1
      reduced = BlockReduce(temp_storage).Sum(reduced);
    }

    if (threadIdx.x==0 && threadIdx.y == 0) {
      for (int s=0; s<coarseSpin; s++) {
	for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	  int v = coarse_color_block + coarse_color_local;
	  arg.out(parity_coarse, x_coarse_cb, s, v) = reduced[s*coarse_colors_per_thread+coarse_color_local];
	}
      }
    }
  }
  
}
