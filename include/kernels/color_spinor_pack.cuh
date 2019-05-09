#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>

namespace quda {

  template <typename Field>
  struct PackGhostArg {

    Field field;
    int_fastdiv X[QUDA_MAX_DIM];
    const int_fastdiv volumeCB;
    const int nDim;
    const int nFace;
    const int parity;
    const int nParity;
    const int dagger;
    const QudaPCType pc_type;
    int commDim[4]; // whether a given dimension is partitioned or not
    int_fastdiv nParity2dim_threads;

    PackGhostArg(Field field, const ColorSpinorField &a, int parity, int nFace, int dagger) :
        field(field),
        volumeCB(a.VolumeCB()),
        nDim(a.Ndim()),
        nFace(nFace),
        parity(parity),
        nParity(a.SiteSubset()),
        dagger(dagger),
        pc_type(a.PCType())
    {
      X[0] = ((nParity == 1) ? 2 : 1) * a.X(0); // set to full lattice dimensions
      for (int d=1; d<nDim; d++) X[d] = a.X(d);
      X[4] = (nDim == 5) ? a.X(4) : 1; // set fifth dimension correctly
      for (int i=0; i<4; i++) {
	commDim[i] = comm_dim_partitioned(i);
      }
    }
  };

  // this is the maximum number of colors for which we support block-float format
#define MAX_BLOCK_FLOAT_NC 32

  /**
     Compute the max element over the spin-color components of a given site.
   */
  template <typename Float, int Ns, int Ms, int Nc, int Mc, typename Arg>
  __device__ __host__ __forceinline__ Float compute_site_max(Arg &arg, int x_cb, int parity, int spinor_parity, int spin_block, int color_block, bool active) {

    Float thread_max = 0.0;
    Float site_max = active ? 0.0 : 1.0;

#ifdef __CUDA_ARCH__
    // workout how big a shared-memory allocation we need
    // just statically compute the largest size needed to avoid templating on block size
    constexpr int max_block_size = 1024; // all supported GPUs have 1024 as their max block size
    constexpr int bank_width = 32; // shared memory has 32 banks
    constexpr int color_spin_threads = Nc <= MAX_BLOCK_FLOAT_NC ? (Ns/Ms) * (Nc/Mc) : 1;
    // this is the largest size of blockDim.x (rounded up to multiples of bank_width)
    constexpr int thread_width_x = ( (max_block_size / color_spin_threads + bank_width-1) / bank_width) * bank_width;
    __shared__ Float v[ (Ns/Ms) * (Nc/Mc) * thread_width_x];
    const auto &rhs = arg.field;
    if (active) {
#pragma unroll
      for (int spin_local=0; spin_local<Ms; spin_local++) {
	int s = spin_block + spin_local;
#pragma unroll
	for (int color_local=0; color_local<Mc; color_local++) {
	  int c = color_block + color_local;
	  complex<Float> z = rhs(spinor_parity, x_cb, s, c);
	  thread_max = thread_max > fabs(z.real()) ? thread_max : fabs(z.real());
	  thread_max = thread_max > fabs(z.imag()) ? thread_max : fabs(z.imag());
	}
      }
      v[ ( (spin_block/Ms) * (Nc/Mc) + (color_block/Mc)) * blockDim.x + threadIdx.x ] = thread_max;
    }

    __syncthreads();
   
    if (active) {
#pragma unroll
      for (int sc=0; sc<(Ns/Ms) * (Nc/Mc); sc++) {
	site_max = site_max > v[sc*blockDim.x + threadIdx.x] ? site_max : v[sc*blockDim.x + threadIdx.x];
      }
    }
#else
    errorQuda("Not supported on CPU");
#endif

    return site_max;
  }


  template <typename Float, bool block_float, int Ns, int Ms, int Nc, int Mc, int nDim, int dim, int dir, typename Arg>
  __device__ __host__ __forceinline__ void packGhost(Arg &arg, int x_cb, int parity, int spinor_parity, int spin_block, int color_block) {

    int x[5] = { };
    if (nDim == 5) getCoords5(x, x_cb, arg.X, parity, arg.pc_type);
    else getCoords(x, x_cb, arg.X, parity);

    const auto &rhs = arg.field;

    {
      Float max = 1.0;
      if (block_float) {
        bool active = ( arg.commDim[dim] && ( (dir == 0 && x[dim] < arg.nFace) || (dir == 1 && x[dim] >= arg.X[dim] - arg.nFace) ) );
        max = compute_site_max<Float,Ns,Ms,Nc,Mc>(arg, x_cb, parity, spinor_parity, spin_block, color_block, active);
      }      

      if (dir == 0 && arg.commDim[dim] && x[dim] < arg.nFace) {
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
            arg.field.Ghost(dim, 0, spinor_parity, ghostFaceIndex<0, nDim>(x, arg.X, dim, arg.nFace), s, c, 0, max)
                = rhs(spinor_parity, x_cb, s, c);
          }
	}
      }

      if (dir == 1 && arg.commDim[dim] && x[dim] >= arg.X[dim] - arg.nFace) {
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
            arg.field.Ghost(dim, 1, spinor_parity, ghostFaceIndex<1, nDim>(x, arg.X, dim, arg.nFace), s, c, 0, max)
                = rhs(spinor_parity, x_cb, s, c);
          }
	}
      }
    }
  }

  template <typename Float, bool block_float, int Ns, int Ms, int Nc, int Mc, int nDim, typename Arg>
  void GenericPackGhost(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      for (int dim=0; dim<4; dim++)
	for (int dir=0; dir<2; dir++)
	  for (int x_cb=0; x_cb<arg.volumeCB; x_cb++)
	    for (int spin_block=0; spin_block<Ns; spin_block+=Ms)
	      for (int color_block=0; color_block<Nc; color_block+=Mc)
		switch(dir) {
		case 0: // backwards pack
		  switch(dim) {
		  case 0: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,0,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 1: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,1,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 2: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,2,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 3: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,3,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  }
		  break;
		case 1: // forwards pack
		  switch(dim) {
		  case 0: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,0,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 1: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,1,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 2: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,2,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  case 3: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,3,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
		  }
		}
    }
  }

  template <typename Float, bool block_float, int Ns, int Ms, int Nc, int Mc, int nDim, int dim_threads, typename Arg>
  __global__ void GenericPackGhostKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int spin_color_block = blockDim.y*blockIdx.y + threadIdx.y;
    int parity_dim_dir = blockDim.z*blockIdx.z + threadIdx.z;

    // ensure all threads are always active so it safe to synchronize
    x_cb %= arg.volumeCB;
    spin_color_block %= (Ns/Ms)*(Nc/Mc);
    parity_dim_dir %= arg.nParity2dim_threads;

    const int dim_dir = parity_dim_dir % (2*dim_threads);
    const int dim0 = dim_dir / 2;
    const int dir = dim_dir % 2;
    const int parity = (arg.nParity == 2) ? (parity_dim_dir / (2*dim_threads) ) : arg.parity;
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int spin_block = (spin_color_block / (Nc / Mc)) * Ms;
    const int color_block = (spin_color_block % (Nc / Mc)) * Mc;

#pragma unroll
    for (int dim=dim0; dim<4; dim+=dim_threads) {
      switch(dir) {
      case 0:
	switch(dim) {
	case 0: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,0,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 1: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,1,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 2: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,2,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 3: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,3,0>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	}
	break;
      case 1:
	switch(dim) {
	case 0: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,0,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 1: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,1,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 2: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,2,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	case 3: packGhost<Float,block_float,Ns,Ms,Nc,Mc,nDim,3,1>(arg, x_cb, parity, spinor_parity, spin_block, color_block); break;
	}
	break;
      }
    }
  }

} // namespace quda
