#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <tune_quda.h>
#include <fast_intdiv.h>

/**
   @file color_spinor_pack.cu

   @brief This is the implementation of the color-spinor halo packer
   for an arbitrary field.  This implementation uses the fine-grained
   accessors and should support all field types reqgardless of
   precision, number of color or spins etc.

   Using a different precision of the field and of the halo is
   supported, though only QUDA_SINGLE_PRECISION fields with
   QUDA_HALF_PRECISION or QUDA_QUARTER_PRECISION halos are
   instantiated. When an integer format is requested for the halos
   then block-float format is used.

   As well as tuning basic block sizes, the autotuner also tunes for
   the dimensions to assign to each thread.  E.g., dim_thread=1 means
   we have one thread for all dimensions, dim_thread=4 means we have
   four threads (e.g., one per dimension).  We always uses seperate
   threads for forwards and backwards directions.  Dimension,
   direction and parity are assigned to the z thread dimension.

   If doing block-float format, since all spin and color components of
   a given site have to reside in the same thread block (to allow us
   to compute the max element) we override the autotuner to keep the z
   thread dimensions in the grid and not the block, and allow for
   smaller tuning increments of the thread block dimension in x to
   ensure that we can always fit within a single thread block.  It is
   this constraint that gives rise for the need to cap the limit for
   block-float support, e.g., MAX_BLOCK_FLOAT_NC.

   At present we launch a volume of threads (actually multiples
   thereof for direction / dimension) and thus we have coalesced reads
   but not coalesced writes.  A more optimal implementation will
   launch a surface of threads for each halo giving coalesced writes.
 */

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
    const QudaDWFPCType pc_type;
    int commDim[4]; // whether a given dimension is partitioned or not
    int_fastdiv nParity2dim_threads;

    PackGhostArg(Field field, const ColorSpinorField &a, int parity, int nFace, int dagger)
      : field(field),
	volumeCB(a.VolumeCB()),
	nDim(a.Ndim()),
	nFace(nFace),
	parity(parity),
	nParity(a.SiteSubset()),
	dagger(dagger),
	pc_type(a.DWFPCtype())
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
  __device__ __host__ inline Float compute_site_max(Arg &arg, int x_cb, int parity, int spinor_parity, int spin_block, int color_block, bool active) {

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
  __device__ __host__ inline void packGhost(Arg &arg, int x_cb, int parity, int spinor_parity, int spin_block, int color_block) {

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
	    arg.field.Ghost(dim, 0, spinor_parity, ghostFaceIndex<0>(x,arg.X,dim,arg.nFace), s, c, 0, max) = rhs(spinor_parity, x_cb, s, c);
	  }
	}
      }

      if (dir == 1 && arg.commDim[dim] && x[dim] >= arg.X[dim] - arg.nFace) {
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
	    arg.field.Ghost(dim, 1, spinor_parity, ghostFaceIndex<1>(x,arg.X,dim,arg.nFace), s, c, 0, max) = rhs(spinor_parity, x_cb, s, c);
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

  template <typename Float, bool block_float, int Ns, int Ms, int Nc, int Mc, typename Arg>
  class GenericPackGhostLauncher : public TunableVectorYZ {
    Arg &arg;
    const ColorSpinorField &meta;
    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; }
    bool tuneAuxDim() const { return true; }

  public:
    inline GenericPackGhostLauncher(Arg &arg, const ColorSpinorField &meta, MemoryLocation *destination)
      : TunableVectorYZ((Ns/Ms)*(Nc/Mc), 2*arg.nParity), arg(arg), meta(meta) {
      strcpy(aux, meta.AuxString());
      strcat(aux,comm_dim_partitioned_string());

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      char label[15] = ",dest=";
      for (int dim=0; dim<4; dim++) {
	for (int dir=0; dir<2; dir++) {
	  label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : destination[2*dim+dir] == Device ? '1' : destination[2*dim+dir] == Host ? '2' : '3';
	}
      }
      label[14] = '\0';
      strcat(aux,label);
    }

    virtual ~GenericPackGhostLauncher() { }

    inline void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (arg.nDim == 5) GenericPackGhost<Float,block_float,Ns,Ms,Nc,Mc,5,Arg>(arg);
	else GenericPackGhost<Float,block_float,Ns,Ms,Nc,Mc,4,Arg>(arg);
      } else {
	const TuneParam &tp = tuneLaunch(*this, getTuning(), getVerbosity());
	arg.nParity2dim_threads = arg.nParity*2*tp.aux.x;
	switch(tp.aux.x) {
	case 1:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,1,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,1,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
	case 2:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,2,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,2,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
	case 4:
	  if (arg.nDim == 5) GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,5,4,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  else GenericPackGhostKernel<Float,block_float,Ns,Ms,Nc,Mc,4,4,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  break;
	}
      }
    }

    // if doing block float then all spin-color components must be within the same block
    void setColorSpinBlock(TuneParam &param) const {
      param.block.y = (Ns/Ms)*(Nc/Mc);
      param.grid.y = 1;
      param.block.z = 1;
      param.grid.z = arg.nParity*2*param.aux.x;
    }

    bool advanceBlockDim(TuneParam &param) const {
      if (!block_float) {
	return TunableVectorYZ::advanceBlockDim(param);
      } else {
	bool advance = Tunable::advanceBlockDim(param);
	setColorSpinBlock(param); // if doing block float then all spin-color components must be within the same block
	return advance;
      }
    }

    int blockStep() const { return block_float ? 2 : TunableVectorYZ::blockStep(); }
    int blockMin() const { return block_float ? 2 : TunableVectorYZ::blockMin(); }

    bool advanceAux(TuneParam &param) const {
      if (param.aux.x < 4) {
	param.aux.x *= 2;
	const_cast<GenericPackGhostLauncher*>(this)->resizeVector((Ns/Ms)*(Nc/Mc), arg.nParity*2*param.aux.x);
	TunableVectorYZ::initTuneParam(param);
	if (block_float) setColorSpinBlock(param);
	return true;
      }
      param.aux.x = 1;
      const_cast<GenericPackGhostLauncher*>(this)->resizeVector((Ns/Ms)*(Nc/Mc), arg.nParity*2*param.aux.x);
      TunableVectorYZ::initTuneParam(param);
      if (block_float) setColorSpinBlock(param);
      return false;
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    virtual void initTuneParam(TuneParam &param) const {
      TunableVectorYZ::initTuneParam(param);
      param.aux = make_int4(1,1,1,1);
      if (block_float) setColorSpinBlock(param);
    }

    virtual void defaultTuneParam(TuneParam &param) const {
      TunableVectorYZ::defaultTuneParam(param);
      param.aux = make_int4(1,1,1,1);
      if (block_float) setColorSpinBlock(param);
    }

    long long flops() const { return 0; }
    long long bytes() const {
      size_t totalBytes = 0;
      for (int d=0; d<4; d++) {
	if (!comm_dim_partitioned(d)) continue;
	totalBytes += arg.nFace*2*Ns*Nc*meta.SurfaceCB(d)*(meta.Precision() + meta.GhostPrecision());
      }
      return totalBytes;
    }
  };

  template <typename Float, typename ghostFloat, QudaFieldOrder order, int Ns, int Nc>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    typedef typename mapper<Float>::type RegFloat;
    typedef typename colorspinor::FieldOrderCB<RegFloat,Ns,Nc,1,order,Float,ghostFloat> Q;
    Q field(a, nFace, 0, ghost);

    constexpr int spins_per_thread = Ns == 1 ? 1 : 2; // make this autotunable?
    constexpr int colors_per_thread = Nc%2 == 0 ? 2 : 1;
    PackGhostArg<Q> arg(field, a, parity, nFace, dagger);

    // if we only have short precision for the ghost then this means we have block-float
    constexpr bool block_float = (sizeof(Float) == QUDA_SINGLE_PRECISION &&
				  (sizeof(ghostFloat) == QUDA_HALF_PRECISION || sizeof(ghostFloat) == QUDA_QUARTER_PRECISION)
                                  && Nc <= MAX_BLOCK_FLOAT_NC) ? true : false;

    // ensure we only compile supported block-float kernels
    constexpr int Nc_ = (sizeof(Float) == QUDA_SINGLE_PRECISION &&
                         (sizeof(ghostFloat) == QUDA_HALF_PRECISION || sizeof(ghostFloat) == QUDA_QUARTER_PRECISION) &&
                         Nc > MAX_BLOCK_FLOAT_NC) ? MAX_BLOCK_FLOAT_NC : Nc;

    if (sizeof(Float) == QUDA_SINGLE_PRECISION &&
        (sizeof(ghostFloat) == QUDA_HALF_PRECISION || sizeof(ghostFloat) == QUDA_QUARTER_PRECISION) && Nc > MAX_BLOCK_FLOAT_NC)
      errorQuda("Block-float format not supported for Nc = %d", Nc);

    GenericPackGhostLauncher<RegFloat,block_float,Ns,spins_per_thread,Nc_,colors_per_thread,PackGhostArg<Q> >
      launch(arg, a, destination);

    launch.apply(0);
  }

  // traits used to ensure we only instantiate arbitrary colors for nSpin=2,4 fields, and only 3 colors otherwise
  template<typename T, typename G, int nSpin, int nColor_> struct precision_spin_color_mapper { static constexpr int nColor = nColor_; };
  template<typename T, typename G, int nColor_> struct precision_spin_color_mapper<T,G,1,nColor_> { static constexpr int nColor = 3; };

  // never need block-float format with nSpin=4 fields for arbitrary colors
  template<int nColor_> struct precision_spin_color_mapper<float,short,4,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<float,char,4,nColor_> { static constexpr int nColor = 3; };

#ifndef GPU_MULTIGRID_DOUBLE
  template<int nColor_> struct precision_spin_color_mapper<double,double,1,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<double,double,2,nColor_> { static constexpr int nColor = 3; };
  template<int nColor_> struct precision_spin_color_mapper<double,double,4,nColor_> { static constexpr int nColor = 3; };
#endif

  template <typename Float, typename ghostFloat, QudaFieldOrder order, int Ns>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {
    
    if (a.Ncolor() != 3 && a.Nspin() == 1)
      errorQuda("Ncolor = %d not supported for Nspin = %d fields", a.Ncolor(), a.Nspin());
    if (a.Ncolor() != 3 && a.Nspin() == 4 && a.Precision() == QUDA_SINGLE_PRECISION &&
        (a.GhostPrecision() == QUDA_HALF_PRECISION || a.GhostPrecision() == QUDA_QUARTER_PRECISION) )
      errorQuda("Ncolor = %d not supported for Nspin = %d fields with precision = %d and ghost_precision = %d",
                a.Ncolor(), a.Nspin(), a.Precision(), a.GhostPrecision());
#ifndef GPU_MULTIGRID_DOUBLE
    if ( a.Ncolor() != 3 && a.Precision() == QUDA_DOUBLE_PRECISION)
      errorQuda("Ncolor = %d not supported for double precision fields", a.Ncolor());
#endif

    if (a.Ncolor() == 2) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,2>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 3) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,3>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_MULTIGRID
    } else if (a.Ncolor() == 4) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,4>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 6) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,6>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 8) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,8>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 12) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,12>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 16) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,16>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 18) { // Needed for two level free field Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,18>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 20) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,20>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 24) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,24>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 28) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,28>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 32) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,32>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 36) { // Needed for three level free field Wilson
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,36>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 48) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,48>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 72) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,72>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 96) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,96>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 128) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,128>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 256) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,256>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 576) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,576>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 768) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,768>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 1024) {
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,1024>::nColor>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_STAGGERED_DIRAC
    } else if (a.Ncolor() == 2304) { // staggered KD 24 -> 96 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,2304>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 3072) { // staggered KD 24 -> 128 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,3072>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 9216) { // staggered 96 -> 96
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,9216>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 12288) { // staggered 96 -> 128 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,12288>::nColor>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 16384) { // staggered 128 -> 128 nvec
      genericPackGhost<Float,ghostFloat,order,Ns,precision_spin_color_mapper<Float,ghostFloat,Ns,16384>::nColor>(ghost, a, parity, nFace, dagger, destination);
#endif // GPU_STAGGERED_DIRAC
#endif // GPU_MULTIGRID
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  // traits used to ensure we only instantiate float4 for spin=4 fields
  template<int nSpin,QudaFieldOrder order_> struct spin_order_mapper { static constexpr QudaFieldOrder order = order_; };
  template<> struct spin_order_mapper<2,QUDA_FLOAT4_FIELD_ORDER> { static constexpr QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER; };
  template<> struct spin_order_mapper<1,QUDA_FLOAT4_FIELD_ORDER> { static constexpr QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER; };

  template <typename Float, typename ghostFloat, QudaFieldOrder order>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    if (a.Nspin() == 4) {
      genericPackGhost<Float,ghostFloat,order,4>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Nspin() == 2) {
      if (order == QUDA_FLOAT4_FIELD_ORDER) errorQuda("Field order %d with nSpin = %d not supported", order, a.Nspin());
      genericPackGhost<Float,ghostFloat,spin_order_mapper<2,order>::order,2>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_STAGGERED_DIRAC
    } else if (a.Nspin() == 1) {
      if (order == QUDA_FLOAT4_FIELD_ORDER) errorQuda("Field order %d with nSpin = %d not supported", order, a.Nspin());
      genericPackGhost<Float,ghostFloat,spin_order_mapper<1,order>::order,1>(ghost, a, parity, nFace, dagger, destination);
#endif
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  // traits used to ensure we only instantiate double and float templates for non-native fields
  template<typename> struct non_native_precision_mapper { };
  template<> struct non_native_precision_mapper<double> { typedef double type; };
  template<> struct non_native_precision_mapper<float> { typedef float type; };
  template<> struct non_native_precision_mapper<short> { typedef float type; };
  template<> struct non_native_precision_mapper<char> { typedef float type; };

  // traits used to ensure we only instantiate float and lower precision for float4 fields
  template<typename T> struct float4_precision_mapper { typedef T type; };
  template<> struct float4_precision_mapper<double> { typedef float type; };
  template<> struct float4_precision_mapper<short> { typedef float type; };
  template<> struct float4_precision_mapper<char> { typedef float type; };

  template <typename Float, typename ghostFloat>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {

      // all precisions, color and spin can use this order
      genericPackGhost<Float,ghostFloat,QUDA_FLOAT2_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);

    } else if (a.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {

      // never have double fields here
      if (typeid(Float) != typeid(typename float4_precision_mapper<Float>::type))
        errorQuda("Precision %d not supported for field type %d", a.Precision(), a.FieldOrder());
      if (typeid(ghostFloat) != typeid(typename float4_precision_mapper<ghostFloat>::type))
        errorQuda("Ghost precision %d not supported for field type %d", a.GhostPrecision(), a.FieldOrder());
      genericPackGhost<typename float4_precision_mapper<Float>::type,
                       typename float4_precision_mapper<ghostFloat>::type,
                       QUDA_FLOAT4_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);

    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      if (typeid(Float) != typeid(typename non_native_precision_mapper<Float>::type))
        errorQuda("Precision %d not supported for field type %d", a.Precision(), a.FieldOrder());
      if (typeid(ghostFloat) != typeid(typename non_native_precision_mapper<ghostFloat>::type))
        errorQuda("Ghost precision %d not supported for field type %d", a.GhostPrecision(), a.FieldOrder());
      genericPackGhost<typename non_native_precision_mapper<Float>::type,
                       typename non_native_precision_mapper<ghostFloat>::type,
                       QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
    } else {
      errorQuda("Unsupported field order = %d", a.FieldOrder());
    }

  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			int nFace, int dagger, MemoryLocation *destination_) {

    if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", a.FieldOrder());
    }

    // set default location to match field type
    MemoryLocation destination[2*QUDA_MAX_DIM];
    for (int i=0; i<4*2; i++) {
      destination[i] = destination_ ? destination_[i] : a.Location() == QUDA_CUDA_FIELD_LOCATION ? Device : Host;
    }

    // only do packing if one of the dimensions is partitioned
    bool partitioned = false;
    for (int d=0; d<4; d++)
      if (comm_dim_partitioned(d)) partitioned = true;
    if (!partitioned) return;

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      if (a.GhostPrecision() == QUDA_DOUBLE_PRECISION) {
        genericPackGhost<double,double>(ghost, a, parity, nFace, dagger, destination);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      if (a.GhostPrecision() == QUDA_SINGLE_PRECISION) {
        genericPackGhost<float,float>(ghost, a, parity, nFace, dagger, destination);
#if 0 // ESW COMPILE TIME
      } else if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
        genericPackGhost<float,short>(ghost, a, parity, nFace, dagger, destination);
      } else if (a.GhostPrecision() == QUDA_QUARTER_PRECISION) {
        genericPackGhost<float,char>(ghost, a, parity, nFace, dagger, destination);
#endif
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
#if 0 // ESW COMPILE TIME
    } else if (a.Precision() == QUDA_HALF_PRECISION) {
      if (a.GhostPrecision() == QUDA_HALF_PRECISION) {
        genericPackGhost<short,short>(ghost, a, parity, nFace, dagger, destination);
      } else {
        errorQuda("precision = %d and ghost precision = %d not supported", a.Precision(), a.GhostPrecision());
      }
#endif
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }

} // namespace quda
