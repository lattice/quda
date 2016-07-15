#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <tune_quda.h>

namespace quda {

  template <typename Field>
  struct PackGhostArg {

    Field field;
    void **ghost;
    const void *v;
    int X[QUDA_MAX_DIM];
    const int volumeCB;
    const int nDim;
    const int nFace;
    const int parity;
    const int nParity;
    const int dagger;
    const QudaDWFPCType pc_type;
    int commDim[4]; // whether a given dimension is partitioned or not

    PackGhostArg(Field field, void **ghost, const ColorSpinorField &a, int parity, int dagger)
      : field(field),
	ghost(ghost),
	v(a.V()),
	volumeCB(a.VolumeCB()),
	nDim(a.Ndim()),
	nFace(a.Nspin() == 1 ? 3 : 1),
	parity(parity),
	nParity(a.SiteSubset()),
	dagger(dagger),
	pc_type(a.DWFPCtype())
    {
      for (int d=0; d<nDim; d++) X[d] = a.X(d);
      X[0] *= (nParity == 1) ? 2 : 1; // set to full lattice dimensions
      X[4] = (nDim == 5) ? a.X(4) : 1; // set fifth dimension correctly
      for (int i=0; i<4; i++) {
	commDim[i] = comm_dim_partitioned(i);
      }
    }
  };

  template <typename Float, int Ns, int Nc, int Mc, typename Arg>
  __device__ __host__ inline void packGhost(Arg &arg, int cb_idx, int parity, int spinor_parity, int color_block) {
    typedef typename mapper<Float>::type RegType;

    const int *X = arg.X;
    int x[5] = { };
    if (arg.nDim == 5) getCoords5(x, cb_idx, X, parity, arg.pc_type);
    else getCoords(x, cb_idx, X, parity);

#pragma unroll
    for (int dim=0; dim<4; dim++) {
      if (arg.commDim[dim] && x[dim] < arg.nFace){
	for (int s=0; s<Ns; s++) {
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
	    arg.field.Ghost(dim, 0, spinor_parity, ghostFaceIndex<0>(x,arg.X,dim,arg.nFace), s, c)
	      = arg.field(spinor_parity, cb_idx, s, c);
	  }
	}
      }
      
      if (arg.commDim[dim] && x[dim] >= X[dim] - arg.nFace){
	for (int s=0; s<Ns; s++) {
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
	    arg.field.Ghost(dim, 1, spinor_parity, ghostFaceIndex<1>(x,arg.X,dim,arg.nFace), s, c)
	      = arg.field(spinor_parity, cb_idx, s, c);
	  }
	}
      }
    }
  }

  template <typename Float, int Ns, int Nc, int Mc, typename Arg>
  void GenericPackGhost(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      for (int i=0; i<arg.volumeCB; i++)
	for (int color_block=0; color_block<Nc; color_block+=Mc)
	  packGhost<Float,Ns,Nc,Mc>(arg, i, parity, spinor_parity, color_block);
    }
  }

  template <typename Float, int Ns, int Nc, int Mc, typename Arg>
  __global__ void GenericPackGhostKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;
    const int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int color_block = (blockDim.z*blockIdx.z + threadIdx.z)*Mc;
    if (color_block >= Nc) return;
    packGhost<Float,Ns,Nc,Mc>(arg, x_cb, parity, spinor_parity, color_block);
  }

  template <typename Float, int Ns, int Nc, int Mc, typename Arg>
  class GenericPackGhostLauncher : public Tunable {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;

    long long flops() const { return 0; }
    long long bytes() const {
      // FIXME take into account paritioning
      size_t totalBytes = 0;
      for (int d=0; d<4; d++) {
	if (!comm_dim_partitioned(d)) continue;
	totalBytes += 2*arg.nFace*2*Ns*Nc*meta.SurfaceCB(d)*meta.Precision();
      }
      return totalBytes;
    }

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = Tunable::advanceBlockDim(param);
      param.block.y = block.y; param.block.z = block.z;
      param.grid.y = grid.y; param.grid.z = grid.z;

      if (ret) { // we advanced the block.x so we're done
	return true;
      } else { // block.x (spacetime) was reset

	// y thread dimension corresponds to parity
	// z thread dimension corresponds to block color (spin is kept thread local)
	if (param.block.y == 1 && arg.nParity == 2) { // advance parity
	  param.block.y = arg.nParity;
	  param.grid.y = 1;
	  return true;
	} else {
	  // reset parity
	  param.block.y = 1;
	  param.grid.y = arg.nParity;

	  // let's try to block-color
	  while(param.block.z <= Nc/Mc) {
	    param.block.z++;
	    if ( (Nc/Mc) % param.block.z == 0) {
	      param.grid.z = (Nc/Mc) / param.block.z;
	      break;
	    }
	  }

	  // we can advance block-color since this is valid
	  if (param.block.z <= Nc/Mc) { //
	    return true;
	  } else { // we have run off the end so let's reset
	    param.block.z = 1;
	    param.grid.z = Nc/Mc;
	    return false;
	  }

	}
      }
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = arg.nParity;
      param.block.z = 1;
      param.grid.z = Nc/Mc;
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.block.y = 1;
      param.grid.y = arg.nParity;
      param.block.z = 1;
      param.grid.z = Nc/Mc;
    }

  public:
    GenericPackGhostLauncher(Arg &arg, const ColorSpinorField &meta) : arg(arg), meta(meta) {
      strcpy(aux, meta.AuxString());
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux,",comm=");
      strcat(aux,comm);
#endif
    }

    virtual ~GenericPackGhostLauncher() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	GenericPackGhost<Float,Ns,Nc,Mc,Arg>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	GenericPackGhostKernel<Float,Ns,Nc,Mc,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }
  };

  template <typename Float, QudaFieldOrder order, int Ns, int Nc>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    typedef typename colorspinor::FieldOrderCB<Float,Ns,Nc,1,order> Q;
    Q field(a, 0, ghost);

    const int colors_per_thread = 1;
    PackGhostArg<Q> arg(field, ghost, a, parity, dagger);
    GenericPackGhostLauncher<Float,Ns,Nc,colors_per_thread,PackGhostArg<Q> > launch(arg, a);
    launch.apply(0);
  }

  template <typename Float, QudaFieldOrder order, int Ns>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {
    
    if (a.Ncolor() == 2) {
      genericPackGhost<Float,order,Ns,2>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 3) {
      genericPackGhost<Float,order,Ns,3>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 4) {
      genericPackGhost<Float,order,Ns,4>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 6) {
      genericPackGhost<Float,order,Ns,6>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 8) {
      genericPackGhost<Float,order,Ns,8>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 12) {
      genericPackGhost<Float,order,Ns,12>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 16) {
      genericPackGhost<Float,order,Ns,16>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 20) {
      genericPackGhost<Float,order,Ns,20>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 24) {
      genericPackGhost<Float,order,Ns,24>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 28) {
      genericPackGhost<Float,order,Ns,28>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 32) {
      genericPackGhost<Float,order,Ns,32>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 48) {
      genericPackGhost<Float,order,Ns,48>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 72) {
      genericPackGhost<Float,order,Ns,72>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 96) {
      genericPackGhost<Float,order,Ns,96>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 256) {
      genericPackGhost<Float,order,Ns,256>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 576) {
      genericPackGhost<Float,order,Ns,576>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 768) {
      genericPackGhost<Float,order,Ns,768>(ghost, a, parity, dagger);
    } else if (a.Ncolor() == 1024) {
      genericPackGhost<Float,order,Ns,1024>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  template <typename Float, QudaFieldOrder order>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.Nspin() == 4) {
      genericPackGhost<Float,order,4>(ghost, a, parity, dagger);
    } else if (a.Nspin() == 2) {
      genericPackGhost<Float,order,2>(ghost, a, parity, dagger);
#ifdef GPU_STAGGERED_DIRAC
    } else if (a.Nspin() == 1) {
      genericPackGhost<Float,order,1>(ghost, a, parity, dagger);
#endif
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  template <typename Float>
  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_FLOAT2_FIELD_ORDER>(ghost, a, parity, dagger);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported field order = %d", a.FieldOrder());
    }

  }

  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger) {

    if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", a.FieldOrder());
    }

    // only do packing if one of the dimensions is partitioned
    bool partitioned = false;
    for (int d=0; d<4; d++)
      if (comm_dim_partitioned(d)) partitioned = true;
    if (!partitioned) return;

    if (a.Precision() == QUDA_DOUBLE_PRECISION) {
      genericPackGhost<double>(ghost, a, parity, dagger);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericPackGhost<float>(ghost, a, parity, dagger);
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }

} // namespace quda
