#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <tune_quda.h>
#include <fast_intdiv.h>

namespace quda {

  template <typename Field>
  struct PackGhostArg {

    Field field;
    void **ghost;
    const void *v;
    int_fastdiv X[QUDA_MAX_DIM];
    const int volumeCB;
    const int nDim;
    const int nFace;
    const int parity;
    const int nParity;
    const int dagger;
    const QudaDWFPCType pc_type;
    int commDim[4]; // whether a given dimension is partitioned or not

    PackGhostArg(Field field, void **ghost, const ColorSpinorField &a, int parity, int nFace, int dagger)
      : field(field),
	ghost(ghost),
	v(a.V()),
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

  template <typename Float, int Ns, int Ms, int Nc, int Mc, int nDim, typename Arg>
  __device__ __host__ inline void packGhost(Arg &arg, int cb_idx, int parity, int spinor_parity, int spin_block, int color_block) {
    typedef typename mapper<Float>::type RegType;

    int x[5] = { };
    if (nDim == 5) getCoords5(x, cb_idx, arg.X, parity, arg.pc_type);
    else getCoords(x, cb_idx, arg.X, parity);

#pragma unroll
    for (int dim=0; dim<4; dim++) {
      if (arg.commDim[dim] && x[dim] < arg.nFace){
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
	    arg.field.Ghost(dim, 0, spinor_parity, ghostFaceIndex<0>(x,arg.X,dim,arg.nFace), s, c)
	      = arg.field(spinor_parity, cb_idx, s, c);
	  }
	}
      }
      
      if (arg.commDim[dim] && x[dim] >= arg.X[dim] - arg.nFace){
	for (int spin_local=0; spin_local<Ms; spin_local++) {
	  int s = spin_block + spin_local;
	  for (int color_local=0; color_local<Mc; color_local++) {
	    int c = color_block + color_local;
	    arg.field.Ghost(dim, 1, spinor_parity, ghostFaceIndex<1>(x,arg.X,dim,arg.nFace), s, c)
	      = arg.field(spinor_parity, cb_idx, s, c);
	  }
	}
      }
    }
  }

  template <typename Float, int Ns, int Ms, int Nc, int Mc, int nDim, typename Arg>
  void GenericPackGhost(Arg &arg) {
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;
      const int spinor_parity = (arg.nParity == 2) ? parity : 0;
      for (int i=0; i<arg.volumeCB; i++)
	for (int spin_block=0; spin_block<Ns; spin_block+=Ms)
	  for (int color_block=0; color_block<Nc; color_block+=Mc)
	    packGhost<Float,Ns,Ms,Nc,Mc,nDim>(arg, i, parity, spinor_parity, spin_block, color_block);
    }
  }

  template <typename Float, int Ns, int Ms, int Nc, int Mc, int nDim, typename Arg>
  __global__ void GenericPackGhostKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.volumeCB) return;

    const int parity = (arg.nParity == 2) ? blockDim.z*blockIdx.z + threadIdx.z : arg.parity;
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;
    const int spin_color_block = blockDim.y*blockIdx.y + threadIdx.y;
    if (spin_color_block >= (Ns/Ms)*(Nc/Mc)) return; // ensure only valid threads
    const int spin_block = (spin_color_block / (Nc / Mc)) * Ms;
    const int color_block = (spin_color_block % (Nc / Mc)) * Mc;
    packGhost<Float,Ns,Ms,Nc,Mc,nDim>(arg, x_cb, parity, spinor_parity, spin_block, color_block);
  }

  template <typename Float, int Ns, int Ms, int Nc, int Mc, typename Arg>
  class GenericPackGhostLauncher : public TunableVectorYZ {
    Arg &arg;
    const ColorSpinorField &meta;
    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; }

  public:
    inline GenericPackGhostLauncher(Arg &arg, const ColorSpinorField &meta, MemoryLocation *destination)
      : TunableVectorYZ((Ns/Ms)*(Nc/Mc), arg.nParity), arg(arg), meta(meta) {
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
	if (arg.nDim == 5) GenericPackGhost<Float,Ns,Ms,Nc,Mc,5,Arg>(arg);
	else GenericPackGhost<Float,Ns,Ms,Nc,Mc,4,Arg>(arg);
      } else {
	const TuneParam &tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (arg.nDim == 5) GenericPackGhostKernel<Float,Ns,Ms,Nc,Mc,5,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	else GenericPackGhostKernel<Float,Ns,Ms,Nc,Mc,4,Arg> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    long long flops() const { return 0; }
    long long bytes() const {
      size_t totalBytes = 0;
      for (int d=0; d<4; d++) {
	if (!comm_dim_partitioned(d)) continue;
	totalBytes += 2*arg.nFace*2*Ns*Nc*meta.SurfaceCB(d)*meta.Precision();
      }
      return totalBytes;
    }
  };

  template <typename Float, QudaFieldOrder order, int Ns, int Nc>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    typedef typename colorspinor::FieldOrderCB<Float,Ns,Nc,1,order> Q;
    Q field(a, nFace, 0, ghost);

    constexpr int spins_per_thread = 1; // make this autotunable
    constexpr int colors_per_thread = 1;
    PackGhostArg<Q> arg(field, ghost, a, parity, nFace, dagger);
    GenericPackGhostLauncher<Float,Ns,spins_per_thread,Nc,colors_per_thread,PackGhostArg<Q> >
      launch(arg, a, destination);
    launch.apply(0);
  }

  template <typename Float, QudaFieldOrder order, int Ns>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {
    
    if (a.Ncolor() == 2) {
      genericPackGhost<Float,order,Ns,2>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 3) {
      genericPackGhost<Float,order,Ns,3>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 4) {
      genericPackGhost<Float,order,Ns,4>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 6) {
      genericPackGhost<Float,order,Ns,6>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 8) {
      genericPackGhost<Float,order,Ns,8>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 12) {
      genericPackGhost<Float,order,Ns,12>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 16) {
      genericPackGhost<Float,order,Ns,16>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 20) {
      genericPackGhost<Float,order,Ns,20>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 24) {
      genericPackGhost<Float,order,Ns,24>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 28) {
      genericPackGhost<Float,order,Ns,28>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 32) {
      genericPackGhost<Float,order,Ns,32>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 48) {
      genericPackGhost<Float,order,Ns,48>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 72) {
      genericPackGhost<Float,order,Ns,72>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 96) {
      genericPackGhost<Float,order,Ns,96>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 256) {
      genericPackGhost<Float,order,Ns,256>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 576) {
      genericPackGhost<Float,order,Ns,576>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 768) {
      genericPackGhost<Float,order,Ns,768>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Ncolor() == 1024) {
      genericPackGhost<Float,order,Ns,1024>(ghost, a, parity, nFace, dagger, destination);
    } else {
      errorQuda("Unsupported nColor = %d", a.Ncolor());
    }

  }

  template <typename Float, QudaFieldOrder order>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    if (a.Nspin() == 4) {
      genericPackGhost<Float,order,4>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Nspin() == 2) {
      genericPackGhost<Float,order,2>(ghost, a, parity, nFace, dagger, destination);
#ifdef GPU_STAGGERED_DIRAC
    } else if (a.Nspin() == 1) {
      genericPackGhost<Float,order,1>(ghost, a, parity, nFace, dagger, destination);
#endif
    } else {
      errorQuda("Unsupported nSpin = %d", a.Nspin());
    }

  }

  template <typename Float>
  inline void genericPackGhost(void **ghost, const ColorSpinorField &a, QudaParity parity,
			       int nFace, int dagger, MemoryLocation *destination) {

    if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_FLOAT2_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_FLOAT4_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      genericPackGhost<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(ghost, a, parity, nFace, dagger, destination);
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
      genericPackGhost<double>(ghost, a, parity, nFace, dagger, destination);
    } else if (a.Precision() == QUDA_SINGLE_PRECISION) {
      genericPackGhost<float>(ghost, a, parity, nFace, dagger, destination);
    } else {
      errorQuda("Unsupported precision %d", a.Precision());
    }

  }

} // namespace quda
