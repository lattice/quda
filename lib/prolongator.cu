#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>

#include <multigrid_helper.cuh>

namespace quda {

#ifdef GPU_MULTIGRID
  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Out, typename In, typename Rotator, int fineSpin, int coarseSpin>
  struct ProlongateArg {
    Out out;
    const In in;
    const Rotator V;
    const int *geo_map;  // need to make a device copy of this
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the output field (if single parity)
    const int nParity; // number of parities of input fine field

    ProlongateArg(Out &out, const In &in, const Rotator &V, 
		  const int *geo_map,  const int parity, const ColorSpinorField &meta) :
      out(out), in(in), V(V), geo_map(geo_map), spin_map(),
      parity(parity), nParity(meta.SiteSubset())
    { }

    ProlongateArg(const ProlongateArg<Out,In,Rotator,fineSpin,coarseSpin> &arg) :
      out(arg.out), in(arg.in), V(arg.V), geo_map(arg.geo_map), spin_map(),
      parity(arg.parity), nParity(arg.nParity) {
    }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <typename Float, int fineSpin, int coarseColor, class Coarse, typename S>
  __device__ __host__ inline void prolongate(complex<Float> out[fineSpin*coarseColor], const Coarse &in, 
					     int parity, int x_cb, const int *geo_map, const S& spin_map, int fineVolume) {
    int x = parity*fineVolume/2 + x_cb;
    int x_coarse = geo_map[x];
    int parity_coarse = (x_coarse >= in.VolumeCB()) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*in.VolumeCB();

#pragma unroll
    for (int s=0; s<fineSpin; s++) {
#pragma unroll
      for (int c=0; c<coarseColor; c++) {
	out[s*coarseColor+c] = in(parity_coarse, x_coarse_cb, spin_map(s), c);
      }
    }
  }

  /**
     Rotates from the coarse-color basis into the fine-color basis.  This
     is the second step of applying the prolongator.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, int fine_colors_per_thread,
	    class FineColor, class Rotator>
  __device__ __host__ inline void rotateFineColor(FineColor &out, const complex<Float> in[fineSpin*coarseColor],
						  const Rotator &V, int parity, int nParity, int x_cb, int fine_color_block) {
    const int spinor_parity = (nParity == 2) ? parity : 0;

#pragma unroll
    for (int s=0; s<fineSpin; s++)
#pragma unroll
      for (int fine_color_local=0; fine_color_local<fine_colors_per_thread; fine_color_local++)
	out(spinor_parity, x_cb, s, fine_color_block+fine_color_local) = 0.0; // global fine color index
    
#pragma unroll
    for (int s=0; s<fineSpin; s++) {
#pragma unroll
      for (int fine_color_local=0; fine_color_local<fine_colors_per_thread; fine_color_local++) {
	int i = fine_color_block + fine_color_local; // global fine color index
#pragma unroll
	for (int j=0; j<coarseColor; j++) {
	  // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	  out(spinor_parity, x_cb, s, i) += V(parity, x_cb, s, i, j) * in[s*coarseColor + j];
	}
      }
    }

  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread, typename Arg>
  void Prolongate(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.out.VolumeCB(); x_cb++) {
	complex<Float> tmp[fineSpin*coarseColor];
	prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.Volume());
	for (int fine_color_block=0; fine_color_block<fineColor; fine_color_block+=fine_colors_per_thread) {
	  rotateFineColor<Float,fineSpin,fineColor,coarseColor,fine_colors_per_thread>
	    (arg.out, tmp, arg.V, parity, arg.nParity, x_cb, fine_color_block);
	}
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread, typename Arg>
  __global__ void ProlongateKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = arg.nParity == 2 ? threadIdx.y : arg.parity; // parity is within the block
    if (x_cb >= arg.out.VolumeCB()) return;

    int fine_color_block = (blockDim.z*blockIdx.z + threadIdx.z) * fine_colors_per_thread;
    if (fine_color_block >= fineColor) return;

    complex<Float> tmp[fineSpin*coarseColor];
    prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.Volume());
    rotateFineColor<Float,fineSpin,fineColor,coarseColor,fine_colors_per_thread>
      (arg.out, tmp, arg.V, parity, arg.nParity, x_cb, fine_color_block);
  }
  
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int fine_colors_per_thread, typename Arg>
  class ProlongateLaunch : public Tunable {

  protected:
    Arg &arg;
    QudaFieldLocation location;
    char vol[TuneKey::volume_n];

    long long flops() const { return 0; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.out.VolumeCB(); } // fine parity is the block y dimension

  public:
    ProlongateLaunch(Arg &arg, const ColorSpinorField &fine, const ColorSpinorField &coarse, 
		     const QudaFieldLocation location) : arg(arg), location(location) { 
      strcpy(vol, fine.VolString());
      strcat(vol, ",");
      strcat(vol, coarse.VolString());

      strcpy(aux, fine.AuxString());
      strcat(aux, ",");
      strcat(aux, coarse.AuxString());
    }

    virtual ~ProlongateLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	Prolongate<Float,fineSpin,fineColor,coarseSpin,coarseColor,fine_colors_per_thread>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	tp.block.y = arg.nParity;
	ProlongateKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,fine_colors_per_thread,Arg>
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(vol, typeid(*this).name(), aux);
    }

    /**
       Specialized variant of advanceBlockDim that tunes for the
       optimal fine colors per block / grid.
     */
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

	// let's try to advance spin/block-color
	while(param.block.z <= fineColor/fine_colors_per_thread) {
	  param.block.z++;
	  if ( (fineColor/fine_colors_per_thread) % param.block.z == 0) {
	    param.grid.z = (fineColor/fine_colors_per_thread) / param.block.z;
	    break;
	  }
	}

	// we can advance spin/block-color since this is valid
	if (param.block.z <= (fineColor/fine_colors_per_thread) ) { //
	  return true;
	} else { // we have run off the end so let's reset
	  param.block.z = 1;
	  param.grid.z = fineColor/fine_colors_per_thread;
	  return false;
	}
      }
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid = dim3( ((arg.out.VolumeCB())+param.block.x-1) / param.block.x, 1, 1);

      param.block.z = 1;
      param.grid.z = fineColor / fine_colors_per_thread;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( ((arg.out.VolumeCB())+param.block.x-1) / param.block.x, 1, 1);

      param.block.z = 1;
      param.grid.z = fineColor / fine_colors_per_thread;
    }

    long long bytes() const {
      return arg.in.Bytes() + arg.out.Bytes() + arg.V.Bytes() + arg.out.Volume()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  const int *fine_to_coarse, int parity) {

    typedef FieldOrderCB<Float,fineSpin,fineColor,1,order> fineSpinor;
    typedef FieldOrderCB<Float,coarseSpin,coarseColor,1,order> coarseSpinor;
    typedef FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order> packedSpinor;
    typedef ProlongateArg<fineSpinor,coarseSpinor,packedSpinor,fineSpin,coarseSpin> Arg;

    fineSpinor   Out(const_cast<ColorSpinorField&>(out));
    coarseSpinor In(const_cast<ColorSpinorField&>(in));
    packedSpinor V(const_cast<ColorSpinorField&>(v));

    // for fine grid we keep 3 colors per thread else use fine grained
    constexpr int fine_colors_per_thread = 1;//fineColor == 3 ? fineColor : 1;

    Arg arg(Out, In, V, fine_to_coarse, parity, out);
    ProlongateLaunch<Float, fineSpin, fineColor, coarseSpin, coarseColor, fine_colors_per_thread, Arg>
      prolongator(arg, out, in, Location(out, in, v));
    prolongator.apply(0);

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }


  template <typename Float, int fineSpin, int fineColor, int coarseSpin, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int nVec, const int *fine_to_coarse, const int *spin_map, int parity) {

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      if (mapper(s) != spin_map[s]) errorQuda("Spin map does not match spin_mapper");

    if (nVec == 2) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,2,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 4) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,4,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 8) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,8,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 12) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,12,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 16) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,16,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 20) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,20,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 24) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,24,order>(out, in, v, fine_to_coarse, parity);
    } else if (nVec == 48) {
      Prolongate<Float,fineSpin,fineColor,coarseSpin,48,order>(out, in, v, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported nVec %d", nVec);
    }
  }

  template <typename Float, int fineSpin, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map, int parity) {

    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());

    if (out.Ncolor() == 3) {
      Prolongate<Float,fineSpin,3,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Ncolor() == 2) {
      Prolongate<Float,fineSpin,2,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Ncolor() == 8) {
      Prolongate<Float,fineSpin,8,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Ncolor() == 16) {
      Prolongate<Float,fineSpin,16,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Ncolor() == 24) {
      Prolongate<Float,fineSpin,24,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Ncolor() == 48) {
      Prolongate<Float,fineSpin,48,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported nColor %d", out.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map, int parity) {

    if (out.Nspin() == 4) {
      Prolongate<Float,4,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Nspin() == 2) {
      Prolongate<Float,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#ifdef GPU_STAGGERED_DIRAC
    } else if (out.Nspin() == 1) {
      Prolongate<Float,1,order>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", out.Nspin());
    }
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map, int parity) {

    if (out.FieldOrder() != in.FieldOrder() || out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)", 
		out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      Prolongate<Float,QUDA_FLOAT2_FIELD_ORDER>
	(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      Prolongate<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
	(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported field type %d", out.FieldOrder());
    }
  }
#endif // GPU_MULTIGRID

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map, int parity) {
#ifdef GPU_MULTIGRID
    if (out.Precision() != in.Precision() || v.Precision() != in.Precision()) 
      errorQuda("Precision mismatch out=%d in=%d v=%d", out.Precision(), in.Precision(), v.Precision());

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      Prolongate<double>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      Prolongate<float>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
#else
    errorQuda("Multigrid has not been built");
#endif
  }

} // end namespace quda
