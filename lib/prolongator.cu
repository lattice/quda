#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Out, typename In, typename Rotator, int fineSpin>
  struct ProlongateArg {
    Out out;
    const In in;
    const Rotator V;
    const int *geo_map;  // need to make a device copy of this
    int spin_map[fineSpin];
    ProlongateArg(Out &out, const In &in, const Rotator &V, 
		  const int *geo_map, const int *spin_map) : 
      out(out), in(in), V(V), geo_map(geo_map)  {
      for (int s=0; s<fineSpin; s++) this->spin_map[s] = spin_map[s];
    }

    ProlongateArg(const ProlongateArg<Out,In,Rotator,fineSpin> &arg) :
      out(arg.out), in(arg.in), V(arg.V), geo_map(arg.geo_map) {
      for (int s=0; s<fineSpin; s++) this->spin_map[s] = arg.spin_map[s];
    }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <typename Float, int fineSpin, int coarseColor, class Coarse>
  __device__ __host__ inline void prolongate(complex<Float> out[fineSpin*coarseColor], const Coarse &in, 
					     int parity, int x_cb, const int *geo_map, const int *spin_map, int fineVolume) {
    int x = parity*fineVolume/2 + x_cb;
    int x_coarse = geo_map[x];
    int parity_coarse = (x_coarse >= in.Volume()/2) ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*in.Volume()/2;

    for (int s=0; s<fineSpin; s++) {
      for (int c=0; c<coarseColor; c++) {
	out[s*coarseColor+c] = in(parity_coarse, x_coarse_cb, spin_map[s], c);
      }
    }
  }

  /**
     Rotates from the coarse-color basis into the fine-color basis.  This
     is the second step of applying the prolongator.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, class FineColor, class Rotator>
  __device__ __host__ inline void rotateFineColor(FineColor &out, const complex<Float> in[fineSpin*coarseColor],
						  const Rotator &V, int parity, int x_cb) {
    for (int s=0; s<out.Nspin(); s++) 
      for (int i=0; i<out.Ncolor(); i++) out(parity, x_cb, s, i) = 0.0;
    
    for (int i=0; i<fineColor; i++) {
      for (int s=0; s<fineSpin; s++) {
	for (int j=0; j<coarseColor; j++) { 
	  // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	  out(parity, x_cb, s, i) += V(parity, x_cb, s, i, j) * in[s*coarseColor + j];
	}
      }
    }

  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void Prolongate(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<arg.out.Volume()/2; x_cb++) {
	complex<Float> tmp[fineSpin*coarseColor];
	prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.Volume());
	rotateFineColor<Float,fineSpin,fineColor,coarseColor>(arg.out, tmp, arg.V, parity, x_cb);
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ProlongateKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity=threadIdx.y; //parity is within the block
    if (x_cb >= arg.out.Volume()/2) return;

    complex<Float> tmp[fineSpin*coarseColor];
    prolongate<Float,fineSpin,coarseColor>(tmp, arg.in, parity, x_cb, arg.geo_map, arg.spin_map, arg.out.Volume());
    rotateFineColor<Float,fineSpin,fineColor,coarseColor>(arg.out, tmp, arg.V, parity, x_cb);
  }
  
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class ProlongateLaunch : public Tunable {

  protected:
    Arg &arg;
    QudaFieldLocation location;

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.out.Volume()/2; } // fine parity is the block y dimension

  public:
    ProlongateLaunch(Arg &arg, const QudaFieldLocation location) 
      : arg(arg), location(location) { }
    virtual ~ProlongateLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	Prolongate<Float,fineSpin,fineColor,coarseSpin,coarseColor>(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	tp.block.y = 2; // need factor of two for parity with in the block
	ProlongateKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,Arg> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.out.Volume(); 
      // FIXME should use stride here
      aux << "out_stride=" << arg.out.Volume() << ",in_stride=" << arg.in.Volume();
      //return TuneKey(vol.str(), typeid(*this).name(), aux.str());
      return TuneKey("fixme", typeid(*this).name(), "fixme");
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid = dim3( ((arg.out.Volume()/2)+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( ((arg.out.Volume()/2)+param.block.x-1) / param.block.x, 1, 1);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseColor, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  const int *fine_to_coarse, const int *spin_map) {
    if (in.Nspin() != 2) errorQuda("coarseSpin != 2 not supported");
    const int coarseSpin = 2;

    typedef FieldOrderCB<Float,fineSpin,fineColor,1,order> fineSpinor;
    typedef FieldOrderCB<Float,coarseSpin,coarseColor,1,order> coarseSpinor;
    typedef FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order> packedSpinor;
    typedef ProlongateArg<fineSpinor,coarseSpinor,packedSpinor,fineSpin> Arg;

    fineSpinor   Out(const_cast<ColorSpinorField&>(out));
    coarseSpinor In(const_cast<ColorSpinorField&>(in));
    packedSpinor V(const_cast<ColorSpinorField&>(v));

    Arg arg(Out, In, V, fine_to_coarse,spin_map);
    ProlongateLaunch<Float, fineSpin, fineColor, coarseSpin, coarseColor, Arg> prolongator(arg, Location(out, in, v));
    prolongator.apply(0);

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }


  template <typename Float, int fineSpin, int fineColor, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int nVec, const int *fine_to_coarse, const int *spin_map) {

    if (nVec == 2) {
      Prolongate<Float,fineSpin,fineColor,2,order>(out, in, v, fine_to_coarse, spin_map);
    } else if (nVec == 24) {
      Prolongate<Float,fineSpin,fineColor,24,order>(out, in, v, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported nVec %d", nVec);
    }
  }

  template <typename Float, int fineSpin, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map) {

    if (out.Ncolor() == 3) {
      Prolongate<Float,fineSpin,3,order>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else if (out.Ncolor() == 2) {
      Prolongate<Float,fineSpin,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else if (out.Ncolor() == 24) {
      Prolongate<Float,fineSpin,24,order>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported nColor %d", out.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map) {

    if (out.Nspin() == 4) {
      Prolongate<Float,4,order>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else if (out.Nspin() == 2) {
      Prolongate<Float,2,order>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported nSpin %d", out.Nspin());
    }
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map) {

    if (out.FieldOrder() != in.FieldOrder() || out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)", 
		out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      Prolongate<Float,QUDA_FLOAT2_FIELD_ORDER>
	(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      Prolongate<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
	(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported field type %d", out.FieldOrder());
    }
  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  int Nvec, const int *fine_to_coarse, const int *spin_map) {
    if (out.Precision() != in.Precision() || v.Precision() != in.Precision()) 
      errorQuda("Precision mismatch out=%d in=%d v=%d", out.Precision(), in.Precision(), v.Precision());

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      Prolongate<double>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      Prolongate<float>(out, in, v, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

} // end namespace quda
