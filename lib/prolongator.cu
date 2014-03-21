#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Out, typename In, typename Rotator, typename Tmp>
  struct ProlongateArg {
    Out out;
    Tmp tmp;
    const In in;
    const Rotator V;
    const int *geo_map;  // need to make a device copy of this
    int spin_map[4];
    ProlongateArg(Out &out, const In &in, const Rotator &V, 
		  Tmp &tmp, const int *geo_map, const int *spin_map) : 
      out(out), tmp(tmp), in(in), V(V), geo_map(geo_map)  {
      for (int s=0; s<4; s++) this->spin_map[s] = spin_map[s];
    }

    ProlongateArg(const ProlongateArg<Out,In,Rotator,Tmp> &arg) : 
      out(arg.out), in(arg.in), V(arg.V), tmp(arg.tmp), geo_map(arg.geo_map) {
      for (int s=0; s<4; s++) this->spin_map[s] = arg.spin_map[s];      
    }
  };

  /**
     Applies the grid prolongation operator (coarse to fine)
  */
  template <class Fine, class Coarse>
  __device__ __host__ void prolongate(Fine &out, const Coarse &in, int x, 
				      const int *geo_map, const int *spin_map) {
    for (int s=0; s<out.Nspin(); s++) {
      for (int c=0; c<in.Ncolor(); c++) {
	out(x, s, c) = in(geo_map[x], spin_map[s], c);
      }
    }
  }

  /**
     Rotates from the coarse-color basis into the fine-color basis.  This
     is the second step of applying the prolongator.
  */
  template <class FineColor, class CoarseColor, class Rotator>
  __device__ __host__ void rotateFineColor(FineColor &out, const CoarseColor &in, const Rotator &V, int x) {
    for (int s=0; s<out.Nspin(); s++) 
      for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;
    
    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) { 
	  // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	  out(x, s, i) += V(x, s, i, j) * in(x, s, j);
	}
      }
    }
  }

  template <typename Arg>
  void Prolongate(Arg &arg) {
    for (int x=0; x<arg.out.Volume(); x++) {
      prolongate(arg.tmp, arg.in, x, arg.geo_map, arg.spin_map);
      rotateFineColor(arg.out, arg.tmp, arg.V, x);
    }
  }

  template <typename Float, typename Arg>
  __global__ void ProlongateKernel(Arg arg) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= arg.out.Volume()) return;
    prolongate(arg.tmp, arg.in, x, arg.geo_map, arg.spin_map);
    rotateFineColor(arg.out, arg.tmp, arg.V, x);    
  }
  
  template <typename Float, typename Arg>
  class ProlongateLaunch : public Tunable {

  protected:
    Arg &arg;
    QudaFieldLocation location;

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.out.Volume(); }

  public:
    ProlongateLaunch(Arg &arg, const QudaFieldLocation location) 
      : arg(arg), location(location) { }
    virtual ~ProlongateLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	Prolongate(arg);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	ProlongateKernel<Float,Arg> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.out.Volume(); 
      // FIXME should use stride here
      aux << "out_stride=" << arg.out.Volume() << ",in_stride=" << arg.in.Volume();
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid = dim3( (arg.out.Volume()+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (arg.out.Volume()+param.block.x-1) / param.block.x, 1, 1);
    }

  };

  template <typename Float, QudaFieldOrder order>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, const int *spin_map) {
    typedef typename accessor<Float,order>::type F;

    F Out(const_cast<ColorSpinorField&>(out));
    F In(const_cast<ColorSpinorField&>(in));
    F V(const_cast<ColorSpinorField&>(v),Nvec);
    F Tmp(const_cast<ColorSpinorField&>(tmp));

    ProlongateArg<F,F,F,F> arg(Out, In, V, Tmp, fine_to_coarse,spin_map);
    ProlongateLaunch<double, ProlongateArg<F, F, F, F> > prolongator(arg, Location(out, in, v));
    prolongator.apply(0);

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, const int *spin_map) {

    if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      Prolongate<Float,QUDA_FLOAT2_FIELD_ORDER>
	(out, in, v, tmp, Nvec, fine_to_coarse, spin_map);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      Prolongate<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
	(out, in, v, tmp, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported field type %d", out.FieldOrder());
    }
  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, const int *spin_map) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      Prolongate<double>(out, in, v, tmp, Nvec, fine_to_coarse, spin_map);
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      Prolongate<float>(out, in, v, tmp, Nvec, fine_to_coarse, spin_map);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  /*  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, int Nvec, const int *geo_map, const int *spin_map) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      typedef FieldOrder<double> Field;
      Field *outOrder = createOrder<double>(out);
      Field *inOrder = createOrder<double>(in);
      Field *vOrder = createOrder<double>(v, Nvec);
      Field *tmpOrder = createOrder<double>(tmp);
      ProlongateArg<Field, Field, Field, Field> 
	arg(*outOrder, *inOrder, *vOrder, *tmpOrder, geo_map, spin_map);
      ProlongateLaunch<ProlongateArg<Field, Field, Field, Field> > 
	prolongator(arg, Location(out, in, v));
      prolongator.apply(0);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
      typedef FieldOrder<float> Field;
      Field *outOrder = createOrder<float>(out);
      Field *inOrder = createOrder<float>(in);
      Field *vOrder = createOrder<float>(v, Nvec);
      Field *tmpOrder = createOrder<float>(tmp);
      ProlongateArg<Field, Field, Field, Field> 
	arg(*outOrder, *inOrder, *vOrder, *tmpOrder, geo_map, spin_map);
      ProlongateLaunch<ProlongateArg<Field, Field, Field, Field> > 
	prolongator(arg, Location(out, in, v));
      prolongator.apply(0);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION)  checkCudaError();
  }
  */
} // end namespace quda
