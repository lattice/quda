#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;
  
  /** 
      Kernel argument struct
  */
  template <typename Out, typename In>
  struct ProlongateArg {
    Out &out;
    const In &in;
    const int *geo_map;
    int spin_map[4];
    ProlongateArg(Out &out, const In &in, const int *geo_map, const int *spin_map) : 
      out(out), in(in), geo_map(geo_map)  {
      //for (int x=0; x<out.Volume(); x++) printfQuda("geo_map[%d] = %d\n",x,this->geo_map[x]);
      for (int s=0; s<4; s++) this->spin_map[s] = spin_map[s];
    }
    ProlongateArg(const ProlongateArg<Out,In> &arg) : out(arg.out), in(arg.in), geo_map(arg.geo_map) {
      //for (int x=0; x<out.Volume(); x++) printfQuda("geo_map[%d] = %d\n",x,this->geo_map[x]);

//      for (int x=0; x<arg.out.Volume(); x++) this->geo_map[x] = arg.geo_map[x];
      for (int s=0; s<4; s++) this->spin_map[s] = arg.spin_map[s];      
    }
  };

  // Applies the grid prolongation operator (coarse to fine)
  template <typename Arg>
  void prolongate(Arg arg) {

    for (int x=0; x<arg.out.Volume(); x++) {
      for (int s=0; s<arg.out.Nspin(); s++) {
	for (int c=0; c<arg.out.Ncolor(); c++) {
	//printfQuda("x=%d, s=%d, c=%d, arg.geo_map[x] = %d, arg.spin_map[s] = %d\n",x,s,c,0,arg.spin_map[s]);
	  arg.out(x, s, c) = arg.in(arg.geo_map[x], arg.spin_map[s], c);
	}
      }
    }

  }

  // Applies the grid prolongation operator (coarse to fine)
  template <typename Arg>
  __global__ void prolongateKernel(Arg arg) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    for (int s=0; s<arg.out.Nspin(); s++) {
      for (int c=0; c<arg.out.Ncolor(); c++) {
	arg.out(x, s, c) = arg.in(arg.geo_map[x], arg.spin_map[s], c);
      }
    }

  }
  
  /*
    Rotates from the coarse-color basis into the fine-color basis.  This
    is the second step of applying the prolongator.
  */
  template <class FineColor, class CoarseColor, class Rotator>
  void rotateFineColor(FineColor &out, const CoarseColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<in.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) { 
	    // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	    out(x, s, i) += V(x, s, i, j) * in(x, s, j);
	  }
	}
      }
      
    }

  }

  /*
    Rotates from the coarse-color basis into the fine-color basis.  This
    is the second step of applying the prolongator.
  */
  template <class FineColor, class CoarseColor, class Rotator>
  __global__ void rotateFineColorKernel(FineColor out, const CoarseColor in, const Rotator V) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;

    for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;
    
    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) { 
	  // V is a ColorMatrixField with internal dimensions Ns * Nc * Nvec
	  out(x, s, i) += V(x, s, i, j) * in(x, s, j);
	}
      }
    }
    
  }

  template <typename Arg>
  class ProlongateLaunch : public Tunable {

  protected:
    const Arg &arg;

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      if (advance) param.grid = dim3( (arg.out.Volume()+param.block.x-1) / param.block.x, 1, 1);
      return advance;
    }

  public:
    ProlongateLaunch(const Arg &arg) : arg(arg) { }
    virtual ~ProlongateLaunch() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_VERBOSE);
      prolongateKernel<Arg> <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
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

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		  ColorSpinorField &tmp, int Nvec, const int *geo_map, const int *spin_map) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
	printfQuda("out.Precision() = QUDA_DOUBLE_PRECISION\n");
      FieldOrder<double> *outOrder = createOrder<double>(out);
      FieldOrder<double> *inOrder = createOrder<double>(in);
      FieldOrder<double> *vOrder = createOrder<double>(v, Nvec);
      FieldOrder<double> *tmpOrder = createOrder<double>(tmp);
      ProlongateArg<FieldOrder<double>, FieldOrder<double> > 
	arg(*outOrder, *inOrder, geo_map, spin_map);
      if (Location(out, in, v) == QUDA_CPU_FIELD_LOCATION) {
	prolongate(arg);
	rotateFineColor(*outOrder, *tmpOrder, *vOrder);
      } else {
	ProlongateLaunch<ProlongateArg<FieldOrder<double>, FieldOrder<double> > > prolongator(arg);
	prolongator.apply(0);
	errorQuda("Need rotation");
      }
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
	//printfQuda("out.Precision() = %d\n", out.Precision());
      FieldOrder<float> *outOrder = createOrder<float>(out);
      FieldOrder<float> *inOrder = createOrder<float>(in);
      FieldOrder<float> *vOrder = createOrder<float>(v, Nvec);
      FieldOrder<float> *tmpOrder = createOrder<float>(tmp);
      ProlongateArg<FieldOrder<float>, FieldOrder<float> > 
	arg(*tmpOrder, *inOrder, geo_map, spin_map);
      if (Location(out, in, v) == QUDA_CPU_FIELD_LOCATION) {
	//printfQuda("tmpOrder = %p, inOrder = %p\n", (void *)tmpOrder, (void *)inOrder);
	//printfQuda("Location(out,in,v) == QUDA_CPU_FIELD_LOCATION\n");
	prolongate(arg);
	//printfQuda("Now rotate fine color\n");
	rotateFineColor(*outOrder, *tmpOrder, *vOrder);
	//printfQuda("End location\n");
      } else {
	//printfQuda("Location(out,in,v) == %d\n",Location(out,in,v));
	ProlongateLaunch<ProlongateArg<FieldOrder<float>, FieldOrder<float> > > prolongator(arg);
	prolongator.apply(0);
	errorQuda("Need rotation");
      }
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }

  }

} // end namespace quda
