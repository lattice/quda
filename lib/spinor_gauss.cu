/*
  Spinor reordering and copying routines.  These are implemented to
  un on both CPU and GPU.  Here we are templating on the following:
  - input precision
  - output precision
  - number of colors
  - number of spins
  - field ordering
*/

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <algorithm> // for std::swap
#include <random_quda.h>

namespace quda {

  using namespace colorspinor;


  template<typename InOrder, typename FloatIn>
  __device__ __host__ void genGauss(InOrder& inOrder, cuRNGState& localState, int x, int s, int c){
      FloatIn phi = 2.0*M_PI*Random<FloatIn>(localState);
      FloatIn radius = Random<FloatIn>(localState);
      radius = sqrt(-1.0 * log(radius));
      inOrder(0, x, s, c) = complex<FloatIn>(radius*cos(phi),radius*sin(phi));
  }

  /** CPU function to reorder spinor fields.  */
  template <typename FloatIn, int Ns, int Nc, typename InOrder>
    void gaussSpinor(InOrder &inOrder, int volume, RNG rngstate) {
    for (int x=0; x<volume; x++) {
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	    //inOrder(0, x, s, c) = complex<FloatIn>(0,0);
	    cuRNGState localState = rngstate.State()[x];
	    genGauss<InOrder, FloatIn>(inOrder, localState, x, s, c);
	    rngstate.State()[x] = localState;
	}
      }
    }
  }

  /** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
  template <typename FloatIn, int Ns, int Nc, typename InOrder>
    __global__ void gaussSpinorKernel(InOrder inOrder, int volume, RNG rngstate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= volume) return;

    cuRNGState localState = rngstate.State()[x];
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	  genGauss<InOrder, FloatIn>(inOrder, localState, x, s, c);
      }
    }
    rngstate.State()[x] = localState;
  }

  template <typename FloatIn, int Ns, int Nc, typename InOrder>
    class GaussSpinor : Tunable {
    InOrder &in;
    const ColorSpinorField &meta; // this reference is for meta data only
    QudaFieldLocation location;
    RNG & rngstate;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    //bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return meta.VolumeCB(); }

  public:
    GaussSpinor(InOrder &in, const ColorSpinorField &meta, QudaFieldLocation location, RNG &rngstate)
      : in(in), meta(meta), location(location), rngstate(rngstate){ }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	gaussSpinor<FloatIn, Ns, Nc>(in, meta.VolumeCB(), rngstate);
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	gaussSpinorKernel<FloatIn, Ns, Nc, InOrder>
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>
	  (in, meta.VolumeCB(), rngstate);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }

    long long flops() const { return 0; }
    long long bytes() const { return in.Bytes(); }

    void preTune(){
	rngstate.backup();
    }
    void postTune(){
	rngstate.restore();
    }
  };

  template <typename FloatIn, int Ns, int Nc, typename InOrder>
    void gaussSpinor(InOrder &inOrder, const ColorSpinorField &meta,
				QudaFieldLocation location, RNG &rngstate) {
    GaussSpinor<FloatIn, Ns, Nc, InOrder> gauss(inOrder, meta, location, rngstate);
    gauss.apply(0);
  }

  /** Decide on the input order*/
  template <typename FloatIn, int Ns, int Nc>
    void gaussSpinor(ColorSpinorField &in, QudaFieldLocation location, RNG &rngstate) {

    if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      typedef typename colorspinor::FieldOrderCB<FloatIn, Ns, Nc, 1, QUDA_FLOAT2_FIELD_ORDER> ColorSpinor;
      ColorSpinor inOrder(in);
      gaussSpinor<FloatIn,Ns,Nc>(inOrder, in, location, rngstate);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      typedef typename colorspinor::FieldOrderCB<FloatIn, Ns, Nc, 1, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> ColorSpinor;
      ColorSpinor inOrder(in);
      gaussSpinor<FloatIn,Ns,Nc>(inOrder, in, location, rngstate);
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", in.FieldOrder(), Ns, Nc);
    }

  }

  void spinorGauss(ColorSpinorField &src, RNG& randstates){

      QudaFieldLocation location;
      if (typeid(src) == typeid(cudaColorSpinorField)){
	  location = QUDA_CUDA_FIELD_LOCATION;
      }else{
	  location = QUDA_CPU_FIELD_LOCATION;
      }
      if (src.Ncolor() != 3 ){
	  errorQuda("%s is not implemented for Ncolor!=3");
      }
      if (src.Nspin() == 4 ){
	  if (src.Precision() == QUDA_SINGLE_PRECISION){
	      gaussSpinor<float, 4, 3>(src, location, randstates);
	  } else if(src.Precision() == QUDA_DOUBLE_PRECISION) {
	      gaussSpinor<double, 4, 3>(src, location, randstates);
	  }
      }else if (src.Nspin() == 1 ){
	  if (src.Precision() == QUDA_SINGLE_PRECISION){
	      gaussSpinor<float, 1, 3>(src, location, randstates);
	  } else if(src.Precision() == QUDA_DOUBLE_PRECISION) {
	      gaussSpinor<double, 1, 3>(src, location, randstates);
	  }
      }else{
	  errorQuda("spinorGauss not implemented for Nspin != 1 or Nspin !=4");
      }

  }

  void spinorGauss(ColorSpinorField &src, int seed)
  {
      RNG* randstates = new RNG(src.VolumeCB(), seed, src.X());
      randstates->Init();
      spinorGauss(src, *randstates);
      randstates->Release();
      delete randstates;
  }
} // namespace quda
