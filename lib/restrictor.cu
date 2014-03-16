#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <cub/cub.cuh>
#include <typeinfo>

namespace quda {

  using namespace quda::colorspinor;

  /** 
      Kernel argument struct
  */
  template <typename Out, typename In, typename Rotator, typename Tmp>
  struct RestrictArg {
    Out &out;
    Tmp &tmp;
    const In &in;
    const Rotator &V;
    const int *geo_map;  // need to make a device copy of this
    int spin_map[4];
    RestrictArg(Out &out, const In &in, const Rotator &V, 
		Tmp &tmp, const int *geo_map, const int *spin_map) : 
      out(out), tmp(tmp), in(in), V(V), geo_map(geo_map)  {
      for (int s=0; s<4; s++) this->spin_map[s] = spin_map[s];
    }

    RestrictArg(const RestrictArg<Out,In,Rotator,Tmp> &arg) : 
      out(arg.out), in(arg.in), V(arg.V), tmp(arg.tmp), geo_map(arg.geo_map) {
      for (int s=0; s<4; s++) this->spin_map[s] = arg.spin_map[s];      
    }
  };

  /**
     Applies the grid restriction operator (fine to coarse)
  */
  __device__ __host__ void restrict(CoarseSpinor &out, const FineSpinor &in, int x, const int* fine_to_coarse, geo_map, const int* spin_map) {
    for (int s=0; s<in.Nspin(); s++) {
      for (int c=0; c<in.Ncolor(); c++) {
	out(geo_map[x], spin_map[s], c) += in(x, s, c);
      }
    }
  }

  /**
    Rotates from the fine-color basis into the coarse-color basis.
  */
  template <class CoarseColor, class FineColor, class Rotator>
  __device__ __host__ void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V, int x) {
    for (int s=0; s<out.Nspin(); s++) 
      for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;
    
    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) {
	  out(x, s, i) += quda::conj(V(x, s, j, i)) * in(x, s, j);
	}
      }
    }
  }

  template <typename Arg>
  void Restrict(Arg &arg) {
    // Zero all elements first, since this is a reduction operation
    for (int x=0; x<arg.out.Volume(); x++)
      for (int s=0; s<arg.out.Nspin(); s++) 
	for (int c=0; c<arg.out.Ncolor(); c++)
	  arg.out(x, s, c) = 0.0;

    for (int x=0; x<arg.in.Volume(); x++) {
      rotateCoarseColor(arg.tmp, arg.in, arg.V, x);
      restrict(arg.out, arg.tmp, x, arg.geo_map, arg.spin_map);
    }
  }

  /**
     Here, we ensure that each thread block maps exactly to a geometric block

     This is wrong, we need an inverse geo_map, since each thread
     block corresponds to one geometric block, we need to map from
     coarse point to fine points so we can assign a fine grid value to
     each thread
   */
  template <typename Float, typename Arg, int block_size>
  __global__ void RestrictKernel(Arg arg) {
    int x_coarse = blockIdx.x;

    // this is the fine geometric index
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    rotateCoarseColor(arg.tmp, arg.in, arg.V, x);
 
    //this is going to be severely sub-optimal until color and spin are templated
    typedef quda::complex<Float> Complex;
    typedef cub::BlockReduce<Complex, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int length = arg.out.Nspin() * arg.out.Ncolor();
    Complex *reduced = (Complex*)malloc(length * sizeof(Complex));
    for (int i=0; i<length; i++) reduced[i] = 0.0;

    for (int s=0; s<arg.tmp.Nspin(); s++) {
      for (int c=0; c<arg.tmp.Ncolor(); c++) {
	reduced[arg.spin_map[s]*arg.tmp.Ncolor()+c] += BlockReduce(temp_storage).Sum( arg.tmp(x, s, c) );
      }
    }

    if (threadIdx.x==0) {
      for (int s=0; s<arg.out.Nspin(); s++) {
	for (int c=0; c<arg.out.Ncolor(); c++) {
	  arg.out(x_coarse, s, c) = reduced[s*arg.out.Ncolor()+c];
	}
      }
    }
    free(reduced);
  }

  template <typename Float, typename Arg>
  class RestrictLaunch : public Tunable {

  protected:
    const Arg &arg;
    QudaFieldLocation location;
    const int block_size;

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.in.Volume(); }

  public:
    RestrictLaunch(const Arg &arg, const QudaFieldLocation location) 
      : arg(arg), location(location), block_size(arg.in.Volume()/arg.out.Volume()) { }
    virtual ~RestrictLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	Restrict(arg);
      } else {
	// no tuning here currently as we're assuming CTA size = geo block size
	// later can think about multiple points per thread
	TuneParam tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());
	if (block_size == 16) {
	  RestrictKernel<Float,Arg,16><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	} else if (block_size == 256) {
	  RestrictKernel<Float,Arg,256><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	} else {
	  errorQuda("Block size not instantiated");
	}
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
      param.block = dim3(arg.in.Volume()/arg.out.Volume(), 1, 1);
      param.grid = dim3( (arg.in.Volume()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(block_size, 1, 1);
      param.grid = dim3( (arg.in.Volume()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;
    }

  };

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map) {

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      typedef FieldOrder<double> Field;
      Field *outOrder = createOrder<double>(out);
      Field *inOrder = createOrder<double>(in);
      Field *vOrder = createOrder<double>(v, Nvec);
      Field *tmpOrder = createOrder<double>(tmp);
      RestrictArg<Field,Field,Field,Field> 
	arg(*outOrder,*inOrder,*vOrder,*tmpOrder,fine_to_coarse,spin_map);
      RestrictLaunch<double, RestrictArg<Field, Field, Field, Field> > 
      (arg, Location(out, in, v));
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
      RestrictArg<Field,Field,Field,Field> 
	arg(*outOrder,*inOrder,*vOrder,*tmpOrder,fine_to_coarse,spin_map);
      RestrictLaunch<float, RestrictArg<Field, Field, Field, Field> > 
	RestrictLaunch(arg, Location(out, in, v));
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION)  
      checkCudaError();
  }

} // namespace quda
