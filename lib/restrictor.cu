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
    Out out;
    Tmp tmp;
    const In in;
    const Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    int spin_map[4];
    RestrictArg(Out &out, const In &in, const Rotator &V, 
		Tmp &tmp, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map) : 
      out(out), tmp(tmp), in(in), V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine)
    { for (int s=0; s<4; s++) this->spin_map[s] = spin_map[s]; }

    RestrictArg(const RestrictArg<Out,In,Rotator,Tmp> &arg) : 
      out(arg.out), in(arg.in), V(arg.V), tmp(arg.tmp), 
      fine_to_coarse(arg.fine_to_coarse), coarse_to_fine(arg.coarse_to_fine) 
    { for (int s=0; s<4; s++) this->spin_map[s] = arg.spin_map[s]; }
  };

  /**
     Applies the grid restriction operator (fine to coarse)
  */
  template <class Coarse, class Fine>
  __device__ __host__ void restrict(Coarse &out, const Fine &in, int x, 
				    const int* fine_to_coarse, const int* spin_map) {
    for (int s=0; s<in.Nspin(); s++) {
      for (int c=0; c<in.Ncolor(); c++) {
	out(fine_to_coarse[x], spin_map[s], c) += in(x, s, c);
      }
    }
  }

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <class CoarseColor, class FineColor, class Rotator>
  __device__ __host__ void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V, int x) {
    for (int s=0; s<out.Nspin(); s++) 
      for (int i=0; i<out.Ncolor(); i++) 
	out(x, s, i) = 0.0;
    
    for (int i=0; i<out.Ncolor(); i++) {
      for (int s=0; s<out.Nspin(); s++) {
	for (int j=0; j<in.Ncolor(); j++) {
	  out(x, s, i) += conj(V(x, s, j, i)) * in(x, s, j);
	}
      }
    }
  }

  template <typename Arg>
  void Restrict(Arg arg) {
    // Zero all elements first, since this is a reduction operation
    for (int x=0; x<arg.out.Volume(); x++)
      for (int s=0; s<arg.out.Nspin(); s++) 
	for (int c=0; c<arg.out.Ncolor(); c++)
	  arg.out(x, s, c) = 0.0;

    for (int x=0; x<arg.in.Volume(); x++) {
      rotateCoarseColor(arg.tmp, arg.in, arg.V, x);
      restrict(arg.out, arg.tmp, x, arg.fine_to_coarse, arg.spin_map);
    }
  }

  /**
     Here, we ensure that each thread block maps exactly to a geometric block

     Each thread block corresponds to one geometric block, with number
     of threads equal to the number of fine grid points per aggregate,
     so each thread represents a fine-grid point.  The look up table
     coarse_to_fine is the mapping to the each fine grid point. 
  */
  template <typename Float, typename Arg, int block_size>
  __global__ void RestrictKernel(Arg arg) {
    int x_coarse = blockIdx.x;

    // obtain fine index from this look up table
    int x_fine = arg.coarse_to_fine[blockIdx.x*blockDim.x + threadIdx.x];
    
    rotateCoarseColor(arg.tmp, arg.in, arg.V, x_fine);
 
    //this is going to be severely sub-optimal until color and spin are templated
    typedef cub::BlockReduce<complex<Float>, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int length = arg.out.Nspin() * arg.out.Ncolor();
    complex<Float> *reduced = (complex<Float>*)malloc(length * sizeof(complex<Float>));
    for (int i=0; i<length; i++) reduced[i] = 0.0;

    for (int s=0; s<arg.tmp.Nspin(); s++) {
      for (int c=0; c<arg.tmp.Ncolor(); c++) {
	reduced[arg.spin_map[s]*arg.out.Ncolor()+c] += 
	  BlockReduce(temp_storage).Sum( arg.tmp(x_fine, s, c) );
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
    Arg &arg;
    QudaFieldLocation location;
    const int block_size;

    long long flops() const { return 0; }
    long long bytes() const { return 0; }
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.in.Volume(); }

  public:
    RestrictLaunch(Arg &arg, const QudaFieldLocation location) 
      : arg(arg), location(location), block_size(arg.in.Volume()/arg.out.Volume()) { }
    virtual ~RestrictLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	Restrict(arg);
      } else {
	// no tuning here currently as we're assuming CTA size = geo
	// block size later can think about multiple points per thread
	// (after we have fully templated the parameters).
	TuneParam tp = tuneLaunch(*this, QUDA_TUNE_NO, getVerbosity());

	if (block_size == 16) {
	  RestrictKernel<Float,Arg,16><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	} else if (block_size == 32) {
	  RestrictKernel<Float,Arg,32><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
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

    void initTuneParam(TuneParam &param) const { defaultTuneParam(param); }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(block_size, 1, 1);
      param.grid = dim3( (arg.in.Volume()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;
    }

  };

  template <typename Float, QudaFieldOrder order>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse,
		const int *coarse_to_fine, const int *spin_map) {

    typedef typename accessor<Float,order>::type F;

    F Out(const_cast<ColorSpinorField&>(out));
    F In(const_cast<ColorSpinorField&>(in));
    F V(const_cast<ColorSpinorField&>(v),Nvec);
    F Tmp(const_cast<ColorSpinorField&>(tmp));

    RestrictArg<F,F,F,F> arg(Out, In, V, Tmp, fine_to_coarse,coarse_to_fine,spin_map);
    RestrictLaunch<Float, RestrictArg<F, F, F, F> > restrictor(arg, Location(out, in, v));
    restrictor.apply(0);

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <typename Float>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, 
		const int *coarse_to_fine, const int *spin_map) {

    if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      Restrict<Float,QUDA_FLOAT2_FIELD_ORDER>
	(out, in, v, tmp, Nvec, fine_to_coarse, coarse_to_fine, spin_map);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      Restrict<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
	(out, in, v, tmp, Nvec, fine_to_coarse, coarse_to_fine, spin_map);
    } else {
      errorQuda("Unsupported field type %d", out.FieldOrder());
    }
  }

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, int Nvec, const int *fine_to_coarse, 
		const int *coarse_to_fine, const int *spin_map) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      Restrict<double>(out, in, v, tmp, Nvec, fine_to_coarse, coarse_to_fine, spin_map);
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      Restrict<float>(out, in, v, tmp, Nvec, fine_to_coarse, coarse_to_fine, spin_map);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }

  }

} // namespace quda
