/*
  Spinor reordering and copying routines.  These are implemented to
  un on both CPU and GPU.  Here we are templating on the following:
  - input precision
  - output precision
  - number of colors
  - number of spins
  - field ordering
*/

#include <color_spinor_field_order.h>

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0/sqrt(2.0))
#define kU (1.0/sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif

namespace quda {

  /** Straight copy with no basis change */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    class PreserveBasis {
  public:
    __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  for (int z=0; z<2; z++) {
	    out[(s*Nc+c)*2+z] = in[(s*Nc+c)*2+z];
	  }
	}
      }
    }
  };

  /** Transform from relativistic into non-relavisitic basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct NonRelBasis {
      __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
	int s1[4] = {1, 2, 3, 0};
	int s2[4] = {3, 0, 1, 2};
	FloatOut K1[4] = {kP, -kP, -kP, -kP};
	FloatOut K2[4] = {kP, -kP, kP, kP};
	for (int s=0; s<Ns; s++) {
	  for (int c=0; c<Nc; c++) {
	    for (int z=0; z<2; z++) {
	      out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	    }
	  }
	}
      }
    };

  /** Transform from non-relativistic into relavisitic basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct RelBasis {
      __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
	int s1[4] = {1, 2, 3, 0};
	int s2[4] = {3, 0, 1, 2};
	FloatOut K1[4] = {-kU, kU,  kU,  kU};
	FloatOut K2[4] = {-kU, kU, -kU, -kU};
	for (int s=0; s<Ns; s++) {
	  for (int c=0; c<Nc; c++) {
	    for (int z=0; z<2; z++) {
	      out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	    }
	  }
	}
      }
    };

  /** CPU function to reorder spinor fields.  */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    void packSpinor(OutOrder &outOrder, const InOrder &inOrder, Basis basis, int volume) {  
    for (int x=0; x<volume; x++) {
      FloatIn in[Ns*Nc*2];
      FloatOut out[Ns*Nc*2];
      inOrder.load(in, x);
      basis(out, in);
      outOrder.save(out, x);
    }
  }

  /** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    __global__ void packSpinorKernel(OutOrder outOrder, const InOrder inOrder, Basis basis, int volume) {  
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    FloatIn in[Ns*Nc*2];
    FloatOut out[Ns*Nc*2];
    inOrder.load(in, x);

    if (x >= volume) return;
    basis(out, in);
    outOrder.save(out, x);
  }

  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    class PackSpinor : Tunable {
    const InOrder &in;
    OutOrder &out;
    Basis &basis;
    int volume;

  private:
    int sharedBytesPerThread() const { 
      size_t regSize = sizeof(FloatOut) > sizeof(FloatIn) ? sizeof(FloatOut) : sizeof(FloatIn);
      return Ns*Nc*2*regSize;
    }

    // the minimum shared memory per block is (block+1) because we pad to avoid bank conflicts
    int sharedBytesPerBlock(const TuneParam &param) const { return (param.block.x+1)*sharedBytesPerThread(); }
    bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      if (advance) param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = sharedBytesPerThread() * (param.block.x+1); // FIXME: use sharedBytesPerBlock
      return advance;
    }

  public:
  PackSpinor(OutOrder &out, const InOrder &in, Basis &basis, int volume) 
    : out(out), in(in), basis(basis), volume(volume) { ; }
    virtual ~PackSpinor() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_VERBOSE);
      packSpinorKernel<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis> 
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>> 
	(out, in, basis, volume);
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << in.volume; 
      aux << "out_stride=" << out.stride << ",in_stride=" << in.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    virtual void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
    }

    long long flops() const { return 0; } 
    long long bytes() const { return in.Bytes() + out.Bytes(); } 
  };


  /** Decide whether we are changing basis or not */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder>
    void packParitySpinor(OutOrder &outOrder, const InOrder &inOrder, int Vh, 
			  QudaGammaBasis dstBasis, QudaGammaBasis srcBasis, QudaFieldLocation location) {
    if (dstBasis==srcBasis) {
      PreserveBasis<FloatOut, FloatIn, Ns, Nc> basis;
      if (location == QUDA_CPU_FIELD_LOCATION) {
	packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);
      } else {
	PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, PreserveBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
	pack.apply(0);
      }
    } else if (dstBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      NonRelBasis<FloatOut, FloatIn, Ns, Nc> basis;
      if (location == QUDA_CPU_FIELD_LOCATION) {
	packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);
      } else {
	PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, NonRelBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
	pack.apply(0);
      }
    } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      RelBasis<FloatOut, FloatIn, Ns, Nc> basis;
      if (location == QUDA_CPU_FIELD_LOCATION) {
	packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);    
      } else {
	PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, RelBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
	pack.apply(0);
      } 
    } else {
      errorQuda("Basis change not supported");
    }
  }

  /** Decide on the output order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename InOrder>
    void packParitySpinor(InOrder &inOrder, FloatOut *Out, ColorSpinorField &out, QudaGammaBasis inBasis, QudaFieldLocation location) {
    if (out.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      FloatNOrder<FloatOut, Ns, Nc, 4> 
	outOrder(Out, out.VolumeCB(), out.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      FloatNOrder<FloatOut, Ns, Nc, 2> 
	outOrder(Out, out.VolumeCB(), out.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_FLOAT_FIELD_ORDER) { 
      FloatNOrder<FloatOut, Ns, Nc, 1> 
	outOrder(Out, out.VolumeCB(), out.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatOut, Ns, Nc>    
	outOrder(Out, out.VolumeCB(), out.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatOut, Ns, Nc>    
	outOrder(Out, out.VolumeCB(), out.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else {
      errorQuda("Order not defined");
    }

  }

  /** Decide on the input order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    void packParitySpinor(FloatOut *Out, FloatIn *In, ColorSpinorField &out, const ColorSpinorField &in, QudaFieldLocation location) {
    if (in.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      FloatNOrder<FloatIn, Ns, Nc, 4> 
	inOrder(In, in.VolumeCB(), in.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, Out, out, in.GammaBasis(), location);    
    } else if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      FloatNOrder<FloatIn, Ns, Nc, 2> 
	inOrder(In, in.VolumeCB(), in.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, Out, out, in.GammaBasis(), location);
    } else if (in.FieldOrder() == QUDA_FLOAT_FIELD_ORDER) { 
      FloatNOrder<FloatIn, Ns, Nc, 1> 
	inOrder(In, in.VolumeCB(), in.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, Out, out, in.GammaBasis(), location);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatIn, Ns, Nc>    
	inOrder(In, in.VolumeCB(), in.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, Out, out, in.GammaBasis(), location);
    } else if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatIn, Ns, Nc>    
	inOrder(In, in.VolumeCB(), in.Stride());
      packParitySpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, Out, out, in.GammaBasis(), location);
    } else {
      errorQuda("Order not defined");
    }

  }


  template <int Ns, typename dstFloat, typename srcFloat>
    void packSpinor(dstFloat *Dst, srcFloat *Src, ColorSpinorField &dst, const ColorSpinorField &src, QudaFieldLocation location) {

    if (dst.Ndim() != src.Ndim())
      errorQuda("Number of dimensions %d %d don't match", dst.Ndim(), src.Ndim());

    if (dst.Volume() != src.Volume())
      errorQuda("Volumes %d %d don't match", dst.Volume(), src.Volume());

    if (!( dst.SiteOrder() == src.SiteOrder() ||
	   (dst.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER && 
	    src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) ||
	   (dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER && 
	    src.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER) ) ) {
      errorQuda("Subset orders %d %d don't match", dst.SiteOrder(), src.SiteOrder());
    }

    if (dst.SiteSubset() != src.SiteSubset())
      errorQuda("Subset types do not match %d %d", dst.SiteSubset(), src.SiteSubset());

    if (dst.Ncolor() != 3 || src.Ncolor() != 3) errorQuda("Nc != 3 not yet supported");

    const int Nc = 3;
 
    int dstLength = dst.Bytes() / dst.Precision(); // cannot use total_length since ALIGNMENT_ADJUST 
    int srcLength = src.Bytes() / src.Precision(); // changes position of odd field

    // We currently only support parity-ordered fields; even-odd or odd-even
    if (dst.SiteOrder() == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
    }

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET) { // full field
      // check what src parity ordering is
      unsigned int evenOff, oddOff;
      if (dst.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER) {
	evenOff = 0;
	oddOff = srcLength/2;
      } else {
	oddOff = 0;
	evenOff = srcLength/2;
      }    
      packParitySpinor<dstFloat, srcFloat, Ns, Nc>(Dst, Src+evenOff, dst, src, location);
      packParitySpinor<dstFloat, srcFloat, Ns, Nc>(Dst+dstLength/2, Src+oddOff, dst, src, location);
    } else { // parity field
      packParitySpinor<dstFloat, srcFloat, Ns, Nc>(Dst, Src, dst, src, location);
    }

  }

} // namespace quda
