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
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
  public:
    __device__ __host__ inline void operator()(RegTypeOut out[Ns*Nc*2], const RegTypeIn in[Ns*Nc*2]) {
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
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
      __device__ __host__ inline void operator()(RegTypeOut out[Ns*Nc*2], const RegTypeIn in[Ns*Nc*2]) {
	int s1[4] = {1, 2, 3, 0};
	int s2[4] = {3, 0, 1, 2};
	RegTypeOut K1[4] = {kP, -kP, -kP, -kP};
	RegTypeOut K2[4] = {kP, -kP, kP, kP};
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
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
      __device__ __host__ inline void operator()(RegTypeOut out[Ns*Nc*2], const RegTypeIn in[Ns*Nc*2]) {
	int s1[4] = {1, 2, 3, 0};
	int s2[4] = {3, 0, 1, 2};
	RegTypeOut K1[4] = {-kU, kU,  kU,  kU};
	RegTypeOut K2[4] = {-kU, kU, -kU, -kU};
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
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
    for (int x=0; x<volume; x++) {
      RegTypeIn in[Ns*Nc*2];
      RegTypeOut out[Ns*Nc*2];
      inOrder.load(in, x);
      basis(out, in);
      outOrder.save(out, x);
    }
  }

  /** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    __global__ void packSpinorKernel(OutOrder outOrder, const InOrder inOrder, Basis basis, int volume) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    RegTypeIn in[Ns*Nc*2];
    RegTypeOut out[Ns*Nc*2];
    inOrder.load(in, x);
    // if (x >= volume) return; all load and save routines are index safe (needed for shared variants)
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
    unsigned int sharedBytesPerThread() const { 
      size_t regSize = sizeof(FloatOut) > sizeof(FloatIn) ? sizeof(FloatOut) : sizeof(FloatIn);
      return Ns*Nc*2*regSize;
    }

    // the minimum shared memory per block is (block+1) because we pad to avoid bank conflicts
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return (param.block.x+1)*sharedBytesPerThread(); }
    bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return volume; }
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      param.shared_bytes = sharedBytesPerThread() * (param.block.x+1); // FIXME: use sharedBytesPerBlock
      return advance;
    }

  public:
  PackSpinor(OutOrder &out, const InOrder &in, Basis &basis, int volume) 
    : out(out), in(in), basis(basis), volume(volume) { ; }
    virtual ~PackSpinor() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      packSpinorKernel<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis> 
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>> 
	(out, in, basis, volume);
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << in.volumeCB; 
      aux << "out_stride=" << out.stride << ",in_stride=" << in.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { return in.Bytes() + out.Bytes(); } 
  };


  /** Decide whether we are changing basis or not */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder>
    void genericCopyColorSpinor(OutOrder &outOrder, const InOrder &inOrder, int Vh, 
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
    void genericCopyColorSpinor(InOrder &inOrder, ColorSpinorField &out, 
				QudaGammaBasis inBasis, QudaFieldLocation location, 
				FloatOut *Out, float *outNorm) {
    if (out.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      FloatNOrder<FloatOut, Ns, Nc, 4> outOrder(out, Out, outNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      FloatNOrder<FloatOut, Ns, Nc, 2> outOrder(out, Out, outNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
    } else if (out.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      QDPJITDiracOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.VolumeCB(), out.GammaBasis(), inBasis, location);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else {
      errorQuda("Order not defined");
    }

  }

  /** Decide on the input order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    void genericCopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, 
				QudaFieldLocation location, FloatOut *Out, FloatIn *In, 
				float *outNorm, float *inNorm) {
    if (in.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      FloatNOrder<FloatIn, Ns, Nc, 4> inOrder(in, In, inNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      FloatNOrder<FloatIn, Ns, Nc, 2> inOrder(in, In, inNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatIn, Ns, Nc> inOrder(in, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatIn, Ns, Nc> inOrder(in, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      QDPJITDiracOrder<FloatIn, Ns, Nc> inOrder(in, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else {
      errorQuda("Order not defined");
    }

  }


  template <int Ns, typename dstFloat, typename srcFloat>
    void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, dstFloat *Dst, srcFloat *Src, 
				float *dstNorm, float *srcNorm) {

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
 
    // We currently only support parity-ordered fields; even-odd or odd-even
    if (dst.SiteOrder() == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
    }

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET) { // full field
      if (src.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER ||
	  dst.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {
	errorQuda("QDPJIT field ordering not supported for full site fields");
      }

      // set for the source subset ordering
      srcFloat *srcEven = Src ? Src : (srcFloat*)src.V();
      srcFloat *srcOdd = (srcFloat*)((char*)srcEven + src.Bytes()/2);
      float *srcNormEven = srcNorm ? srcNorm : (float*)src.Norm();
      float *srcNormOdd = (float*)((char*)srcNormEven + src.NormBytes()/2);
      if (src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) {
	std::swap<srcFloat*>(srcEven, srcOdd);
	std::swap<float*>(srcNormEven, srcNormOdd);
      }

      // set for the destination subset ordering
      dstFloat *dstEven = Dst ? Dst : (dstFloat*)dst.V();
      dstFloat *dstOdd = (dstFloat*)((char*)dstEven + dst.Bytes()/2);
      float *dstNormEven = dstNorm ? dstNorm : (float*)dst.Norm();
      float *dstNormOdd = (float*)((char*)dstNormEven + dst.NormBytes()/2);
      if (dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) {
	std::swap<dstFloat*>(dstEven, dstOdd);
	std::swap<float*>(dstNormEven, dstNormOdd);
      }

      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
	(dst, src, location, dstEven, srcEven, dstNormEven, srcNormEven);
      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
	(dst, src, location,  dstOdd,  srcOdd,  dstNormOdd,  srcNormOdd);
    } else { // parity field
      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>
	(dst, src, location, Dst, Src, dstNorm, srcNorm);
    }

  }

  template <typename dstFloat, typename srcFloat>
  void CopyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, dstFloat *Dst, srcFloat *Src, 
			      float *dstNorm=0, float *srcNorm=0) {

    if (dst.Nspin() != src.Nspin())
      errorQuda("source and destination spins must match");

    if (dst.Nspin() == 4) {
      copyGenericColorSpinor<4>(dst, src, location, Dst, Src, dstNorm, srcNorm);
    } else if (dst.Nspin() == 1) {
      copyGenericColorSpinor<1>(dst, src, location, Dst, Src, dstNorm, srcNorm);    
    } else {
      errorQuda("Nspin=%d unsupported", dst.Nspin());
    }
    
  }
  
  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, void *Dst, void *Src, 
			      void *dstNorm, void *srcNorm) {
    
    if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {	
	CopyGenericColorSpinor(dst, src, location, (double*)Dst, (double*)Src);
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (double*)Dst, (float*)Src);
      } else if (src.Precision() == QUDA_HALF_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (double*)Dst, (short*)Src, 0, (float*)srcNorm);
      } else {
	errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (float*)Dst, (double*)Src);
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (float*)Dst, (float*)Src);
      } else if (src.Precision() == QUDA_HALF_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (float*)Dst, (short*)Src, 0, (float*)srcNorm);
      } else {
	errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else if (dst.Precision() == QUDA_HALF_PRECISION) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (short*)Dst, (double*)Src, (float*)dstNorm, 0);
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (short*)Dst, (float*)Src, (float*)dstNorm, 0);
      } else if (src.Precision() == QUDA_HALF_PRECISION) {
	CopyGenericColorSpinor(dst, src, location, (short*)Dst, (short*)Src, (float*)dstNorm, (float*)srcNorm);
      } else {
	errorQuda("Unsupported Precision %d", src.Precision());
      }
    } else {
      errorQuda("Unsupported Precision %d", dst.Precision());
    }
  }  

} // namespace quda
