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

  using namespace colorspinor;

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
	RegTypeOut K1[4] = {static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(-kP), 
			    static_cast<RegTypeOut>(-kP), static_cast<RegTypeOut>(-kP)};
	RegTypeOut K2[4] = {static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(-kP), 
			    static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(kP)};
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
	RegTypeOut K1[4] = {static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(kU),
			    static_cast<RegTypeOut>(kU),  static_cast<RegTypeOut>(kU)};
	RegTypeOut K2[4] = {static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(kU),
			    static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(-kU)};
	for (int s=0; s<Ns; s++) {
	  for (int c=0; c<Nc; c++) {
	    for (int z=0; z<2; z++) {
	      out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	    }
	  }
	}
      }
    };

  /** Transform from chiral into non-relavisitic basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct ChiralToNonRelBasis {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
    __device__ __host__ inline void operator()(RegTypeOut out[Ns*Nc*2], const RegTypeIn in[Ns*Nc*2]) {
	int s1[4] = {0, 1, 0, 1};
	int s2[4] = {2, 3, 2, 3};
	RegTypeOut K1[4] = {static_cast<RegTypeOut>(-kP), static_cast<RegTypeOut>(-kP),
			    static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(kP)};
	RegTypeOut K2[4] = {static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(kP),
			    static_cast<RegTypeOut>(kP), static_cast<RegTypeOut>(kP)};
	for (int s=0; s<Ns; s++) {
	  for (int c=0; c<Nc; c++) {
	    for (int z=0; z<2; z++) {
	      out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	    }
	  }
	}
      }
  };

  /** Transform from non-relativistic into chiral basis */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    struct NonRelToChiralBasis {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
    __device__ __host__ inline void operator()(RegTypeOut out[Ns*Nc*2], const RegTypeIn in[Ns*Nc*2]) {
	int s1[4] = {0, 1, 0, 1};
	int s2[4] = {2, 3, 2, 3};
	RegTypeOut K1[4] = {static_cast<RegTypeOut>(-kU), static_cast<RegTypeOut>(-kU),
			    static_cast<RegTypeOut>(kU), static_cast<RegTypeOut>(kU)};
	RegTypeOut K2[4] = {static_cast<RegTypeOut>(kU),static_cast<RegTypeOut>(kU),
			    static_cast<RegTypeOut>(kU), static_cast<RegTypeOut>(kU)};
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
    if (x >= volume) return;

    RegTypeIn in[Ns*Nc*2];
    RegTypeOut out[Ns*Nc*2];
    inOrder.load(in, x);
    basis(out, in);
    outOrder.save(out, x);
  }

  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
    class PackSpinor : Tunable {
    const InOrder &in;
    OutOrder &out;
    Basis &basis;
    const ColorSpinorField &meta; // this reference is for meta data only
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return meta.VolumeCB(); }

  public:
    PackSpinor(OutOrder &out, const InOrder &in, Basis &basis, const ColorSpinorField &meta,
	       QudaFieldLocation location)
      : out(out), in(in), basis(basis), meta(meta), location(location) {
      writeAuxString("out_stride=%d,in_stride=%d", out.stride, in.stride);
    }
    virtual ~PackSpinor() { ; }
  
    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	packSpinor<FloatOut, FloatIn, Ns, Nc>(out, in, basis, meta.VolumeCB());
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	packSpinorKernel<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis>
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>
	  (out, in, basis, meta.VolumeCB());
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; } 
    long long bytes() const { return in.Bytes() + out.Bytes(); } 
  };


  /** Decide whether we are changing basis or not */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder>
    void genericCopyColorSpinor(OutOrder &outOrder, const InOrder &inOrder, 
				QudaGammaBasis dstBasis, QudaGammaBasis srcBasis, 
				const ColorSpinorField &out, QudaFieldLocation location) {
    if (dstBasis==srcBasis) {
      PreserveBasis<FloatOut, FloatIn, Ns, Nc> basis;
      PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, PreserveBasis<FloatOut, FloatIn, Ns, Nc> >
	pack(outOrder, inOrder, basis, out, location);
      pack.apply(0);
    } else if (dstBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      if (Nc != 3) errorQuda("Can only change basis with Ncolor = 4, not Ncolor = %d", Nc);
      NonRelBasis<FloatOut, FloatIn, 4, 3> basis;
      PackSpinor<FloatOut, FloatIn, 4, 3, OutOrder, InOrder, NonRelBasis<FloatOut, FloatIn, 4, 3> >
	pack(outOrder, inOrder, basis, out, location);
      pack.apply(0);
    } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      if (Nc != 3) errorQuda("Can only change basis with Ncolor = 4, not Ncolor = %d", Nc);
      RelBasis<FloatOut, FloatIn, 4, 3> basis;
      PackSpinor<FloatOut, FloatIn, 4, 3, OutOrder, InOrder, RelBasis<FloatOut, FloatIn, 4, 3> >
	pack(outOrder, inOrder, basis, out, location);
      pack.apply(0);
    } else if (dstBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_CHIRAL_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      if (Nc != 3) errorQuda("Can only change basis with Ncolor = 4, not Ncolor = %d", Nc);
      ChiralToNonRelBasis<FloatOut, FloatIn, 4, 3> basis;
      PackSpinor<FloatOut, FloatIn, 4, 3, OutOrder, InOrder, ChiralToNonRelBasis<FloatOut, FloatIn, 4, 3> >
	pack(outOrder, inOrder, basis, out, location);
      pack.apply(0);
    } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_CHIRAL_GAMMA_BASIS) {
      if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
      if (Nc != 3) errorQuda("Can only change basis with Ncolor = 4, not Ncolor = %d", Nc);
      NonRelToChiralBasis<FloatOut, FloatIn, 4, 3> basis;
      PackSpinor<FloatOut, FloatIn, 4, 3, OutOrder, InOrder, NonRelToChiralBasis<FloatOut, FloatIn, 4, 3> >
	pack(outOrder, inOrder, basis, out, location);
      pack.apply(0);
    } else {
      errorQuda("Basis change from %d to %d not supported", srcBasis, dstBasis);
    }
  }

  /** Decide on the output order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename InOrder>
    void genericCopyColorSpinor(InOrder &inOrder, ColorSpinorField &out, 
				QudaGammaBasis inBasis, QudaFieldLocation location, 
				FloatOut *Out, float *outNorm) {

    if (out.isNative()) {
      typedef typename colorspinor_mapper<FloatOut,Ns,Nc>::type ColorSpinor;
      ColorSpinor outOrder(out, Out, outNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, out, location);
    } else if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Ns == 4) {
      // this is needed for single-precision mg for changing basis in the transfer
      typedef typename colorspinor::FloatNOrder<float, 4, Nc, 2> ColorSpinor;
      ColorSpinor outOrder(out, (float*)Out, outNorm);
      genericCopyColorSpinor<float,FloatIn,4,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, out, location);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, out, location);
    } else if (out.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, out, location);
    } else if (out.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      QDPJITDiracOrder<FloatOut, Ns, Nc> outOrder(out, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out.GammaBasis(), inBasis, out, location);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", out.FieldOrder(), Ns, Nc);
    }

  }

  /** Decide on the input order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
    void genericCopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, 
				QudaFieldLocation location, FloatOut *Out, FloatIn *In, 
				float *outNorm, float *inNorm) {

    if (in.isNative()) {
      typedef typename colorspinor_mapper<FloatIn,Ns,Nc>::type ColorSpinor;
      ColorSpinor inOrder(in, In, inNorm);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Ns == 4) {
      // this is needed for single-precision mg for changing basis in the transfer
      typedef typename colorspinor::FloatNOrder<float, 4, Nc, 2> ColorSpinor;
      ColorSpinor inOrder(in, (float*)In, inNorm);
      genericCopyColorSpinor<FloatOut,float,4,Nc>(inOrder, out, in.GammaBasis(), location, Out, outNorm);
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
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", in.FieldOrder(), Ns, Nc);
    }

  }


  template <int Ns, int Nc, typename dstFloat, typename srcFloat>
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

  template <int Nc, typename dstFloat, typename srcFloat>
  void CopyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, dstFloat *Dst, srcFloat *Src, 
			      float *dstNorm=0, float *srcNorm=0) {

    if (dst.Nspin() != src.Nspin())
      errorQuda("source and destination spins must match");

    if (dst.Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
      copyGenericColorSpinor<4,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 2) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
      copyGenericColorSpinor<2,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
      copyGenericColorSpinor<1,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else {
      errorQuda("Nspin=%d unsupported", dst.Nspin());
    }
    
  }

} // namespace quda
