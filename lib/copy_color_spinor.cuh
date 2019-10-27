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
#include <utility> // for std::swap

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

  template <typename Out, typename In>
  struct CopyColorSpinorArg {
    Out out;
    const In in;
    const int volumeCB;
    const int nParity;
    const int outParity;
    const int inParity;
    CopyColorSpinorArg(const Out &out, const In &in, const ColorSpinorField &out_, const ColorSpinorField &in_)
      : out(out), in(in), volumeCB(in_.VolumeCB()), nParity(in_.SiteSubset()),
	outParity(out_.SiteOrder()==QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0),
	inParity(in_.SiteOrder()==QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0) { }
  };

  /** Straight copy with no basis change */
  template <int Ns, int Nc>
  struct PreserveBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      for (int s=0; s<Ns; s++) for (int c=0; c<Nc; c++) out[s*Nc+c] = in[s*Nc+c];
    }
  };

  /** Transform from relativistic into non-relavisitic basis */
  template <int Ns, int Nc>
  struct NonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP)};
      FloatOut K2[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from non-relativistic into relavisitic basis */
  template <int Ns, int Nc>
  struct RelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU),  static_cast<FloatOut>(kU)};
      FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from chiral into non-relavisitic basis */
  template <int Ns, int Nc>
  struct ChiralToNonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4] = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      FloatOut K2[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** Transform from non-relativistic into chiral basis */
  template <int Ns, int Nc>
  struct NonRelToChiralBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns*Nc], const complex<FloatIn> in[Ns*Nc]) const {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      FloatOut K2[4] = {static_cast<FloatOut>(kU),static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      for (int s=0; s<Ns; s++) {
	for (int c=0; c<Nc; c++) {
	  out[s*Nc+c] = K1[s]*static_cast<complex<FloatOut> >(in[s1[s]*Nc+c]) + K2[s]*static_cast<complex<FloatOut> >(in[s2[s]*Nc+c]);
	}
      }
    }
  };

  /** CPU function to reorder spinor fields.  */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename Arg, typename Basis>
  void copyColorSpinor(Arg &arg, const Basis &basis) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity = 0; parity<arg.nParity; parity++) {
      for (int x=0; x<arg.volumeCB; x++) {
	ColorSpinor<RegTypeIn, Nc, Ns> in = arg.in(x, (parity+arg.inParity)&1);
	ColorSpinor<RegTypeOut, Nc, Ns> out;
	basis(out.data, in.data);
	arg.out(x, (parity+arg.outParity)&1) = out;
      }
    }
  }

  /** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename Arg, typename Basis>
  __global__ void copyColorSpinorKernel(Arg arg, Basis basis) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= arg.volumeCB) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;

    ColorSpinor<RegTypeIn, Nc, Ns> in = arg.in(x, (parity+arg.inParity)&1);
    ColorSpinor<RegTypeOut, Nc, Ns> out;
    basis(out.data, in.data);
    arg.out(x, (parity+arg.outParity)&1) = out;
  }

  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename Arg>
    class CopyColorSpinor : TunableVectorY {
    Arg &arg;
    const ColorSpinorField &meta;
    const QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return meta.VolumeCB(); }

  public:
    CopyColorSpinor(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in,
		    QudaFieldLocation location)
      : TunableVectorY(arg.nParity), arg(arg), meta(in), location(location) {
      if (out.GammaBasis()!=in.GammaBasis()) errorQuda("Cannot change gamma basis for nSpin=%d\n", Ns);
      writeAuxString("out_stride=%d,in_stride=%d", arg.out.stride, arg.in.stride);
    }
    virtual ~CopyColorSpinor() { ; }
  
    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, PreserveBasis<Ns,Nc>());
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, PreserveBasis<Ns,Nc>());
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; } 
    long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
  };

  template <typename FloatOut, typename FloatIn, int Nc, typename Arg>
  class CopyColorSpinor<FloatOut,FloatIn,4,Nc,Arg> : TunableVectorY {
    static constexpr int Ns = 4;
    Arg &arg;
    const ColorSpinorField &out;
    const ColorSpinorField &in;
    const QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    CopyColorSpinor(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in,
		    QudaFieldLocation location)
      : TunableVectorY(arg.nParity), arg(arg), out(out), in(in), location(location) {

      if (out.GammaBasis()==in.GammaBasis()) {
	writeAuxString("out_stride=%d,in_stride=%d,PreserveBasis", arg.out.stride, arg.in.stride);
      } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	writeAuxString("out_stride=%d,in_stride=%d,NonRelBasis", arg.out.stride, arg.in.stride);
      } else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	writeAuxString("out_stride=%d,in_stride=%d,RelBasis", arg.out.stride, arg.in.stride);
      } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	writeAuxString("out_stride=%d,in_stride=%d,ChiralToNonRelBasis", arg.out.stride, arg.in.stride);
      } else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	writeAuxString("out_stride=%d,in_stride=%d,NonRelToChiralBasis", arg.out.stride, arg.in.stride);
      } else {
	errorQuda("Basis change from %d to %d not supported", in.GammaBasis(), out.GammaBasis());
      }
    }
    virtual ~CopyColorSpinor() { ; }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	if (out.GammaBasis()==in.GammaBasis()) {
	  copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, PreserveBasis<Ns,Nc>());
	} else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	  copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, NonRelBasis<Ns,Nc>());
	} else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	  copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, RelBasis<Ns,Nc>());
	} else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	  copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, ChiralToNonRelBasis<Ns,Nc>());
	} else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	  copyColorSpinor<FloatOut, FloatIn, Ns, Nc>(arg, NonRelToChiralBasis<Ns,Nc>());
	}
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	if (out.GammaBasis()==in.GammaBasis()) {
	  copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	    <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, PreserveBasis<Ns,Nc>());
	} else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	  copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	    <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, NonRelBasis<Ns,Nc>());
	} else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
	  copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	    <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, RelBasis<Ns,Nc>());
	} else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	  copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	    <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, ChiralToNonRelBasis<Ns,Nc>());
	} else if (in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
	  copyColorSpinorKernel<FloatOut, FloatIn, Ns, Nc>
	    <<<tp.grid, tp.block, tp.shared_bytes, stream>>> (arg, NonRelToChiralBasis<Ns,Nc>());
	}
      }
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; }
    long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
  };


  /** Decide whether we are changing basis or not */
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename Out, typename In>
  void genericCopyColorSpinor(Out &outOrder, const In &inOrder, const ColorSpinorField &out,
			      const ColorSpinorField &in, QudaFieldLocation location) {

    typedef CopyColorSpinorArg<Out,In> Arg;
    Arg arg(outOrder, inOrder, out, in);
    CopyColorSpinor<FloatOut, FloatIn, Ns, Nc, Arg> copy(arg, out, in, location);
    copy.apply(0);

  }

  /** Decide on the output order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename InOrder>
    void genericCopyColorSpinor(InOrder &inOrder, ColorSpinorField &out, 
				const ColorSpinorField &in, QudaFieldLocation location,
				FloatOut *Out, float *outNorm) {
    const bool override = true;
    if (out.isNative()) {
      typedef typename colorspinor_mapper<FloatOut,Ns,Nc>::type ColorSpinor;
      ColorSpinor outOrder(out, 1, Out, outNorm, nullptr, override);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out, in, location);
    } else if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Ns == 4) {
      // this is needed for single-precision mg for changing basis in the transfer
      typedef typename colorspinor::FloatNOrder<FloatOut, 4, Nc, 2> ColorSpinor;
      ColorSpinor outOrder(out, 1, Out, outNorm, nullptr, override);
      genericCopyColorSpinor<FloatOut,FloatIn,4,Nc>
	(outOrder, inOrder, out, in, location);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatOut, Ns, Nc> outOrder(out, 1, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out, in, location);
    } else if (out.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatOut, Ns, Nc> outOrder(out, 1, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out, in, location);
    } else if (out.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      PaddedSpaceSpinorColorOrder<FloatOut, Ns, Nc> outOrder(out, 1, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out, in, location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (out.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      QDPJITDiracOrder<FloatOut, Ns, Nc> outOrder(out, 1, Out);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>
	(outOrder, inOrder, out, in, location);
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
    const bool override = true;
    if (in.isNative()) {
      typedef typename colorspinor_mapper<FloatIn,Ns,Nc>::type ColorSpinor;
      ColorSpinor inOrder(in, 1, In, inNorm, nullptr, override);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in, location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Ns == 4) {
      // this is needed for single-precision mg for changing basis in the transfer
      typedef typename colorspinor::FloatNOrder<FloatIn, 4, Nc, 2> ColorSpinor;
      ColorSpinor inOrder(in, 1, In, inNorm, nullptr, override);
      genericCopyColorSpinor<FloatOut,FloatIn,4,Nc>(inOrder, out, in, location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<FloatIn, Ns, Nc> inOrder(in, 1, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in, location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<FloatIn, Ns, Nc> inOrder(in, 1, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in, location, Out, outNorm);
    } else if (in.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      PaddedSpaceSpinorColorOrder<FloatIn, Ns, Nc> inOrder(in, 1, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in, location, Out, outNorm);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (in.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      QDPJITDiracOrder<FloatIn, Ns, Nc> inOrder(in, 1, In);
      genericCopyColorSpinor<FloatOut,FloatIn,Ns,Nc>(inOrder, out, in, location, Out, outNorm);
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

    if (dst.Volume() != src.Volume()) errorQuda("Volumes %lu %lu don't match", dst.Volume(), src.Volume());

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

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET && (src.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER || dst.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER)) {
      errorQuda("QDPJIT field ordering not supported for full site fields");
    }

    genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);

  }

  template <int Nc, typename dstFloat, typename srcFloat>
  void CopyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, dstFloat *Dst, srcFloat *Src, 
			      float *dstNorm=0, float *srcNorm=0) {

    if (dst.Nspin() != src.Nspin())
      errorQuda("source and destination spins must match");

    if (dst.Nspin() == 4) {
#if defined(NSPIN4)
      copyGenericColorSpinor<4,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 2) {
#if defined(NSPIN2)
      copyGenericColorSpinor<2,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 1) {
#if defined(NSPIN1)
      copyGenericColorSpinor<1,Nc>(dst, src, location, Dst, Src, dstNorm, srcNorm);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else {
      errorQuda("Nspin=%d unsupported", dst.Nspin());
    }
    
  }

} // namespace quda
