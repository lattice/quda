/*
  Spinor reordering and copying routines.  These are implemented to
  un on both CPU and GPU.  Here we are templating on the following:
  - input precision
  - output precision
  - number of colors
  - number of spins
  - field ordering
*/

#include <utility> // for std::swap
#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/copy_color_spinor_mg.cuh>
#include <multigrid.h>

namespace quda {

  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder>
  class CopySpinor : TunableKernel1D {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    FloatOut *Out;
    FloatIn *In;

    bool advanceSharedBytes(TuneParam &) const { return false; } // Don't tune shared mem
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    CopySpinor(ColorSpinorField &out, const ColorSpinorField &in, QudaFieldLocation location, FloatOut* Out, FloatIn* In) :
      TunableKernel1D(in, location),
      out(out),
      in(in),
      Out(Out),
      In(In)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      constexpr bool enable_host = true;
      launch<CopySpinor_, enable_host>(tp, stream, CopyArg<Ns, Nc, OutOrder, InOrder>(out, in, Out, In));
    }

    long long flops() const { return 0; }
    long long bytes() const { return in.Bytes() + out.Bytes(); }
  };

  /** Decide on the output order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename I>
  void genericCopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, QudaFieldLocation location,
                              FloatOut *Out, FloatIn *In)
  {
    if (out.isNative()) {
      using O = FieldOrderCB<typename mapper<FloatOut>::type, Ns, Nc, 1, colorspinor::getNative<FloatOut>(Ns), FloatOut>;
      CopySpinor<FloatOut, FloatIn, Ns, Nc, O, I>(out, in, location, Out, In);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using O = FieldOrderCB<typename mapper<FloatOut>::type, Ns, Nc, 1, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,FloatOut>;
      CopySpinor<FloatOut, FloatIn, Ns, Nc, O, I>(out, in, location, Out, In);
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", out.FieldOrder(), Ns, Nc);
    }
  }

  /** Decide on the input order*/
  template <typename FloatOut, typename FloatIn, int Ns, int Nc>
  void genericCopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, QudaFieldLocation location,
                              FloatOut *Out, FloatIn *In)
  {
    if (in.isNative()) {
      using I = FieldOrderCB<typename mapper<FloatIn>::type, Ns, Nc, 1, colorspinor::getNative<FloatIn>(Ns), FloatIn>;
      genericCopyColorSpinor<FloatOut, FloatIn, Ns, Nc, I>(out, in, location, Out, In);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using I = FieldOrderCB<typename mapper<FloatIn>::type, Ns, Nc, 1, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,FloatIn>;
      genericCopyColorSpinor<FloatOut, FloatIn, Ns, Nc, I>(out, in, location, Out, In);
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", in.FieldOrder(), Ns, Nc);
    }
  }

  using copy_pack_t = std::tuple<ColorSpinorField &, const ColorSpinorField &, QudaFieldLocation, void *, const void *>;

  template <int Ns, int Nc, typename dstFloat, typename srcFloat>
  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, const copy_pack_t &pack)
  {
    auto &location = std::get<2>(pack);
    dstFloat *Dst = static_cast<dstFloat*>(std::get<3>(pack));
    srcFloat *Src = const_cast<srcFloat*>(static_cast<const srcFloat*>(std::get<4>(pack)));

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

    if (dst.SiteSubset() == QUDA_FULL_SITE_SUBSET) { // full field
      if (src.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER ||
	  dst.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {
	errorQuda("QDPJIT field ordering not supported for full site fields");
      }

      // set for the source subset ordering
      srcFloat *srcEven = Src ? Src : (srcFloat*)src.V();
      srcFloat *srcOdd = (srcFloat*)((char*)srcEven + src.Bytes()/2);
      if (src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) {
	std::swap<srcFloat*>(srcEven, srcOdd);
      }

      // set for the destination subset ordering
      dstFloat *dstEven = Dst ? Dst : (dstFloat*)dst.V();
      dstFloat *dstOdd = (dstFloat*)((char*)dstEven + dst.Bytes()/2);
      if (dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER) {
	std::swap<dstFloat*>(dstEven, dstOdd);
      }

      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>(dst, src, location, dstEven, srcEven);
      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>(dst, src, location,  dstOdd,  srcOdd);
    } else { // parity field
      genericCopyColorSpinor<dstFloat, srcFloat, Ns, Nc>(dst, src, location, Dst, Src);
    }
  }

  template <int Nc, typename dst_t, typename src_t>
  void CopyGenericColorSpinor(const copy_pack_t &pack)
  {
    auto &dst = std::get<0>(pack);
    auto &src = std::get<1>(pack);
    if (dst.Nspin() != src.Nspin()) errorQuda("source and destination spins must match");

    if (dst.Nspin() == 4) {
#if defined(NSPIN4)
      copyGenericColorSpinor<4, Nc, dst_t, src_t>(dst, src, pack);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 2) {
#if defined(NSPIN2)
      copyGenericColorSpinor<2, Nc, dst_t, src_t>(dst, src, pack);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else if (dst.Nspin() == 1) {
#if defined(NSPIN1)
      copyGenericColorSpinor<1, Nc, dst_t, src_t>(dst, src, pack);
#else
      errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());
#endif
    } else {
      errorQuda("Nspin=%d unsupported", dst.Nspin());
    }

  }

  template <int...> struct IntList {
  };

  template <int fineColor, typename dst_t, typename src_t, typename param_t, int coarseColor, int... N>
  bool instantiateColor(const ColorSpinorField &field, const param_t &param, IntList<coarseColor, N...>)
  {
    if (field.Ncolor() == fineColor * coarseColor) {
      CopyGenericColorSpinor<fineColor * coarseColor, dst_t, src_t>(param);
      return true;
    } else {
      if constexpr (sizeof...(N) > 0) {
        return instantiateColor<fineColor, dst_t, src_t>(field, param, IntList<N...>());
      }
    }
    return false;
  }

  template <typename dst_t, typename src_t, typename param_t, int fineColor, int... N>
  bool instantiateColor(const ColorSpinorField &field, const param_t &param, IntList<fineColor, N...>)
  {
    // 1 ensures we generate templates for just the fineColor with no multiplication by coarseColor
    // clang-format off
    IntList<1, @QUDA_MULTIGRID_NVEC_LIST@> coarseColors;
    // clang-format on
    bool success = instantiateColor<fineColor, dst_t, src_t>(field, param, coarseColors);

    if (!success) {
      if constexpr (sizeof...(N) > 0) success = instantiateColor<dst_t, src_t>(field, param, IntList<N...>());
    }
    return success;
  }

  template <typename dst_t, typename src_t, typename param_t>
  void instantiateColor(const ColorSpinorField &field, const param_t &param)
  {
    // clang-format off
    IntList<@QUDA_MULTIGRID_NC_NVEC_LIST@> fineColors;
    // clang-format on
    if (!instantiateColor<dst_t, src_t>(field, param, fineColors)) {
      errorQuda("Nc = %d has not been instantiated", field.Ncolor());
    }
  }

} // namespace quda
