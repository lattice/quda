#include <tuple>
#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/copy_color_spinor.cuh>
#include <instantiate.h>

namespace quda
{

  template <int nSpin, int nColor, typename Out, typename In, typename param_t> class CopyColorSpinor : TunableKernel2D
  {
    using FloatOut = std::remove_pointer_t<typename std::tuple_element<3, param_t>::type>;
    using FloatIn = std::remove_const_t<std::remove_pointer_t<typename std::tuple_element<4, param_t>::type>>;
    template <template <int, int> class Basis>
    using Arg = CopyColorSpinorArg<FloatOut, FloatIn, nSpin, nColor, Out, In, Basis>;
    FloatOut *Out_;
    const FloatIn *In_;
    ColorSpinorField &out;
    const ColorSpinorField &in;

    bool advanceSharedBytes(TuneParam &) const { return false; } // Don't tune shared mem
    unsigned int minThreads() const { return in.VolumeCB(); }

    std::string get_basis_str(QudaGammaBasis basis)
    {
      switch (basis) {
      case QUDA_DEGRAND_ROSSI_GAMMA_BASIS: return "degrand_rossi";
      case QUDA_UKQCD_GAMMA_BASIS: return "ukqcd";
      case QUDA_CHIRAL_GAMMA_BASIS: return "chiral";
      case QUDA_DIRAC_PAULI_GAMMA_BASIS: return "dirac_pauli";
      default: errorQuda("Unknown gamma basis %d", basis);
      }
      return "unknown";
    }

  public:
    CopyColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, const param_t &param) :
      TunableKernel2D(in, in.SiteSubset(), std::get<2>(param)),
      Out_(std::get<3>(param)),
      In_(std::get<4>(param)),
      out(out),
      in(in)
    {
      strcat(aux, out.AuxString().c_str());
      if (out.GammaBasis() != in.GammaBasis()) {
        strcat(aux, "to_");
        strcat(aux, get_basis_str(out.GammaBasis()).c_str());
        strcat(aux, ",from_");
        strcat(aux, get_basis_str(in.GammaBasis()).c_str());
      }

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      constexpr bool enable_host = true;
      if constexpr (nSpin == 4) {
        if (out.GammaBasis() == in.GammaBasis()) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<PreserveBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<NonRelBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS && in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<RelBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<ChiralToNonRelBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS && in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<NonRelToChiralBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_DIRAC_PAULI_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<DegrandRossiToDiracPaulBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS && in.GammaBasis() == QUDA_DIRAC_PAULI_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<DiracPaulToDegrandRossiBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<ChiralToFromDegrandRossiBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<ChiralToFromDegrandRossiBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_DIRAC_PAULI_GAMMA_BASIS && in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<UKQCDToFromDiracPauliBasis>(out, in, Out_, In_));
        } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DIRAC_PAULI_GAMMA_BASIS) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<UKQCDToFromDiracPauliBasis>(out, in, Out_, In_));
        } else {
          errorQuda("Unexpected basis change from %d to %d", in.GammaBasis(), out.GammaBasis());
        }
      } else {
        if (out.GammaBasis() == in.GammaBasis()) {
          launch<CopyColorSpinor_, enable_host>(tp, stream, Arg<PreserveBasis>(out, in, Out_, In_));
        } else {
          errorQuda("Unexpected basis change from %d to %d", in.GammaBasis(), out.GammaBasis());
        }
      }
    }

    long long flops() const { return 0; }
    long long bytes() const { return in.Bytes() + out.Bytes(); }
  };

  /** Decide on the output order*/
  template <int Ns, int Nc, typename I, typename param_t> void genericCopyColorSpinor(const param_t &param)
  {
    auto &out = std::get<0>(param);
    auto &in = std::get<1>(param);
    using FloatOut = std::remove_pointer_t<typename std::tuple_element<3, param_t>::type>;
    if (out.isNative()) {
      using O = typename colorspinor_mapper<FloatOut, Ns, Nc>::type;
      CopyColorSpinor<Ns, Nc, O, I, param_t>(out, in, param);
    } else if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using O = SpaceSpinorColorOrder<FloatOut, Ns, Nc>;
      CopyColorSpinor<Ns, Nc, O, I, param_t>(out, in, param);
    } else if (out.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      using O = SpaceColorSpinorOrder<FloatOut, Ns, Nc>;
      CopyColorSpinor<Ns, Nc, O, I, param_t>(out, in, param);
    } else if (out.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using O = PaddedSpaceSpinorColorOrder<FloatOut, Ns, Nc>;
      if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>())
        CopyColorSpinor<Ns, Nc, O, I, param_t>(out, in, param);
      else
        errorQuda("TIFR interface has not been built");
    } else if (out.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {
      using O = QDPJITDiracOrder<FloatOut, Ns, Nc>;
      if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>())
        CopyColorSpinor<Ns, Nc, O, I, param_t>(out, in, param);
      else
        errorQuda("QDPJIT interface has not been built");
    } else {
      errorQuda("Order %d not defined (Ns = %d, Nc = %d, precision = %d)", out.FieldOrder(), Ns, Nc, out.Precision());
    }
  }

  /** Decide on the input order*/
  template <int Ns, int Nc, typename param_t> void genericCopyColorSpinor(const param_t &param)
  {
    auto &in = std::get<1>(param);
    using FloatIn = std::remove_const_t<std::remove_pointer_t<typename std::tuple_element<4, param_t>::type>>;
    if (in.isNative()) {
      using I = typename colorspinor_mapper<FloatIn, Ns, Nc>::type;
      genericCopyColorSpinor<Ns, Nc, I>(param);
    } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using I = SpaceSpinorColorOrder<FloatIn, Ns, Nc>;
      genericCopyColorSpinor<Ns, Nc, I>(param);
    } else if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      using I = SpaceColorSpinorOrder<FloatIn, Ns, Nc>;
      genericCopyColorSpinor<Ns, Nc, I>(param);
    } else if (in.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using ColorSpinor = PaddedSpaceSpinorColorOrder<FloatIn, Ns, Nc>;
      if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>())
        genericCopyColorSpinor<Ns, Nc, ColorSpinor>(param);
      else
        errorQuda("TIFR interface has not been built");
    } else if (in.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {
      using ColorSpinor = QDPJITDiracOrder<FloatIn, Ns, Nc>;
      if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>())
        genericCopyColorSpinor<Ns, Nc, ColorSpinor>(param);
      else
        errorQuda("QDPJIT interface has not been built");
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d, precision = %d)", in.FieldOrder(), Ns, Nc, in.Precision());
    }
  }

  template <int Ns, int Nc, typename param_t> void copyGenericColorSpinor(const param_t &param)
  {
    auto &dst = std::get<0>(param);
    auto &src = std::get<1>(param);

    if (dst.Ndim() != src.Ndim()) errorQuda("Number of dimensions %d %d don't match", dst.Ndim(), src.Ndim());

    if (dst.Volume() != src.Volume()) errorQuda("Volumes %lu %lu don't match", dst.Volume(), src.Volume());

    if (!(dst.SiteOrder() == src.SiteOrder()
          || (dst.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER && src.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER)
          || (dst.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER && src.SiteOrder() == QUDA_EVEN_ODD_SITE_ORDER))) {
      errorQuda("Subset orders %d %d don't match", dst.SiteOrder(), src.SiteOrder());
    }

    if (dst.SiteSubset() != src.SiteSubset())
      errorQuda("Subset types do not match %d %d", dst.SiteSubset(), src.SiteSubset());

    // We currently only support parity-ordered fields; even-odd or odd-even
    if (dst.SiteOrder() == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
    }

    genericCopyColorSpinor<Ns, Nc>(param);
  }

  template <typename dst_t, typename src_t>
  using param_t = std::tuple<ColorSpinorField &, const ColorSpinorField &, QudaFieldLocation, dst_t *, const src_t *>;
  using copy_pack_t = std::tuple<ColorSpinorField &, const ColorSpinorField &, QudaFieldLocation, void *, const void *>;

  template <int Nc, typename dst_t, typename src_t> void CopyGenericColorSpinor(const copy_pack_t &pack)
  {
    auto &dst = std::get<0>(pack);
    auto &src = std::get<1>(pack);
    param_t<dst_t, src_t> param(std::get<0>(pack), std::get<1>(pack), std::get<2>(pack),
                                static_cast<dst_t *>(std::get<3>(pack)), static_cast<const src_t *>(std::get<4>(pack)));

    if (dst.Nspin() != src.Nspin()) errorQuda("source and destination spins must match");
    if (!is_enabled_spin(dst.Nspin())) errorQuda("%s has not been built for Nspin=%d fields", __func__, src.Nspin());

    switch (dst.Nspin()) {
    case 1:
      if constexpr (is_enabled_spin(1)) copyGenericColorSpinor<1, Nc>(param);
      break;
    case 2:
      if constexpr (is_enabled_spin(2)) copyGenericColorSpinor<2, Nc>(param);
      break;
    case 4:
      if constexpr (is_enabled_spin(4)) copyGenericColorSpinor<4, Nc>(param);
      break;
    default: errorQuda("Nspin=%d unsupported", dst.Nspin());
    }
  }

} // namespace quda
