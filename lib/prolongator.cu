#include <color_spinor_field.h>
#include <multigrid.h>
#include <tunable_nd.h>
#include <kernels/prolongator.cuh>

namespace quda {

  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  class ProlongateLaunch : public TunableKernel3D {
    template <QudaFieldOrder order> using Arg = ProlongateArg<Float,vFloat,fineSpin,fineColor,coarseSpin,coarseColor,order>;

    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &V;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;

    unsigned int minThreads() const { return out.VolumeCB(); } // fine parity is the block y dimension

  public:
    ProlongateLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
                     const int *fine_to_coarse, int parity)
      : TunableKernel3D(in, out.SiteSubset(), fineColor/fine_colors_per_thread<fineColor, coarseColor>()), out(out), in(in), V(V),
        fine_to_coarse(fine_to_coarse), parity(parity), location(checkLocation(out, in, V))
    {
      strcat(vol, ",");
      strcat(vol, out.VolString());
      strcat(aux, ",");
      strcat(aux, out.AuxString());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<Prolongator>(tp, stream, Arg<QUDA_FLOAT2_FIELD_ORDER>(out, in, V, fine_to_coarse, parity));
      } else {
        errorQuda("Unsupported field order %d", out.FieldOrder());
      }
    }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * out.SiteSubset()*(long long)out.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = V.Bytes() / (V.SiteSubset() == out.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + out.SiteSubset()*out.VolumeCB()*sizeof(int);
    }

  };

  template <int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  struct enabled : std::false_type { };

#ifdef NSPIN4
  template <> struct enabled<4,  3, 2,  6> : std::true_type { };
  template <> struct enabled<4,  3, 2, 24> : std::true_type { };
  template <> struct enabled<4,  3, 2, 32> : std::true_type { };
  template <> struct enabled<2, 24, 2, 32> : std::true_type { };
  template <> struct enabled<2, 32, 2, 32> : std::true_type { };
  template <> struct enabled<2,  6, 2,  6> : std::true_type { };
#endif
#ifdef NSPIN1
  template <> struct enabled<1,  3, 2, 24> : std::true_type { };
  template <> struct enabled<1,  3, 2, 64> : std::true_type { };
  template <> struct enabled<1,  3, 2, 96> : std::true_type { };
  template <> struct enabled<2, 24, 2, 64> : std::true_type { };
  template <> struct enabled<2, 24, 2, 96> : std::true_type { };
  template <> struct enabled<2, 64, 2, 64> : std::true_type { };
  template <> struct enabled<2, 64, 2, 96> : std::true_type { };
  template <> struct enabled<2, 96, 2, 96> : std::true_type { };
#endif
#if defined(NSPIN1) || defined(NSPIN4)
  template <> struct enabled<2, 24, 2, 24> : std::true_type { };
#endif

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  std::enable_if_t<enabled<fineSpin, fineColor, coarseSpin, coarseColor>::value, void>
  Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
             const int *fine_to_coarse, int parity)
  {
    if (v.Precision() == QUDA_HALF_PRECISION) {
      if constexpr(is_enabled(QUDA_HALF_PRECISION)) {
        ProlongateLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor>
          prolongator(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in.Precision()) {
      ProlongateLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor>
        prolongator(out, in, v, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  std::enable_if_t<!enabled<fineSpin, fineColor, coarseSpin, coarseColor>::value, void>
  Prolongate(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &, const int *, int)
  {
    errorQuda("Not enabled");
  }

  template <typename Float, int fineSpin>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int nVec, const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());
    constexpr int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++)
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (out.Ncolor() == 3) {
      constexpr int fineColor = 3;
      switch (nVec) {
      case 6: Prolongate<Float, fineSpin, fineColor, coarseSpin, 6>(out, in, v, fine_to_coarse, parity); break;
      case 24: Prolongate<Float, fineSpin, fineColor, coarseSpin, 24>(out, in, v, fine_to_coarse, parity); break;
      case 32: Prolongate<Float, fineSpin, fineColor, coarseSpin, 32>(out, in, v, fine_to_coarse, parity); break;
      case 64: Prolongate<Float, fineSpin, fineColor, coarseSpin, 64>(out, in, v, fine_to_coarse, parity); break;
      case 96: Prolongate<Float, fineSpin, fineColor, coarseSpin, 96>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 6) { // for coarsening coarsened Wilson free field.
      constexpr int fineColor = 6;
      switch (nVec) {
      case 6: Prolongate<Float, fineSpin, fineColor, coarseSpin, 6>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 24) {
      constexpr int fineColor = 24;
      switch (nVec) {
      case 24: Prolongate<Float, fineSpin, fineColor, coarseSpin, 24>(out, in, v, fine_to_coarse, parity); break;
      case 32: Prolongate<Float, fineSpin, fineColor, coarseSpin, 32>(out, in, v, fine_to_coarse, parity); break;
      case 64: Prolongate<Float, fineSpin, fineColor, coarseSpin, 64>(out, in, v, fine_to_coarse, parity); break;
      case 96: Prolongate<Float, fineSpin, fineColor, coarseSpin, 96>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 32) {
      constexpr int fineColor = 32;
      switch (nVec) {
      case 32: Prolongate<Float, fineSpin, fineColor, coarseSpin, 32>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 64) {
      constexpr int fineColor = 64;
      switch (nVec) {
      case 64: Prolongate<Float, fineSpin, fineColor, coarseSpin, 64>(out, in, v, fine_to_coarse, parity); break;
      case 96: Prolongate<Float, fineSpin, fineColor, coarseSpin, 96>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 96) {
      constexpr int fineColor = 96;
      switch (nVec) {
      case 96: Prolongate<Float, fineSpin, fineColor, coarseSpin, 96>(out, in, v, fine_to_coarse, parity); break;
      default: errorQuda("Unsupported nVec %d", nVec);
      }
    } else {
      errorQuda("Unsupported nColor %d", out.Ncolor());
    }
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if (out.Nspin() == 2) {
      Prolongate<Float,2>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else if (out.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) Prolongate<Float,4>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
      else errorQuda("nSpin 4 has not been built");
    } else if (out.Nspin() == 1) {
      if constexpr (is_enabled_spin(1)) Prolongate<Float,1>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
      else errorQuda("nSpin 1 has not been built");
    } else {
      errorQuda("Unsupported nSpin %d", out.Nspin());
    }
  }

  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      if (out.FieldOrder() != in.FieldOrder() || out.FieldOrder() != v.FieldOrder())
        errorQuda("Field orders do not match (out=%d, in=%d, v=%d)",
                  out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

      QudaPrecision precision = checkPrecision(out, in);

      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) Prolongate<double>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
        else errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        Prolongate<float>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", out.Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // end namespace quda
