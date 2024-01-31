#include <color_spinor_field.h>
#include <multigrid.h>
#include <tunable_nd.h>
#include <kernels/prolongator.cuh>

namespace quda {

  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  class ProlongateLaunch : public TunableKernel3D {
    template <bool to_non_rel>
    using Arg = ProlongateArg<Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor, to_non_rel>;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    const ColorSpinorField &V;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;

    unsigned int minThreads() const { return out[0].VolumeCB(); } // fine parity is the block y dimension

  public:
    ProlongateLaunch(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &V,
                     const int *fine_to_coarse, int parity)
      : TunableKernel3D(in[0], out[0].SiteSubset() * out.size(), fineColor/fine_colors_per_thread<fineColor, coarseColor>()),
        out(out),
        in(in),
        V(V),
        fine_to_coarse(fine_to_coarse),
        parity(parity),
        location(checkLocation(out[0], in[0], V))
    {
      strcat(vol, ",");
      strcat(vol, out[0].VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out[0].AuxString().c_str());
      strcat(aux, ",n_rhs=");
      char rhs_str[16];
      i32toa(rhs_str, out.size());
      strcat(aux, rhs_str);
      if (out[0].GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) strcat(aux, ",to_non_rel");

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      if (checkNative(out[0], in[0], V)) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if constexpr (fineSpin == 4) {
          if (out[0].GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
            launch<Prolongator>(tp, stream, Arg<true>(out, in, V, fine_to_coarse, parity));
          } else {
            launch<Prolongator>(tp, stream, Arg<false>(out, in, V, fine_to_coarse, parity));
          }
        } else {
          launch<Prolongator>(tp, stream, Arg<false>(out, in, V, fine_to_coarse, parity));
        }
      }
    }

    long long flops() const
    {
      return out.size() * 8 * fineSpin * fineColor * coarseColor * out[0].SiteSubset() * out[0].VolumeCB();
    }

    long long bytes() const {
      size_t v_bytes = V.Bytes() / (V.SiteSubset() == out[0].SiteSubset() ? 1 : 2);
      return out.size() * (in[0].Bytes() + out[0].Bytes() + v_bytes + out[0].SiteSubset() * out[0].VolumeCB() * sizeof(int));
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseColor>
  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if (in[0].Nspin() != 2) errorQuda("Coarse spin %d is not supported", in[0].Nspin());
    constexpr int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++)
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");


    if (v.Precision() == QUDA_HALF_PRECISION) {
      if constexpr(is_enabled(QUDA_HALF_PRECISION)) {
        ProlongateLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor>
          prolongator(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in[0].Precision()) {
      ProlongateLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor>
        prolongator(out, in, v, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename Float, int fineColor, int coarseColor>
  void Prolongate(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if (!is_enabled_spin(out[0].Nspin())) errorQuda("nSpin %d has not been built", in[0].Nspin());

    if (out[0].Nspin() == 2) {
      Prolongate<Float, 2, fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
    } else if constexpr (fineColor == 3) {
      if (out[0].Nspin() == 4) {
        if constexpr (is_enabled_spin(4))
          Prolongate<Float, 4, fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
      } else if (out[0].Nspin() == 1) {
        if constexpr (is_enabled_spin(1))
          Prolongate<Float, 1, fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Unsupported nSpin %d", out[0].Nspin());
      }
    } else {
      errorQuda("Unexpected spin %d and color %d combination", out[0].Nspin(), out[0].Ncolor());
    }
  }

  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;

  template <>
  void Prolongate<fineColor, coarseColor>(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &v,
                                          const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
      QudaPrecision precision = checkPrecision(out[0], in[0]);

      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          Prolongate<double, fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        Prolongate<float, fineColor, coarseColor>(out, in, v, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", out[0].Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // end namespace quda
