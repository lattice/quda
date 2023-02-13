#include <color_spinor_field.h>
#include <multigrid.h>
#include <power_of_two_array.h>
#include <tunable_block_reduction.h>
#include <kernels/restrictor.cuh>

namespace quda {

  // this is a dummy structure for the restrictor to give a compatible
  // interface with TunableBlock2D
  struct Aggregates {
    using array_type = PowerOfTwoArray<1, 1>;
    static constexpr array_type block = array_type();
  };

  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  class RestrictLaunch : public TunableBlock2D {
    using Arg = RestrictArg<Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor>;
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &v;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int parity;

    bool tuneSharedBytes() const { return false; }
    bool tuneAuxDim() const { return true; }
    unsigned int minThreads() const { return in.Volume(); } // fine parity is the block y dimension

  public:
    RestrictLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      TunableBlock2D(in, false, coarseColor / coarse_colors_per_thread<fineColor, coarseColor>(), max_z_block()),
      out(out), in(in), v(v), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      parity(parity)
    {
      strcat(vol, ",");
      strcat(vol, out.VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (checkNative(out, in, v)) {
        Arg arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        arg.swizzle_factor = tp.aux.x;
        launch<Restrictor, Aggregates>(tp, stream, arg);
      }
    }

    bool advanceAux(TuneParam &param) const
    {
      if (Arg::swizzle) {
        if (param.aux.x < 2 * (int)device::processor_count()) {
          param.aux.x++;
          return true;
        } else {
          param.aux.x = 1;
          return false;
        }
      } else {
        return false;
      }
    }

    /**
       @brief Find the smallest block size that is larger than the
       aggregate size.  If the aggregate size is larger than the
       maximum, then the maximum is returned and the thread block will
       rake over the aggregate.
     */
    unsigned int blockMapper() const
    {
      auto aggregate_size = in.Volume() / out.Volume();
      auto max_block = 128u;
      for (uint32_t b = blockMin(); b < max_block; b += blockStep()) if (aggregate_size < b) return b;
      return max_block;
    }

    void initTuneParam(TuneParam &param) const {
      TunableBlock2D::initTuneParam(param);
      param.block.x = blockMapper();
      param.grid.x = out.Volume();
      param.shared_bytes = 0;
      param.aux.x = 2; // swizzle factor
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableBlock2D::defaultTuneParam(param);
      param.block.x = blockMapper();
      param.grid.x = out.Volume();
      param.shared_bytes = 0;
      param.aux.x = 2; // swizzle factor
    }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * in.SiteSubset()*(long long)in.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset()*in.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseColor>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    if (out.Nspin() != 2) errorQuda("Unsupported nSpin %d", out.Nspin());
    constexpr int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++)
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (v.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
        RestrictLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor>
          restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in.Precision()) {
      RestrictLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor>
        restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename Float, int fineColor, int coarseColor>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    if (!is_enabled_spin(in.Nspin())) errorQuda("nSpin %d has not been built", in.Nspin());

    if (in.Nspin() == 2) {
      Restrict<Float, 2, fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
    } else if constexpr (fineColor == 3) {
      if (in.Nspin() == 4) {
        if constexpr (is_enabled_spin(4))
          Restrict<Float, 4, fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else if (in.Nspin() == 1) {
        if constexpr (is_enabled_spin(1))
          Restrict<Float, 1, fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else {
        errorQuda("Unexpected nSpin = %d", in.Nspin());
      }
    } else {
      errorQuda("Unexpected spin %d and color %d combination", in.Nspin(), in.Ncolor());
    }
  }

  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;

  template <>
  void Restrict<fineColor, coarseColor>(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                                        const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    checkLocation(out, in, v);
    QudaPrecision precision = checkPrecision(out, in);

    if constexpr (is_enabled_multigrid()) {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          Restrict<double, fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
        else errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        Restrict<float, fineColor, coarseColor>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", out.Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
