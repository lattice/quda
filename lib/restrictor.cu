#include <array>

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
    template <QudaFieldOrder order = QUDA_FLOAT2_FIELD_ORDER> using Arg =
      RestrictArg<Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor, order>;
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
      strcat(vol, out.VolString());
      strcat(aux, ",");
      strcat(aux, out.AuxString());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
        Arg<QUDA_FLOAT2_FIELD_ORDER> arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        arg.swizzle_factor = tp.aux.x;
        launch<Restrictor, Aggregates>(tp, stream, arg);
      } else {
        errorQuda("Unsupported field order %d", out.FieldOrder());
      }
    }

    bool advanceAux(TuneParam &param) const
    {
      if (Arg<>::swizzle) {
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
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, int parity)
  {
    if constexpr (enabled<fineSpin, fineColor, coarseSpin, coarseColor>::value) {
      if (v.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled<QUDA_HALF_PRECISION>()) {
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
    } else {
      errorQuda("Not enabled");
    }
  }

  template <typename Float>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                int nVec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    if (out.Nspin() != 2) errorQuda("Unsupported nSpin %d", out.Nspin());
    constexpr int coarseSpin = 2;

    // Template over fine color
    if (in.Ncolor() == 3) { // standard QCD
      if (in.Nspin() == 4) {
        constexpr int fineColor = 3;
        constexpr int fineSpin = 4;

        // first check that the spin_map matches the spin_mapper
        spin_mapper<fineSpin,coarseSpin> mapper;
        for (int s=0; s<fineSpin; s++)
          for (int p=0; p<2; p++)
            if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

        switch (nVec) {
        case 6: Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 24: Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 32: Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Nspin() == 1) {
        constexpr int fineColor = 3;
        constexpr int fineSpin = 1;

        // first check that the spin_map matches the spin_mapper
        spin_mapper<fineSpin,coarseSpin> mapper;
        for (int s=0; s<fineSpin; s++)
          for (int p=0; p<2; p++)
            if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

        switch (nVec) {
        case 24: Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 64: Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 96: Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else {
        errorQuda("Unexpected nSpin = %d", in.Nspin());
      }

    } else { // Nc != 3

      if (in.Nspin() != 2) errorQuda("Unexpected nSpin = %d", in.Nspin());
      constexpr int fineSpin = 2;

      // first check that the spin_map matches the spin_mapper
      spin_mapper<fineSpin,coarseSpin> mapper;
      for (int s=0; s<fineSpin; s++)
        for (int p=0; p<2; p++)
          if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

      if (in.Ncolor() == 6) { // Coarsen coarsened Wilson free field
        const int fineColor = 6;
        switch (nVec) {
        case 6: Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 24) { // to keep compilation under control coarse grids have same or more colors
        const int fineColor = 24;
        switch (nVec) {
        case 24: Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 32: Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 64: Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 96: Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 32) {
        const int fineColor = 32;
        switch (nVec) {
        case 32: Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 64) {
        const int fineColor = 64;
        switch (nVec) {
        case 64: Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        case 96: Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 96) {
        const int fineColor = 96;
        switch (nVec) {
        case 96: Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity); break;
        default: errorQuda("Unsupported nVec %d", nVec);
        }
      } else {
        errorQuda("Unsupported nColor %d", in.Ncolor());
      }
    } // Nc != 3
  }

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    checkOrder(out, in, v);
    checkLocation(out, in, v);
    QudaPrecision precision = checkPrecision(out, in);

    if constexpr (is_enabled_multigrid()) {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) Restrict<double>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
        else errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        Restrict<float>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", out.Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
