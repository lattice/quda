#include <color_spinor_field.h>
#include <tunable_block_reduction.h>
#include <kernels/restrictor.cuh>

namespace quda {

  struct Aggregates {
    // List of block sizes we wish to instantiate.  The required block
    // size is equal to number of fine points per aggregate, rounded
    // up to a whole power of two.  So for example, 2x2x2x2 and
    // 3x3x3x1 aggregation would both use the same block size 32
#ifndef QUDA_FAST_COMPILE_REDUCE
    static constexpr std::array<unsigned int, 6> block = {32, 64, 128, 256, 512, 1024};
#else
    static constexpr std::array<unsigned int, 1> block = {1024};
#endif

    /**
       @brief Return the first power of two block that is larger than the required size
    */
    static unsigned int block_mapper(unsigned int raw_block)
    {
      for (auto block_ : block) if (raw_block <= block_) return block_;
      errorQuda("Invalid raw block size %d\n", raw_block);
      return 0;
    }
  };

#ifndef QUDA_FAST_COMPILE_REDUCE
  constexpr std::array<unsigned int, 6> Aggregates::block;
#else
  constexpr std::array<unsigned int, 1> Aggregates::block;
#endif

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
      TunableBlock2D(in, coarseColor / coarse_colors_per_thread<fineColor, coarseColor>(), max_y_block()),
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

    int tuningIter() const { return 3; }

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

    void initTuneParam(TuneParam &param) const {
      TunableBlock2D::initTuneParam(param);
      param.block.x = Aggregates::block_mapper(in.Volume() / out.Volume());
      param.grid.x = out.Volume();
      param.shared_bytes = 0;
      param.aux.x = 1; // swizzle factor
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableBlock2D::defaultTuneParam(param);
      param.block.x = Aggregates::block_mapper(in.Volume() / out.Volume());
      param.grid.x = out.Volume();
      param.shared_bytes = 0;
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * in.SiteSubset()*(long long)in.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset()*in.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                const int *fine_to_coarse, const int *coarse_to_fine, int parity) {

    if (v.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      RestrictLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor>
        restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (v.Precision() == in.Precision()) {
      RestrictLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor>
        restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
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
#ifdef NSPIN4
      if (in.Nspin() == 4) {
        constexpr int fineColor = 3;
        constexpr int fineSpin = 4;

        // first check that the spin_map matches the spin_mapper
        spin_mapper<fineSpin,coarseSpin> mapper;
        for (int s=0; s<fineSpin; s++)
          for (int p=0; p<2; p++)
            if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

        if (nVec == 6) { // free field Wilson
          Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 24) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 32) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else
#endif // NSPIN4
#ifdef NSPIN1
      if (in.Nspin() == 1) {
        constexpr int fineColor = 3;
        constexpr int fineSpin = 1;

        // first check that the spin_map matches the spin_mapper
        spin_mapper<fineSpin,coarseSpin> mapper;
        for (int s=0; s<fineSpin; s++)
          for (int p=0; p<2; p++)
            if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

        if (nVec == 24) { // free field staggered
          Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 64) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else
#endif
      {
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

#ifdef NSPIN4
      if (in.Ncolor() == 6) { // Coarsen coarsened Wilson free field
        const int fineColor = 6;
        if (nVec == 6) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else
#endif // NSPIN4
      if (in.Ncolor() == 24) { // to keep compilation under control coarse grids have same or more colors
        const int fineColor = 24;
        if (nVec == 24) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#ifdef NSPIN4
        } else if (nVec == 32) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#endif // NSPIN4
#ifdef NSPIN1
        } else if (nVec == 64) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
#endif // NSPIN1
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#ifdef NSPIN4
      } else if (in.Ncolor() == 32) {
        const int fineColor = 32;
        if (nVec == 32) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#endif // NSPIN4
#ifdef NSPIN1
      } else if (in.Ncolor() == 64) {
        const int fineColor = 64;
        if (nVec == 64) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
      } else if (in.Ncolor() == 96) {
        const int fineColor = 96;
        if (nVec == 96) {
          Restrict<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        } else {
          errorQuda("Unsupported nVec %d", nVec);
        }
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nColor %d", in.Ncolor());
      }
    } // Nc != 3
  }

#ifdef GPU_MULTIGRID
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int * const * spin_map, int parity)
  {
    checkOrder(out, in, v);
    checkLocation(out, in, v);
    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Restrict<double>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      Restrict<float>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
  }
#else
  void Restrict(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &,
                int, const int *, const int *, const int * const *, int)
  {
    errorQuda("Multigrid has not been built");
  }
#endif

} // namespace quda
