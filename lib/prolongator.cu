#include <color_spinor_field.h>
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

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  const int *fine_to_coarse, int parity) {

    if (v.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      ProlongateLaunch<Float, short, fineSpin, fineColor, coarseSpin, coarseColor>
        prolongator(out, in, v, fine_to_coarse, parity);
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else if (v.Precision() == in.Precision()) {
      ProlongateLaunch<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor>
        prolongator(out, in, v, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename Float, int fineSpin>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int nVec, const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (out.Ncolor() == 3) {
      const int fineColor = 3;
#ifdef NSPIN4
      if (nVec == 6) { // Free field Wilson
        Prolongate<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, parity);
      } else
#endif // NSPIN4
      if (nVec == 24) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, parity);
#ifdef NSPIN4
      } else if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN4
#ifdef NSPIN1
      } else if (nVec == 64) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, parity);
      } else if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#ifdef NSPIN4
    } else if (out.Ncolor() == 6) { // for coarsening coarsened Wilson free field.
      const int fineColor = 6;
      if (nVec == 6) { // these are probably only for debugging only
        Prolongate<Float,fineSpin,fineColor,coarseSpin,6>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN4
    } else if (out.Ncolor() == 24) {
      const int fineColor = 24;
      if (nVec == 24) { // to keep compilation under control coarse grids have same or more colors
        Prolongate<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, parity);
#ifdef NSPIN4
      } else if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN4
#ifdef NSPIN1
      } else if (nVec == 64) { 
        Prolongate<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, parity);
      } else if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
#endif // NSPIN1
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#ifdef NSPIN4
    } else if (out.Ncolor() == 32) {
      const int fineColor = 32;
      if (nVec == 32) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN4
#ifdef NSPIN1
    } else if (out.Ncolor() == 64) {
      const int fineColor = 64;
      if (nVec == 64) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,64>(out, in, v, fine_to_coarse, parity);
      } else if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (out.Ncolor() == 96) {
      const int fineColor = 96;
      if (nVec == 96) {
        Prolongate<Float,fineSpin,fineColor,coarseSpin,96>(out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("Unsupported nVec %d", nVec);
      }
#endif // NSPIN1
    } else {
      errorQuda("Unsupported nColor %d", out.Ncolor());
    }
  }

  template <typename Float>
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity) {

    if (out.Nspin() == 2) {
      Prolongate<Float,2>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#ifdef NSPIN4
    } else if (out.Nspin() == 4) {
      Prolongate<Float,4>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#endif
#ifdef NSPIN1
    } else if (out.Nspin() == 1) {
      Prolongate<Float,1>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", out.Nspin());
    }
  }

#ifdef GPU_MULTIGRID
  void Prolongate(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                  int Nvec, const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    if (out.FieldOrder() != in.FieldOrder() || out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)", 
                out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Prolongate<double>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      Prolongate<float>(out, in, v, Nvec, fine_to_coarse, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
  }
#else
  void Prolongate(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &,
                  int, const int *, const int * const *, int)
  {
    errorQuda("Multigrid has not been built");
  }
#endif

} // end namespace quda
