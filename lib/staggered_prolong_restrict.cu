#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/staggered_prolong_restrict.cuh>

namespace quda {

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, StaggeredTransferType transferType>
  class StaggeredProlongRestrictLaunch : public TunableKernel3D {
    template <bool native> using Arg = StaggeredProlongRestrictArg<Float, fineSpin, fineColor, coarseSpin, coarseColor, transferType, native>;

    ColorSpinorField &out;
    const ColorSpinorField &in;
    const int *fine_to_coarse;
    int parity;

    unsigned int minThreads() const { return fineColorSpinorField<transferType>(in,out).VolumeCB(); } // fine parity is the block y dimension

  public:
    StaggeredProlongRestrictLaunch(ColorSpinorField &out, const ColorSpinorField &in,
                                   const int *fine_to_coarse, int parity) :
      TunableKernel3D(fineColorSpinorField<transferType>(in,out), fineColorSpinorField<transferType>(in,out).SiteSubset(), fineColor),
      out(out),
      in(in),
      fine_to_coarse(fine_to_coarse),
      parity(parity)
    {
      strcat(vol, ",");
      strcat(vol, coarseColorSpinorField<transferType>(in,out).VolString().c_str());
      strcat(aux, ",");
      strcat(aux, coarseColorSpinorField<transferType>(in,out).AuxString().c_str());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
          launch_host<StaggeredProlongRestrict>(tp, stream, Arg<false>(out, in, fine_to_coarse, parity));
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      } else {
        if (out.isNative()) {
          launch<StaggeredProlongRestrict>(tp, stream, Arg<true>(out, in, fine_to_coarse, parity));
        } else {
          errorQuda("Unsupported field order %d", out.FieldOrder());
        }
      }
    }

    long long bytes() const {
      return in.Bytes() + out.Bytes() + fineColorSpinorField<transferType>(in,out).SiteSubset()*fineColorSpinorField<transferType>(in,out).VolumeCB()*sizeof(int);
    }

  };

  template <int fineSpin, int fineColor, int coarseSpin, int coarseColor, StaggeredTransferType transferType>
  void StaggeredProlongateRestrict(ColorSpinorField &out, const ColorSpinorField &in, const int *fine_to_coarse, int parity)
  {
    // check precision
    QudaPrecision precision = checkPrecision(out, in);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      StaggeredProlongRestrictLaunch<double,fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      StaggeredProlongRestrictLaunch<float,fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
  }

  template <StaggeredTransferType transferType>
  void StaggeredProlongateRestrict(ColorSpinorField &out, const ColorSpinorField &in, const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    checkOrder(out, in);
    checkLocation(out, in);

    if (fineColorSpinorField<transferType>(in,out).Nspin() != 1)
      errorQuda("Fine spin %d is not supported", fineColorSpinorField<transferType>(in,out).Nspin());
    const int fineSpin = 1;

    if (coarseColorSpinorField<transferType>(in,out).Nspin() != 2)
      errorQuda("Coarse spin %d is not supported", coarseColorSpinorField<transferType>(in,out).Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      for (int p=0; p<2; p++)
        if (mapper(s,p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (fineColorSpinorField<transferType>(in,out).Ncolor() != 3)
      errorQuda("Unsupported fine nColor %d",fineColorSpinorField<transferType>(in,out).Ncolor());
    const int fineColor = 3;

    if (coarseColorSpinorField<transferType>(in,out).Ncolor() != 8*fineColor)
      errorQuda("Unsupported coarse nColor %d", coarseColorSpinorField<transferType>(in,out).Ncolor());
    const int coarseColor = 8*fineColor;

    StaggeredProlongateRestrict<fineSpin,fineColor,coarseSpin,coarseColor,transferType>(out, in, fine_to_coarse, parity);
  }

#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)
  void StaggeredProlongate(ColorSpinorField &out, const ColorSpinorField &in,
                           const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    StaggeredProlongateRestrict<StaggeredTransferType::STAGGERED_TRANSFER_PROLONG>(out, in, fine_to_coarse, spin_map, parity);
  }

  void StaggeredRestrict(ColorSpinorField &out, const ColorSpinorField &in,
                         const int *fine_to_coarse, const int * const * spin_map, int parity)
  {
    StaggeredProlongateRestrict<StaggeredTransferType::STAGGERED_TRANSFER_RESTRICT>(out, in, fine_to_coarse, spin_map, parity);
  }
#else
  void StaggeredProlongate(ColorSpinorField &, const ColorSpinorField &, const int *, const int * const *, int) { errorQuda("Staggered multigrid has not been build");  }
  void StaggeredRestrict(ColorSpinorField &, const ColorSpinorField &, const int *, const int * const *, int) { errorQuda("Staggered multigrid has not been build"); }
#endif

} // end namespace quda
