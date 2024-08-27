#include <color_spinor_field.h>
#include <multigrid.h>
#include <tunable_nd.h>
#include <kernels/prolongator_mma.cuh>
#include <device.hpp>

namespace quda
{

  template <typename Float, typename vFloat, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int nVec>
  class ProlongateLaunchMma : public TunableKernel
  {

    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &V;
    const int *fine_to_coarse;
    int parity;
    QudaFieldLocation location;

    bool checkParam(const TuneParam &param) const { return true; }

    unsigned int sharedBytesPerThread() const { return 0; }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    void initTuneParam(TuneParam &param) const
    {
      param.aux.x = 0;
      param.aux.y = 0;
      param.aux.z = 0;
      param.aux.w = 0;
      set_mma_param(param);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      param.aux.x = 0;
      param.aux.y = 0;
      param.aux.z = 0;
      param.aux.w = 0;
      set_mma_param(param);
    }

  public:
    ProlongateLaunchMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
                        const int *fine_to_coarse, int parity) :
      TunableKernel(in),
      out(out),
      in(in),
      V(V),
      fine_to_coarse(fine_to_coarse),
      parity(parity),
      location(checkLocation(out, in, V))
    {
      printf("out.Location() = %d, parity = %d\n", out.Location(), parity);
      strcat(vol, ",");
      strcat(vol, out.VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());
      if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) strcat(aux, ",to_non_rel");

      apply(device::get_default_stream());
    }

    // using mma_t = simt::simt_t<float, 8, 4, 2, 2>;
    // using mma_t = smma::smma_t<mma::tfloat32, 4, 1, 1>;  // 3xTF32
    using mma_t = typename mma::smma_dispatch<Float>::type;
    static constexpr int n_atom_size = nVec;
    static constexpr int m_atom_size = fineColor;
    static constexpr int k_atom_size = coarseColor;

    long long flops() const
    {
      return nVec * 8 * fineSpin * fineColor * coarseColor * out.SiteSubset() * out.VolumeCB();
    }

    long long bytes() const
    {
      size_t v_bytes = V.Bytes() / (V.SiteSubset() == out.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + nVec * (v_bytes + out.SiteSubset() * out.VolumeCB() * sizeof(int));
    }

    bool set_mma_param(TuneParam &tp) const
    {
      tp.block.x = 1;
      tp.block.y = 16;
      tp.block.z = 8;

      int bN = fineColor;
      int bM = nVec;
      int bK = coarseColor;

      tp.grid = dim3(out.SiteSubset() * out.VolumeCB() * fineSpin, nVec / bM, fineColor / bN);
      tp.set_max_shared_bytes = true;

      int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      return shared_bytes <= device::maximum_dynamic_shared_memory();
    }

    static constexpr int shared_bytes_per_block(int bM, int bN, int bK)
    {
      return mma::shared_memory_bytes<mma_t>(bM, bN, bK) + (bM + 4) * (bK + 4) * 2 * sizeof(vFloat)
        + (bK + 4) * (bN + 4) * 2 * sizeof(Float);
    }

    template <int bN, int bM, int bK, int block_y, int block_z>
    void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      constexpr int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      if constexpr (shared_bytes <= device::maximum_dynamic_shared_memory()) {
        constexpr bool to_non_rel = false;
        using Arg = ProlongateMmaArg<mma_t, Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor, nVec,
                                     to_non_rel, bN, bM, bK, block_y, block_z>;
        Arg arg(out, in, V, fine_to_coarse, parity);
        tp.set_max_shared_bytes = true;
        launch_cuda<ProlongatorMma>(tp, stream, arg);
      } else {
        errorQuda("Using too many shared memory bytes per block: %d", shared_bytes);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      constexpr int block_y = 16;
      constexpr int block_z = 8;
      constexpr int bN = fineColor;
      constexpr int bM = nVec;
      constexpr int bK = coarseColor;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_mma<bN, bM, bK, block_y, block_z>(tp, stream);
    }
  };

  template <typename Float, int fineSpin, int fineColor, int coarseColor, int nVec>
  void ProlongateMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                     const int *fine_to_coarse, const int *const *spin_map, int parity)
  {
    if (in.Nspin() != 2) errorQuda("Coarse spin %d is not supported", in.Nspin());
    constexpr int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin, coarseSpin> mapper;
    for (int s = 0; s < fineSpin; s++)
      for (int p = 0; p < 2; p++)
        if (mapper(s, p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (v.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
        ProlongateLaunchMma<Float, short, fineSpin, fineColor, coarseSpin, coarseColor, nVec> prolongator(
          out, in, v, fine_to_coarse, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in.Precision()) {
      ProlongateLaunchMma<Float, Float, fineSpin, fineColor, coarseSpin, coarseColor, nVec> prolongator(
        out, in, v, fine_to_coarse, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename Float, int fineColor, int coarseColor, int nVec>
  void ProlongateMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                     const int *fine_to_coarse, const int *const *spin_map, int parity)
  {
    if (!is_enabled_spin(out.Nspin())) errorQuda("nSpin %d has not been built", in.Nspin());

    if (out.Nspin() == 2) {
      ProlongateMma<Float, 2, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, spin_map, parity);
    } else if constexpr (fineColor == 3) {
      if (out.Nspin() == 4) {
        if constexpr (is_enabled_spin(4))
          ProlongateMma<Float, 4, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, spin_map, parity);
      } else if (out.Nspin() == 1) {
        if constexpr (is_enabled_spin(1))
          ProlongateMma<Float, 1, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Unsupported nSpin %d", out.Nspin());
      }
    } else {
      errorQuda("Unexpected spin %d and color %d combination", out.Nspin(), out.Ncolor());
    }
  }

  // clang-format on
  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;
  // clang-format off

  template <>
  void ProlongateMma<fineColor, coarseColor, nVec>(ColorSpinorField &out, const ColorSpinorField &in,
                                                   const ColorSpinorField &v, const int *fine_to_coarse,
                                                   const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid() && fineColor > 3) {
      QudaPrecision precision = checkPrecision(out, in);

      if (precision == QUDA_DOUBLE_PRECISION) {
        errorQuda("ProlongateMma with double precision has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        ProlongateMma<float, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", out.Precision());
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // end namespace quda
