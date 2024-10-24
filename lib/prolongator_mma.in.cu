#include <color_spinor_field.h>
#include <multigrid.h>
#include <tunable_nd.h>
#include <kernels/prolongator_mma.cuh>
#include <device.hpp>
#include <expand_list.hpp>
#include <mma_tensor_op/smma_m16n8k8_sm70.cuh>

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

    // using mma_t = typename mma::smma_dispatch<Float>::type;
    // using mma_t = simt::simt_t<float, 8, 4, 2, 2>;
    // using mma_t = smma::smma_x_t<mma::half, 8, 1, 1>;
    // using mma_t = hmma::hmma_t<16, 16, 4, mma::half, mma::half2>;
#if (__COMPUTE_CAPABILITY__ >= 800)
    using mma_t = hmma::hmma_tfloat32_t<4, 1, 1>;
#else
    using mma_t = hmma::hmma_x_t<16, 8, 8, mma::half, mma::half2>;
#endif

    static constexpr int spin_block_factor = spin_mapper<fineSpin, coarseSpin>::get_spin_block_factor();

    static constexpr int m = nVec;
    static constexpr int n = fineColor * spin_block_factor;
    static constexpr int k = coarseColor;

    static constexpr int n_atom_size = mma_t::MMA_N;
    static constexpr int m_atom_size = mma_t::MMA_M;
    static constexpr int k_atom_size = mma_t::MMA_K;
    static constexpr int block_atom_size = 32 / 8;

    using this_t = ProlongateLaunchMma<Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor, nVec>;
    expand_aux_t<this_t, k, block_atom_size, n, n_atom_size, m, m_atom_size, k, k_atom_size> expand;

    bool checkParam(const TuneParam &param) const { return true; }

    unsigned int sharedBytesPerThread() const { return 0; }

    bool advanceTuneParam(TuneParam &param) const { return expand.advance_aux(param); }

    void initTuneParam(TuneParam &param) const
    {
      expand.init_aux(param);
      set_mma_param(param);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      expand.init_aux(param);
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
      location(checkLocation(out, in, V)),
      expand(*this)
    {
      strcat(vol, ",");
      strcat(vol, out.VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());
      setRHSstring(aux, in.Nvec());

      strcat(aux, mma_t::get_type_name().c_str());

      apply(device::get_default_stream());
    }

    long long flops() const
    {
      return nVec * 8 * fineSpin * fineColor * coarseColor * out.SiteSubset() * out.VolumeCB();
    }

    long long bytes() const
    {
      size_t v_bytes = V.Bytes() / (V.SiteSubset() == out.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + nVec * (v_bytes + out.SiteSubset() * out.VolumeCB() * sizeof(int));
    }

    static constexpr int shared_bytes_per_block(int bM, int bN, int bK)
    {
      return mma::shared_memory_bytes<mma_t>(bM, bN, bK);
    }

    bool set_mma_param(TuneParam &tp) const
    {
      static_assert(k % k_atom_size == 0, "k modulo k_atom_size == 0");

      tp.block.x = 1;
      tp.block.y = expand.get_x(tp);
      tp.block.z = 8;

      int bN = expand.get_y(tp);
      int bM = expand.get_z(tp);

      tp.grid
        = dim3(out.SiteSubset() * out.VolumeCB() * fineSpin / spin_block_factor, (m + bM - 1) / bM, (n + bN - 1) / bN);
      tp.set_max_shared_bytes = true;

      int bK = expand.get_w(tp);
      int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      return shared_bytes <= device::maximum_dynamic_shared_memory();
    }

    template <int block_y, int bN, int bM, int bK> void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      constexpr int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      if constexpr (shared_bytes <= device::maximum_dynamic_shared_memory()) {
        constexpr bool to_non_rel = false;
        constexpr int block_z = 8;
        using Arg = ProlongateMmaArg<mma_t, Float, vFloat, fineSpin, fineColor, coarseSpin, coarseColor, nVec,
                                     to_non_rel, bN, bM, bK, block_y, block_z>;
        Arg arg(out, in, V, fine_to_coarse, parity);
        tp.set_max_shared_bytes = true;
        launch_cuda<ProlongatorMma>(tp, stream, arg);
      } else {
        errorQuda("Using too many shared memory bytes per block: %d", shared_bytes);
      }
    }

    void launch_mma(TuneParam &tp, const qudaStream_t &stream) { expand.expand(tp, stream); }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_mma(tp, stream);
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

  // clang-format off
  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;
  // clang-format on

  template <>
  void ProlongateMma<fineColor, coarseColor, nVec>(ColorSpinorField &out, const ColorSpinorField &in,
                                                   const ColorSpinorField &v, const int *fine_to_coarse,
                                                   const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {
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
