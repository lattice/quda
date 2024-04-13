#include <color_spinor_field.h>
#include <kernels/spinor_dilute.cuh>
#include <tunable_nd.h>
#include <instantiate.h>

namespace quda
{

  template <typename real, int Ns, int Nc> class SpinorDilute : TunableKernel2D
  {
    std::vector<ColorSpinorField> &v;
    const ColorSpinorField &src;
    QudaDilutionType type;
    const lat_dim_t &local_block;
    unsigned int minThreads() const { return src.VolumeCB(); }
    template <QudaDilutionType type> using Arg = SpinorDiluteArg<real, Ns, Nc, type>;

  public:
    SpinorDilute(const ColorSpinorField &src, std::vector<ColorSpinorField> &v, QudaDilutionType type,
                 const lat_dim_t &local_block) :
      TunableKernel2D(src, src.SiteSubset()), v(v), src(src), type(type), local_block(local_block)
    {
      switch (type) {
      case QUDA_DILUTION_SPIN: strcat(aux, ",spin_dilution"); break;
      case QUDA_DILUTION_COLOR: strcat(aux, ",color_dilution"); break;
      case QUDA_DILUTION_SPIN_COLOR: strcat(aux, ",spin_color_dilution"); break;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: strcat(aux, ",spin_color_even_odd_dilution"); break;
      case QUDA_DILUTION_BLOCK: strcat(aux, ",block_dilution"); break;
      default: errorQuda("Unsupported dilution type %d", type);
      }
      if (type != QUDA_DILUTION_BLOCK && v.size() != static_cast<unsigned int>(get_size<Ns, Nc>(type)))
        errorQuda("Input container size %lu does not match expected size %d for dilution type", v.size(),
                  get_size<Ns, Nc>(type));

      size_t block_volume = 1;
      for (int i = 0; i < src.Ndim(); i++) block_volume *= local_block[i];
      size_t n_blocks = comm_size() * src.Volume() / block_volume;
      if (type == QUDA_DILUTION_BLOCK) {
        if (v.size() != n_blocks)
          errorQuda("Input container size %lu does not match expected size %lu for dilution block size (%d,%d,%d,%d)",
                    v.size(), n_blocks, local_block[0], local_block[1], local_block[2], local_block[3]);
        if (v.size() > Arg<QUDA_DILUTION_BLOCK>::max_dilution_size)
          errorQuda("Container size %lu exceeds maximum size %d", v.size(), Arg<QUDA_DILUTION_BLOCK>::max_dilution_size);

        for (auto i = 0; i < src.Ndim(); i++) {
          if (local_block[i] == 0) errorQuda("Dim %d: Dilution block size = 0", i);
          if ((src.X(i) * comm_dim(i)) % local_block[i] != 0)
            errorQuda("Dim %d: Invalid dilution block size %d for global lattice dim = %d", i, local_block[i],
                      src.X(i) * comm_dim(i));
        }
      }

      apply(device::get_default_stream());
    }

    template <QudaDilutionType type> auto constexpr sequence()
    {
      return std::make_index_sequence<get_size<Ns, Nc>(type)>();
    }

    template <QudaDilutionType type> void apply(TuneParam &tp, const qudaStream_t &stream)
    {
      launch<DiluteSpinor>(tp, stream, Arg<type>(v, src, local_block, sequence<type>()));
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (type) {
      case QUDA_DILUTION_SPIN: apply<QUDA_DILUTION_SPIN>(tp, stream); break;
      case QUDA_DILUTION_COLOR: apply<QUDA_DILUTION_COLOR>(tp, stream); break;
      case QUDA_DILUTION_SPIN_COLOR: apply<QUDA_DILUTION_SPIN_COLOR>(tp, stream); break;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: apply<QUDA_DILUTION_SPIN_COLOR_EVEN_ODD>(tp, stream); break;
      case QUDA_DILUTION_BLOCK: apply<QUDA_DILUTION_BLOCK>(tp, stream); break;
      default: errorQuda("Dilution type %d not supported", type);
      }
    }

    long long bytes() const { return v.size() * v[0].Bytes() + src.Bytes(); }
  };

  template <int...> struct IntList {
  };

  template <typename real, int Ns, int Nc, int... N>
  void spinorDilute(const ColorSpinorField &src, std::vector<ColorSpinorField> &v, QudaDilutionType type,
                    const lat_dim_t &local_block, IntList<Nc, N...>)
  {
    if (src.Ncolor() == Nc) {
      if constexpr (Nc <= 32) {
        SpinorDilute<real, Ns, Nc>(src, v, type, local_block);
      } else {
        errorQuda(
          "nColor = %d is too large to compile, see QUDA issue #1422 (https://github.com/lattice/quda/issues/1422)");
      }
    } else {
      if constexpr (sizeof...(N) > 0)
        spinorDilute<real, Ns>(src, v, type, local_block, IntList<N...>());
      else
        errorQuda("nColor = %d not implemented", src.Ncolor());
    }
  }

  template <typename real>
  void spinorDilute(const ColorSpinorField &src, std::vector<ColorSpinorField> &v, QudaDilutionType type,
                    const lat_dim_t &local_block)
  {
    checkNative(src);
    if (!is_enabled_spin(src.Nspin())) errorQuda("spinorNoise has not been built for nSpin=%d fields", src.Nspin());

    if (src.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) spinorDilute<real, 4>(src, v, type, local_block, IntList<3>());
    } else if (src.Nspin() == 2) {
      if constexpr (is_enabled_spin(2))
        spinorDilute<real, 2>(src, v, type, local_block, IntList<3, @QUDA_MULTIGRID_NVEC_LIST@>());
    } else if (src.Nspin() == 1) {
      if constexpr (is_enabled_spin(1)) spinorDilute<real, 1>(src, v, type, local_block, IntList<3>());
    } else {
      errorQuda("Nspin = %d not implemented", src.Nspin());
    }
  }

  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type,
                    const lat_dim_t &local_block)
  {
    switch (src.Precision()) {
    case QUDA_DOUBLE_PRECISION: spinorDilute<double>(src, v, type, local_block); break;
    case QUDA_SINGLE_PRECISION: spinorDilute<float>(src, v, type, local_block); break;
    default: errorQuda("Not instantiated %d\n", src.Precision());
    }
  }

} // namespace quda
