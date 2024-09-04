#include <color_spinor_field.h>
#include <multigrid.h>
#include <power_of_two_array.h>
#include <tunable_block_reduction.h>
#include <kernels/restrictor_mma.cuh>
#include <device.hpp>
#include <int_factor_array.hpp>

namespace quda
{

  template <typename out_t, typename in_t, typename v_t, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
            int nVec, int aggregate_size>
  class RestrictMmaLaunch : public TunableKernel
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &v;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int parity;

    bool checkParam(const TuneParam &param) const { return true; }

    unsigned int sharedBytesPerThread() const { return 0; }

    bool advanceTuneParam(TuneParam &param) const
    {
      auto advancer = [&](int &i, int limit) -> bool {
        if (i < limit) {
          i++;
          return set_mma_param(param);
        } else {
          return false;
        }
      };

      if (advancer(param.aux.x, numFactors((block_limit + block_atom_size - 1) / block_atom_size) - 1)) {
        return true;
      } else {
        param.aux.x = 0;
        if (advancer(param.aux.y, numFactors((n + n_atom_size - 1) / n_atom_size) - 1)) {
          return true;
        } else {
          param.aux.y = 0;
          if (advancer(param.aux.z, numFactors((m + m_atom_size - 1) / m_atom_size) - 1)) {
            return true;
          } else {
            param.aux.z = 0;
            if (advancer(param.aux.w, numFactors((k + k_atom_size - 1) / k_atom_size) - 1)) {
              return true;
            } else {
              param.aux.w = 0;
              return false;
            }
          }
        }
      }
    }

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
    RestrictMmaLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                      const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      TunableKernel(in),
      out(out),
      in(in),
      v(v),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      parity(parity)
    {
      strcat(vol, ",");
      strcat(vol, out.VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());
      setRHSstring(aux, in.Nvec());

      strcat(aux, mma_t::get_type_name().c_str());

      apply(device::get_default_stream());
    }

    using mma_t = typename mma::smma_dispatch<out_t>::type;
    // using mma_t = simt::simt_t<float, 8, 4, 2, 2>;

    static constexpr int spin_block_factor = spin_mapper<fineSpin, coarseSpin>::get_spin_block_factor();

    static constexpr int m = nVec;
    static constexpr int n = coarseColor;
    static constexpr int k = fineColor * spin_block_factor * aggregate_size;

    static constexpr int n_atom_size = mma_t::MMA_N;
    static constexpr int m_atom_size = mma_t::MMA_M;
    static constexpr int k_atom_size = fineColor * spin_block_factor * 4;
    static constexpr int block_atom_size = 32 / 8;
    static constexpr int block_limit = 32;

    long long flops() const { return nVec * 8 * fineSpin * fineColor * coarseColor * in.SiteSubset() * in.VolumeCB(); }

    long long bytes() const
    {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return nVec * (in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset() * in.VolumeCB() * sizeof(int));
    }

    static constexpr int shared_bytes_per_block(int bM, int bN, int bK)
    {
      return mma::shared_memory_bytes<mma_t>(bM, bN, bK);
    }

    bool set_mma_param(TuneParam &tp) const
    {
      static_assert(m % m_atom_size == 0, "m modulo m_atom_size == 0");
      static_assert(n % n_atom_size == 0, "n modulo n_atom_size == 0");
      static_assert(k % k_atom_size == 0, "k modulo k_atom_size == 0");

      tp.block.x = 1;
      tp.block.y
        = block_atom_size * get_int_factor_array((block_limit + block_atom_size - 1) / block_atom_size)[tp.aux.x];
      tp.block.z = 8;

      int bN = n_atom_size * get_int_factor_array((n + n_atom_size - 1) / n_atom_size)[tp.aux.y];
      int bM = m_atom_size * get_int_factor_array((m + m_atom_size - 1) / m_atom_size)[tp.aux.z];

      tp.grid = dim3(out.Volume() * coarseSpin, (m + bM - 1) / bM, (n + bN - 1) / bN);
      tp.set_max_shared_bytes = true;

      int bK = k_atom_size * get_int_factor_array(k / k_atom_size)[tp.aux.w];
      int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      return shared_bytes <= device::maximum_dynamic_shared_memory();
    }

    template <int bN, int bM, int bK, int block_y, int block_z>
    void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      constexpr int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      if constexpr (shared_bytes <= device::maximum_dynamic_shared_memory()) {
        using Arg = RestrictMmaArg<mma_t, out_t, in_t, v_t, fineSpin, fineColor, coarseSpin, coarseColor, nVec,
                                   aggregate_size, bN, bM, bK, block_y, block_z>;
        Arg arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        tp.set_max_shared_bytes = true;
        launch_cuda<RestrictorMma>(tp, stream, arg);
      } else {
        errorQuda("Using too many shared memory bytes per block: %d", shared_bytes);
      }
    }

    template <int bN, int bM, int block_y, int block_z, size_t d, size_t... Ds>
    void launch_mma_span_k(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>)
    {
      if (tp.aux.w == d) {
        constexpr IntFactorArray<k / k_atom_size> k_factors;
        launch_mma<bN, bM, k_factors[d] * k_atom_size, block_y, block_z>(tp, stream);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_k<bN, bM, block_y, block_z>(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.z.");
        }
      }
    }

    template <int bN, int block_y, int block_z, size_t d, size_t... Ds>
    void launch_mma_span_m(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>)
    {
      if (tp.aux.z == d) {
        constexpr IntFactorArray<(m + m_atom_size - 1) / m_atom_size> m_factors;
        std::make_index_sequence<IntFactorArray<k / k_atom_size>().size()> k_indices;
        launch_mma_span_k<bN, m_factors[d] * m_atom_size, block_y, block_z>(tp, stream, k_indices);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_m<bN, block_y, block_z>(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.z.");
        }
      }
    }

    template <int block_y, int block_z, size_t d, size_t... Ds>
    void launch_mma_span_n(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>)
    {
      if (tp.aux.y == d) {
        constexpr IntFactorArray<(n + n_atom_size - 1) / n_atom_size> n_factors;
        std::make_index_sequence<IntFactorArray<(m + m_atom_size - 1) / m_atom_size>().size()> m_indices;
        launch_mma_span_m<n_factors[d] * n_atom_size, block_y, block_z>(tp, stream, m_indices);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_n<block_y, block_z>(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.y.");
        }
      }
    }

    template <size_t d, size_t... Ds>
    void launch_mma_span_block(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>)
    {
      if (tp.aux.x == d) {
        constexpr IntFactorArray<(block_limit + block_atom_size - 1) / block_atom_size> block_factors;
        std::make_index_sequence<IntFactorArray<(n + n_atom_size - 1) / n_atom_size>().size()> n_indices;
        launch_mma_span_n<block_factors[d] * block_atom_size, 8>(tp, stream, n_indices);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_block(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.x.");
        }
      }
    }

    void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      std::make_index_sequence<IntFactorArray<(block_limit + block_atom_size - 1) / block_atom_size>().size()> block_indices;
      launch_mma_span_block(tp, stream, block_indices);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_mma(tp, stream);
    }
  };

  template <typename store_t, typename in_t, int fineSpin, int fineColor, int coarseColor, int nVec, int aggregate_size>
  void RestrictMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if (out.Nspin() != 2) errorQuda("Unsupported nSpin %d", out.Nspin());
    constexpr int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin, coarseSpin> mapper;
    for (int s = 0; s < fineSpin; s++)
      for (int p = 0; p < 2; p++)
        if (mapper(s, p) != spin_map[s][p]) errorQuda("Spin map does not match spin_mapper");

    if (v.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
        RestrictMmaLaunch<store_t, in_t, short, fineSpin, fineColor, coarseSpin, coarseColor, nVec, aggregate_size> restrictor(
          out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in.Precision()) {
      RestrictMmaLaunch<store_t, in_t, store_t, fineSpin, fineColor, coarseSpin, coarseColor, nVec, aggregate_size> restrictor(
        out, in, v, fine_to_coarse, coarse_to_fine, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename store_t, int fineColor, int coarseColor, int nVec, int aggregate_size>
  void RestrictMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if (!is_enabled_spin(in.Nspin())) errorQuda("nSpin %d has not been built", in.Nspin());

    if (in.Nspin() == 2) {
      RestrictMma<store_t, store_t, 2, fineColor, coarseColor, nVec, aggregate_size>(out, in, v, fine_to_coarse,
                                                                                     coarse_to_fine, spin_map, parity);
    } else if constexpr (fineColor == 3) {
      if (in.Nspin() == 4) {
        if constexpr (is_enabled_spin(4)) {
          if (in.Precision() == out.Precision()) {
            RestrictMma<store_t, store_t, 4, fineColor, coarseColor, nVec, aggregate_size>(
              out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
          } else if (in.Precision() == QUDA_HALF_PRECISION) {
#if 0
            if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
              RestrictMma<store_t, short, 4, fineColor, coarseColor, nVec, aggregate_size>(out, in, v,
              fine_to_coarse, coarse_to_fine, spin_map,
                                                                  parity);
            } else {
#endif
              errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#if 0
            }
#endif
          } else {
            errorQuda("Unsupported precision %d", in.Precision());
          }
        }
      } else if (in.Nspin() == 1) {
#if 0
        if constexpr (is_enabled_spin(1)) {
          if (in.Precision() == out.Precision()) {
            RestrictMma<store_t, store_t, 1, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
                                                                  parity);
          } else if (in.Precision() == QUDA_HALF_PRECISION) {
            if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
              RestrictMma<store_t, short, 1, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
                                                                  parity);
            } else {
              errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
            }
          } else {
            errorQuda("Unsupported precision %d", in.Precision());
          }
        }
#else
        errorQuda("Unexpected nSpin = %d", in.Nspin());
#endif
      } else {
        errorQuda("Unexpected nSpin = %d", in.Nspin());
      }
    } else {
      errorQuda("Unexpected spin %d and color %d combination", in.Nspin(), in.Ncolor());
    }
  }

  // clang-format off
  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;
  // clang-format on

  template <typename store_t, int fineColor, int coarseColor, int nVec>
  void RestrictMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    int aggregate_size = in.Volume() / out.Volume();
    if (aggregate_size == 128) {
      if constexpr (fineColor == 3 && coarseColor == 24) {
        RestrictMma<store_t, fineColor, coarseColor, nVec, 128>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
                                                                parity);
      } else {
        errorQuda("Unexpected aggregate_size = %d\n", aggregate_size);
      }
    } else if (aggregate_size == 16) {
      if constexpr (fineColor == 24 && coarseColor == 32) {
        RestrictMma<store_t, fineColor, coarseColor, nVec, 16>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
                                                               parity);
      } else {
        errorQuda("Unexpected aggregate_size = %d\n", aggregate_size);
      }
    } else{
      errorQuda("Unexpected aggregate_size = %d\n", aggregate_size);
    }
  }

  template <>
  void RestrictMma<fineColor, coarseColor, nVec>(ColorSpinorField &out, const ColorSpinorField &in,
                                                 const ColorSpinorField &v, const int *fine_to_coarse,
                                                 const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {

      checkLocation(out, in, v);
      if (in.Nspin() == 2) checkPrecision(in, out);
      QudaPrecision precision = out.Precision();

      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double())
          RestrictMma<double, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
        else
          errorQuda("Double precision multigrid has not been enabled");
      } else if (precision == QUDA_SINGLE_PRECISION) {
        RestrictMma<float, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
      } else {
        errorQuda("Unsupported precision %d", precision);
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
