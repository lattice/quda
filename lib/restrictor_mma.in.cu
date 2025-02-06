#include <color_spinor_field.h>
#include <multigrid.h>
#include <power_of_two_array.h>
#include <tunable_block_reduction.h>
#include <expand_list.hpp>

#ifdef QUDA_MMA_AVAILABLE
#include <device.hpp>
#include <kernels/restrictor_mma.cuh>
#include <mma_tensor_op/smma_m16n8k8_sm70.cuh>
#endif

namespace quda
{

#ifdef QUDA_MMA_AVAILABLE

  template <typename out_t, typename in_t, typename v_t, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int nVec>
  class RestrictMmaLaunch : public TunableKernel
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &v;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int parity;
    int aggregate_size;

    using mma_t = typename mma::mg_mma_restrictor_t<out_t>::type;

    static constexpr int spin_block_factor = spin_mapper<fineSpin, coarseSpin>::get_spin_block_factor();
    // The number of fine grid aggregate to be inlucded in an thread block per iteration: this number
    // cannot be too large due to shared memory restriction.
    static constexpr int aggregate_size_block_max = 16;

    static constexpr int m = nVec;
    static constexpr int n = coarseColor;
    static constexpr int k = fineColor * spin_block_factor * aggregate_size_block_max;

    static constexpr int n_atom_size = mma_t::MMA_N;
    static constexpr int m_atom_size = mma_t::MMA_M;
    static constexpr int k_atom_size = fineColor * spin_block_factor * mma_t::MMA_K;
    // The atom for `block_y` that determines the number of threads in a thread block:
    //    # of threads = block_y * block_z (which always equals to 8)
    // As a result the number of threads goes from 32 to 256 threads.
    static constexpr int block_atom_size = 32 / 8;
    static constexpr int block_limit = 32;

    using this_t = RestrictMmaLaunch<out_t, in_t, v_t, fineSpin, fineColor, coarseSpin, coarseColor, nVec>;
    expand_aux_t<this_t, block_limit, block_atom_size, n, n_atom_size, m, m_atom_size, k, k_atom_size> expand;

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
    RestrictMmaLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                      const int *fine_to_coarse, const int *coarse_to_fine, int parity) :
      TunableKernel(in),
      out(out),
      in(in),
      v(v),
      fine_to_coarse(fine_to_coarse),
      coarse_to_fine(coarse_to_fine),
      parity(parity),
      aggregate_size(in.Volume() / out.Volume()),
      expand(*this)
    {
      strcat(vol, ",");
      strcat(vol, out.VolString().c_str());
      strcat(aux, ",");
      strcat(aux, out.AuxString().c_str());
      setRHSstring(aux, out.Nvec_actual());

      strcat(aux, mma_t::get_type_name().c_str());
      strcat(aux, ",aggregate_size_block_max=");
      i32toa(aux + strlen(aux), aggregate_size_block_max);

      apply(device::get_default_stream());
    }

    long long flops() const
    {
      return out.Nvec_actual() * 8 * fineSpin * fineColor * coarseColor * in.SiteSubset() * in.VolumeCB();
    }

    long long bytes() const
    {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return nVec * (in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset() * in.VolumeCB() * sizeof(int));
    }

    int shared_bytes_per_block(int bM, int bN, int bK) const
    {
      return mma::shared_memory_bytes<mma_t>(bM, bN, bK) + aggregate_size * sizeof(int);
    }

    bool set_mma_param(TuneParam &tp) const
    {
      static_assert(k % k_atom_size == 0, "k modulo k_atom_size == 0");

      tp.block.x = 1;
      tp.block.y = expand.get_x(tp);
      tp.block.z = 8;

      int bN = expand.get_y(tp);
      int bM = expand.get_z(tp);

      tp.grid = dim3(out.Volume() * coarseSpin, (m + bM - 1) / bM, (n + bN - 1) / bN);
      tp.set_max_shared_bytes = true;

      int bK = expand.get_w(tp);
      int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      return shared_bytes <= device::maximum_dynamic_shared_memory();
    }

    template <int block_y, int bN, int bM, int bK> void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      constexpr int shared_bytes = mma::shared_memory_bytes<mma_t>(bM, bN, bK);
      if constexpr (shared_bytes <= device::maximum_dynamic_shared_memory()) {
        constexpr int block_z = 8;
        using Arg = RestrictMmaArg<mma_t, out_t, in_t, v_t, fineSpin, fineColor, coarseSpin, coarseColor, nVec, bN, bM,
                                   bK, block_y, block_z>;
        Arg arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
        tp.set_max_shared_bytes = true;
        launch_cuda<RestrictorMma>(tp, stream, arg);
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

  template <typename store_t, typename in_t, int fineSpin, int fineColor, int coarseColor, int nVec>
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
        RestrictMmaLaunch<store_t, in_t, short, fineSpin, fineColor, coarseSpin, coarseColor, nVec> restrictor(
          out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (v.Precision() == in.Precision()) {
      RestrictMmaLaunch<store_t, in_t, store_t, fineSpin, fineColor, coarseSpin, coarseColor, nVec> restrictor(
        out, in, v, fine_to_coarse, coarse_to_fine, parity);
    } else {
      errorQuda("Unsupported V precision %d", v.Precision());
    }
  }

  template <typename store_t, int fineColor, int coarseColor, int nVec>
  void RestrictMma(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
                   const int *fine_to_coarse, const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if (!is_enabled_spin(in.Nspin())) errorQuda("nSpin %d has not been built", in.Nspin());

    if (in.Nspin() == 2) {
      RestrictMma<store_t, store_t, 2, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine,
                                                                     spin_map, parity);
    } else if constexpr (fineColor == 3) {
      if (in.Nspin() == 4) {
        if constexpr (is_enabled_spin(4)) {
          if (in.Precision() == out.Precision()) {
            RestrictMma<store_t, store_t, 4, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine,
                                                                           spin_map, parity);
          } else if (in.Precision() == QUDA_HALF_PRECISION) {
            // The half precision instantiations are disabled here and below: fixed point format with variadic
            // scale cannot have nVec as the fastest running index since that is incompatible with the idea of
            // having one scale float for one color spinor.
#if 0
            if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
              RestrictMma<store_t, short, 4, fineColor, coarseColor, nVec>(out, in, v,
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
        if constexpr (is_enabled_spin(1)) {
          if (in.Precision() == out.Precision()) {
            RestrictMma<store_t, store_t, 1, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine,
                                                                           spin_map, parity);
          } else if (in.Precision() == QUDA_HALF_PRECISION) {
#if 0
            if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
              RestrictMma<store_t, short, 1, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
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
      } else {
        errorQuda("Unexpected nSpin = %d", in.Nspin());
      }
    } else {
      errorQuda("Unexpected spin %d and color %d combination", in.Nspin(), in.Ncolor());
    }
  }
#endif

  // clang-format off
  constexpr int fineColor = @QUDA_MULTIGRID_NC_NVEC@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC2@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;
  // clang-format on

#ifdef QUDA_MMA_AVAILABLE
  template <>
  void RestrictMma<fineColor, coarseColor, nVec>(ColorSpinorField &out, const ColorSpinorField &in,
                                                 const ColorSpinorField &v, const int *fine_to_coarse,
                                                 const int *coarse_to_fine, const int *const *spin_map, int parity)
  {
    if constexpr (is_enabled_multigrid()) {

      if constexpr ((fineColor == 3 || fineColor == 24 || fineColor == 32 || fineColor == 64 || fineColor == 96)
                    && (coarseColor == 24 || coarseColor == 32 || coarseColor == 64 || coarseColor == 96)
                    && (fineColor <= coarseColor)) {
        checkLocation(out, in, v);
        if (in.Nspin() == 2) checkPrecision(in, out);
        QudaPrecision precision = out.Precision();

        if (precision == QUDA_DOUBLE_PRECISION) {
          if constexpr (is_enabled_multigrid_double())
            RestrictMma<double, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map,
                                                              parity);
          else
            errorQuda("Double precision multigrid has not been enabled");
        } else if (precision == QUDA_SINGLE_PRECISION) {
          RestrictMma<float, fineColor, coarseColor, nVec>(out, in, v, fine_to_coarse, coarse_to_fine, spin_map, parity);
        } else {
          errorQuda("Unsupported precision %d", precision);
        }
      } else {
        errorQuda("fineColor=%d or coarseColor=%d have not been implemented yet.", fineColor, coarseColor);
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }
#else
  template <>
  void RestrictMma<fineColor, coarseColor, nVec>(ColorSpinorField &, const ColorSpinorField &, const ColorSpinorField &,
                                                 const int *, const int *, const int *const *, int)
  {
    errorQuda("RestrictMma is not instantiated.");
  }
#endif

} // namespace quda
