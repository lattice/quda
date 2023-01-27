#include <gauge_field.h>
#include <color_spinor_field.h>
#include <uint_to_char.h>
#include <worker.h>
#include <tunable_nd.h>
#include <kernels/dslash_coarse.cuh>
#include <kernels/dslash_coarse_mma.cuh>
#include <shmem_helper.cuh>
#include <dslash_quda.h>
#include <dslash_shmem.h>
#include <multigrid.h>

#include <int_factor_array.hpp>
#include <utility>
#include <tunable_kernel.h>

namespace quda {

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, int Nc, bool dslash, bool clover, bool dagger,
            DslashType type>
  class DslashCoarse <Float, yFloat, ghostFloat, Ns, Nc, dslash, clover, dagger, type, true> : public TunableKernel
  {
    static constexpr int nDim = 4;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const GaugeField &Y;
    const GaugeField &X;
    const double kappa;
    const int parity;
    const int nParity;
    const int nSrc;
    const ColorSpinorField &halo;

    const int max_color_col_stride = 8;
    mutable int color_col_stride;
    mutable int dim_threads;

    long long flops() const
    {
      return ((dslash * 2 * nDim + clover * 1) * (8 * Ns * Nc * Ns * Nc) - 2 * Ns * Nc) * nParity
        * static_cast<long long>(out[0].VolumeCB()) * out.size() * out[0].Nvec();
    }

    long long bytes() const
    {
        return (dslash || clover) * out[0].Bytes() + dslash * 8 * inA[0].Bytes() + clover * inB[0].Bytes() +
              nSrc * nParity * (dslash * Y.Bytes() * Y.VolumeCB() / (2 * Y.Stride()) + clover * X.Bytes() / 2);
    }

    unsigned int sharedBytesPerThread() const {
        return 0;
    }

    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions
    unsigned int minThreads() const {
        return X.VolumeCB();
    }

    /**
       @param Helper function to check that the present launch parameters are valid
    */
    bool checkParam(const TuneParam &param) const
    {
        return true;
    }

    bool advanceTuneParam(TuneParam &param) const {
        if (param.aux.x < 2) {
          param.aux.x++;
          set_mma_param(param);
          return true;
        } else {
          param.aux.x = 0;
          if (static_cast<unsigned int>(param.aux.y) < numFactors(out[0].Nvec() / n_atom_size) - 1) {
            param.aux.y++;
            set_mma_param(param);
            return true;
          } else {
            param.aux.y = 0;
            if (static_cast<unsigned int>(param.aux.z) < numFactors((Ns * Nc) / m_atom_size) - 1) {
              param.aux.z++;
              set_mma_param(param);
              return true;
            } else {
              return false;
            }
          }
        }
    }

    void initTuneParam(TuneParam &param) const
    {
        param.aux.x = 0;
        param.aux.y = 0;
        param.aux.z = 0;
        set_mma_param(param);
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
        param.aux.x = 0;
        param.aux.y = 0;
        param.aux.z = 0;
        set_mma_param(param);
    }

  public:
    DslashCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                 cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y,
                 const GaugeField &X, double kappa, int parity, MemoryLocation *halo_location,
                 const ColorSpinorField &halo) :
      TunableKernel(out[0].Location()),
      out(out),
      inA(inA),
      inB(inB),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      nParity(out[0].SiteSubset()),
      nSrc(out[0].Ndim() == 5 ? out[0].X(4) : 1),
      halo(halo),
      color_col_stride(-1)
    {
      strcpy(vol, out[0].VolString().c_str());
      strcpy(aux, (std::string("policy_kernel,") + vol).c_str());
      strcat(aux, comm_dim_partitioned_string());

      switch(type) {
      case DSLASH_INTERIOR: strcat(aux,",interior"); break;
      case DSLASH_EXTERIOR: strcat(aux,",exterior"); break;
      case DSLASH_FULL:     strcat(aux,",full"); break;
      }

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      if (doHalo<type>()) {
        char label[15] = ",halo=";
        for (int dim=0; dim<4; dim++) {
          for (int dir=0; dir<2; dir++) {
            label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : halo_location[2*dim+dir] == Device ? '1' : halo_location[2*dim+dir] == Host ? '2' : '3';
          }
        }
        label[14] = '\0';
        strcat(aux,label);
      }
      strcat(aux, ",mma");
      strcat(aux, mma_t::get_type_name().c_str());

      if (dslash) { strcat(aux, ",dslash"); }
      if (clover) { strcat(aux, ",clover"); }

      strcat(aux, ",n_rhs=");
      char rhs_str[8];
      i32toa(rhs_str, out[0].Nvec());
      strcat(aux, rhs_str);

      apply(device::get_default_stream());
    }

    // using mma_t = smma::smma_t<mma::bfloat16, 8, 1, 1>;  // 3xBF16
    // using mma_t = smma::smma_t<mma::tfloat32, 4, 1, 1>;  // 3xTF32
    // using mma_t = simt::simt_t<float, 8, 4, 2, 2>;       // SIMT
    // using mma_t = hmma::hmma_tfloat32_t<4, 1, 1>;        // 1xTF32
    // using mma_t = mma::smma_half_t;                      // 3xFP16
    // using mma_t = mma::hmma_t;                           // 1xFP16
    using mma_t = typename mma::smma_dispatch<yFloat>::type;
    static constexpr int n_atom_size = mma_t::MMA_N;
    static constexpr int m_atom_size = mma_t::MMA_M;

    void set_mma_param(TuneParam &tp) const {
      tp.block.x = 1;
      tp.block.y = Ns * Nc / (1 << tp.aux.x);
      tp.block.z = 8;

      if (out[0].Nvec() % n_atom_size != 0) {
        errorQuda("out[0].Nvec() %% n_atom_size != 0");
      }
      int bN = n_atom_size * get_int_factor_array(out[0].Nvec() / n_atom_size)[tp.aux.y];
      if (out[0].Nvec() % bN != 0) {
        errorQuda("Invalid bN.");
      }

      if ((Ns * Nc) % m_atom_size != 0) {
        errorQuda("(Ns * Nc) %% m_atom_size != 0");
      }
      int bM = m_atom_size * get_int_factor_array((Ns * Nc) / m_atom_size)[tp.aux.z];
      if ((Ns * Nc) % bM != 0) {
        errorQuda("Invalid bM");
      }

      tp.grid = dim3(out[0].SiteSubset() * out[0].VolumeCB(), (Ns * Nc) / bM, out[0].Nvec() / bN);
      tp.set_max_shared_bytes = true;

      int shared_bytes = mma::shared_memory_bytes<mma_t>(bM, bN, Ns * Nc);
      tp.shared_bytes = shared_bytes;
    }

    template <int nVec, int bN, int bM, int block_y, int block_z>
    void launch_mma(TuneParam &tp, const qudaStream_t &stream) {
      using Arg = DslashCoarseMmaArg<mma_t, dslash, clover, dagger, type, Float, yFloat, ghostFloat, Ns, Nc, nVec, bN, bM, block_y, block_z>;
      Arg arg(out[0], inA[0], inB[0], Y, X, (Float)kappa, parity, halo);
      tp.set_max_shared_bytes = true;
      launch_cuda<CoarseDslashMma>(tp, stream, arg);
    }

    template <int nVec, int bN, int block_y, int block_z, size_t d, size_t... Ds>
    void launch_mma_span_m(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>) {
      if (tp.aux.z == d) {
        constexpr IntFactorArray<(Ns * Nc) / m_atom_size> a;
        launch_mma<nVec, bN, a[d] * m_atom_size, block_y, block_z>(tp, stream);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_m<nVec, bN, block_y, block_z>(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.y.");
        }
      }
    }

    template <int nVec, int block_y, int block_z, size_t d, size_t... Ds>
    void launch_mma_span_n(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<d, Ds...>) {
      if (tp.aux.y == d) {
        constexpr IntFactorArray<nVec / n_atom_size> a;
        std::make_index_sequence<IntFactorArray<(Ns * Nc) / m_atom_size>().size()> xt;
        launch_mma_span_m<nVec, a[d] * n_atom_size, block_y, block_z>(tp, stream, xt);
      } else {
        if constexpr (sizeof...(Ds) > 0) {
          launch_mma_span_n<nVec, block_y, block_z>(tp, stream, std::index_sequence<Ds...>());
        } else {
          errorQuda("Invalid tp.aux.y.");
        }
      }
    }

    template <int nVec>
    void launch_mma(TuneParam &tp, const qudaStream_t &stream) {
      std::make_index_sequence<IntFactorArray<nVec / n_atom_size>().size()> xt;

      switch (tp.aux.x) {
        case 0: launch_mma_span_n<nVec, Ns * Nc / 1, 8>(tp, stream, xt); break;
        case 1: launch_mma_span_n<nVec, Ns * Nc / 2, 8>(tp, stream, xt); break;
        case 2: launch_mma_span_n<nVec, Ns * Nc / 4, 8>(tp, stream, xt); break;
        default: errorQuda("tp.aux.x = %d not supported", tp.aux.x);
      }
    }

    template <int...> struct IntList { };

    template <int nVec, int... N>
    void launch_mma_span_nVec(TuneParam &tp, const qudaStream_t &stream, IntList<nVec, N...>) {
      if (out[0].Nvec() == nVec) {
        launch_mma<nVec>(tp, stream);
      } else {
        IntList<N...> nVecs;
        if constexpr (sizeof...(N) > 0) {
          launch_mma_span_nVec(tp, stream, nVecs);
        } else {
          errorQuda("nVec = %d not instantiated\n", out[0].Nvec());
        }
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        IntList<@QUDA_MULTIGRID_MRHS_LIST@> nVecs;
        launch_mma_span_nVec(tp, stream, nVecs);
    }

  };

} // namespace quda
