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

#include <expand_list.hpp>
#include <utility>
#include <tunable_kernel.h>

#include <device.hpp>
#include <mma_tensor_op/smma_m16n8k8_sm70.cuh>

namespace quda
{

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, int Nc, bool dslash, bool clover, bool dagger,
            DslashType type, int nVec>
  class DslashCoarseMma : public TunableKernel
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
    const ColorSpinorField &halo;

    const int max_color_col_stride = 8;
    mutable int color_col_stride;
    mutable int dim_threads;

    // using mma_t = smma::smma_t<mma::bfloat16, 8, 1, 1>;  // 3xBF16
    // using mma_t = smma::smma_t<mma::tfloat32, 4, 1, 1>;  // 3xTF32
    // using mma_t = smma::smma_x_t<mma::half, 8, 1, 1>;    // 3xFP16 - m16n8k8 variant for sm70
    // using mma_t = hmma::hmma_tfloat32_t<4, 1, 1>;        // 1xTF32
    // using mma_t = mma::smma_half_t;                      // 3xFP16
    // using mma_t = mma::hmma_t;                           // 1xFP16
#if (__COMPUTE_CAPABILITY__ >= 800)
    using mma_t = hmma::hmma_tfloat32_t<4, 1, 1>;
#else
    using mma_t = mma::hmma_t;
#endif
    static constexpr int n_atom_size = mma_t::MMA_N;
    static constexpr int m_atom_size = mma_t::MMA_M;
    static constexpr int k_atom_size = Ns * Nc / 2;

    static constexpr int n = nVec;
    static constexpr int m = Ns * Nc;
    static constexpr int k = Ns * Nc;
    static constexpr int block_atom_size = Ns * Nc / (Nc > 64 ? 8 : 4);
    static constexpr int block_limit = Ns * Nc / (Nc > 64 ? 2 : 1);

    using this_t = DslashCoarseMma<Float, yFloat, ghostFloat, Ns, Nc, dslash, clover, dagger, type, nVec>;
    expand_aux_t<this_t, block_limit, block_atom_size, n, n_atom_size, m, m_atom_size, k, k_atom_size> expand;

    long long flops() const
    {
      return ((dslash * 2 * nDim + clover * 1) * (8 * Ns * Nc * Ns * Nc) - 2 * Ns * Nc) * nParity
        * static_cast<long long>(out.VolumeCB()) * out.size() * out[0].Nvec();
    }

    long long bytes() const
    {
      return (dslash || clover) * out.Bytes() + dslash * 8 * inA.Bytes() + clover * inB.Bytes()
        + (nParity * (dslash * Y.Bytes() * Y.VolumeCB() / (2 * Y.Stride()) + clover * X.Bytes() / 2)) * out.size();
    }

    unsigned int sharedBytesPerThread() const { return 0; }

    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions
    unsigned int minThreads() const { return X.VolumeCB(); }

    /**
       @param Helper function to check that the present launch parameters are valid
    */
    bool checkParam(const TuneParam &param) const { return true; }

    bool advanceTuneParam(TuneParam &param) const
    {
      return expand.advance_aux(param);
    }

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
    DslashCoarseMma(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                    cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, double kappa,
                    int parity, MemoryLocation *halo_location, const ColorSpinorField &halo) :
      TunableKernel(out[0].Location()),
      out(out),
      inA(inA),
      inB(inB),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      halo(halo),
      color_col_stride(-1),
      expand(*this)
    {
      strcpy(vol, out.VolString().c_str());
      strcpy(aux, (std::string("policy_kernel,") + vol).c_str());
      strcat(aux, comm_dim_partitioned_string());

      switch (type) {
      case DSLASH_INTERIOR: strcat(aux, ",interior"); break;
      case DSLASH_EXTERIOR: strcat(aux, ",exterior"); break;
      case DSLASH_FULL: strcat(aux, ",full"); break;
      }

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      if (doHalo<type>()) {
        char label[15] = ",halo=";
        for (int dim = 0; dim < 4; dim++) {
          for (int dir = 0; dir < 2; dir++) {
            label[2 * dim + dir + 6] = !comm_dim_partitioned(dim) ? '0' :
              halo_location[2 * dim + dir] == Device              ? '1' :
              halo_location[2 * dim + dir] == Host                ? '2' :
                                                                    '3';
          }
        }
        label[14] = '\0';
        strcat(aux, label);
      }
      strcat(aux, ",mma");
      strcat(aux, mma_t::get_type_name().c_str());

      if (dslash) { strcat(aux, ",dslash"); }
      if (clover) { strcat(aux, ",clover"); }

      strcat(aux, ",n_rhs=");
#ifdef USE_TENSOR_MEMORY_ACCELERATOR
      strcat(aux, ",use_tma");
#endif
      char rhs_str[16];
      i32toa(rhs_str, out[0].Nvec());
      strcat(aux, rhs_str);

      apply(device::get_default_stream());
    }

    static constexpr int shared_bytes_per_block(int bM, int bN, int bK)
    {
      int bytes = mma::shared_memory_bytes<mma_t>(bM, bN, bK) + (bM + mma::get_tmp_pad()) * (bK + mma::get_tmp_pad()) * 2 * sizeof(yFloat)
        + (bK + mma::get_tmp_pad()) * (bN + mma::get_tmp_pad()) * 2 * sizeof(Float);
#ifdef USE_TENSOR_MEMORY_ACCELERATOR
      return bytes + sizeof(barrier_t);
#else
      return bytes;
#endif
    }

    bool set_mma_param(TuneParam &tp) const
    {
      tp.block.x = 1;
      tp.block.y = expand.get_x(tp);
      tp.block.z = 8;

      if (out[0].Nvec() % n_atom_size != 0) { errorQuda("out[0].Nvec() %% n_atom_size != 0"); }
      int bN = expand.get_y(tp);
      if (out[0].Nvec() % bN != 0) { errorQuda("Invalid bN."); }

      if ((Ns * Nc) % m_atom_size != 0) { errorQuda("(Ns * Nc) %% m_atom_size != 0"); }
      int bM = expand.get_z(tp);
      if ((Ns * Nc) % bM != 0) { errorQuda("Invalid bM"); }

      tp.grid = dim3(out.SiteSubset() * out.VolumeCB(), (Ns * Nc) / bM, out[0].Nvec() / bN);
      tp.set_max_shared_bytes = true;

      if ((Ns * Nc) % k_atom_size != 0) { errorQuda("(Ns * Nc) %% k_atom_size != 0"); }
      int bK = expand.get_w(tp);
      if ((Ns * Nc) % bK != 0) { errorQuda("Invalid bK"); }
      int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      tp.shared_bytes = shared_bytes;

      return shared_bytes <= device::maximum_dynamic_shared_memory();
    }

    template <int block_y, int bN, int bM, int bK>
    void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      constexpr int shared_bytes = shared_bytes_per_block(bM, bN, bK);
      if constexpr (shared_bytes <= device::maximum_dynamic_shared_memory()) {
        constexpr int block_z = 8;
        using Arg = DslashCoarseMmaArg<mma_t, dslash, clover, dagger, type, Float, yFloat, ghostFloat, Ns, Nc, nVec, bN,
                                       bM, bK, block_y, block_z>;
        Arg arg(out[0], inA[0], inB[0], Y, X, (Float)kappa, parity, halo);
        tp.set_max_shared_bytes = true;
        launch_cuda<CoarseDslashMma>(tp, stream, arg);
      } else {
        errorQuda("Using too many shared memory bytes per block: %d", shared_bytes);
      }
    }

    void launch_mma(TuneParam &tp, const qudaStream_t &stream)
    {
      expand.expand(tp, stream);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch_mma(tp, stream);
    }
  };

} // namespace quda
