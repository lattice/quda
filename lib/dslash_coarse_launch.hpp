#pragma once

#include <cstring>
#include <cstdlib>
#include <sstream>
#include <vector>
#include "dslash_coarse_decl.hpp"
#include <tune_quda.h>
#include <comm_quda.h>
#include <worker.h>
#include <uint_to_char.h>
#include <instantiate.h>
#include <multigrid.h>

namespace quda
{

  namespace dslash
  {
    extern Worker *aux_worker;
    void shmem_signal_wait_all();
  } // namespace dslash

  enum class DslashCoarsePolicy {
    DSLASH_COARSE_BASIC,                   // stage both sends and recvs in host memory using memcpys
    DSLASH_COARSE_ZERO_COPY_PACK,          // zero copy write pack buffers
    DSLASH_COARSE_ZERO_COPY_READ,          // zero copy read halos in dslash kernel
    DSLASH_COARSE_ZERO_COPY,               // full zero copy
    DSLASH_COARSE_SHMEM,                   // non overlapping shmem exchange
    DSLASH_COARSE_SHMEM_OVERLAP,           // overlapping shmem exchange
    DSLASH_COARSE_GDR_SEND,                // GDR send
    DSLASH_COARSE_GDR_RECV,                // GDR recv
    DSLASH_COARSE_GDR,                     // full GDR
    DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV, // zero copy write and GDR recv
    DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ, // GDR send and zero copy read
    DSLASH_COARSE_POLICY_DISABLED
  };

  template <bool dagger, int coarseColor, bool use_mma, int nVec>
  void apply_coarse_store(QudaPrecision precision, QudaPrecision y_prec, QudaPrecision halo_prec,
                          cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                          cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                          real_t kappa, int parity, bool dslash, bool clover, DslashType type,
                          MemoryLocation *halo_location, const ColorSpinorField &halo)
  {
    auto call = [&]<typename Float, typename yFloat, typename ghostFloat>() {
      if constexpr (use_mma) {
        ApplyCoarseMma_t<dagger, coarseColor, nVec, Float, yFloat, ghostFloat>(
          out, inA, inB, Y, X, kappa, parity, dslash, clover, type, halo_location, halo);
      } else {
        ApplyCoarse_t<dagger, coarseColor, Float, yFloat, ghostFloat>(out, inA, inB, Y, X, kappa, parity, dslash, clover,
                                                                      type, halo_location, halo);
      }
    };

    if (precision == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled_multigrid_double()) {
        if (y_prec != QUDA_DOUBLE_PRECISION) errorQuda("Y Precision %d not supported", y_prec);
        if (halo_prec != QUDA_DOUBLE_PRECISION)
          errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_prec,
                    precision, y_prec);
        call.template operator()<double, double, double>();
      } else {
        errorQuda("Double precision multigrid has not been enabled");
      }
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if (y_prec == QUDA_SINGLE_PRECISION) {
        if (halo_prec == QUDA_SINGLE_PRECISION) {
          call.template operator()<float, float, float>();
        } else {
          errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_prec,
                    precision, y_prec);
        }
      } else if (y_prec == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
          if (halo_prec == QUDA_HALF_PRECISION) {
            call.template operator()<float, short, short>();
          } else if (halo_prec == QUDA_QUARTER_PRECISION) {
            if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) {
              call.template operator()<float, short, int8_t>();
            } else {
              errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
            }
          } else {
            errorQuda("Halo precision %d not supported with field precision %d and link precision %d", halo_prec,
                      precision, y_prec);
          }
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        }
      } else {
        errorQuda("Unsupported precision %d", y_prec);
      }
    } else {
      errorQuda("Unsupported precision %d", precision);
    }
  }

  template <bool dagger, int coarseColor, bool use_mma_, int nVec> struct DslashCoarseLaunch {

    constexpr static bool use_mma = use_mma_;

    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &inA;
    cvector_ref<const ColorSpinorField> &inB;
    const ColorSpinorField &halo;
    const GaugeField &Y;
    const GaugeField &X;
    real_t kappa;
    int parity;
    bool dslash;
    bool clover;
    const int *commDim;
    const QudaPrecision halo_precision;
    static constexpr bool enable_coarse_shmem_overlap() { return false; }

    DslashCoarseLaunch(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                       cvector_ref<const ColorSpinorField> &inB, const ColorSpinorField &halo, const GaugeField &Y,
                       const GaugeField &X, real_t kappa, int parity, bool dslash, bool clover, const int *commDim,
                       QudaPrecision halo_precision) :
      out(out),
      inA(inA),
      inB(inB),
      halo(halo),
      Y(Y),
      X(X),
      kappa(kappa),
      parity(parity),
      dslash(dslash),
      clover(clover),
      commDim(commDim),
      halo_precision(halo_precision == QUDA_INVALID_PRECISION ? Y.Precision() : halo_precision)
    {
    }

    /**
       @brief Execute the coarse dslash using the given policy
     */
    inline void operator()(DslashCoarsePolicy policy)
    {
      if (inA[0].data() == out[0].data()) errorQuda("Aliasing pointers");

      QudaPrecision precision = checkPrecision(out[0], inA[0], inB[0]);
      checkPrecision(Y, X);
      checkLocation(out[0], inA[0], inB[0], Y, X);

      int comm_sum = 4;
      if (commDim)
        for (int i = 0; i < 4; i++) comm_sum -= (1 - commDim[i]);
      if (comm_sum != 4 && comm_sum != 0) errorQuda("Unsupported comms %d", comm_sum);
      bool comms = comm_sum;
      int shmem = 0;

      MemoryLocation pack_destination[2 * QUDA_MAX_DIM];
      MemoryLocation halo_location[2 * QUDA_MAX_DIM];
      bool gdr_send = false;
      bool gdr_recv = false;
      if (policy == DslashCoarsePolicy::DSLASH_COARSE_SHMEM || policy == DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP) {
        for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) {
          pack_destination[i] = Shmem;
          halo_location[i] = Device;
        }
        shmem = 1;
      } else {
        for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) {
          pack_destination[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK
                                 || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
                                 || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ?
            Host :
            Device;
          halo_location[i] = (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ
                              || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
                              || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ?
            Host :
            Device;
        }
        gdr_send = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR
                    || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ);
        gdr_recv = (policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR
                    || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV);
      }
      bool p2p_enabled = comm_peer2peer_enabled_global();
      if (policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY
          || policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV
          || policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ)
        comm_enable_peer2peer(false);

      auto launch_kernels = [&](DslashType type) {
        apply_coarse_store<dagger, coarseColor, use_mma, nVec>(precision, Y.Precision(), halo_precision, out, inA, inB,
                                                               Y, X, kappa, parity, dslash, clover, type,
                                                               halo_location, halo);
      };

      if (policy != DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP) {
        if (dslash && comm_partitioned() && comms) {
          const int nFace = 1;
          halo.exchangeGhost((QudaParity)(inA.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? (1 - parity) : 0), nFace, dagger,
                             pack_destination, halo_location, gdr_send, gdr_recv, halo_precision, shmem, inA);
        }

        if (dslash::aux_worker) dslash::aux_worker->apply(device::get_default_stream());
        launch_kernels(comms ? DSLASH_FULL : DSLASH_INTERIOR);
      } else if constexpr (DslashCoarseLaunch<dagger, coarseColor, use_mma, nVec>::enable_coarse_shmem_overlap()) {
#ifdef NVSHMEM_COMMS
        if (dslash && comm_partitioned() && comms) {
          const int nFace = 1;
          shmem += 2;
          halo.exchangeGhost((QudaParity)(inA.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? (1 - parity) : 0), nFace, dagger,
                             pack_destination, halo_location, gdr_send, gdr_recv, halo_precision, shmem, inA);
        }
        launch_kernels(DSLASH_INTERIOR);
        if (dslash::aux_worker) dslash::aux_worker->apply(device::get_default_stream());
        if (dslash && comm_partitioned() && comms) {
          quda::dslash::shmem_signal_wait_all();
          launch_kernels(DSLASH_EXTERIOR);
        }
#else
        errorQuda("NVSHMEM policy called but NVSHMEM not enabled.");
#endif
      }
      if (dslash && comm_partitioned() && comms) inA[0].bufferIndex = (1 - inA[0].bufferIndex);

      comm_enable_peer2peer(p2p_enabled);
    }
  };

  template <typename Launch> class DslashCoarsePolicyTune : public Tunable {

    static inline bool dslash_init = false;
    static inline int first_active_policy = static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED);
    static inline char policy_string[TuneKey::aux_n] = {};
    static inline std::vector<DslashCoarsePolicy> policies
      = {static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED), DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED};

    static void enable_policy(DslashCoarsePolicy p) { policies[static_cast<std::size_t>(p)] = p; }

    Launch &dslash;

    bool tuneGridDim() const { return false; }
    bool tuneAuxDim() const { return true; }
    static constexpr bool enable_coarse_shmem_overlap = Launch::enable_coarse_shmem_overlap();

  public:
    DslashCoarsePolicyTune(Launch &dslash) : dslash(dslash)
    {
      if (!dslash_init) {

        static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_COARSE_POLICY");

        if (dslash_policy_env) {
          std::stringstream policy_list(dslash_policy_env);

          int policy_;
          while (policy_list >> policy_) {
            DslashCoarsePolicy dslash_policy = static_cast<DslashCoarsePolicy>(policy_);

            if ((dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND
                 || dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV
                 || dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR
                 || dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV
                 || dslash_policy == DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ)
                && !comm_gdr_enabled()) {
              errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", static_cast<int>(dslash_policy));
            }

            enable_policy(dslash_policy);
            first_active_policy = policy_ < first_active_policy ? policy_ : first_active_policy;
            if (policy_list.peek() == ',') policy_list.ignore();
          }
          if (first_active_policy == static_cast<int>(DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED))
            errorQuda("No valid policy found in QUDA_ENABLE_DSLASH_COARSE_POLICY");
        } else {
          first_active_policy = 0;
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_BASIC);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_READ);
          enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY);
          if (comm_nvshmem_enabled()) {
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_SHMEM);
            if constexpr (enable_coarse_shmem_overlap) enable_policy(DslashCoarsePolicy::DSLASH_COARSE_SHMEM_OVERLAP);
          }
          if (comm_gdr_enabled()) {
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_RECV);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV);
            enable_policy(DslashCoarsePolicy::DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ);
          }
        }

        strcat(policy_string, ",pol=");
        for (int i = 0; i < (int)DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED; i++) {
          strcat(policy_string, (int)policies[i] == i ? "1" : "0");
        }

        dslash_init = true;
      }

      strcpy(aux, "policy,");
      if (dslash.dslash) strcat(aux, "dslash");
      strcat(aux, dslash.clover ? "clover," : ",");
      strcat(aux, dslash.inA.AuxString().c_str());
      strcat(aux, ",gauge_prec=");

      char prec_str[16];
      i32toa(prec_str, dslash.Y.Precision());
      strcat(aux, prec_str);
      strcat(aux, ",halo_prec=");
      i32toa(prec_str, dslash.halo_precision);
      strcat(aux, prec_str);
      strcat(aux, comm_dim_partitioned_string(dslash.commDim));
      strcat(aux, comm_dim_topology_string());
      strcat(aux, comm_config_string());
      strcat(aux, policy_string);

      int comm_sum = 4;
      if (dslash.commDim)
        for (int i = 0; i < 4; i++) comm_sum -= (1 - dslash.commDim[i]);
      strcat(aux, comm_sum ? ",full" : ",interior");

      if (Launch::use_mma) { strcat(aux, ",mma"); }
      strcat(aux, ",n_rhs=");
      char rhs_str[16];
      i32toa(rhs_str, dslash.out.size() * dslash.out[0].Nvec());
      strcat(aux, rhs_str);

#ifdef QUDA_FAST_COMPILE_DSLASH
      strcat(aux, ",fast_compile");
#endif

      if (!tuned()) {
        disableProfileCount();
        for (auto &i : policies)
          if (i != DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) dslash(i);
        enableProfileCount();
        setPolicyTuning(true);
      }
    }

    virtual ~DslashCoarsePolicyTune() { setPolicyTuning(false); }

    inline void apply(const qudaStream_t &)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (tp.aux.x >= (int)policies.size()) errorQuda("Requested policy that is outside of range");
      if (policies[tp.aux.x] == DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED)
        errorQuda("Requested policy is disabled");
      dslash(policies[tp.aux.x]);
    }

    bool advanceAux(TuneParam &param) const
    {
      while ((unsigned)param.aux.x < policies.size() - 1) {
        param.aux.x++;
        if (policies[param.aux.x] != DslashCoarsePolicy::DSLASH_COARSE_POLICY_DISABLED) return true;
      }
      param.aux.x = 0;
      return false;
    }

    bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.aux = make_int4(first_active_policy, 0, 0, 0);
    }

    void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.aux = make_int4(first_active_policy, 0, 0, 0);
    }

    TuneKey tuneKey() const { return TuneKey(dslash.inA.VolString().c_str(), typeid(*this).name(), aux); }

    long long flops() const
    {
      int nDim = 4;
      int Ns = dslash.inA.Nspin();
      int Nc = dslash.inA.Ncolor() / dslash.inA[0].Nvec();
      int nParity = dslash.inA.SiteSubset();
      long long volumeCB = dslash.inA.VolumeCB();
      return ((dslash.dslash * 2 * nDim + dslash.clover * 1) * (8 * Ns * Nc * Ns * Nc) - 2 * Ns * Nc) * nParity
        * volumeCB * dslash.out.size() * dslash.out[0].Nvec();
    }

    long long bytes() const
    {
      int nParity = dslash.inA.SiteSubset();
      return (dslash.dslash || dslash.clover) * dslash.out.Bytes() + dslash.dslash * 8 * dslash.inA.Bytes()
        + dslash.clover * dslash.inB.Bytes()
        + (nParity
           * (dslash.dslash * dslash.Y.Bytes() * dslash.Y.VolumeCB() / (2 * dslash.Y.Stride())
              + dslash.clover * dslash.X.Bytes() / 2))
        * dslash.out.size() * dslash.out[0].Nvec();
    }
  };

  template <bool dagger, int coarseColor>
  void ApplyCoarse(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                   cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                   int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      if (inA.size() > get_max_multi_rhs()) {
        ApplyCoarse<dagger, coarseColor>(
          {out.begin(), out.begin() + out.size() / 2}, {inA.begin(), inA.begin() + inA.size() / 2},
          {inB.begin(), inB.begin() + inB.size() / 2}, Y, X, kappa, parity, dslash, clover, commDim, halo_precision);
        ApplyCoarse<dagger, coarseColor>({out.begin() + out.size() / 2, out.end()},
                                         {inA.begin() + inA.size() / 2, inA.end()},
                                         {inB.begin() + inB.size() / 2, inB.end()}, Y, X, kappa, parity, dslash, clover,
                                         commDim, halo_precision);
        return;
      }

      auto halo = ColorSpinorField::create_comms_batch(inA, 1, false);
      DslashCoarseLaunch<dagger, coarseColor, false, 1> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash, clover,
                                                               commDim, halo_precision);
      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

  template <bool dagger, int coarseColor, int nVec>
  void ApplyCoarseMma(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                      cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X, real_t kappa,
                      int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      auto halo = ColorSpinorField::create_comms_batch(inA, 1, false);
      DslashCoarseLaunch<dagger, coarseColor, true, nVec> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                                                 clover, commDim, halo_precision);
      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
