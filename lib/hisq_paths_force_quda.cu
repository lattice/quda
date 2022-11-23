#include <utility>
#include <quda_internal.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <tune_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/hisq_paths_force.cuh>

namespace quda {

  namespace fermion_force {

    typedef std::reference_wrapper<GaugeField> GaugeField_ref;

    template <typename Arg> class OneLinkForce : public TunableKernel3D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      OneLinkForce(Arg &arg, const GaugeField &link, const GaugeField &force) :
        TunableKernel3D(link, 2, 4),
        arg(arg),
        force(force),
        link(link)
      {
        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<OneLinkTerm>(tp, stream, arg);
      }

      void preTune() { force.backup(); }
      void postTune() { force.restore(); }

      long long flops() const {
        // all four directions are handled in one kernel
        long long adds_per_site = 4ll;
        long long rescales_per_site = 4ll;
        return 2 * arg.threads.x * ( 18ll * adds_per_site + 18ll * rescales_per_site );
      }

      long long bytes() const { return 2*4*arg.threads.x*( arg.oProd.Bytes() + 2*arg.force.Bytes() ); }
    };

    template <typename Arg> class AllThreeAllLepageLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &p3;
      const GaugeField &pMu_2;
      const GaugeField &link;
      const bool has_lepage;
      unsigned int minThreads() const override { return arg.threads.x; }

      unsigned int sharedBytesPerThread() const override { return sizeof(typename Arg::Link); }

      unsigned int maxSharedBytesPerBlock() const override { return maxDynamicSharedBytesPerBlock(); }

    public:
      AllThreeAllLepageLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, int mu_next,
                         const PathCoefficients<typename Arg::real> &act_path_coeff, const GaugeField &force,
                         const GaugeField &p3, const GaugeField &pMu_2) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        p3(p3),
        pMu_2(pMu_2),
        link(link),
        has_lepage(act_path_coeff.lepage != 0.)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.mu_next = mu_next;
        arg.compute_lepage = has_lepage ? 1 : 0;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        strcat(aux, ",sig=");
        u32toa(aux2, arg.sig);
        strcat(aux, aux2);
        if (mu != MU_IGNORED) {
          strcat(aux, ",mu=");
          u32toa(aux2, arg.mu);
          strcat(aux, aux2);
          if (has_lepage) {
            strcat(aux, ",lepage");
          }
        }
        if (mu_next != MU_NEXT_IGNORED) {
          strcat(aux, ",mu_next=");
          u32toa(aux2, arg.mu_next);
          strcat(aux, aux2);
        }

        apply(device::get_default_stream());
      }

      template <int sig, int mu, int mu_next>
      void instantiate(TuneParam &tp, const qudaStream_t &stream) {
        if (has_lepage) launch<AllThreeAllLepageLink>(tp, stream, FatLinkParam<Arg, sig, mu, mu_next, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_YES>(arg));
        else launch<AllThreeAllLepageLink>(tp, stream, FatLinkParam<Arg, sig, mu, mu_next, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_NO>(arg));
      }

      template <int sig, int mu>
      void instantiate(int mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (mu_next == MU_NEXT_IGNORED) instantiate<sig, mu, MU_NEXT_IGNORED>(tp, stream);
        else if (goes_forward(mu_next)) instantiate<sig, mu, MU_NEXT_POSITIVE>(tp, stream);
        else instantiate<sig, mu, MU_NEXT_NEGATIVE>(tp, stream);
      }

      template <int sig>
      void instantiate(int mu, int mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (mu == MU_IGNORED) instantiate<sig, MU_IGNORED>(mu_next, tp, stream);
        else if (goes_forward(mu)) instantiate<sig, MU_POSITIVE>(mu_next, tp, stream);
        else instantiate<sig, MU_NEGATIVE>(mu_next, tp, stream);
      }

      void instantiate(int sig, int mu, int mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(sig)) instantiate<SIG_POSITIVE>(mu, mu_next, tp, stream);
        else instantiate<SIG_NEGATIVE>(mu, mu_next, tp, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true; // maximize the shared memory pool
        instantiate(arg.sig, arg.mu, arg.mu_next, tp, stream);
      }

      void preTune() {
        force.backup();
        p3.backup();
        pMu_2.backup();
      }

      void postTune() {
        force.restore();
        p3.restore();
        pMu_2.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;
        // Three link side link, all Lepage
        if (arg.mu != -1) {
          adds_per_site += 1ll;
          rescales_per_site += 1ll;
          if (has_lepage) {
            multiplies_per_site += 6ll;
            adds_per_site += 2ll;
            rescales_per_site += 2ll;
            if (goes_forward(arg.sig)) {
              multiplies_per_site += 2ll;
              adds_per_site += 1ll;
              rescales_per_site += 1ll;
            }
          }
        }
        // Three link middle link
        if (arg.mu_next != -1) {
          multiplies_per_site += 2ll;
          if (goes_forward(arg.sig)) {
            multiplies_per_site += 2ll;
            adds_per_site += 1ll;
            rescales_per_site += 1ll;
          }
        }
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        long long bytes_per_site = ( arg.link.Bytes() );
        // Three link side link, all Lepage
        if (arg.mu != -1) {
          bytes_per_site += ( arg.p3.Bytes() + 2 * arg.force.Bytes() );
          if (has_lepage) {
            bytes_per_site += ( 5 * arg.link.Bytes() + 2 * arg.pMu.Bytes() );
            if (goes_forward(arg.sig)) {
              bytes_per_site += ( arg.link.Bytes() + 2 * arg.force.Bytes() );
            }
          }
        }
        // Three link middle link
        if (arg.mu_next != -1) {
          bytes_per_site += ( arg.link.Bytes() + arg.oProd.Bytes() + arg.pMu_2.Bytes() + arg.p3.Bytes() );
          if (goes_forward(arg.sig)) {
            bytes_per_site += ( arg.link.Bytes() + 2 * arg.force.Bytes() );
          }
        }
        // logic correction
        if (arg.mu_next == -1 && !has_lepage)
          bytes_per_site -= ( arg.link.Bytes() );
        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Arg> class AllFiveAllSevenLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &p5;
      const GaugeField &pNuMu_next;
      const GaugeField &qNuMu_next;
      const GaugeField &link;
      unsigned int minThreads() const override { return arg.threads.x; }

      unsigned int sharedBytesPerThread() const override { return (goes_forward(arg.sig) ? 3 : 2) * sizeof(typename Arg::Link); }

      unsigned int maxSharedBytesPerBlock() const override { return maxDynamicSharedBytesPerBlock(); }

    public:
      AllFiveAllSevenLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, int nu,
                   int nu_next, const GaugeField &force, const GaugeField &shortP,
                   const GaugeField &P5, const GaugeField &pNuMu_next, const GaugeField &qNuMu_next) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        p5(P5),
        pNuMu_next(pNuMu_next),
        qNuMu_next(qNuMu_next),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.nu = nu;
        arg.nu_next = nu_next;

        if (nu != NU_IGNORED) {
          // rho is the "last" direction that's orthogonal to sig, mu, and nu
          // it's only relevant for the side 5 + All-7 part of the calculation
          for (arg.rho = 0; arg.rho < 4; arg.rho++) {
            if (arg.rho != pos_dir(sig) && arg.rho != pos_dir(mu) && arg.rho != pos_dir(nu))
              break;
          }
        } else {
          arg.rho = -1;
        }

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        strcat(aux, ",sig=");
        u32toa(aux2, arg.sig);
        strcat(aux, aux2);
        strcat(aux, ",mu=");
        u32toa(aux2, arg.mu);
        strcat(aux, aux2);
        if (nu != NU_IGNORED) {
          strcat(aux, ",nu=");
          u32toa(aux2, arg.nu);
          strcat(aux, aux2);
        }
        if (nu_next != NU_NEXT_IGNORED) {
          strcat(aux, ",nu_next=");
          u32toa(aux2, arg.nu_next);
          strcat(aux, aux2);
        }
        
        apply(device::get_default_stream());
      }

      template <int sig, int mu, int nu>
      void instantiate(int nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (nu_next == NU_NEXT_IGNORED) {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, MU_NEXT_IGNORED, nu, NU_NEXT_IGNORED>(arg));
        } else if (goes_forward(nu_next)) {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, MU_NEXT_IGNORED, nu, NU_NEXT_POSITIVE>(arg));
        } else {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, MU_NEXT_IGNORED, nu, NU_NEXT_NEGATIVE>(arg));
        }
      }

      template <int sig, int mu>
      void instantiate(int nu, int nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (nu == NU_IGNORED) {
          instantiate<sig, mu, NU_IGNORED>(nu_next, tp, stream);
        } else if (goes_forward(nu)) {
          instantiate<sig, mu, NU_POSITIVE>(nu_next, tp, stream);
        } else {
          instantiate<sig, mu, NU_NEGATIVE>(nu_next, tp, stream);
        }
      }

      template <int sig>
      void instantiate(int mu, int nu, int nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(mu)) instantiate<sig, MU_POSITIVE>(nu, nu_next, tp, stream);
        else instantiate<sig, MU_NEGATIVE>(nu, nu_next, tp, stream);
      }

      void instantiate(int sig, int mu, int nu, int nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(sig)) instantiate<SIG_POSITIVE>(mu, nu, nu_next, tp, stream);
        else instantiate<SIG_NEGATIVE>(mu, nu, nu_next, tp, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true; // maximize the shared memory pool
        instantiate(arg.sig, arg.mu, arg.nu, arg.nu_next, tp, stream);
      }

      void preTune() {
        force.backup();
        shortP.backup();
        p5.backup();
        pNuMu_next.backup();
        qNuMu_next.backup();
      }

      void postTune() {
        force.restore();
        shortP.restore();
        p5.restore();
        pNuMu_next.restore();
        qNuMu_next.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;

        // SideFiveAllSeven contribution
        if (arg.nu != NU_IGNORED) {
          multiplies_per_site += 12ll;
          adds_per_site += 6ll;
          rescales_per_site += 6ll;
          if (goes_forward(arg.sig)) {
            multiplies_per_site += 4ll;
            adds_per_site += 2ll;
            rescales_per_site += 2ll;
          }
        }

        // MiddleFive contribution
        if (arg.nu_next != NU_NEXT_IGNORED) {
          multiplies_per_site += 3ll;
          if (goes_forward(arg.sig)) {
            multiplies_per_site += 1ll;
            adds_per_site += 1ll;
            rescales_per_site += 1ll;
          }
        }

        return 2*arg.threads.x*(198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        long long bytes_per_site = arg.link.Bytes();
        if (goes_forward(arg.sig)) bytes_per_site += 2 * arg.force.Bytes();

        // SideFiveAllSeven contribution
        if (arg.nu != NU_IGNORED) {
          bytes_per_site += 8 * arg.link.Bytes() + 2 * arg.qNuMu.Bytes() + 2 * arg.pNuMu.Bytes() +
                            arg.p5.Bytes() + 2 * arg.shortP.Bytes() + 4 * arg.force.Bytes();
          if (goes_forward(arg.sig))
            bytes_per_site += arg.qNuMu.Bytes() + arg.pNuMu.Bytes();
        }

        // MiddleFive contribution
        if (arg.nu_next != NU_NEXT_IGNORED) {
          bytes_per_site += 3 * arg.link.Bytes() + arg.pMu.Bytes() + arg.p5.Bytes() +
                            arg.pNuMu_next.Bytes() + arg.qNuMu_next.Bytes();
        }

        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      using real = typename mapper<Float>::type;

      template <bool low_memory>
      void hisqFiveSeven(GaugeField &newOprod, GaugeField &P3, GaugeField_ref &P5, GaugeField_ref &Pnumu, GaugeField_ref &Qnumu,
                         GaugeField_ref &Pnumu_2, GaugeField_ref &Qnumu_2, const GaugeField &Pmu,
                         const GaugeField &link, const PathCoefficients<real> &act_path_coeff, int sig, int mu) {
        if constexpr (low_memory) {
          for (int nu=0; nu < 8; nu++) {
            if (pos_dir(nu) == pos_dir(sig) || pos_dir(nu) == pos_dir(mu)) continue;

            // 5-link: middle link
            // In/out: newOprod
            // Out: P5, Pnumu, Qnumu
            // In: Pmu, link
            // Ignored: Pnumu_2, Qnumu_2
            AllFiveAllSevenLinkArg<Float, nColor, recon> middleFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu_2, Qnumu_2, Pnumu, Qnumu, link, act_path_coeff);
            AllFiveAllSevenLinkForce<decltype(middleFiveLinkArg)> middleFiveLink(middleFiveLinkArg, link, sig, mu, -1, -1, nu, newOprod, P3, P5, Pnumu, Qnumu);

            // All 7 link, 5-link side-link
            // In/out: newOprod, P3 (called shortP)
            // In: P5, Qnumu_2, link
            // Out: none
            // Ignored: Pmu, Pnumu, Qnumu
            AllFiveAllSevenLinkArg<Float, nColor, recon> allSevenSideFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, link, act_path_coeff);
            AllFiveAllSevenLinkForce<decltype(allSevenSideFiveLinkArg)> allSevenSideLinkFive(allSevenSideFiveLinkArg, link, sig, mu, nu, -1, newOprod, P3, P5, Pnumu_2, Qnumu_2);
          } //nu
        } else {
          // optimized, more fused path
          // Uses a "double buffer" for Pnumu/Qnumu

          // unroll the nu loop
          std::vector<int> nu_vals;
          nu_vals.reserve(4);
          for (int nu = 0; nu < 8; nu++) {
            if (pos_dir(nu) == pos_dir(sig) || pos_dir(nu) == pos_dir(mu)) continue;
            nu_vals.emplace_back(nu);
          }

          // first: just MiddleFiveLink
          // In/out: newOprod
          // Out: P5, Pnumu, Qnumu
          // In: Pmu, link
          // Ignored: Pnumu_2, Qnumu_2 (since this is MiddleFive only)
          AllFiveAllSevenLinkArg<Float, nColor, recon> middleFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu_2, Qnumu_2, Pnumu, Qnumu, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(middleFiveLinkArg)> middleFiveArg(middleFiveLinkArg, link, sig, mu, -1, nu_vals[0], newOprod, P3, P5, Pnumu, Qnumu);

          for (int i = 0; i < 3; i++) {
            // next: fully fused kernels
            // In/out: new Oprod, P3 (called shortP), P5
            // In: Pmu, Pnumu, Qnumu, link
            // Out: Pnumu_2, Qnumu_2
            AllFiveAllSevenLinkArg<Float, nColor, recon> allFiveAllSevenLinkArg(newOprod, P3, Pmu, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, link, act_path_coeff);
            AllFiveAllSevenLinkForce<decltype(allFiveAllSevenLinkArg)> allFiveAllSevenLink(allFiveAllSevenLinkArg, link, sig, mu, nu_vals[i], nu_vals[i+1], newOprod, P3, P5, Pnumu_2, Qnumu_2);

            std::swap(Pnumu, Pnumu_2);
            std::swap(Qnumu, Qnumu_2);
          }

          // last: just SideFiveAllSevenLink
          // In/out: newOprod, P3 (called shortP)
          // In: P5, Pnumu, Qnumu, link
          // Out: none
          // Ignored: Pmu, Pnumu_2, Qnumu_2
          AllFiveAllSevenLinkArg<Float, nColor, recon> allSevenSideFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(allSevenSideFiveLinkArg)> allSevenSideFiveLink(allSevenSideFiveLinkArg, link, sig, mu, nu_vals[3], -1, newOprod, P3, P5, Pnumu, Qnumu);
        }
      }

      HisqStaplesForce(GaugeField &P3, GaugeField_ref &Pmu, GaugeField_ref &P5, GaugeField_ref &Pnumu, GaugeField_ref &Qnumu,
                       GaugeField_ref &Pmu_2, GaugeField_ref &Pnumu_2, GaugeField_ref &Qnumu_2,
                       GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {
        PathCoefficients<real> act_path_coeff(path_coeff_array);

        {
          // Out: newOprod
          // In: oprod, link
          OneLinkArg<Float, nColor, recon> arg(newOprod, oprod, link, act_path_coeff);
          OneLinkForce<decltype(arg)> oneLink(arg, link, newOprod);
        }

        constexpr bool low_memory_path = false;

        for (int sig=0; sig<8; sig++) {
          if constexpr (low_memory_path) {
            for (int mu=0; mu<8; mu++) {
              if (pos_dir(mu) == pos_dir(sig)) continue;

              // 3-link: middle link
              // In/out: newOprod
              // Out: (first) Pmu, P3
              // In: oprod, link
              AllThreeAllLepageLinkArg<Float, nColor, recon> middleThreeLinkArg(newOprod, P3, oprod, Pmu, Pmu, link, act_path_coeff);
              AllThreeAllLepageLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig, -1, mu, act_path_coeff, newOprod, P3, Pmu);

              // All 5 and 7 link contributions
              // In/out: newOprod, P3
              // In: Pmu, link
              // Internal only: P5, Pnumu, Qnumu, and the double-buffer flavors
              hisqFiveSeven<low_memory_path>(newOprod, P3, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, Pmu, link, act_path_coeff, sig, mu);

              // Side 3-link, fused with Lepage all link when the lepage coeff != 0.
              // In/out: newOprod
              // In: P3, (second) Pmu, link
              AllThreeAllLepageLinkArg<Float, nColor, recon> allLepageSideThreeLinkArg(newOprod, P3, oprod, Pmu, Pmu, link, act_path_coeff);
              AllThreeAllLepageLinkForce<decltype(allLepageSideThreeLinkArg)> allLepageSideThreeLink(allLepageSideThreeLinkArg, link, sig, mu, -1, act_path_coeff, newOprod, P3, Pmu);
            }//mu
          } else {
            // optimized, more fused path
            // Uses a "double buffer" for Pmu

            // unroll the mu loop
            std::vector<int> mu_vals;
            mu_vals.reserve(6);
            for (int mu = 0; mu < 8; mu++) {
              if (pos_dir(mu) == pos_dir(sig)) continue;
              mu_vals.emplace_back(mu);
            }

            // 3-link: middle link only
            // In/out: newOprod
            // Out: (first) Pmu, P3
            // In: oprod, link
            // Ignored: Pmu_2
            AllThreeAllLepageLinkArg<Float, nColor, recon> middleThreeLinkArg(newOprod, P3, oprod, Pmu_2, Pmu, link, act_path_coeff);
            AllThreeAllLepageLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig, -1, mu_vals[0], act_path_coeff, newOprod, P3, Pmu);

            // All 5 and 7 link contributions
            // In/out: newOprod, P3
            // In: Pmu, link
            // Internal only: P5, Pnumu, Qnumu, and the double-buffer flavors
            hisqFiveSeven<low_memory_path>(newOprod, P3, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, Pmu, link, act_path_coeff, sig, mu_vals[0]);

            for (int i = 0; i < 5; i++) {
              std::swap(Pmu, Pmu_2);

              // Fully fused 3-link and Lepage contributions (when Lepage coeff != 0.)
              // In/out: oProd, P3 (read + overwritten)
              // In: (first) Pmu, oProd, link
              // Out: (second) Pmu
              AllThreeAllLepageLinkArg<Float, nColor, recon> allThreeAllLepageLinkArg(newOprod, P3, oprod, Pmu_2, Pmu, link, act_path_coeff);
              AllThreeAllLepageLinkForce<decltype(allThreeAllLepageLinkArg)> allLepageAllThreeLink(allThreeAllLepageLinkArg, link, sig, mu_vals[i], mu_vals[i+1], act_path_coeff, newOprod, P3, Pmu);

              // All 5 and 7 link contributions, as above
              hisqFiveSeven<low_memory_path>(newOprod, P3, P5, Pnumu, Qnumu, Pnumu_2, Qnumu_2, Pmu, link, act_path_coeff, sig, mu_vals[i+1]);
            }

            std::swap(Pmu, Pmu_2);

            // Side 3-link, fused with Lepage all link when the lepage coeff != 0.
            // In/out: newOprod
            // In: P3, (second) Pmu, link
            // Ignored: (first) Pmu, oProd
            AllThreeAllLepageLinkArg<Float, nColor, recon> allLepageSideThreeLinkArg(newOprod, P3, oprod, Pmu_2, Pmu, link, act_path_coeff);
            AllThreeAllLepageLinkForce<decltype(allLepageSideThreeLinkArg)> allLepageSideThreeLink(allLepageSideThreeLinkArg, link, sig, mu_vals[5], -1, act_path_coeff, newOprod, P3, Pmu);


          }
        }//sig
      }
    };

#ifdef GPU_STAGGERED_DIRAC
    void hisqStaplesForce(GaugeField &newOprod, const GaugeField &oprod, const GaugeField &link, const double path_coeff_array[6])
    {
      checkNative(link, oprod, newOprod);
      checkLocation(newOprod, oprod, link);
      checkPrecision(oprod, link, newOprod);

      // create color matrix fields with zero padding
      GaugeFieldParam gauge_param(link);
      gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      gauge_param.geometry = QUDA_SCALAR_GEOMETRY;
      gauge_param.setPrecision(gauge_param.Precision(), true);

      auto P3 = GaugeField::Create(gauge_param);

      auto Pmu = GaugeField::Create(gauge_param);
      auto P5 = GaugeField::Create(gauge_param);
      auto Pnumu = GaugeField::Create(gauge_param);
      auto Qnumu = GaugeField::Create(gauge_param);

      // need double buffers for these fields to fuse "middle link" terms with
      // subsequent "side link" in a different direction
      auto Pmu_2 = GaugeField::Create(gauge_param);
      auto Pnumu_2 = GaugeField::Create(gauge_param);
      auto Qnumu_2 = GaugeField::Create(gauge_param);

      instantiate<HisqStaplesForce, ReconstructNone>(*P3, GaugeField_ref(*Pmu),
        GaugeField_ref(*P5), GaugeField_ref(*Pnumu), GaugeField_ref(*Qnumu),
        GaugeField_ref(*Pmu_2), GaugeField_ref(*Pnumu_2), GaugeField_ref(*Qnumu_2),
        newOprod, oprod, link, path_coeff_array);

      delete Pmu;
      delete P3;
      delete P5;
      delete Pnumu;
      delete Qnumu;
      delete Pmu_2;
      delete Pnumu_2;
      delete Qnumu_2;
    }
#else
    void hisqStaplesForce(GaugeField &, const GaugeField &, const GaugeField &, const double[6])
    {
      errorQuda("HISQ force not enabled");
    }
#endif

    template <typename Arg>
    class HisqLongForce : public TunableKernel2D {

      Arg &arg;
      GaugeField &force;
      const GaugeField &meta;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      HisqLongForce(Arg &arg, GaugeField &force, const GaugeField &meta) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<LongLink>(tp, stream, arg);
      }

      void preTune() {
        force.backup();
      }

      void postTune() {
        force.restore();
      }

      long long flops() const {
        // all 4 directions
        long long multiplies_per_site = 4ll * 6ll;
        long long adds_per_site = 4ll * 3ll;
        long long rescales_per_site = 4ll;
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        return 4 * 2 * arg.threads.x * (2 * arg.force.Bytes() + 4 * arg.link.Bytes() + 3 * arg.oProd.Bytes());
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon>
    struct HisqLongLinkForce {
      HisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        LongLinkArg<Float, nColor, recon> arg(newOprod, link, oldOprod, coeff);
        HisqLongForce<decltype(arg)> longLink(arg, newOprod, link);
      }
    };

#ifdef GPU_STAGGERED_DIRAC
    void hisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
    {
      checkNative(link, oldOprod, newOprod);
      checkLocation(newOprod, oldOprod, link);
      checkPrecision(newOprod, link, oldOprod);
      instantiate<HisqLongLinkForce, ReconstructNone>(newOprod, oldOprod, link, coeff);
    }
#else
    void hisqLongLinkForce(GaugeField &, const GaugeField &, const GaugeField &, double)
    {
      errorQuda("HISQ force not enabled");
    }
#endif

    template <typename Arg>
    class HisqCompleteLinkForce : public TunableKernel2D {

      Arg &arg;
      GaugeField &force;
      const GaugeField &meta;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      HisqCompleteLinkForce(Arg &arg, GaugeField &force, const GaugeField &meta) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<CompleteForce>(tp, stream, arg);
      }

      void preTune() {
        force.backup();
      }

      void postTune() {
        force.restore();
      }

      long long flops() const {
        // all 4 directions
        int multiplies_per_site = 4ll;
        int rescales_per_site = 4ll;
        int antiherm_per_site = 4ll;
        // the flops counts for antiherm_per_site assumes the rescale by 1/2 is fused into the coefficient rescale
        return 2ll * arg.threads.x * (198ll * multiplies_per_site + 18ll * rescales_per_site + 23ll * antiherm_per_site);
      }

      long long bytes() const {
        return 4*2*arg.threads.x*(arg.force.Bytes() + arg.link.Bytes() + arg.oProd.Bytes());
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqCompleteForce {
      HisqCompleteForce(GaugeField &force, const GaugeField &link)
      {
        CompleteForceArg<real, nColor, recon> arg(force, link);
        HisqCompleteLinkForce<decltype(arg)> completeForce(arg, force, link);
      }
    };

#ifdef GPU_STAGGERED_DIRAC
    void hisqCompleteForce(GaugeField &force, const GaugeField &link)
    {
      checkNative(link, force);
      checkLocation(force, link);
      checkPrecision(link, force);
      instantiate<HisqCompleteForce, ReconstructNone>(force, link);
    }
#else
    void hisqCompleteForce(GaugeField &, const GaugeField &)
    {
      errorQuda("HISQ force not enabled");
    }
#endif

  } // namespace fermion_force

} // namespace quda
