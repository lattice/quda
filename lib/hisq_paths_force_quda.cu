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

    template <typename Arg> class MiddleThreeLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &pMu;
      const GaugeField &p3;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      MiddleThreeLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &pMu,
                   const GaugeField &p3) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        pMu(pMu),
        p3(p3),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;

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

        apply(device::get_default_stream());
      }

      template <int sig>
      void instantiate(int mu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(mu)) launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, sig, MU_POSITIVE>(arg));
        else launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, sig, MU_NEGATIVE>(arg));
      }

      void instantiate(int sig, int mu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(sig)) instantiate<SIG_POSITIVE>(mu, tp, stream);
        else instantiate<SIG_NEGATIVE>(mu, tp, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        instantiate(arg.sig, arg.mu, tp, stream);
      }

      void preTune() {
        pMu.backup();
        force.backup();
        p3.backup();
      }

      void postTune() {
        pMu.restore();
        force.restore();
        p3.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 2ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;
        if (goes_forward(arg.sig)) {
          multiplies_per_site += 1ll;
          adds_per_site += 1ll;
          rescales_per_site += 1ll;
        }
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        long long bytes_per_site = 2 * arg.link.Bytes() + arg.oProd.Bytes() + arg.p3.Bytes() + arg.pMu.Bytes();
        if (goes_forward(arg.sig))
          bytes_per_site += arg.link.Bytes() + 2 * arg.force.Bytes();
        return 2 * arg.threads.x * bytes_per_site;
      }

    };

    template <typename Arg> class AllLepageSideThreeLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &link;
      const bool has_lepage;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      AllLepageSideThreeLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                         const PathCoefficients<typename Arg::real> &act_path_coeff, const GaugeField &force) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        link(link),
        has_lepage(act_path_coeff.lepage != 0.)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.compute_lepage = has_lepage ? 1 : 0;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        if (has_lepage) {
          strcat(aux, ",lepage,sig=");
          u32toa(aux2, arg.sig);
          strcat(aux, aux2);
        }
        strcat(aux, ",mu=");
        u32toa(aux2, arg.mu);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (has_lepage) {
          if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_POSITIVE, MU_POSITIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_YES>(arg));
          } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_POSITIVE, MU_NEGATIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_YES>(arg));
          } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_NEGATIVE, MU_POSITIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_YES>(arg));
          } else {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_NEGATIVE, MU_NEGATIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_YES>(arg));
          }
        } else {
          if (goes_forward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_IGNORED, MU_POSITIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_NO>(arg));
          } else {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, SIG_IGNORED, MU_NEGATIVE, NU_IGNORED, NU_NEXT_IGNORED, COMPUTE_LEPAGE_NO>(arg));
          }
        }
      }

      void preTune() {
        force.backup();
      }

      void postTune() {
        force.restore();
      }

      long long flops() const {
        // 3-link side contribution: 1 add, 1 rescale
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 1ll;
        long long rescales_per_site = 1ll;
        // Lepage contributions
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
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        // 3-link side contribution
        long long bytes_per_site = ( 2 * arg.force.Bytes() + arg.p3.Bytes() );
        // Lepage contributions
        if (has_lepage) {
          bytes_per_site += ( 6 * arg.link.Bytes() + 2 * arg.pMu.Bytes() );
          if (goes_forward(arg.sig)) {
            bytes_per_site += ( 2 * arg.force.Bytes() + arg.link.Bytes() );
          }
        }
        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Arg> class MiddleFiveLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &pNuMu;
      const GaugeField &p5;
      const GaugeField &qNuMu;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      MiddleFiveLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, int nu,
                   const GaugeField &force, const GaugeField &pNuMu,
                   const GaugeField &p5, const GaugeField &qNuMu) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        pNuMu(pNuMu),
        p5(p5),
        qNuMu(qNuMu),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.nu = nu;

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
        strcat(aux, ",nu=");
        u32toa(aux2, arg.nu);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      template <int sig, int mu>
      void instantiate(int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(nu)) launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, sig, mu, NU_POSITIVE>(arg));
        else launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, sig, mu, NU_NEGATIVE>(arg));
      }

      template <int sig>
      void instantiate(int mu, int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(mu)) instantiate<sig, MU_POSITIVE>(nu, tp, stream);
        else instantiate<sig, MU_NEGATIVE>(nu, tp, stream);
      }

      void instantiate(int sig, int mu, int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(sig)) instantiate<SIG_POSITIVE>(mu, nu, tp, stream);
        else instantiate<SIG_NEGATIVE>(mu, nu, tp, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        instantiate(arg.sig, arg.mu, arg.nu, tp, stream);
      }

      void preTune() {
        pNuMu.backup();
        p5.backup();
        force.backup();
        qNuMu.backup();
      }

      void postTune() {
        pNuMu.restore();
        p5.restore();
        force.restore();
        qNuMu.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 3ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;
        if (goes_forward(arg.sig)) {
          multiplies_per_site += 1ll;
          adds_per_site += 1ll;
          rescales_per_site += 1ll;
        }
        return 2*arg.threads.x*(198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        long long bytes_per_site = 4 * arg.link.Bytes() + arg.pMu.Bytes() + arg.p5.Bytes() +
                                   arg.pNuMu.Bytes() + arg.qNuMu.Bytes();
        if (goes_forward(arg.sig))
          bytes_per_site += 2 * arg.force.Bytes();
        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Arg> class AllSevenSideFiveLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      AllSevenSideFiveLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, int nu, int rho,
                   const GaugeField &force, const GaugeField &shortP) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.nu = nu;
        arg.rho = rho;

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
        strcat(aux, ",nu=");
        u32toa(aux2, arg.nu);
        strcat(aux, aux2);
        strcat(aux, ",rho=");
        u32toa(aux2, arg.rho);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      template <int sig, int mu>
      void instantiate(int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(nu)) launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, sig, mu, NU_POSITIVE>(arg));
        else launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, sig, mu, NU_NEGATIVE>(arg));
      }

      template <int sig>
      void instantiate(int mu, int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(mu)) instantiate<sig, MU_POSITIVE>(nu, tp, stream);
        else instantiate<sig, MU_NEGATIVE>(nu, tp, stream);
      }

      void instantiate(int sig, int mu, int nu, TuneParam &tp, const qudaStream_t &stream) {
        if (goes_forward(sig)) instantiate<SIG_POSITIVE>(mu, nu, tp, stream);
        else instantiate<SIG_NEGATIVE>(mu, nu, tp, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        instantiate(arg.sig, arg.mu, arg.nu, tp, stream);
      }

      void preTune() {
        force.backup();
        shortP.backup();
      }

      void postTune() {
        force.restore();
        shortP.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 12ll;
        long long adds_per_site = 8ll;
        long long rescales_per_site = 6ll;
        if (goes_forward(arg.sig)) {
          multiplies_per_site += 5ll;
          adds_per_site += 3ll;
          rescales_per_site += 2ll;
        }
        return 2*arg.threads.x*(198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const {
        long long bytes_per_site = 9 * arg.link.Bytes() + 2 * arg.qNuMu.Bytes() + 2 * arg.pNuMu.Bytes() +
                                   arg.p5.Bytes() + 2 * arg.shortP.Bytes() + 4 * arg.force.Bytes();
        if (goes_forward(arg.sig))
          bytes_per_site += arg.qNuMu.Bytes() + arg.pNuMu.Bytes() + 2 * arg.force.Bytes();
        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Arg> class AllFiveAllSevenLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &p5_next;
      const GaugeField &pNuMu_next;
      const GaugeField &qNuMu_next;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      AllFiveAllSevenLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, int nu, int rho,
                   int nu_next, const GaugeField &force, const GaugeField &shortP,
                   const GaugeField &P5_next, const GaugeField &pNuMu_next, const GaugeField &qNuMu_next) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        p5_next(P5_next),
        pNuMu_next(pNuMu_next),
        qNuMu_next(qNuMu_next),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;
        arg.nu = nu;
        arg.rho = rho;
        arg.nu_next = nu_next;

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
          strcat(aux, ",rho=");
          u32toa(aux2, arg.rho);
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
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, nu, NU_NEXT_IGNORED>(arg));
        } else if (goes_forward(nu_next)) {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, nu, NU_NEXT_POSITIVE>(arg));
        } else {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, nu, NU_NEXT_NEGATIVE>(arg));
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
        instantiate(arg.sig, arg.mu, arg.nu, arg.nu_next, tp, stream);
      }

      void preTune() {
        force.backup();
        shortP.backup();
      }

      void postTune() {
        force.restore();
        shortP.restore();
      }

      long long flops() const {
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;

        // SideFiveAllSeven contribution
        if (arg.nu != NU_IGNORED) {
          multiplies_per_site += 12ll;
          adds_per_site += 8ll;
          rescales_per_site += 6ll;
          if (goes_forward(arg.sig)) {
            multiplies_per_site += 5ll;
            adds_per_site += 3ll;
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
        // FIXME update
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
          bytes_per_site += 3 * arg.link.Bytes() + arg.pMu.Bytes() + arg.p5_next.Bytes() +
                            arg.pNuMu_next.Bytes() + arg.qNuMu_next.Bytes();
        }

        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      using real = typename mapper<Float>::type;

      template <bool low_memory>
      void hisqFiveSeven(GaugeField &newOprod, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu, GaugeField &Qnumu,
                         GaugeField &P5_2, GaugeField &Pnumu_2, GaugeField &Qnumu_2, const GaugeField &Pmu,
                         const GaugeField &link, const PathCoefficients<real> &act_path_coeff, int sig, int mu) {
        if constexpr (low_memory) {
          for (int nu=0; nu < 8; nu++) {
            if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

            // 5-link: middle link
            // In/out: newOprod
            // Out: Pnumu, P5, Qnumu
            // In: Pmu, link
            MiddleFiveLinkArg<Float, nColor, recon> middleFiveLinkArg(newOprod, Pnumu, P5, Qnumu, Pmu, link, act_path_coeff);
            MiddleFiveLinkForce<decltype(middleFiveLinkArg)> middleFiveLink(middleFiveLinkArg, link, sig, mu, nu, newOprod, Pnumu, P5, Qnumu);

            // determine the remaining orthogonal direction
            int rho;
            for (rho = 0; rho < 4; rho++) {
              if (rho != pos_dir(sig) && rho != pos_dir(mu) && rho != pos_dir(nu))
                break;
            }

            // Fused 7-link middle and side link with 5-link side link:
            // In/out: newOprod, P3 (called shortP), 
            // In: P5, Pmunu (called "oprod"), Qnumu, link
            AllSevenSideFiveLinkArg<Float, nColor, recon> argAll(newOprod, P3, P5, Pnumu, Qnumu, link, act_path_coeff);
            AllSevenSideFiveLinkForce<decltype(argAll)> all(argAll, link, sig, mu, nu, rho, newOprod, P3);

          } //nu
        } else {
          // optimized, more fused path
          // Uses a "double buffer" for P5/Pnumu/Qnumu

          // unroll the nu loop
          std::vector<std::pair<int, int> > nu_rho_pairs;
          nu_rho_pairs.reserve(4);
          for (int nu = 0; nu < 8; nu++) {
            if (pos_dir(nu) == pos_dir(sig) || pos_dir(nu) == pos_dir(mu)) continue;
            int rho;
            for (rho = 0; rho < 4; rho++) {
              if (rho != pos_dir(sig) && rho != pos_dir(mu) && rho != pos_dir(nu))
                break;
            }
            nu_rho_pairs.emplace_back(std::make_pair(nu, rho));
          }

          // first: just MiddleFiveLink
          // In/out: newOprod
          // Out: P5, Pnumu, Qnumu
          // In: Pmu, link
          // Ignored: P5_2, Pnumu_2, Qnumu_2 (since this is MiddleFive only)
          AllFiveAllSevenLinkArg<Float, nColor, recon> midarg0(newOprod, P3, Pmu, P5_2, Pnumu_2, Qnumu_2, P5, Pnumu, Qnumu, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(midarg0)> midarglink0(midarg0, link, sig, mu, -1, -1, nu_rho_pairs[0].first, newOprod, P3, P5, Pnumu, Qnumu);

          // next: fully fused kernels
          // In/out: new Oprod, P3 (called shortP)
          // In: Pmu, P5, Pnumu, Qnumu, link
          // Out: P5_2, Pnumu_2, Qnumu_2
          AllFiveAllSevenLinkArg<Float, nColor, recon> midarg1(newOprod, P3, Pmu, P5, Pnumu, Qnumu, P5_2, Pnumu_2, Qnumu_2, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(midarg1)> midarglink1(midarg1, link, sig, mu, nu_rho_pairs[0].first, nu_rho_pairs[0].second, nu_rho_pairs[1].first, newOprod, P3, P5_2, Pnumu_2, Qnumu_2);

          AllFiveAllSevenLinkArg<Float, nColor, recon> midarg2(newOprod, P3, Pmu, P5_2, Pnumu_2, Qnumu_2, P5, Pnumu, Qnumu, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(midarg2)> midarglink2(midarg2, link, sig, mu, nu_rho_pairs[1].first, nu_rho_pairs[1].second, nu_rho_pairs[2].first, newOprod, P3, P5, Pnumu, Qnumu);

          AllFiveAllSevenLinkArg<Float, nColor, recon> midarg3(newOprod, P3, Pmu, P5, Pnumu, Qnumu, P5_2, Pnumu_2, Qnumu_2, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(midarg3)> midarglink3(midarg3, link, sig, mu, nu_rho_pairs[2].first, nu_rho_pairs[2].second, nu_rho_pairs[3].first, newOprod, P3, P5_2, Pnumu_2, Qnumu_2);

          // last: just SideFiveAllSevenLink
          // In/out: newOprod, P3 (called shortP)
          // In: P5_2, Pnumu_2, Qnumu_2, link
          // Out: none
          // Ignored: Pmu, P5, Pnumu, Qnumu
          AllFiveAllSevenLinkArg<Float, nColor, recon> midarg4(newOprod, P3, Pmu, P5_2, Pnumu_2, Qnumu_2, P5, Pnumu, Qnumu, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(midarg4)> midarglink4(midarg4, link, sig, mu, nu_rho_pairs[3].first, nu_rho_pairs[3].second, -1, newOprod, P3, P5, Pnumu, Qnumu);

        }
      }

      HisqStaplesForce(GaugeField &Pmu, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu, GaugeField &Qnumu,
                       GaugeField &P5_2, GaugeField &Pnumu_2, GaugeField &Qnumu_2, GaugeField &newOprod,
                       const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {
        PathCoefficients<real> act_path_coeff(path_coeff_array);

        {
          // Out: newOprod
          // In: oprod, link
          OneLinkArg<Float, nColor, recon> arg(newOprod, oprod, link, act_path_coeff);
          OneLinkForce<decltype(arg)> oneLink(arg, link, newOprod);
        }

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) continue;

            // 3-link: middle link
            // In/out: newOprod
            // Out: Pmu, P3
            // In: oprod, link
            MiddleThreeLinkArg<Float, nColor, recon> middleThreeLinkArg(newOprod, Pmu, P3, oprod, link, act_path_coeff);
            MiddleThreeLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig, mu, newOprod, Pmu, P3);

            constexpr bool low_memory_path = false;

            // All 5 and 7 link contributions
            // In/out: newOprod, P3
            // In: Pmu, link
            // Internal only: P5, Pnumu, Qnumu, and the double-buffer flavors
            hisqFiveSeven<low_memory_path>(newOprod, P3, P5, Pnumu, Qnumu, P5_2, Pnumu_2, Qnumu_2, Pmu, link, act_path_coeff, sig, mu);

            // Side 3-link, fused with Lepage all link when the lepage coeff != 0.
            // In/out: newOprod
            // In: P3, Pmu, link
            AllLepageSideThreeLinkArg<Float, nColor, recon> allLepageSideThreeLinkArg(newOprod, P3, Pmu, link, act_path_coeff);
            AllLepageSideThreeLinkForce<decltype(allLepageSideThreeLinkArg)> allLepageSideThreeLink(allLepageSideThreeLinkArg, link, sig, mu, act_path_coeff, newOprod);
          }//mu
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

      auto Pmu = GaugeField::Create(gauge_param);
      auto P3 = GaugeField::Create(gauge_param);
      auto P5 = GaugeField::Create(gauge_param);
      auto Pnumu = GaugeField::Create(gauge_param);
      auto Qnumu = GaugeField::Create(gauge_param);

      // need some double buffer going on
      auto P5_2 = GaugeField::Create(gauge_param);
      auto Pnumu_2 = GaugeField::Create(gauge_param);
      auto Qnumu_2 = GaugeField::Create(gauge_param);

      instantiate<HisqStaplesForce, ReconstructNone>(*Pmu, *P3, *P5, *Pnumu, *Qnumu, *P5_2, *Pnumu_2, *Qnumu_2, newOprod, oprod, link, path_coeff_array);

      delete Pmu;
      delete P3;
      delete P5;
      delete Pnumu;
      delete Qnumu;
      delete P5_2;
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
