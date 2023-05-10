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

    struct dim_dir_pair {
      int dim, dir;
      static dim_dir_pair make_pair(int signed_dir) {
        dim_dir_pair pr = { signed_dir > 3 ? 7 - signed_dir : signed_dir, signed_dir > 3 ? 0 : 1 };
        return pr;
      }

      static dim_dir_pair invalid_pair() {
        dim_dir_pair pr = { -1, -1 };
        return pr;
      }

      int signed_dir() {
        return is_invalid() ? -1 : ((dir == 1) ? dim : (7 - dim));
      }

      bool is_forward() const noexcept { return dir == 1; }
      bool is_backward() const noexcept { return dir == 0; }
      bool is_valid() const noexcept { return (is_forward() || is_backward()) && dim >= 0 && dim < 4; }
      bool is_invalid() const noexcept { return !is_valid(); }
    };

    template <typename Arg> class OneLinkForce : public TunableKernel3D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &link;
      unsigned int minThreads() const override { return arg.threads.x; }

    public:
      OneLinkForce(Arg &arg, const GaugeField &link, const GaugeField &force) :
        TunableKernel3D(link, 2, 4),
        arg(arg),
        force(force),
        link(link)
      {
        strcat(aux, comm_dim_partitioned_string());

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<OneLinkTerm>(tp, stream, arg);
      }

      void preTune() override { force.backup(); }
      void postTune() override { force.restore(); }

      long long flops() const override {
        // all four directions are handled in one kernel
        long long adds_per_site = 4ll;
        long long rescales_per_site = 4ll;
        return 2 * arg.threads.x * ( 18ll * adds_per_site + 18ll * rescales_per_site );
      }

      long long bytes() const override {
        long long link_bytes_per_site = 0ll;
        long long cm_bytes_per_site = 4ll * (arg.oProd.Bytes() + 2 * arg.force.Bytes());
        return 2 * arg.threads.x * (link_bytes_per_site + cm_bytes_per_site);
      }
    };

    template <typename Arg> class AllThreeAllLepageLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &p3;
      const GaugeField &pMu_next;
      const GaugeField &link;

      const dim_dir_pair sig, mu, mu_next;
      const bool has_lepage;
      unsigned int minThreads() const override { return arg.threads.x; }

      unsigned int sharedBytesPerThread() const override { return sizeof(Matrix<complex<typename Arg::real>, Arg::nColor>); }

      unsigned int maxSharedBytesPerBlock() const override { return maxDynamicSharedBytesPerBlock(); }

    public:
      AllThreeAllLepageLinkForce(Arg &arg, const GaugeField &link, dim_dir_pair sig, dim_dir_pair mu, dim_dir_pair mu_next,
                         const PathCoefficients<typename Arg::real> &act_path_coeff, const GaugeField &force,
                         const GaugeField &p3, const GaugeField &pMu_next) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        p3(p3),
        pMu_next(pMu_next),
        link(link),
        sig(sig),
        mu(mu),
        mu_next(mu_next),
        has_lepage(act_path_coeff.lepage != 0.)
      {
        arg.sig = sig.dim;
        arg.mu = mu.dim;
        arg.mu_next = mu_next.dim;
        arg.compute_lepage = has_lepage ? 1 : 0;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",sig=");
        strcat(aux, sig.is_forward() ? "+" : "-");
        u32toa(aux2, sig.dim);
        strcat(aux, aux2);
        if (mu.is_valid()) {
          strcat(aux, ",mu=");
          strcat(aux, mu.is_forward() ? "+" : "-");
          u32toa(aux2, mu.dim);
          strcat(aux, aux2);
          if (has_lepage) {
            strcat(aux, ",lepage");
          }
        }
        if (mu_next.is_valid()) {
          strcat(aux, ",mu_next=");
          strcat(aux, mu_next.is_forward() ? "+" : "-");
          u32toa(aux2, mu_next.dim);
          strcat(aux, aux2);
        }

        apply(device::get_default_stream());
      }

      template <int sig, int mu, int mu_next>
      void instantiate(TuneParam &tp, const qudaStream_t &stream) {
        if (has_lepage) launch<AllThreeAllLepageLink>(tp, stream, FatLinkParam<Arg, sig, mu, mu_next, DIR_IGNORED, DIR_IGNORED, COMPUTE_LEPAGE_YES>(arg));
        else launch<AllThreeAllLepageLink>(tp, stream, FatLinkParam<Arg, sig, mu, mu_next, DIR_IGNORED, DIR_IGNORED, COMPUTE_LEPAGE_NO>(arg));
      }

      template <int sig, int mu>
      void instantiate(dim_dir_pair mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (mu_next.is_invalid()) instantiate<sig, mu, DIR_IGNORED>(tp, stream);
        else if (mu_next.is_forward()) instantiate<sig, mu, DIR_POSITIVE>(tp, stream);
        else instantiate<sig, mu, DIR_NEGATIVE>(tp, stream);
      }

      template <int sig>
      void instantiate(dim_dir_pair mu, dim_dir_pair mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (mu.is_invalid()) instantiate<sig, DIR_IGNORED>(mu_next, tp, stream);
        else if (mu.is_forward()) instantiate<sig, DIR_POSITIVE>(mu_next, tp, stream);
        else instantiate<sig, DIR_NEGATIVE>(mu_next, tp, stream);
      }

      void instantiate(dim_dir_pair sig, dim_dir_pair mu, dim_dir_pair mu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (sig.is_forward()) instantiate<DIR_POSITIVE>(mu, mu_next, tp, stream);
        else instantiate<DIR_NEGATIVE>(mu, mu_next, tp, stream);
      }

      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true; // maximize the shared memory pool
        instantiate(sig, mu, mu_next, tp, stream);
      }

      void preTune() override {
        force.backup();
        p3.backup();
        pMu_next.backup();
      }

      void postTune() override {
        force.restore();
        p3.restore();
        pMu_next.restore();
      }

      long long flops() const override {
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;
        // Three link side link, all Lepage
        if (mu.is_valid()) {
          adds_per_site += 1ll;
          rescales_per_site += 1ll;
          if (has_lepage) {
            multiplies_per_site += 6ll;
            adds_per_site += 2ll;
            rescales_per_site += 2ll;
            if (sig.is_forward()) {
              multiplies_per_site += 2ll;
              adds_per_site += 1ll;
              rescales_per_site += 1ll;
            }
          }
        }
        // Three link middle link
        if (mu_next.is_valid()) {
          multiplies_per_site += 2ll;
          if (sig.is_forward()) {
            multiplies_per_site += 2ll;
            adds_per_site += 1ll;
            rescales_per_site += 1ll;
          }
        }
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const override {
        long long link_bytes_per_site = arg.link.Bytes();
        long long cm_bytes_per_site = 0ll;

        // Three link side link, all Lepage
        if (mu.is_valid()) {
          cm_bytes_per_site += arg.p3.Bytes() + 2 * arg.force.Bytes();
          if (has_lepage) {
            link_bytes_per_site += 5 * arg.link.Bytes();
            cm_bytes_per_site += 2 * arg.pMu.Bytes();
            if (sig.is_forward()) {
              link_bytes_per_site += arg.link.Bytes();
              cm_bytes_per_site += 2 * arg.force.Bytes();
            }
          }
        }
        // Three link middle link
        if (mu_next.is_valid()) {
          link_bytes_per_site += arg.link.Bytes();
          cm_bytes_per_site += arg.oProd.Bytes() + arg.pMu_next.Bytes() + arg.p3.Bytes();
          if (sig.is_forward()) {
            link_bytes_per_site += arg.link.Bytes();
            cm_bytes_per_site += 2 * arg.force.Bytes();
          }
        }

        // logic correction
        if (mu_next.is_invalid() && !has_lepage) {
          link_bytes_per_site -= arg.link.Bytes();
        }

        return 2 * arg.threads.x * (link_bytes_per_site + cm_bytes_per_site);
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

      const dim_dir_pair sig, mu, nu, nu_next;

      unsigned int minThreads() const override { return arg.threads.x; }

      unsigned int sharedBytesPerThread() const override { return (sig.is_forward() ? 3 : 2) * sizeof(Matrix<complex<typename Arg::real>, Arg::nColor>); }

      unsigned int maxSharedBytesPerBlock() const override { return maxDynamicSharedBytesPerBlock(); }

    public:
      AllFiveAllSevenLinkForce(Arg &arg, const GaugeField &link, dim_dir_pair sig, dim_dir_pair mu, dim_dir_pair nu,
                   dim_dir_pair nu_next, const GaugeField &force, const GaugeField &shortP,
                   const GaugeField &P5, const GaugeField &pNuMu_next, const GaugeField &qNuMu_next) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        p5(P5),
        pNuMu_next(pNuMu_next),
        qNuMu_next(qNuMu_next),
        link(link),
        sig(sig),
        mu(mu),
        nu(nu),
        nu_next(nu_next)
      {
        arg.sig = sig.dim;
        arg.mu = mu.dim;
        arg.nu = nu.dim;
        arg.nu_next = nu_next.dim;

        if (nu.is_valid()) {
          // rho is the "last" direction that's orthogonal to sig, mu, and nu
          // it's only relevant for the side 5 + All-7 part of the calculation
          for (arg.rho = 0; arg.rho < 4; arg.rho++) {
            if (arg.rho != sig.dim && arg.rho != mu.dim && arg.rho != nu.dim)
              break;
          }
        } else {
          arg.rho = -1;
        }

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",sig=");
        strcat(aux, sig.is_forward() ? "+" : "-");
        u32toa(aux2, sig.dim);
        strcat(aux, aux2);
        strcat(aux, ",mu=");
        strcat(aux, mu.is_forward() ? "+" : "-");
        u32toa(aux2, mu.dim);
        strcat(aux, aux2);
        if (nu.is_valid()) {
          strcat(aux, ",nu=");
          strcat(aux, nu.is_forward() ? "+" : "-");
          u32toa(aux2, nu.dim);
          strcat(aux, aux2);
        }
        if (nu_next.is_valid()) {
          strcat(aux, ",nu_next=");
          strcat(aux, nu_next.is_forward() ? "+" : "-");
          u32toa(aux2, nu_next.dim);
          strcat(aux, aux2);
        }
        
        apply(device::get_default_stream());
      }

      template <int sig, int mu, int nu>
      void instantiate(dim_dir_pair nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (nu_next.is_invalid()) {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, DIR_IGNORED, nu, DIR_IGNORED>(arg));
        } else if (nu_next.is_forward()) {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, DIR_IGNORED, nu, DIR_POSITIVE>(arg));
        } else {
          launch<AllFiveAllSevenLink>(tp, stream, FatLinkParam<Arg, sig, mu, DIR_IGNORED, nu, DIR_NEGATIVE>(arg));
        }
      }

      template <int sig, int mu>
      void instantiate(dim_dir_pair nu, dim_dir_pair nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (nu.is_invalid()) {
          instantiate<sig, mu, DIR_IGNORED>(nu_next, tp, stream);
        } else if (nu.is_forward()) {
          instantiate<sig, mu, DIR_POSITIVE>(nu_next, tp, stream);
        } else {
          instantiate<sig, mu, DIR_NEGATIVE>(nu_next, tp, stream);
        }
      }

      template <int sig>
      void instantiate(dim_dir_pair mu, dim_dir_pair nu, dim_dir_pair nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (mu.is_forward()) instantiate<sig, DIR_POSITIVE>(nu, nu_next, tp, stream);
        else instantiate<sig, DIR_NEGATIVE>(nu, nu_next, tp, stream);
      }

      void instantiate(dim_dir_pair sig, dim_dir_pair mu, dim_dir_pair nu, dim_dir_pair nu_next, TuneParam &tp, const qudaStream_t &stream) {
        if (sig.is_forward()) instantiate<DIR_POSITIVE>(mu, nu, nu_next, tp, stream);
        else instantiate<DIR_NEGATIVE>(mu, nu, nu_next, tp, stream);
      }

      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true; // maximize the shared memory pool
        instantiate(sig, mu, nu, nu_next, tp, stream);
      }

      void preTune() override {
        force.backup();
        shortP.backup();
        p5.backup();
        pNuMu_next.backup();
        qNuMu_next.backup();
      }

      void postTune() override {
        force.restore();
        shortP.restore();
        p5.restore();
        pNuMu_next.restore();
        qNuMu_next.restore();
      }

      long long flops() const override {
        long long multiplies_per_site = 0ll;
        long long adds_per_site = 0ll;
        long long rescales_per_site = 0ll;

        // SideFiveAllSeven contribution
        if (nu.is_valid()) {
          multiplies_per_site += 12ll;
          adds_per_site += 6ll;
          rescales_per_site += 6ll;
          if (sig.is_forward()) {
            multiplies_per_site += 4ll;
            adds_per_site += 2ll;
            rescales_per_site += 2ll;
          }
        }

        // MiddleFive contribution
        if (nu_next.is_valid()) {
          multiplies_per_site += 3ll;
          if (sig.is_forward()) {
            multiplies_per_site += 1ll;
            adds_per_site += 1ll;
            rescales_per_site += 1ll;
          }
        }

        return 2*arg.threads.x*(198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const override {
        long long link_bytes_per_site = arg.link.Bytes();
        long long cm_bytes_per_site = 0ll;
        if (sig.is_forward()) cm_bytes_per_site += 2 * arg.force.Bytes();

        // SideFiveAllSeven contribution
        if (nu.is_valid()) {
          link_bytes_per_site += 8 * arg.link.Bytes();
          cm_bytes_per_site += 2 * arg.qNuMu.Bytes() + 2 * arg.pNuMu.Bytes() +
                            arg.p5.Bytes() + 2 * arg.shortP.Bytes() + 4 * arg.force.Bytes();
          if (sig.is_forward())
            cm_bytes_per_site += arg.qNuMu.Bytes() + arg.pNuMu.Bytes();
        }

        // MiddleFive contribution
        if (nu_next.is_valid()) {
          link_bytes_per_site += 3 * arg.link.Bytes();
          cm_bytes_per_site += arg.pMu.Bytes() + arg.p5.Bytes() +
                            arg.pNuMu_next.Bytes() + arg.qNuMu_next.Bytes();
        }

        return 2 * arg.threads.x * (link_bytes_per_site + cm_bytes_per_site);
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO>
    struct HisqStaplesForce {
      using real = typename mapper<Float>::type;

      void hisqFiveSeven(GaugeField &newOprod, GaugeField &P3, GaugeField_ref &P5, GaugeField_ref &Pnumu, GaugeField_ref &Qnumu,
                         GaugeField_ref &Pnumu_next, GaugeField_ref &Qnumu_next, const GaugeField &Pmu,
                         const GaugeField &link, const PathCoefficients<real> &act_path_coeff, dim_dir_pair sig_pair, dim_dir_pair mu_pair) {

        // unroll the nu loop
        std::vector<dim_dir_pair> nu_vals;
        nu_vals.reserve(4);
        for (int nu = 0; nu < 8; nu++) {
          auto nu_pair = dim_dir_pair::make_pair(nu);
          if (nu_pair.dim == sig_pair.dim || nu_pair.dim == mu_pair.dim) continue;
          nu_vals.emplace_back(nu_pair);
        }

        // first: just MiddleFiveLink
        // In/out: newOprod
        // Out: P5, Pnumu, Qnumu
        // In: Pmu, link
        // Ignored: Pnumu_next, Qnumu_next (since this is MiddleFive only)
        AllFiveAllSevenLinkArg<Float, nColor, recon, phase> middleFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu_next, Qnumu_next, Pnumu, Qnumu, link, act_path_coeff);
        AllFiveAllSevenLinkForce<decltype(middleFiveLinkArg)> middleFiveArg(middleFiveLinkArg, link, sig_pair, mu_pair, dim_dir_pair::invalid_pair(), nu_vals[0], newOprod, P3, P5, Pnumu, Qnumu);

        for (int i = 0; i < 3; i++) {
          // next: fully fused kernels
          // In/out: new Oprod, P3 (called shortP), P5
          // In: Pmu, Pnumu, Qnumu, link
          // Out: Pnumu_next, Qnumu_next
          AllFiveAllSevenLinkArg<Float, nColor, recon, phase> allFiveAllSevenLinkArg(newOprod, P3, Pmu, P5, Pnumu, Qnumu, Pnumu_next, Qnumu_next, link, act_path_coeff);
          AllFiveAllSevenLinkForce<decltype(allFiveAllSevenLinkArg)> allFiveAllSevenLink(allFiveAllSevenLinkArg, link, sig_pair, mu_pair, nu_vals[i], nu_vals[i+1], newOprod, P3, P5, Pnumu_next, Qnumu_next);

          std::swap(Pnumu, Pnumu_next);
          std::swap(Qnumu, Qnumu_next);
        }

        // last: just SideFiveAllSevenLink
        // In/out: newOprod, P3 (called shortP)
        // In: P5, Pnumu, Qnumu, link
        // Out: none
        // Ignored: Pmu, Pnumu_next, Qnumu_next
        AllFiveAllSevenLinkArg<Float, nColor, recon, phase> allSevenSideFiveLinkArg(newOprod, P3, Pmu, P5, Pnumu, Qnumu, Pnumu_next, Qnumu_next, link, act_path_coeff);
        AllFiveAllSevenLinkForce<decltype(allSevenSideFiveLinkArg)> allSevenSideFiveLink(allSevenSideFiveLinkArg, link, sig_pair, mu_pair, nu_vals[3], dim_dir_pair::invalid_pair(), newOprod, P3, P5, Pnumu, Qnumu);
      }

      HisqStaplesForce(const GaugeField &link, GaugeField &P3, GaugeField_ref &Pmu, GaugeField_ref &P5, GaugeField_ref &Pnumu, GaugeField_ref &Qnumu,
                       GaugeField_ref &Pmu_next, GaugeField_ref &Pnumu_next, GaugeField_ref &Qnumu_next,
                       GaugeField &newOprod, const GaugeField &oprod, const double *path_coeff_array)
      {
        PathCoefficients<real> act_path_coeff(path_coeff_array);

        {
          // Out: newOprod
          // In: oprod, link
          OneLinkArg<Float, nColor, recon, phase> arg(newOprod, oprod, link, act_path_coeff);
          OneLinkForce<decltype(arg)> oneLink(arg, link, newOprod);
        }

        for (int sig = 0; sig < 8; sig++) {
          auto sig_pair = dim_dir_pair::make_pair(sig);

          // unroll the mu loop
          std::vector<dim_dir_pair> mu_vals;
          mu_vals.reserve(6);
          for (int mu = 0; mu < 8; mu++) {
            auto mu_pair = dim_dir_pair::make_pair(mu);
            if (sig_pair.dim == mu_pair.dim) continue;
            mu_vals.emplace_back(mu_pair);
          }

          // 3-link: middle link only
          // In/out: newOprod
          // Out: (first) Pmu, P3
          // In: oprod, link
          // Ignored: Pmu_next
          AllThreeAllLepageLinkArg<Float, nColor, recon, phase> middleThreeLinkArg(newOprod, P3, oprod, Pmu, Pmu_next, link, act_path_coeff);
          AllThreeAllLepageLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig_pair, dim_dir_pair::invalid_pair(), mu_vals[0], act_path_coeff, newOprod, P3, Pmu_next);

          // All 5 and 7 link contributions
          // In/out: newOprod, P3
          // In: Pmu, link
          // Internal only: P5, Pnumu, Qnumu, and the double-buffer flavors
          hisqFiveSeven(newOprod, P3, P5, Pnumu, Qnumu, Pnumu_next, Qnumu_next, Pmu_next, link, act_path_coeff, sig_pair, mu_vals[0]);

          for (int i = 0; i < 5; i++) {
            std::swap(Pmu, Pmu_next);

            // Fully fused 3-link and Lepage contributions (when Lepage coeff != 0.)
            // In/out: oProd, P3 (read + overwritten)
            // In: (first) Pmu, oProd, link
            // Out: (second) Pmu
            AllThreeAllLepageLinkArg<Float, nColor, recon, phase> allThreeAllLepageLinkArg(newOprod, P3, oprod, Pmu, Pmu_next, link, act_path_coeff);
            AllThreeAllLepageLinkForce<decltype(allThreeAllLepageLinkArg)> allLepageAllThreeLink(allThreeAllLepageLinkArg, link, sig_pair, mu_vals[i], mu_vals[i+1], act_path_coeff, newOprod, P3, Pmu_next);

            // All 5 and 7 link contributions, as above
            hisqFiveSeven(newOprod, P3, P5, Pnumu, Qnumu, Pnumu_next, Qnumu_next, Pmu_next, link, act_path_coeff, sig_pair, mu_vals[i+1]);
          }

          std::swap(Pmu, Pmu_next);

          // Side 3-link, fused with Lepage all link when the lepage coeff != 0.
          // In/out: newOprod
          // In: P3, (second) Pmu, link
          // Ignored: (first) Pmu, oProd
          AllThreeAllLepageLinkArg<Float, nColor, recon, phase> allLepageSideThreeLinkArg(newOprod, P3, oprod, Pmu, Pmu_next, link, act_path_coeff);
          AllThreeAllLepageLinkForce<decltype(allLepageSideThreeLinkArg)> allLepageSideThreeLink(allLepageSideThreeLinkArg, link, sig_pair, mu_vals[5], dim_dir_pair::invalid_pair(), act_path_coeff, newOprod, P3, Pmu_next);
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

      // need double buffers for these fields to fuse "side link" terms with
      // subsequent "middle link" terms in a different direction
      auto Pmu_next = GaugeField::Create(gauge_param);
      auto Pnumu_next = GaugeField::Create(gauge_param);
      auto Qnumu_next = GaugeField::Create(gauge_param);

      instantiateGaugeStaggered<HisqStaplesForce>(link, *P3, GaugeField_ref(*Pmu),
        GaugeField_ref(*P5), GaugeField_ref(*Pnumu), GaugeField_ref(*Qnumu),
        GaugeField_ref(*Pmu_next), GaugeField_ref(*Pnumu_next), GaugeField_ref(*Qnumu_next),
        newOprod, oprod, path_coeff_array);

      delete Pmu;
      delete P3;
      delete P5;
      delete Pnumu;
      delete Qnumu;
      delete Pmu_next;
      delete Pnumu_next;
      delete Qnumu_next;
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
      unsigned int minThreads() const override { return arg.threads.x; }

    public:
      HisqLongForce(Arg &arg, GaugeField &force, const GaugeField &meta) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        strcat(aux, comm_dim_partitioned_string());

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) override {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<LongLink>(tp, stream, arg);
      }

      void preTune() override {
        force.backup();
      }

      void postTune() override {
        force.restore();
      }

      long long flops() const override {
        // all 4 directions
        long long multiplies_per_site = 4ll * 6ll;
        long long adds_per_site = 4ll * 3ll;
        long long rescales_per_site = 4ll;
        return 2 * arg.threads.x * (198ll * multiplies_per_site + 18ll * adds_per_site + 18ll * rescales_per_site);
      }

      long long bytes() const override {
        long long link_bytes_per_site = 4ll * (4 * arg.link.Bytes());
        long long cm_bytes_per_site = 4ll * (2 * arg.force.Bytes() + 3 * arg.oProd.Bytes());
        return 2 * arg.threads.x * (link_bytes_per_site + cm_bytes_per_site);
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO>
    struct HisqLongLinkForce {
      HisqLongLinkForce(const GaugeField &link, GaugeField &newOprod, const GaugeField &oldOprod, double coeff)
      {
        LongLinkArg<Float, nColor, recon, phase> arg(newOprod, link, oldOprod, coeff);
        HisqLongForce<decltype(arg)> longLink(arg, newOprod, link);
      }
    };

#ifdef GPU_STAGGERED_DIRAC
    void hisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
    {
      checkNative(link, oldOprod, newOprod);
      checkLocation(newOprod, oldOprod, link);
      checkPrecision(newOprod, link, oldOprod);
      instantiateGaugeStaggered<HisqLongLinkForce>(link, newOprod, oldOprod, coeff);
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
      unsigned int minThreads() const override { return arg.threads.x; }

    public:
      HisqCompleteLinkForce(Arg &arg, GaugeField &force, const GaugeField &meta) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        strcat(aux, comm_dim_partitioned_string());

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) override {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<CompleteForce>(tp, stream, arg);
      }

      void preTune() override {
        force.backup();
      }

      void postTune() override {
        force.restore();
      }

      long long flops() const override {
        // all 4 directions
        long long multiplies_per_site = 4ll;
        long long rescales_per_site = 4ll;
        long long antiherm_per_site = 4ll;

        // the flops counts for antiherm_per_site assumes the rescale by 1/2 is fused into the coefficient rescale
        return 2ll * arg.threads.x * (198ll * multiplies_per_site + 18ll * rescales_per_site + 23ll * antiherm_per_site);
      }

      long long bytes() const override {
        long long link_bytes_per_site = 4ll * arg.link.Bytes();
        long long cm_bytes_per_site = 4ll * (arg.force.Bytes() + arg.oProd.Bytes());
        return 2 * arg.threads.x * (link_bytes_per_site + cm_bytes_per_site);
      }
    };

    template <typename real, int nColor, QudaReconstructType recon, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO>
    struct HisqCompleteForce {
      HisqCompleteForce(const GaugeField &link, GaugeField &force)
      {
        CompleteForceArg<real, nColor, recon, phase> arg(force, link);
        HisqCompleteLinkForce<decltype(arg)> completeForce(arg, force, link);
      }
    };

#ifdef GPU_STAGGERED_DIRAC
    void hisqCompleteForce(GaugeField &force, const GaugeField &link)
    {
      checkNative(link, force);
      checkLocation(force, link);
      checkPrecision(link, force);
      instantiateGaugeStaggered<HisqCompleteForce>(link, force);
    }
#else
    void hisqCompleteForce(GaugeField &, const GaugeField &)
    {
      errorQuda("HISQ force not enabled");
    }
#endif

  } // namespace fermion_force

} // namespace quda
