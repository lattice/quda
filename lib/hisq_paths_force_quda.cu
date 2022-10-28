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

      long long flops() const { return 2*4*arg.threads.x*36ll; }

      long long bytes() const { return 2*4*arg.threads.x*( arg.oProd.Bytes() + 2*arg.force.Bytes() ); }
    };

    template <typename Arg> class MiddleThreeLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &pMu;
      const GaugeField &qMu;
      const GaugeField &p3;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      MiddleThreeLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &pMu,
                   const GaugeField &qMu, const GaugeField &p3) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        pMu(pMu),
        qMu(qMu),
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

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 1, 1>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 1, 0>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 0, 1>(arg));
        } else {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 0, 0>(arg));
        }
      }

      void preTune() {
        pMu.backup();
        qMu.backup();
        force.backup();
        p3.backup();
      }

      void postTune() {
        pMu.restore();
        qMu.restore();
        force.restore();
        p3.restore();
      }

      long long flops() const {
        return 2*arg.threads.x*(2 * 198 + (goes_forward(arg.sig) ? 414 : 0));
      }

      long long bytes() const {
        return 2*arg.threads.x*( ( goes_forward(arg.sig) ? 2 * arg.force.Bytes() : 0 ) +
                               arg.pMu.Bytes() + arg.qMu.Bytes() +
                               arg.p3.Bytes() + 3 * arg.link.Bytes() + arg.oProd.Bytes() );
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
      MiddleFiveLinkForce(Arg &arg, const GaugeField &link, int sig, int nu,
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
        arg.nu = nu;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        strcat(aux, ",sig=");
        u32toa(aux2, arg.sig);
        strcat(aux, aux2);
        strcat(aux, ",nu=");
        u32toa(aux2, arg.nu);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.sig) && goes_forward(arg.nu)) {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 1, -1, 1>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.nu)) {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 1, -1, 0>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.nu)) {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 0, -1, 1>(arg));
        } else {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 0, -1, 0>(arg));
        }
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
        return 2*arg.threads.x*(3 * 198 + (goes_forward(arg.sig) ? 414 : 0) );
      }

      long long bytes() const {
        return 2*arg.threads.x*( ( goes_forward(arg.sig) ? 2 * arg.force.Bytes() : 0 ) +
                               arg.pNuMu.Bytes() + arg.qNuMu.Bytes() + arg.qMu.Bytes() +
                               arg.p5.Bytes() + 3 * arg.link.Bytes() + arg.pMu.Bytes() );
      }
    };

    template <typename Arg> class AllSevenSideFiveLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      AllSevenSideFiveLinkForce(Arg &arg, const GaugeField &link, int sig, int nu, int rho,
                   const GaugeField &force, const GaugeField &shortP) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        link(link)
      {
        arg.sig = sig;
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
        strcat(aux, ",nu=");
        u32toa(aux2, arg.nu);
        strcat(aux, aux2);
        strcat(aux, ",rho=");
        u32toa(aux2, arg.rho);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.sig)) {
          if (goes_forward(arg.nu)) {
            launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, 1, -1, 1>(arg));
          } else {
            launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, 1, -1, 0>(arg));
          }
        } else {
          if (goes_forward(arg.nu)) {
            launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, 0, -1, 1>(arg));
          } else {
            launch<AllSevenSideFiveLink>(tp, stream, FatLinkParam<Arg, 0, -1, 0>(arg));
          }
        }
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
        int multiplies = (goes_forward(arg.sig) ? 17 : 12);
        int adds = (goes_forward(arg.sig) ? 11 : 8);
        int rescales = (goes_forward(arg.sig) ? 8 : 6);
        return 2*arg.threads.x*(198ll * multiplies + 18ll * adds + 18ll * rescales);
      }

      long long bytes() const {
        return 2*arg.threads.x*( (goes_forward(arg.sig) ? 6 : 4) * arg.force.Bytes() +
                                 (goes_forward(arg.sig) ? 3 : 2) * arg.qNuMu.Bytes() +
                                 (goes_forward(arg.sig) ? 3 : 2) * arg.oProd.Bytes() +
                                 8 * arg.link.Bytes() + 2 * arg.shortP.Bytes() +
                                 arg.qProd.Bytes() + arg.p5.Bytes() );
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
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, 1, 1, -1, 1>(arg));
          } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, 1, 0, -1, 1>(arg));
          } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, 0, 1, -1, 1>(arg));
          } else {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, 0, 0, -1, 1>(arg));
          }
        } else {
          if (goes_forward(arg.mu)) {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, -1, 1, -1, 0>(arg));
          } else {
            launch<AllLepageSideThreeLink>(tp, stream, FatLinkParam<Arg, -1, 0, -1, 0>(arg));
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
          bytes_per_site += ( 5 * arg.link.Bytes() + 2 * arg.oProd.Bytes() + arg.qProd.Bytes() );
          if (goes_forward(arg.sig)) {
            bytes_per_site += ( 2 * arg.force.Bytes() + arg.link.Bytes() );
            if (goes_forward(arg.mu)) {
              bytes_per_site += arg.qProd.Bytes();
            }
          }
        }
        return 2 * arg.threads.x * bytes_per_site;
      }
    };

    template <typename Float, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      HisqStaplesForce(GaugeField &Pmu, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu,
                       GaugeField &Qmu, GaugeField &Qnumu, GaugeField &newOprod,
                       const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {
        using real = typename mapper<Float>::type;
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
            // Out: Pmu, P3, Qmu
            // In: oprod, link
            MiddleThreeLinkArg<Float, nColor, recon> middleThreeLinkArg(newOprod, Pmu, P3, Qmu, oprod, link, act_path_coeff);
            MiddleThreeLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig, mu, newOprod, Pmu, P3, Qmu);

            for (int nu=0; nu < 8; nu++) {
              if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

              // 5-link: middle link
              // In/out: newOprod
              // Out: Pnumu, P5, Qnumu
              // In: Pmu, Qmu, link
              MiddleFiveLinkArg<Float, nColor, recon> middleFiveLinkArg(newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, act_path_coeff);
              MiddleFiveLinkForce<decltype(middleFiveLinkArg)> middleFiveLink(middleFiveLinkArg, link, sig, nu, newOprod, Pnumu, P5, Qnumu);

              // determine the remaining orthogonal direction
              int rho;
              for (rho = 0; rho < 4; rho++) {
                if (rho != pos_dir(sig) && rho != pos_dir(mu) && rho != pos_dir(nu))
                  break;
              }

              // Fused 7-link middle and side link with 5-link side link:
              // In/out: newOprod, P3 (called shortP), 
              // In: P5, Pmunu (called "oprod"), Qnumu, Qmu (called qProd), link
              AllSevenSideFiveLinkArg<Float, nColor, recon> argAll(newOprod, P3, P5, Pnumu, Qnumu, Qmu, link, act_path_coeff);
              AllSevenSideFiveLinkForce<decltype(argAll)> all(argAll, link, sig, nu, rho, newOprod, P3);

            } //nu

            // Side 3-link, fused with Lepage all link when the lepage coeff != 0.
            // In/out: newOprod
            // In: P3, Pmu (called "oProd"), Qmu (called "qProd"), link
            AllLepageSideThreeLinkArg<Float, nColor, recon> allLepageSideThreeLinkArg(newOprod, P3, Pmu, Qmu, link, act_path_coeff);
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
      auto Qmu = GaugeField::Create(gauge_param);
      auto Qnumu = GaugeField::Create(gauge_param);

      instantiate<HisqStaplesForce, ReconstructNone>(*Pmu, *P3, *P5, *Pnumu, *Qmu, *Qnumu, newOprod, oprod, link, path_coeff_array);

      delete Pmu;
      delete P3;
      delete P5;
      delete Pnumu;
      delete Qmu;
      delete Qnumu;
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
      HisqLongForce(Arg &arg, GaugeField &force, const GaugeField &meta, int sig, int mu) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        arg.sig = sig;
        arg.mu = mu;

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
        return 2*arg.threads.x*4968ll;
      }

      long long bytes() const {
        return 4*2*arg.threads.x*(2*arg.outA.Bytes() + 4*arg.link.Bytes() + 3*arg.oProd.Bytes());
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqLongLinkForce {
      HisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        LongLinkArg<real, nColor, recon> arg(newOprod, link, oldOprod, coeff);
        HisqLongForce<decltype(arg)> longLink(arg, newOprod, link, 0, 0);
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
      HisqCompleteLinkForce(Arg &arg, GaugeField &force, const GaugeField &meta, int sig, int mu) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta)
      {
        arg.sig = sig;
        arg.mu = mu;

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
        return 2*arg.threads.x*792ll;
      }

      long long bytes() const {
        return 4*2*arg.threads.x*(arg.outA.Bytes() + arg.link.Bytes() + arg.oProd.Bytes());
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqCompleteForce {
      HisqCompleteForce(GaugeField &force, const GaugeField &link)
      {
        CompleteForceArg<real, nColor, recon> arg(force, link);
        HisqCompleteLinkForce<decltype(arg)> completeForce(arg, force, link, 0, 0);
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
