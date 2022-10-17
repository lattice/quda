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
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 1, 1, true, true, false>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 0, 1, true, true, false>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 1, 0, true, true, false>(arg));
        } else {
          launch<MiddleThreeLink>(tp, stream, FatLinkParam<Arg, 0, 0, true, true, false>(arg));
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

    template <typename Arg> class MiddleFiveLinkForce : public TunableKernel3D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &pNuMu;
      const GaugeField &p5;
      const GaugeField &qNuMu;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      MiddleFiveLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &pNuMu,
                   const GaugeField &p5, const GaugeField &qNuMu) :
        TunableKernel3D(link, 2, 1),
        arg(arg),
        force(force),
        pNuMu(pNuMu),
        p5(p5),
        qNuMu(qNuMu),
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
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 1, 1, true, true, true>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 0, 1, true, true, true>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 1, 0, true, true, true>(arg));
        } else {
          launch<MiddleFiveLink>(tp, stream, FatLinkParam<Arg, 0, 0, true, true, true>(arg));
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

    template <typename Arg> class AllLinkForce : public TunableKernel3D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      AllLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &shortP) :
        TunableKernel3D(link, 2, 1),
        arg(arg),
        force(force),
        shortP(shortP),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        strcat(aux, ",mu=");
        u32toa(aux2, arg.mu);
        strcat(aux, aux2);
        // no sig dependence needed for side link
        // does the all link force also need sigma dependence??

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
          launch<AllLink>(tp, stream, FatLinkParam<Arg, 1, 1>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
          launch<AllLink>(tp, stream, FatLinkParam<Arg, 0, 1>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
          launch<AllLink>(tp, stream, FatLinkParam<Arg, 1, 0>(arg));
        } else {
          launch<AllLink>(tp, stream, FatLinkParam<Arg, 0, 0>(arg));
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
        return 2*arg.threads.x*(goes_forward(arg.sig) ? 1242ll : 828ll);
      }

      long long bytes() const {
        return 2*arg.threads.x*( (goes_forward(arg.sig) ? 4 : 2)*arg.force.Bytes() + 3*arg.link.Bytes()
                               + arg.oProd.Bytes() + arg.p5.Bytes() + 2*arg.shortP.Bytes());
      }
    };

    template <typename Arg> class SideLinkForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &shortP;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      SideLinkForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &shortP) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
        shortP(shortP),
        link(link)
      {
        arg.sig = sig;
        arg.mu = mu;

        char aux2[16];
        strcat(aux, comm_dim_partitioned_string());
        strcat(aux, ",threads=");
        u32toa(aux2, arg.threads.x);
        strcat(aux, aux2);
        strcat(aux, ",mu=");
        u32toa(aux2, arg.mu);
        strcat(aux, aux2);
        // no sig dependence needed for side link

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.mu)) {
          launch<SideLink>(tp, stream, FatLinkParam<Arg, 1>(arg));
        } else {
          launch<SideLink>(tp, stream, FatLinkParam<Arg, 0>(arg));
        }
      }

      void preTune() {
        shortP.backup();
        force.backup();
      }

      void postTune() {
        shortP.restore();
        force.restore();
      }

      long long flops() const {
        return 2*arg.threads.x*2*234;
      }

      long long bytes() const {
        return 2*arg.threads.x*( 2*arg.force.Bytes() + 2*arg.shortP.Bytes() +
                               arg.p3.Bytes() + arg.link.Bytes() + arg.qProd.Bytes() );
      }
    };

    template <typename Arg> class LepageMiddleLinkForce : public TunableKernel3D {
      Arg &arg;
      const GaugeField &outA;
      const GaugeField &outB;
      const GaugeField &pMu;
      const GaugeField &qMu;
      const GaugeField &p3;
      const GaugeField &link;
      const HisqForceType type;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      LepageMiddleLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, HisqForceType type,
                   const GaugeField &outA, const GaugeField &outB, const GaugeField &pMu,
                   const GaugeField &qMu, const GaugeField &p3) :
        TunableKernel3D(link, 2, 1),
        arg(arg),
        outA(outA),
        outB(outB),
        pMu(pMu),
        qMu(qMu),
        p3(p3),
        link(link),
        type(type)
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
        // no need to embed p_mu, q_mu, q_prev because certain values are enforced in `apply`

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (arg.p_mu || arg.q_mu || !arg.q_prev)
          errorQuda("Expect p_mu=%d and q_mu=%d to both be false and q_prev=%d true", arg.p_mu, arg.q_mu, arg.q_prev);
        if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
          launch<LepageMiddleLink>(tp, stream, FatLinkParam<Arg, 1, 1, false, false, true>(arg));
        } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
          launch<LepageMiddleLink>(tp, stream, FatLinkParam<Arg, 0, 1, false, false, true>(arg));
        } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
          launch<LepageMiddleLink>(tp, stream, FatLinkParam<Arg, 1, 0, false, false, true>(arg));
        } else {
          launch<LepageMiddleLink>(tp, stream, FatLinkParam<Arg, 0, 0, false, false, true>(arg));
        }
      }

      void preTune() {
        outA.backup();
        p3.backup();
      }

      void postTune() {
        outA.restore();
        p3.restore();
      }

      long long flops() const {
        return 2*arg.threads.x*(2 * 198 +
                              (!arg.q_prev && goes_forward(arg.sig) ? 198 : 0) +
                              (arg.q_prev && (arg.q_mu || goes_forward(arg.sig) ) ? 198 : 0) +
                              ((arg.q_prev && goes_forward(arg.sig) ) ?  198 : 0) +
                              ( goes_forward(arg.sig) ? 216 : 0) );
      }

      long long bytes() const {
        return 2*arg.threads.x*( ( goes_forward(arg.sig) ? 2*arg.outA.Bytes() : 0 ) +
                               (arg.p_mu ? arg.pMu.Bytes() : 0) +
                               (arg.q_mu ? arg.qMu.Bytes() : 0) +
                               ( ( goes_forward(arg.sig) || arg.q_mu ) ? arg.qPrev.Bytes() : 0) +
                               arg.p3.Bytes() + 3*arg.link.Bytes() + arg.oProd.Bytes() );
      }
    };
    

    template <typename Arg> class SideLinkShortForce : public TunableKernel2D {
      Arg &arg;
      const GaugeField &force;
      const GaugeField &p3;
      const GaugeField &link;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      SideLinkShortForce(Arg &arg, const GaugeField &link, int sig, int mu,
                   const GaugeField &force, const GaugeField &p3) :
        TunableKernel2D(link, 2),
        arg(arg),
        force(force),
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
        strcat(aux, ",mu=");
        u32toa(aux2, arg.mu);
        strcat(aux, aux2);

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (goes_forward(arg.mu)) {
          launch<SideLinkShort>(tp, stream, FatLinkParam<Arg, 1>(arg));
        } else {
          launch<SideLinkShort>(tp, stream, FatLinkParam<Arg, 0>(arg));
        }
      }

      void preTune() {
        force.backup();
      }

      void postTune() {
        force.restore();
      }

      long long flops() const {
        return 2*arg.threads.x*36;
      }

      long long bytes() const {
        return 2*arg.threads.x*( 2*arg.force.Bytes() + arg.p3.Bytes() );
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
        real OneLink = act_path_coeff.one;
        real ThreeSt = act_path_coeff.three;
        real mThreeSt = -ThreeSt;
        real FiveSt  = act_path_coeff.five;
        real mFiveSt  = -FiveSt;
        real SevenSt = act_path_coeff.seven;
        real Lepage  = act_path_coeff.lepage;
        real mLepage  = -Lepage;

        {
          OneLinkArg<real, nColor, recon> arg(newOprod, oprod, link, OneLink);
          OneLinkForce<decltype(arg)> oneLink(arg, link, newOprod);
        }

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) continue;

            //3-link
            //Kernel A: middle link
            MiddleThreeLinkArg<Float, nColor, recon> middleThreeLinkArg(newOprod, Pmu, P3, Qmu, oprod, link, mThreeSt, 2);
            MiddleThreeLinkForce<decltype(middleThreeLinkArg)> middleThreeLink(middleThreeLinkArg, link, sig, mu, newOprod, Pmu, P3, Qmu);

            for (int nu=0; nu < 8; nu++) {
              if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

              //5-link: middle link
              //Kernel B
              MiddleFiveLinkArg<Float, nColor, recon> middleFiveLinkArg(newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, FiveSt, 1);
              MiddleFiveLinkForce<decltype(middleFiveLinkArg)> middleFiveLink(middleFiveLinkArg, link, sig, nu, newOprod, Pnumu, P5, Qnumu);

              for (int rho = 0; rho < 8; rho++) {
                if (rho == sig || rho == opp_dir(sig) || rho == mu || rho == opp_dir(mu) || rho == nu || rho == opp_dir(nu)) continue;

                //7-link: middle link and side link
                AllLinkArg<Float, nColor, recon> arg(newOprod, P5, Pnumu, Qnumu, link, SevenSt, FiveSt != 0 ? SevenSt/FiveSt : 0, 1);
                AllLinkForce<decltype(arg)> all(arg, link, sig, rho, newOprod, P5);
              }//rho

              //5-link: side link
              SideLinkArg<Float, nColor, recon> arg(newOprod, P3, P5, Qmu, link, mFiveSt, (ThreeSt != 0 ? FiveSt/ThreeSt : 0), 1);
              SideLinkForce<decltype(arg)> side(arg, link, sig, nu, newOprod, P3);

            } //nu

            //lepage
            if (Lepage != 0.) {
              LepageMiddleLinkArg<Float, nColor, recon> middleLinkArg(newOprod, P5, Pmu, Qmu, link, Lepage, 2, FORCE_LEPAGE_MIDDLE_LINK);
              LepageMiddleLinkForce<decltype(middleLinkArg)> middleLink(middleLinkArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK, newOprod, newOprod, P5, P5, Qmu);

              SideLinkArg<Float, nColor, recon> arg(newOprod, P3, P5, Qmu, link, mLepage, (ThreeSt != 0 ? Lepage/ThreeSt : 0), 2);
              SideLinkForce<decltype(arg)> side(arg, link, sig, mu, newOprod, P3);
            } // Lepage != 0.0

            // 3-link side link
            SideLinkShortArg<Float, nColor, recon> arg(newOprod, P3, link, ThreeSt, 1);
            SideLinkShortForce<decltype(arg)> side(arg, P3, sig, mu, newOprod, P3);
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
