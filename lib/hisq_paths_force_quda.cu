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

    template <typename Arg> class FatLinkForce : public TunableKernel3D {
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
      FatLinkForce(Arg &arg, const GaugeField &link, int sig, int mu, HisqForceType type,
                   const GaugeField &outA, const GaugeField &outB, const GaugeField &pMu,
                   const GaugeField &qMu, const GaugeField &p3) :
        TunableKernel3D(link, 2, type == FORCE_ONE_LINK ? 4 : 1),
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

        strcat(aux, (std::string(comm_dim_partitioned_string()) + "threads=" + std::to_string(arg.threads.x)).c_str());
        if (type == FORCE_MIDDLE_LINK || type == FORCE_LEPAGE_MIDDLE_LINK)
          strcat(aux, (std::string(",sig=") + std::to_string(arg.sig) +
                       std::string(",mu=") + std::to_string(arg.mu) +
                       std::string(",pMu=") + std::to_string(arg.p_mu) +
                       std::string(",q_mu=") + std::to_string(arg.q_mu) +
                       std::string(",q_prev=") + std::to_string(arg.q_prev)).c_str());
        else if (type != FORCE_ONE_LINK)
          strcat(aux, (std::string(",mu=") + std::to_string(arg.mu)).c_str()); // no sig dependence needed for side link

        switch (type) {
        case FORCE_ONE_LINK:           strcat(aux, ",ONE_LINK");           break;
        case FORCE_ALL_LINK:           strcat(aux, ",ALL_LINK");           break;
        case FORCE_MIDDLE_LINK:        strcat(aux, ",MIDDLE_LINK");        break;
        case FORCE_LEPAGE_MIDDLE_LINK: strcat(aux, ",LEPAGE_MIDDLE_LINK"); break;
        case FORCE_SIDE_LINK:          strcat(aux, ",SIDE_LINK");          break;
        case FORCE_SIDE_LINK_SHORT:    strcat(aux, ",SIDE_LINK_SHORT");    break;
        default: errorQuda("Undefined force type %d", type);
        }

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_ONE_LINK:
          launch<OneLinkTerm>(tp, stream, arg);
          break;
        case FORCE_ALL_LINK:
          if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
            launch<AllLink>(tp, stream, FatLinkParam<Arg, 1, 1>(arg));
          } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
            launch<AllLink>(tp, stream, FatLinkParam<Arg, 0, 1>(arg));
          } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
            launch<AllLink>(tp, stream, FatLinkParam<Arg, 1, 0>(arg));
          } else {
            launch<AllLink>(tp, stream, FatLinkParam<Arg, 0, 0>(arg));
          }
          break;
        case FORCE_MIDDLE_LINK:
          if (!arg.p_mu || !arg.q_mu) errorQuda("Expect p_mu=%d and q_mu=%d to both be true", arg.p_mu, arg.q_mu);
          if (arg.q_prev) {
            if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 1, true, true, true>(arg));
            } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 1, true, true, true>(arg));
            } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 0, true, true, true>(arg));
            } else {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 0, true, true, true>(arg));
            }
          } else {
            if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 1, true, true, false>(arg));
            } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 1, true, true, false>(arg));
            } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 0, true, true, false>(arg));
            } else {
              launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 0, true, true, false>(arg));
            }
          }
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
          if (arg.p_mu || arg.q_mu || !arg.q_prev)
            errorQuda("Expect p_mu=%d and q_mu=%d to both be false and q_prev=%d true", arg.p_mu, arg.q_mu, arg.q_prev);
          if (goes_forward(arg.sig) && goes_forward(arg.mu)) {
            launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 1, false, false, true>(arg));
          } else if (goes_forward(arg.sig) && goes_backward(arg.mu)) {
            launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 1, false, false, true>(arg));
          } else if (goes_backward(arg.sig) && goes_forward(arg.mu)) {
            launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 1, 0, false, false, true>(arg));
          } else {
            launch<MiddleLink>(tp, stream, FatLinkParam<Arg, 0, 0, false, false, true>(arg));
          }
          break;
        case FORCE_SIDE_LINK:
          if (goes_forward(arg.mu)) {
            launch<SideLink>(tp, stream, FatLinkParam<Arg, 1>(arg));
          } else {
            launch<SideLink>(tp, stream, FatLinkParam<Arg, 0>(arg));
          }
          break;
        case FORCE_SIDE_LINK_SHORT:
          if (goes_forward(arg.mu)) {
            launch<SideLinkShort>(tp, stream, FatLinkParam<Arg, 1>(arg));
          } else {
            launch<SideLinkShort>(tp, stream, FatLinkParam<Arg, 0>(arg));
          }
          break;
        default:
          errorQuda("Undefined force type %d", type);
        }
      }

      void preTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          outA.backup();
          break;
        case FORCE_ALL_LINK:
          outA.backup();
          outB.backup();
          break;
        case FORCE_MIDDLE_LINK:
          pMu.backup();
          qMu.backup();
          outA.backup();
          p3.backup();
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
          outA.backup();
          p3.backup();
          break;
        case FORCE_SIDE_LINK:
          outB.backup();
          outA.backup();
          break;
        case FORCE_SIDE_LINK_SHORT:
          outA.backup();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_ONE_LINK:
          outA.restore();
          break;
        case FORCE_ALL_LINK:
          outA.restore();
          outB.restore();
          break;
        case FORCE_MIDDLE_LINK:
          pMu.restore();
          qMu.restore();
          outA.restore();
          p3.restore();
          break;
        case FORCE_LEPAGE_MIDDLE_LINK:
          outA.restore();
          p3.restore();
          break;
        case FORCE_SIDE_LINK:
          outB.restore();
          outA.restore();
          break;
        case FORCE_SIDE_LINK_SHORT:
          outA.restore();
          break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads.x*36ll;
        case FORCE_ALL_LINK:
          return 2*arg.threads.x*(goes_forward(arg.sig) ? 1242ll : 828ll);
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads.x*(2 * 198 +
                                (!arg.q_prev && goes_forward(arg.sig) ? 198 : 0) +
                                (arg.q_prev && (arg.q_mu || goes_forward(arg.sig) ) ? 198 : 0) +
                                ((arg.q_prev && goes_forward(arg.sig) ) ?  198 : 0) +
                                ( goes_forward(arg.sig) ? 216 : 0) );
        case FORCE_SIDE_LINK:       return 2*arg.threads.x*2*234;
        case FORCE_SIDE_LINK_SHORT: return 2*arg.threads.x*36;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_ONE_LINK:
          return 2*4*arg.threads.x*( arg.oProd.Bytes() + 2*arg.outA.Bytes() );
        case FORCE_ALL_LINK:
          return 2*arg.threads.x*( (goes_forward(arg.sig) ? 4 : 2)*arg.outA.Bytes() + 3*arg.link.Bytes()
                                 + arg.oProd.Bytes() + arg.qPrev.Bytes() + 2*arg.outB.Bytes());
        case FORCE_MIDDLE_LINK:
        case FORCE_LEPAGE_MIDDLE_LINK:
          return 2*arg.threads.x*( ( goes_forward(arg.sig) ? 2*arg.outA.Bytes() : 0 ) +
                                 (arg.p_mu ? arg.pMu.Bytes() : 0) +
                                 (arg.q_mu ? arg.qMu.Bytes() : 0) +
                                 ( ( goes_forward(arg.sig) || arg.q_mu ) ? arg.qPrev.Bytes() : 0) +
                                 arg.p3.Bytes() + 3*arg.link.Bytes() + arg.oProd.Bytes() );
        case FORCE_SIDE_LINK:
          return 2*arg.threads.x*( 2*arg.outA.Bytes() + 2*arg.outB.Bytes() +
                                 arg.p3.Bytes() + arg.link.Bytes() + arg.qProd.Bytes() );
        case FORCE_SIDE_LINK_SHORT:
          return 2*arg.threads.x*( 2*arg.outA.Bytes() + arg.p3.Bytes() );
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqStaplesForce {
      HisqStaplesForce(GaugeField &Pmu, GaugeField &P3, GaugeField &P5, GaugeField &Pnumu,
                       GaugeField &Qmu, GaugeField &Qnumu, GaugeField &newOprod,
                       const GaugeField &oprod, const GaugeField &link,
                       const double *path_coeff_array)
      {
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
          FatLinkArg<real, nColor> arg(newOprod, oprod, link, OneLink, FORCE_ONE_LINK);
          arg.threads.z = 4;
          FatLinkForce<decltype(arg)> oneLink(arg, link, 0, 0, FORCE_ONE_LINK, newOprod, newOprod, oprod, oprod, oprod);
        }

        for (int sig=0; sig<8; sig++) {
          for (int mu=0; mu<8; mu++) {
            if ( (mu == sig) || (mu == opp_dir(sig))) continue;

            //3-link
            //Kernel A: middle link
            FatLinkArg<real, nColor> middleLinkArg(newOprod, Pmu, P3, Qmu, oprod, link, mThreeSt, 2, FORCE_MIDDLE_LINK);
            FatLinkForce<decltype(middleLinkArg)> middleLink(middleLinkArg, link, sig, mu, FORCE_MIDDLE_LINK, newOprod, newOprod, Pmu, P3, Qmu);

            for (int nu=0; nu < 8; nu++) {
              if (nu == sig || nu == opp_dir(sig) || nu == mu || nu == opp_dir(mu)) continue;

              //5-link: middle link
              //Kernel B
              FatLinkArg<real, nColor> middleLinkArg(newOprod, Pnumu, P5, Qnumu, Pmu, Qmu, link, FiveSt, 1, FORCE_MIDDLE_LINK);
              FatLinkForce<decltype(middleLinkArg)> middleLink(middleLinkArg, link, sig, nu, FORCE_MIDDLE_LINK, newOprod, newOprod, Pnumu, P5, Qnumu);

              for (int rho = 0; rho < 8; rho++) {
                if (rho == sig || rho == opp_dir(sig) || rho == mu || rho == opp_dir(mu) || rho == nu || rho == opp_dir(nu)) continue;

                //7-link: middle link and side link
                FatLinkArg<real, nColor> arg(newOprod, P5, Pnumu, Qnumu, link, SevenSt, FiveSt != 0 ? SevenSt/FiveSt : 0, 1, FORCE_ALL_LINK, true);
                FatLinkForce<decltype(arg)> all(arg, link, sig, rho, FORCE_ALL_LINK, newOprod, P5, P5, P5, Qnumu);

              }//rho

              //5-link: side link
              FatLinkArg<real, nColor> arg(newOprod, P3, P5, Qmu, link, mFiveSt, (ThreeSt != 0 ? FiveSt/ThreeSt : 0), 1, FORCE_SIDE_LINK);
              FatLinkForce<decltype(arg)> side(arg, link, sig, nu, FORCE_SIDE_LINK, newOprod, P3, P5, P5, Qmu);

            } //nu

            //lepage
            if (Lepage != 0.) {
              FatLinkArg<real, nColor> middleLinkArg(newOprod, P5, Pmu, Qmu, link, Lepage, 2, FORCE_LEPAGE_MIDDLE_LINK);
              FatLinkForce<decltype(middleLinkArg)> middleLink(middleLinkArg, link, sig, mu, FORCE_LEPAGE_MIDDLE_LINK, newOprod, newOprod, P5, P5, Qmu);

              FatLinkArg<real, nColor> arg(newOprod, P3, P5, Qmu, link, mLepage, (ThreeSt != 0 ? Lepage/ThreeSt : 0), 2, FORCE_SIDE_LINK);
              FatLinkForce<decltype(arg)> side(arg, link, sig, mu, FORCE_SIDE_LINK, newOprod, P3, P5, P5, Qmu);
            } // Lepage != 0.0

            // 3-link side link
            FatLinkArg<real, nColor> arg(newOprod, P3, link, ThreeSt, 1, FORCE_SIDE_LINK_SHORT);
            FatLinkForce<decltype(arg)> side(arg, P3, sig, mu, FORCE_SIDE_LINK_SHORT, newOprod, newOprod, P3, P3, P3);
          }//mu
        }//sig
      }
    };

#ifdef GPU_HISQ_FORCE
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
    class HisqForce : public TunableKernel2D {

      Arg &arg;
      GaugeField &force;
      const GaugeField &meta;
      const HisqForceType type;
      unsigned int minThreads() const { return arg.threads.x; }

    public:
      HisqForce(Arg &arg, GaugeField &force, const GaugeField &meta, int sig, int mu, HisqForceType type) :
        TunableKernel2D(meta, 2),
        arg(arg),
        force(force),
        meta(meta),
        type(type)
      {
        arg.sig = sig;
        arg.mu = mu;
        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        switch (type) {
        case FORCE_LONG_LINK: launch<LongLink>(tp, stream, arg); break;
        case FORCE_COMPLETE:  launch<CompleteForce>(tp, stream, arg); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      TuneKey tuneKey() const {
        std::stringstream aux;
        aux << meta.AuxString() << comm_dim_partitioned_string() << ",threads=" << arg.threads.x;
        switch (type) {
        case FORCE_LONG_LINK: aux << ",LONG_LINK"; break;
        case FORCE_COMPLETE:  aux << ",COMPLETE";  break;
        default: errorQuda("Undefined force type %d", type);
        }
        return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
      }

      void preTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          force.backup(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      void postTune() {
        switch (type) {
        case FORCE_LONG_LINK:
        case FORCE_COMPLETE:
          force.restore(); break;
        default: errorQuda("Undefined force type %d", type);
        }
      }

      long long flops() const {
        switch (type) {
        case FORCE_LONG_LINK: return 2*arg.threads.x*4968ll;
        case FORCE_COMPLETE:  return 2*arg.threads.x*792ll;
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }

      long long bytes() const {
        switch (type) {
        case FORCE_LONG_LINK: return 4*2*arg.threads.x*(2*arg.outA.Bytes() + 4*arg.link.Bytes() + 3*arg.oProd.Bytes());
        case FORCE_COMPLETE:  return 4*2*arg.threads.x*(arg.outA.Bytes() + arg.link.Bytes() + arg.oProd.Bytes());
        default: errorQuda("Undefined force type %d", type);
        }
        return 0;
      }
    };

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqLongLinkForce {
      HisqLongLinkForce(GaugeField &newOprod, const GaugeField &oldOprod, const GaugeField &link, double coeff)
      {
        LongLinkArg<real, nColor, recon> arg(newOprod, link, oldOprod, coeff);
        HisqForce<decltype(arg)> longLink(arg, newOprod, link, 0, 0, FORCE_LONG_LINK);
      }
    };

#ifdef GPU_HISQ_FORCE
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

    template <typename real, int nColor, QudaReconstructType recon>
    struct HisqCompleteForce {
      HisqCompleteForce(GaugeField &force, const GaugeField &link)
      {
        CompleteForceArg<real, nColor, recon> arg(force, link);
        HisqForce<decltype(arg)> completeForce(arg, force, link, 0, 0, FORCE_COMPLETE);
      }
    };

#ifdef GPU_HISQ_FORCE
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
