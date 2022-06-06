#include <cstdio>

#include <quda_internal.h>
#include <gauge_field.h>
#include <llfat_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/llfat.cuh>

#define MIN_COEFF 1e-7

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class LongLink : public TunableKernel3D {
    LinkArg<Float, nColor, recon> arg;
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    LongLink(const GaugeField &u, GaugeField &lng, double coeff) :
      TunableKernel3D(lng, 2, 4),
      arg(lng, u, coeff)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeLongLink>(tp, stream, arg);
    }

    long long flops() const { return 2*4*arg.threads.x*198; }
    long long bytes() const { return 2*4*arg.threads.x*(3*arg.u.Bytes()+arg.link.Bytes()); }
  };

  void computeLongLink(GaugeField &lng, const GaugeField &u, double coeff)
  {
    instantiate<LongLink, ReconstructNo12>(u, lng, coeff); // u first arg so we pick its recon
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class OneLink : public TunableKernel3D {
    LinkArg<Float, nColor, recon> arg;
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    OneLink(const GaugeField &u, GaugeField &fat, double coeff) :
      TunableKernel3D(fat, 2, 4),
      arg(fat, u, coeff)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeOneLink>(tp, stream, arg);
    }

    long long flops() const { return 2*4*arg.threads.x*18; }
    long long bytes() const { return 2*4*arg.threads.x*(arg.u.Bytes()+arg.link.Bytes()); }
  };

  void computeOneLink(GaugeField &fat, const GaugeField &u, double coeff)
  {
    if (u.StaggeredPhase() != QUDA_STAGGERED_PHASE_MILC && u.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
    instantiate<OneLink, ReconstructNo12>(u, fat, coeff);
  }

  template <typename Float, int nColor, QudaReconstructType recon> class Staple : public TunableKernel3D {
    GaugeField &fat;
    GaugeField &staple;
    const GaugeField &mulink;
    const GaugeField &u;
    int nu;
    int mu_map[4];
    int dir1;
    int dir2;
    Float coeff;
    bool save_staple;

    dim3 threads() const
    {
      dim3 t(1, 2, 1);
      for (int d = 0; d < 4; d++) t.x *= (fat.X()[d] + u.X()[d]) / 2;
      t.x /= 2; // account for parity in y dimension
      t.z = (3 - ( (dir1 > -1) ? 1 : 0 ) - ( (dir2 > -1) ? 1 : 0 ));
      return t;
    }
    unsigned int minThreads() const { return threads().x; }

  public:
    Staple(const GaugeField &u, GaugeField &fat, GaugeField &staple, const GaugeField &mulink,
           int nu, int dir1, int dir2, double coeff, bool save_staple) :
      TunableKernel3D(fat, 2, (3 - ( (dir1 > -1) ? 1 : 0 ) - ( (dir2 > -1) ? 1 : 0 ))),
      fat(fat),
      staple(staple),
      mulink(mulink),
      u(u),
      nu(nu),
      dir1(dir1),
      dir2(dir2),
      coeff(static_cast<Float>(coeff)),
      save_staple(save_staple)
    {
      // compute the map for z thread index to mu index in the kernel
      // mu != nu 3 -> n_mu = 3
      // mu != nu != rho 2 -> n_mu = 2
      // mu != nu != rho != sig 1 -> n_mu = 1
      int j=0;
      for (int i=0; i<4; i++) {
        if (i==nu || i==dir1 || i==dir2) continue; // skip these dimensions
        mu_map[j++] = i;
      }
      assert((unsigned)j == threads().z);

      if (mulink.Reconstruct() != QUDA_RECONSTRUCT_12) strcat(aux, ",mulink_recon=12");
      strcat(aux, comm_dim_partitioned_string());
      std::stringstream aux_;
      aux_ << ",nu=" << nu << ",dir1=" << dir1 << ",dir2=" << dir2 << ",save=" << save_staple;
      strcat(aux, aux_.str().c_str());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (mulink.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        if (save_staple) {
          StapleArg<Float, nColor, recon, QUDA_RECONSTRUCT_NO, true> arg(fat, staple, mulink, u, coeff, nu, mu_map);
          launch<ComputeStaple>(tp, stream, arg);
        } else {
          StapleArg<Float, nColor, recon, QUDA_RECONSTRUCT_NO, false> arg(fat, staple, mulink, u, coeff, nu, mu_map);
          launch<ComputeStaple>(tp, stream, arg);
        }
      } else if (mulink.Reconstruct() == recon) {
        if (save_staple) {
          StapleArg<Float, nColor, recon, recon, true> arg(fat, staple, mulink, u, coeff, nu, mu_map);
          launch<ComputeStaple>(tp, stream, arg);
        } else {
          StapleArg<Float, nColor, recon, recon, false> arg(fat, staple, mulink, u, coeff, nu, mu_map);
          launch<ComputeStaple>(tp, stream, arg);
        }
      } else {
        errorQuda("Reconstruct %d is not supported\n", u.Reconstruct());
      }
    }

    void preTune() { fat.backup(); staple.backup(); }
    void postTune() { fat.restore(); staple.restore(); }
    long long flops() const { return threads().x * threads().y * threads().z * (4 * 198 + 18 + 36 ); }
    long long bytes() const {
      return (fat.VolumeCB() * fat.Reconstruct() * 2 // fat load/store is only done on interior
              + threads().x * (4 * u.Reconstruct() + 2 * mulink.Reconstruct() + (save_staple ? staple.Reconstruct() : 0))) *
        threads().y * threads().z * u.Precision();
    }
  };

  // Compute the staple field for direction nu,excluding the directions dir1 and dir2.
  void computeStaple(GaugeField &fat, GaugeField &staple, const GaugeField &mulink, const GaugeField &u,
		     int nu, int dir1, int dir2, double coeff, bool save_staple)
  {
    instantiate<Staple, ReconstructNo12>(u, fat, staple, mulink, nu, dir1, dir2, coeff, save_staple);
  }

#ifdef GPU_STAGGERED_DIRAC
  void longKSLink(GaugeField *lng, const GaugeField &u, const double *coeff)
  {
    computeLongLink(*lng, u, coeff[1]);
  }

  void fatKSLink(GaugeField *fat, const GaugeField& u, const double *coeff)
  {
    GaugeFieldParam gParam(u);
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.setPrecision(gParam.Precision());
    gParam.create = QUDA_NULL_FIELD_CREATE;
    auto staple = GaugeField::Create(gParam);
    auto staple1 = GaugeField::Create(gParam);

    if ( ((fat->X()[0] % 2 != 0) || (fat->X()[1] % 2 != 0) || (fat->X()[2] % 2 != 0) || (fat->X()[3] % 2 != 0))
	&& (u.Reconstruct()  != QUDA_RECONSTRUCT_NO)){
      errorQuda("Reconstruct %d and odd dimensionsize is not supported by link fattening code (yet)\n",
		u.Reconstruct());
    }

    computeOneLink(*fat, u, coeff[0]-6.0*coeff[5]);

    // Check the coefficients. If all of the following are zero, return.
    if (fabs(coeff[2]) >= MIN_COEFF || fabs(coeff[3]) >= MIN_COEFF ||
	fabs(coeff[4]) >= MIN_COEFF || fabs(coeff[5]) >= MIN_COEFF) {

      for (int nu = 0; nu < 4; nu++) {
        computeStaple(*fat, *staple, u, u, nu, -1, -1, coeff[2], 1);

        if (coeff[5] != 0.0) computeStaple(*fat, *staple, *staple, u, nu, -1, -1, coeff[5], 0);

        for (int rho = 0; rho < 4; rho++) {
          if (rho != nu) {

            computeStaple(*fat, *staple1, *staple, u, rho, nu, -1, coeff[3], 1);

            if (fabs(coeff[4]) > MIN_COEFF) {
              for (int sig = 0; sig < 4; sig++) {
                if (sig != nu && sig != rho) {
                  computeStaple(*fat, *staple, *staple1, u, sig, nu, rho, coeff[4], 0);
                }
              } //sig
            } // MIN_COEFF
          }
        } //rho
      } //nu
    }

    delete staple;
    delete staple1;
  }
#else
  void longKSLink(GaugeField *, const GaugeField&, const double *)
  {
    errorQuda("Long-link computation not enabled");
  }

  void fatKSLink(GaugeField *, const GaugeField&, const double *)
  {
    errorQuda("Fat-link computation not enabled");
  }
#endif

#undef MIN_COEFF

} // namespace quda
