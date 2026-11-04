#pragma once

#include <quda_internal.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/llfat.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class LongLink : public TunableKernel3D
  {
    LinkArg<Float, nColor, recon> arg;
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    LongLink(const GaugeField &u, GaugeField &lng, double coeff) : TunableKernel3D(lng, 2, 4), arg(lng, u, coeff)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeLongLink>(tp, stream, arg);
    }

    long long flops() const { return 2 * 4 * arg.threads.x * 198; }
    long long bytes() const { return 2 * 4 * arg.threads.x * (3 * arg.u.Bytes() + arg.link.Bytes()); }
  };

  template <typename Float, QudaReconstructType recon>
  void applyLongLink(const GaugeField &u, GaugeField &lng, double coeff)
  {
    LongLink<Float, 3, recon>(u, lng, coeff);
  }

  template <typename Float, int nColor, QudaReconstructType recon> class OneLink : public TunableKernel3D
  {
    LinkArg<Float, nColor, recon> arg;
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    OneLink(const GaugeField &u, GaugeField &fat, double coeff) : TunableKernel3D(fat, 2, 4), arg(fat, u, coeff)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeOneLink>(tp, stream, arg);
    }

    long long flops() const { return 2 * 4 * arg.threads.x * 18; }
    long long bytes() const { return 2 * 4 * arg.threads.x * (arg.u.Bytes() + arg.link.Bytes()); }
  };

  template <typename Float, QudaReconstructType recon>
  void applyOneLink(const GaugeField &u, GaugeField &fat, double coeff)
  {
    OneLink<Float, 3, recon>(u, fat, coeff);
  }

  template <typename Float, int nColor, QudaReconstructType recon> class Staple : public TunableKernel3D
  {
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
      t.z = (3 - ((dir1 > -1) ? 1 : 0) - ((dir2 > -1) ? 1 : 0));
      return t;
    }
    unsigned int minThreads() const { return threads().x; }

  public:
    Staple(const GaugeField &u, GaugeField &fat, GaugeField &staple, const GaugeField &mulink, int nu, int dir1,
           int dir2, double coeff, bool save_staple) :
      TunableKernel3D(fat, 2, (3 - ((dir1 > -1) ? 1 : 0) - ((dir2 > -1) ? 1 : 0))),
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
      int j = 0;
      for (int i = 0; i < 4; i++) {
        if (i == nu || i == dir1 || i == dir2) continue; // skip these dimensions
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

    void apply(const qudaStream_t &stream)
    {
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

    void preTune()
    {
      fat.backup();
      staple.backup();
    }
    void postTune()
    {
      fat.restore();
      staple.restore();
    }
    long long flops() const { return threads().x * threads().y * threads().z * (4 * 198 + 18 + 36); }
    long long bytes() const
    {
      return (fat.VolumeCB() * fat.Reconstruct() * 2 // fat load/store is only done on interior
              + threads().x * (4 * u.Reconstruct() + 2 * mulink.Reconstruct() + (save_staple ? staple.Reconstruct() : 0)))
        * threads().y * threads().z * u.Precision();
    }
  };

  template <typename Float, QudaReconstructType recon>
  void applyStaple(const GaugeField &u, GaugeField &fat, GaugeField &staple, const GaugeField &mulink, int nu, int dir1,
                   int dir2, double coeff, bool save_staple)
  {
    Staple<Float, 3, recon>(u, fat, staple, mulink, nu, dir1, dir2, coeff, save_staple);
  }

} // namespace quda
