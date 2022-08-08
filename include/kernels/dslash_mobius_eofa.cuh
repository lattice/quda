#pragma once

#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.h>
#include <math_helper.cuh>
#include <domain_wall_helper.h>
#include <kernel.h>
#include <dslash_quda.h>

namespace quda
{
  namespace mobius_eofa
  {

    /**
      @brief Structure containing the EOFA coefficients
     */
    template <typename real> struct eofa_coeff {
      real u[QUDA_MAX_DWF_LS]; // xpay coefficients
      real x[QUDA_MAX_DWF_LS];
      real y[QUDA_MAX_DWF_LS];
    };

    template <typename storage_type, int nColor_, bool pm_, bool dagger_, bool xpay_, Dslash5Type type_>
    struct Dslash5Arg : kernel_param<> {
      static constexpr int nColor = nColor_;
      static constexpr bool pm = pm_;
      static constexpr bool dagger = dagger_;
      static constexpr bool xpay = xpay_;
      static constexpr Dslash5Type type = type_;

      typedef typename colorspinor_mapper<storage_type, 4, nColor>::type F;
      typedef typename mapper<storage_type>::type real;

      F out;                  // output vector field
      const F in;             // input vector field
      const F x;              // auxiliary input vector field
      const int nParity;      // number of parities we're working on
      const int volume_cb;    // checkerboarded volume
      const int volume_4d_cb; // 4-d checkerboarded volume
      const int_fastdiv Ls;   // length of 5th dimension

      const real m_f; // fermion mass parameter
      const real m_5; // Wilson mass shift

      real a; // real xpay coefficient

      real kappa;
      real inv;

      real sherman_morrison;

      eofa_coeff<real> coeff;

      Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, const double m_f_,
                 const double m_5_, const Complex */*b_5_*/, const Complex */*c_5_*/, double a_, double inv_, double kappa_,
                 const double *eofa_u, const double *eofa_x, const double *eofa_y, double sherman_morrison_) :
        kernel_param(dim3(in.VolumeCB() / in.X(4), in.X(4), in.SiteSubset())),
        out(out),
        in(in),
        x(x),
        nParity(in.SiteSubset()),
        volume_cb(in.VolumeCB()),
        volume_4d_cb(volume_cb / in.X(4)),
        Ls(in.X(4)),
        m_f(m_f_),
        m_5(m_5_),
        a(a_),
        kappa(kappa_),
        inv(inv_),
        sherman_morrison(sherman_morrison_)
      {
        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

        switch (type) {
        case Dslash5Type::M5_EOFA:
          for (int s = 0; s < Ls; s++) { coeff.u[s] = eofa_u[s]; }
          break;
        case Dslash5Type::M5INV_EOFA:
          for (int s = 0; s < Ls; s++) {
            coeff.u[s] = eofa_u[s];
            coeff.x[s] = eofa_x[s];
            coeff.y[s] = eofa_y[s];
          }
          break;
        default: errorQuda("Unexpected EOFA Dslash5Type %d", static_cast<int>(type));
        }
      }
    };

    /**
      @brief Apply the D5 operator at given site
      @param[in] arg    Argument struct containing any meta data and accessors
      @param[in] parity Parity we are on
      @param[in] x_cb   Checkerboarded 4-d space-time index
      @param[in] s      Ls dimension coordinate
     */
    template <typename Arg> struct eofa_dslash5 {
      const Arg &arg;
      constexpr eofa_dslash5(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline void operator()(int x_cb, int s, int parity)
      {
        using real = typename Arg::real;
        typedef ColorSpinor<real, Arg::nColor, 4> Vector;

        SharedMemoryCache<Vector> cache(target::block_dim());

        Vector out;
        cache.save(arg.in(s * arg.volume_4d_cb + x_cb, parity));
        cache.sync();

        auto Ls = arg.Ls;

        { // forwards direction
          const Vector in = cache.load(threadIdx.x, (s + 1) % Ls, threadIdx.z);
          constexpr int proj_dir = Arg::dagger ? +1 : -1;
          if (s == Ls - 1) {
            out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
          } else {
            out += in.project(4, proj_dir).reconstruct(4, proj_dir);
          }
        }

        { // backwards direction
          const Vector in = cache.load(threadIdx.x, (s + Ls - 1) % Ls, threadIdx.z);
          constexpr int proj_dir = Arg::dagger ? -1 : +1;
          if (s == 0) {
            out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
          } else {
            out += in.project(4, proj_dir).reconstruct(4, proj_dir);
          }
        }

        if (Arg::type == Dslash5Type::M5_EOFA) {
          Vector diagonal = cache.load(threadIdx.x, s, threadIdx.z);
          out = (static_cast<real>(0.5) * arg.kappa) * out + diagonal; // 1 + kappa*D5; the 0.5 for spin projection

          constexpr int proj_dir = Arg::pm ? +1 : -1;

          if (Arg::dagger) {
            if (s == (Arg::pm ? Ls - 1 : 0)) {
              for (int sp = 0; sp < Ls; sp++) {
                out += (static_cast<real>(0.5) * arg.coeff.u[sp])
                  * cache.load(threadIdx.x, sp, threadIdx.z).project(4, proj_dir).reconstruct(4, proj_dir);
              }
            }
          } else {
            out += (static_cast<real>(0.5) * arg.coeff.u[s])
              * cache.load(threadIdx.x, Arg::pm ? Ls - 1 : 0, threadIdx.z).project(4, proj_dir).reconstruct(4, proj_dir);
          }

          if (Arg::xpay) { // really axpy
            Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
            out = arg.a * x + out;
          }
        }
        arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
      }
    };

    /**
      @brief Apply the M5 inverse operator at a given site on the
      lattice.  This is the original algorithm as described in Kim and
      Izubushi (LATTICE 2013_033), where the b and c coefficients are
      constant along the Ls dimension, so is suitable for Shamir and
      Mobius domain-wall fermions.

      @param[in] arg    Argument struct containing any meta data and accessors
      @param[in] parity Parity we are on
      @param[in] x_cb   Checkerboarded 4-d space-time index
      @param[in] s      Ls dimension coordinate
     */
    template <typename Arg> struct eofa_dslash5inv {
      const Arg &arg;
      constexpr eofa_dslash5inv(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline void operator()(int x_cb, int s, int parity)
      {
        using real = typename Arg::real;
        typedef ColorSpinor<real, Arg::nColor, 4> Vector;

        const auto sherman_morrison = arg.sherman_morrison;
        SharedMemoryCache<Vector> cache(target::block_dim());
        cache.save(arg.in(s * arg.volume_4d_cb + x_cb, parity));
        cache.sync();

        Vector out;

        for (int sp = 0; sp < arg.Ls; sp++) {
          Vector in = cache.load(threadIdx.x, sp, threadIdx.z);
          {
            int exp = s < sp ? arg.Ls - sp + s : s - sp;
            real factorR = 0.5 * arg.coeff.y[Arg::pm ? arg.Ls - exp - 1 : exp] * (s < sp ? -arg.m_f : static_cast<real>(1.0));
            constexpr int proj_dir = Arg::dagger ? -1 : +1;
            out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
          }
          {
            int exp = s > sp ? arg.Ls - s + sp : sp - s;
            real factorL = 0.5 * arg.coeff.y[Arg::pm ? arg.Ls - exp - 1 : exp] * (s > sp ? -arg.m_f : static_cast<real>(1.0));
            constexpr int proj_dir = Arg::dagger ? +1 : -1;
            out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
          }
          // The EOFA stuff
          {
            constexpr int proj_dir = Arg::pm ? +1 : -1;
            real t = Arg::dagger ? arg.coeff.y[s] * arg.coeff.x[sp] : arg.coeff.x[s] * arg.coeff.y[sp];
            out += (t * sherman_morrison) * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
          }
        }
        if (Arg::xpay) { // really axpy
          Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
          out = x + arg.a * out;
        }
        arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
      }
    };

  } // namespace mobius_eofa
} // namespace quda
