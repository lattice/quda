#pragma once

#include <shared_memory_cache_helper.cuh>
#include <math_helper.cuh>

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

    constexpr int size = 4096;
    static __constant__ char mobius_eofa_d[size];
    static char mobius_eofa_h[size];

    /**
      @brief Helper function for grabbing the constant struct, whether
      we are on the GPU or CPU.
     */
    template <typename real> inline __device__ __host__ const eofa_coeff<real> *get_eofa_coeff()
    {
#ifdef __CUDA_ARCH__
      return reinterpret_cast<const eofa_coeff<real> *>(mobius_eofa_d);
#else
      return reinterpret_cast<const eofa_coeff<real> *>(mobius_eofa_h);
#endif
    }

    template <typename storage_type, int nColor> struct Dslash5Arg {
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

      const bool dagger; // dagger
      const bool xpay;   // whether we are doing xpay or not

      real a; // real xpay coefficient

      real kappa;
      real inv;

      int eofa_pm;
      real sherman_morrison;

      Dslash5Type type;

      Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, const double m_f_,
                 const double m_5_, const Complex *b_5_, const Complex *c_5_, double a_, int eofa_pm_, double inv_,
                 double kappa_, const double *eofa_u, const double *eofa_x, const double *eofa_y,
                 double sherman_morrison_, bool dagger_, Dslash5Type type_) :
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
        dagger(dagger_),
        xpay(a_ == 0. ? false : true),
        type(type_),
        eofa_pm(eofa_pm_),
        inv(inv_),
        kappa(kappa_),
        sherman_morrison(sherman_morrison_)
      {
        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
        if (sizeof(eofa_coeff<real>) > size)
          errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(eofa_coeff<real>));

        eofa_coeff<real> *eofa_coeffs = reinterpret_cast<eofa_coeff<real> *>(mobius_eofa_h);

        switch (type) {
        case M5_EOFA:
          for (int s = 0; s < Ls; s++) { eofa_coeffs->u[s] = eofa_u[s]; }
          cudaMemcpyToSymbolAsync(mobius_eofa_d, mobius_eofa_h, sizeof(eofa_coeff<real>) / 3, 0, cudaMemcpyHostToDevice,
                                  streams[Nstream - 1]);
          break;
        case M5INV_EOFA:
          for (int s = 0; s < Ls; s++) {
            eofa_coeffs->u[s] = eofa_u[s];
            eofa_coeffs->x[s] = eofa_x[s];
            eofa_coeffs->y[s] = eofa_y[s];
          }
          cudaMemcpyToSymbolAsync(mobius_eofa_d, mobius_eofa_h, sizeof(eofa_coeff<real>), 0, cudaMemcpyHostToDevice,
                                  streams[Nstream - 1]);
          break;
        default: errorQuda("Unknown EOFA Dslash5Type %d", type);
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
    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
    __device__ inline void dslash5(Arg &arg, int parity, int x_cb, int s)
    {
      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, nColor, 4> Vector;

      VectorCache<real, Vector> cache;

      Vector out;
      cache.save(arg.in(s * arg.volume_4d_cb + x_cb, parity));
      cache.sync();

      auto Ls = arg.Ls;

      { // forwards direction
        const Vector in = cache.load(threadIdx.x, (s + 1) % Ls, threadIdx.z);
        constexpr int proj_dir = dagger ? +1 : -1;
        if (s == Ls - 1) {
          out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
        } else {
          out += in.project(4, proj_dir).reconstruct(4, proj_dir);
        }
      }

      { // backwards direction
        const Vector in = cache.load(threadIdx.x, (s + Ls - 1) % Ls, threadIdx.z);
        constexpr int proj_dir = dagger ? -1 : +1;
        if (s == 0) {
          out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
        } else {
          out += in.project(4, proj_dir).reconstruct(4, proj_dir);
        }
      }

      if (type == M5_EOFA) {
        const eofa_coeff<real> *eofa_coeffs = get_eofa_coeff<real>();
        Vector diagonal = cache.load(threadIdx.x, s, threadIdx.z);
        out = (static_cast<real>(0.5) * arg.kappa) * out + diagonal; // 1 + kappa*D5; the 0.5 for spin projection

        constexpr int proj_dir = pm ? +1 : -1;

        if (dagger) {
          if (s == (pm ? Ls - 1 : 0)) {
            for (int sp = 0; sp < Ls; sp++) {
              out += (static_cast<real>(0.5) * eofa_coeffs->u[sp])
                * cache.load(threadIdx.x, sp, threadIdx.z).project(4, proj_dir).reconstruct(4, proj_dir);
            }
          }
        } else {
          out += (static_cast<real>(0.5) * eofa_coeffs->u[s])
            * cache.load(threadIdx.x, pm ? Ls - 1 : 0, threadIdx.z).project(4, proj_dir).reconstruct(4, proj_dir);
        }

        if (xpay) { // really axpy
          Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
          out = arg.a * x + out;
        }
      }
      arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
    }

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

    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
    __device__ __host__ inline void dslash5inv(Arg &arg, int parity, int x_cb, int s)
    {
      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, nColor, 4> Vector;

      const auto sherman_morrison = arg.sherman_morrison;
      VectorCache<real, Vector> cache;
      cache.save(arg.in(s * arg.volume_4d_cb + x_cb, parity));
      cache.sync();

      Vector out;
      const eofa_coeff<real> *eofa_coeffs = get_eofa_coeff<real>();

      for (int sp = 0; sp < arg.Ls; sp++) {
        Vector in = cache.load(threadIdx.x, sp, threadIdx.z);
        {
          int exp = s < sp ? arg.Ls - sp + s : s - sp;
          real factorR = 0.5 * eofa_coeffs->y[pm ? arg.Ls - exp - 1 : exp] * (s < sp ? -arg.m_f : static_cast<real>(1.0));
          constexpr int proj_dir = dagger ? -1 : +1;
          out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
        }
        {
          int exp = s > sp ? arg.Ls - s + sp : sp - s;
          real factorL = 0.5 * eofa_coeffs->y[pm ? arg.Ls - exp - 1 : exp] * (s > sp ? -arg.m_f : static_cast<real>(1.0));
          constexpr int proj_dir = dagger ? +1 : -1;
          out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
        }
        // The EOFA stuff
        {
          constexpr int proj_dir = pm ? +1 : -1;
          real t = dagger ? eofa_coeffs->y[s] * eofa_coeffs->x[sp] : eofa_coeffs->x[s] * eofa_coeffs->y[sp];
          out += (t * sherman_morrison) * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
        }
      }
      if (xpay) { // really axpy
        Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
        out = x + arg.a * out;
      }
      arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
    }

    /**
      @brief GPU kernel for applying the D5 operator
      @param[in] arg Argument struct containing any meta data and accessors
     */
    template <typename storage_type, int nColor, bool dagger, bool pm, bool xpay, Dslash5Type type, typename Arg>
    __global__ void dslash5GPU(Arg arg)
    {
      int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
      int s = blockIdx.y * blockDim.y + threadIdx.y;
      int parity = blockIdx.z * blockDim.z + threadIdx.z;

      if (x_cb >= arg.volume_4d_cb) return;
      if (s >= arg.Ls) return;
      if (parity >= arg.nParity) return;

      if (type == M5_EOFA) {
        dslash5<storage_type, nColor, dagger, pm, xpay, type>(arg, parity, x_cb, s);
      } else if (type == M5INV_EOFA) {
        dslash5inv<storage_type, nColor, dagger, pm, xpay, type>(arg, parity, x_cb, s);
      }
    }

  } // namespace mobius_eofa
} // namespace quda
