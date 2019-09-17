#pragma once

#include <shared_memory_cache_helper.cuh>
#include <math_helper.cuh>

namespace quda
{

  /**
     @brief Structure containing zMobius / Zolotarev coefficients
  */
  template <typename real> struct coeff_5 {
    complex<real> a[QUDA_MAX_DWF_LS]; // xpay coefficients
    complex<real> b[QUDA_MAX_DWF_LS];
    complex<real> c[QUDA_MAX_DWF_LS];
  };

  constexpr int size = 4096;
  static __constant__ char mobius_d[size]; // buffer used for Mobius coefficients for GPU kernel

  // helper trait for determining if we are using variable coefficients
  template <Dslash5Type type> struct is_variable {
    static constexpr bool value = false;
  };
  template <> struct is_variable<DSLASH5_MOBIUS_PRE> {
    static constexpr bool value = true;
  };
  template <> struct is_variable<DSLASH5_MOBIUS> {
    static constexpr bool value = true;
  };
  template <> struct is_variable<M5_INV_ZMOBIUS> {
    static constexpr bool value = true;
  };

  /**
     @brief Helper class for grabbing the constant struct, whether
     we are on the GPU or CPU.
  */
  template <typename real, bool is_variable, typename Arg> class coeff_type
  {
public:
    const Arg &arg;
    __device__ __host__ coeff_type(const Arg &arg) : arg(arg) {}
    __device__ __host__ real a(int s) { return arg.a; }
    __device__ __host__ real b(int s) { return arg.b; }
    __device__ __host__ real c(int s) { return arg.c; }
  };

  /**
     @brief Specialization for variable complex coefficients
  */
  template <typename real, typename Arg> class coeff_type<real, true, Arg>
  {
    /**
       @brief Helper function for grabbing the constant struct, whether
       we are on the GPU or CPU.
    */
    inline __device__ __host__ const coeff_5<real> *coeff() const
    {
#ifdef __CUDA_ARCH__
      return reinterpret_cast<const coeff_5<real> *>(mobius_d);
#else
      return arg.mobius_h;
#endif
    }

public:
    const Arg &arg;
    __device__ __host__ inline coeff_type(const Arg &arg) : arg(arg) {}
    __device__ __host__ complex<real> a(int s) { return coeff()->a[s]; }
    __device__ __host__ complex<real> b(int s) { return coeff()->b[s]; }
    __device__ __host__ complex<real> c(int s) { return coeff()->c[s]; }
  };

  /**
     @brief Parameter structure for applying the Dslash
   */
  template <typename Float, int nColor> struct Dslash5Arg {
    typedef typename colorspinor_mapper<Float, 4, nColor>::type F;
    typedef typename mapper<Float>::type real;

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

    real b; // real constant Mobius coefficient
    real c; // real constant Mobius coefficient
    real a; // real xpay coefficient

    Dslash5Type type;

    coeff_5<real> *mobius_h; // constant buffer used for Mobius coefficients for CPU kernel

    Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f, double m_5,
        const Complex *b_5_, const Complex *c_5_, double a_, bool dagger, Dslash5Type type) :
        out(out),
        in(in),
        x(x),
        nParity(in.SiteSubset()),
        volume_cb(in.VolumeCB()),
        volume_4d_cb(volume_cb / in.X(4)),
        Ls(in.X(4)),
        m_f(m_f),
        m_5(m_5),
        a(a_),
        dagger(dagger),
        xpay(a_ == 0.0 ? false : true),
        type(type)
    {
      if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
      if (!in.isNative() || !out.isNative())
        errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      if (sizeof(coeff_5<real>) > size) errorQuda("Coefficient buffer too large at %lu bytes\n", sizeof(coeff_5<real>));
      mobius_h = new coeff_5<real>;
      auto *a_5 = mobius_h->a;
      auto *b_5 = mobius_h->b;
      auto *c_5 = mobius_h->c;

      switch (type) {
      case DSLASH5_DWF: break;
      case DSLASH5_MOBIUS_PRE:
        for (int s = 0; s < Ls; s++) {
          b_5[s] = b_5_[s];
          c_5[s] = 0.5 * c_5_[s];

          // xpay
          a_5[s] = 0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0);
          a_5[s] *= a_5[s] * static_cast<real>(a);
        }
        break;
      case DSLASH5_MOBIUS:
        for (int s = 0; s < Ls; s++) {
          b_5[s] = 1.0;
          c_5[s] = 0.5 * (c_5_[s] * (m_5 + 4.0) - 1.0) / (b_5_[s] * (m_5 + 4.0) + 1.0);

          // axpy
          a_5[s] = 0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0);
          a_5[s] *= a_5[s] * static_cast<real>(a);
        }
        break;
      case M5_INV_DWF:
        b = 2.0 * (0.5 / (5.0 + m_5)); // 2  * kappa_5
        c = 0.5 / (1.0 + std::pow(b, (int)Ls) * m_f);
        break;
      case M5_INV_MOBIUS:
        b = -(c_5_[0].real() * (4.0 + m_5) - 1.0) / (b_5_[0].real() * (4.0 + m_5) + 1.0);
        c = 0.5 / (1.0 + std::pow(b, (int)Ls) * m_f);
        a *= pow(0.5 / (b_5_[0].real() * (m_5 + 4.0) + 1.0), 2);
        break;
      case M5_INV_ZMOBIUS: {
        complex<double> k = 1.0;
        for (int s = 0; s < Ls; s++) {
          b_5[s] = -(c_5_[s] * (4.0 + m_5) - 1.0) / (b_5_[s] * (4.0 + m_5) + 1.0);
          k *= b_5[s];
        }
        c_5[0] = 0.5 / (1.0 + k * m_f);

        for (int s = 0; s < Ls; s++) { // axpy coefficients
          a_5[s] = 0.5 / (b_5_[s] * (m_5 + 4.0) + 1.0);
          a_5[s] *= a_5[s] * static_cast<real>(a);
        }
      } break;
      default: errorQuda("Unknown Dslash5Type %d", type);
      }

      cudaMemcpyToSymbolAsync(mobius_d, mobius_h, sizeof(coeff_5<real>), 0, cudaMemcpyHostToDevice, streams[Nstream - 1]);
    }

    virtual ~Dslash5Arg() { delete mobius_h; }
  };

  /**
     @brief Apply the D5 operator at given site
     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s Ls dimension coordinate
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __device__ __host__ inline void dslash5(Arg &arg, int parity, int x_cb, int s)
  {
    typedef typename mapper<Float>::type real;
    coeff_type<real, is_variable<type>::value, Arg> coeff(arg);
    typedef ColorSpinor<real, nColor, 4> Vector;

    Vector out;

    { // forwards direction
      const int fwd_idx = ((s + 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(fwd_idx, parity);
      constexpr int proj_dir = dagger ? +1 : -1;
      if (s == arg.Ls - 1) {
        out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
        out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    { // backwards direction
      const int back_idx = ((s + arg.Ls - 1) % arg.Ls) * arg.volume_4d_cb + x_cb;
      const Vector in = arg.in(back_idx, parity);
      constexpr int proj_dir = dagger ? -1 : +1;
      if (s == 0) {
        out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
      } else {
        out += in.project(4, proj_dir).reconstruct(4, proj_dir);
      }
    }

    if (type == DSLASH5_DWF && xpay) {
      Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
      out = x + arg.a * out;
    } else if (type == DSLASH5_MOBIUS_PRE) {
      Vector diagonal = arg.in(s * arg.volume_4d_cb + x_cb, parity);
      out = coeff.c(s) * out + coeff.b(s) * diagonal;

      if (xpay) {
        Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
        out = x + coeff.a(s) * out;
      }
    } else if (type == DSLASH5_MOBIUS) {
      Vector diagonal = arg.in(s * arg.volume_4d_cb + x_cb, parity);
      out = coeff.c(s) * out + diagonal;

      if (xpay) { // really axpy
        Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
        out = coeff.a(s) * x + out;
      }
    }

    arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
  }

  /**
     @brief CPU kernel for applying the D5 operator
     @param[in] arg Argument struct containing any meta data and accessors
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  void dslash5CPU(Arg &arg)
  {
    for (int parity = 0; parity < arg.nParity; parity++) {
      for (int s = 0; s < arg.Ls; s++) {
        for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) { // 4-d volume
          dslash5<Float, nColor, dagger, xpay, type>(arg, parity, x_cb, s);
        } // 4-d volumeCB
      }   // ls
    }     // parity
  }

  /**
     @brief GPU kernel for applying the D5 operator
     @param[in] arg Argument struct containing any meta data and accessors
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, typename Arg>
  __global__ void dslash5GPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int parity = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5<Float, nColor, dagger, xpay, type>(arg, parity, x_cb, s);
  }

  /**
     @brief Apply the M5 inverse operator at a given site on the
     lattice.  This is the original algorithm as described in Kim and
     Izubushi (LATTICE 2013_033), where the b and c coefficients are
     constant along the Ls dimension, so is suitable for Shamir and
     Mobius domain-wall fermions.

     @tparam shared Whether to use a shared memory scratch pad to
     store the input field acroos the Ls dimension to minimize global
     memory reads.
     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s_ Ls dimension coordinate
  */
  template <typename real, int nColor, bool dagger, Dslash5Type type, bool shared, typename Vector, typename Arg>
  __device__ __host__ inline Vector constantInv(Arg &arg, int parity, int x_cb, int s_)
  {
    const auto k = arg.b;
    const auto inv = arg.c;

    // if using shared-memory caching then load spinor field for my site into cache
    VectorCache<real, Vector> cache;
    if (shared) {
      cache.save(arg.in(s_ * arg.volume_4d_cb + x_cb, parity));
      cache.sync();
    }

    Vector out;

    for (int s = 0; s < arg.Ls; s++) {

      Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s * arg.volume_4d_cb + x_cb, parity);

      {
        int exp = s_ < s ? arg.Ls - s + s_ : s_ - s;
        real factorR = inv * __fast_pow(k, exp) * (s_ < s ? -arg.m_f : static_cast<real>(1.0));
        constexpr int proj_dir = dagger ? -1 : +1;
        out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

      {
        int exp = s_ > s ? arg.Ls - s_ + s : s - s_;
        real factorL = inv * __fast_pow(k, exp) * (s_ > s ? -arg.m_f : static_cast<real>(1.0));
        constexpr int proj_dir = dagger ? +1 : -1;
        out += factorL * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }
    }

    return out;
  }

  /**
     @brief Apply the M5 inverse operator at a given site on the
     lattice.  This is an alternative algorithm that is applicable to
     variable b and c coefficients: here each thread in the s
     dimension starts computing at s = s_, and computes the left- and
     right-handed contributions in two separate passes.  For the
     left-handed contribution we sweep through increasing s, e.g.,
     s=s_, s_+1, s_+2, and for the right-handed one we do the
     transpose, s=s_, s_-1, s_-2.  This allows us to progressively
     build up the scalar coefficients needed in a SIMD-friendly
     fashion.

     @tparam shared Whether to use a shared memory scratch pad to
     store the input field acroos the Ls dimension to minimize global
     memory reads.
     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s_ Ls dimension coordinate
  */
  template <typename real, int nColor, bool dagger, Dslash5Type type, bool shared, typename Vector, typename Arg>
  __device__ __host__ inline Vector variableInv(Arg &arg, int parity, int x_cb, int s_)
  {
    constexpr int nSpin = 4;
    typedef ColorSpinor<real, nColor, nSpin / 2> HalfVector;
    coeff_type<real, is_variable<type>::value, Arg> coeff(arg);
    Vector in = arg.in(s_ * arg.volume_4d_cb + x_cb, parity);
    Vector out;

    VectorCache<real, HalfVector> cache;

    { // first do R
      constexpr int proj_dir = dagger ? -1 : +1;

      if (shared) {
        cache.save(in.project(4, proj_dir));
        cache.sync();
      }

      int s = s_;
      auto R = coeff.c(0);
      HalfVector r;
      for (int s_count = 0; s_count < arg.Ls; s_count++) {
        auto factorR = (s_ < s ? -arg.m_f * R : R);

        if (shared) {
          r += factorR * cache.load(threadIdx.x, s, parity);
        } else {
          Vector in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
          r += factorR * in.project(4, proj_dir);
        }

        R *= coeff.b(s);
        s = (s + arg.Ls - 1) % arg.Ls;
      }

      out += r.reconstruct(4, proj_dir);
    }

    { // second do L
      constexpr int proj_dir = dagger ? +1 : -1;
      if (shared) {
        cache.sync(); // ensure we finish R before overwriting cache
        cache.save(in.project(4, proj_dir));
        cache.sync();
      }

      int s = s_;
      auto L = coeff.c(0);
      HalfVector l;
      for (int s_count = 0; s_count < arg.Ls; s_count++) {
        auto factorL = (s_ > s ? -arg.m_f * L : L);

        if (shared) {
          l += factorL * cache.load(threadIdx.x, s, parity);
        } else {
          Vector in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
          l += factorL * in.project(4, proj_dir);
        }

        L *= coeff.b(s);
        s = (s + 1) % arg.Ls;
      }

      out += l.reconstruct(4, proj_dir);
    }

    return out;
  }

  /**
     @brief Apply the M5 inverse operator at a given site on the
     lattice.
     @tparam shared Whether to use a shared memory scratch pad to
     store the input field across the Ls dimension to minimize global
     memory reads.
     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s Ls dimension coordinate
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, bool var_inverse, typename Arg>
  __device__ __host__ inline void dslash5inv(Arg &arg, int parity, int x_cb, int s)
  {
    constexpr int nSpin = 4;
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, nSpin> Vector;
    coeff_type<real, is_variable<type>::value, Arg> coeff(arg);

    Vector out;
    if (var_inverse) { // zMobius, must call variableInv
      out = variableInv<real, nColor, dagger, type, shared, Vector>(arg, parity, x_cb, s);
    } else {
      out = constantInv<real, nColor, dagger, type, shared, Vector>(arg, parity, x_cb, s);
    }

    if (xpay) {
      Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
      out = x + coeff.a(s) * out;
    }

    arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
  }

  /**
     @brief CPU kernel for applying the M5 inverse operator
     @param[in] arg Argument struct containing any meta data and accessors
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, bool var_inverse, typename Arg>
  void dslash5invCPU(Arg &arg)
  {
    for (int parity = 0; parity < arg.nParity; parity++) {
      for (int s = 0; s < arg.Ls; s++) {
        for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) { // 4-d volume
          dslash5inv<Float, nColor, dagger, xpay, type, shared, var_inverse>(arg, parity, x_cb, s);
        } // 4-d volumeCB
      }   // ls
    }     // parity
  }

  /**
     @brief CPU kernel for applying the M5 inverse operator
     @tparam shared Whether to use a shared memory scratch pad to
     store the input field acroos the Ls dimension to minimize global
     memory reads.
     @param[in] arg Argument struct containing any meta data and accessors
  */
  template <typename Float, int nColor, bool dagger, bool xpay, Dslash5Type type, bool shared, bool var_inverse, typename Arg>
  __global__ void dslash5invGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int parity = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    dslash5inv<Float, nColor, dagger, xpay, type, shared, var_inverse>(arg, parity, x_cb, s);
  }

} // namespace quda
