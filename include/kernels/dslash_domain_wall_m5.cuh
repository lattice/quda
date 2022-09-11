#pragma once

#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.h>
#include <math_helper.cuh>
#include <index_helper.cuh>
#include <kernel.h>
#include <domain_wall_helper.h>

namespace quda
{

  namespace mobius_m5
  {
    /** Whether to use a shared memory scratch pad to store the input
      field acrosss the Ls dimension to minimize global memory
      reads. */
    constexpr bool shared() { return true; }

    /** Whether to use variable or fixed coefficient algorithm.  Must be
      true if using ZMOBIUS */
    constexpr bool var_inverse() { return true; }

    /** Whether to use half vector, i.e. utilizing spin projection properties.
      Half vector uses less shared memory but results in more shared memory stores/loads. */
    constexpr bool use_half_vector() { return true; }
  } // namespace mobius_m5

  /**
     @brief Structure containing zMobius / Zolotarev coefficients
  */
  template <typename real> struct coeff_5 {
    complex<real> a[QUDA_MAX_DWF_LS];     // xpay coefficients
    complex<real> alpha[QUDA_MAX_DWF_LS]; // alpha * D5 + beta
    complex<real> beta[QUDA_MAX_DWF_LS];
    complex<real> kappa[QUDA_MAX_DWF_LS];
    complex<real> inv;
  };

  // helper trait for determining if we are using variable coefficients
  template <Dslash5Type type> struct is_variable {
    static constexpr bool value = false;
  };
  template <> struct is_variable<Dslash5Type::DSLASH5_MOBIUS_PRE> {
    static constexpr bool value = true;
  };
  template <> struct is_variable<Dslash5Type::DSLASH5_MOBIUS> {
    static constexpr bool value = true;
  };
  template <> struct is_variable<Dslash5Type::M5_INV_ZMOBIUS> {
    static constexpr bool value = true;
  };

  /**
     @brief Helper class for grabbing the constant struct, whether
     we are on the GPU or CPU.
  */
  template <typename real, bool is_variable, typename Arg> struct coeff_type {
    const Arg &arg;
    __device__ __host__ coeff_type(const Arg &arg) : arg(arg) { }
    __device__ __host__ real a(int) { return arg.a; }
    __device__ __host__ real alpha(int) { return arg.alpha; }
    __device__ __host__ real beta(int) { return arg.beta; }
    __device__ __host__ real kappa(int) { return arg.kappa; }
    __device__ __host__ real inv() { return arg.inv; }
  };

  /**
     @brief Specialization for variable complex coefficients
  */
  template <typename real, typename Arg> struct coeff_type<real, true, Arg> {
    const Arg &arg;
    __device__ __host__ inline coeff_type(const Arg &arg) : arg(arg) { }
    __device__ __host__ complex<real> a(int s) { return arg.coeff.a[s]; }
    __device__ __host__ complex<real> alpha(int s) { return arg.coeff.alpha[s]; }
    __device__ __host__ complex<real> beta(int s) { return arg.coeff.beta[s]; }
    __device__ __host__ complex<real> kappa(int s) { return arg.coeff.kappa[s]; }
    __device__ __host__ complex<real> inv() { return arg.coeff.inv; }
  };

  /**
     @brief Parameter structure for applying the Dslash
   */
  template <typename Float, int nColor_, bool dagger_, bool xpay_, Dslash5Type type_>
  struct Dslash5Arg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr bool dagger = dagger_;
    static constexpr bool xpay = xpay_;
    static constexpr Dslash5Type type = type_;
    using F = typename colorspinor_mapper<Float, 4, nColor>::type;

    F out;                  // output vector field
    const F in;             // input vector field
    const F x;              // auxiliary input vector field
    const int nParity;      // number of parities we're working on
    const int volume_cb;    // checkerboarded volume
    const int volume_4d_cb; // 4-d checkerboarded volume
    const int_fastdiv Ls;   // length of 5th dimension

    const real m_f; // fermion mass parameter
    const real m_5; // Wilson mass shift

    // real constant Mobius coefficient - there are the zMobius counterparts in the `coeff_5` struct
    real a;     // real xpay coefficient
    real alpha; // out = alpha * op(in) + beta * in
    real beta;
    real kappa; // kappa = kappa_b / kappa_c
    real inv;   // The denominator for the M5inv

    coeff_5<real> coeff; // constant buffer used for Mobius coefficients for CPU kernel

    void compute_coeff_mobius_pre(const Complex *b_5, const Complex *c_5)
    {
      // out = (b + c * D5) * in
      for (int s = 0; s < Ls; s++) {
        coeff.beta[s] = b_5[s];
        coeff.alpha[s] = 0.5 * c_5[s]; // 0.5 from gamma matrices
        // xpay
        coeff.a[s] = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        coeff.a[s] *= coeff.a[s] * static_cast<real>(a); // kappa_b * kappa_b * a
      }
    }

    void compute_coeff_mobius(const Complex *b_5, const Complex *c_5)
    {
      // out = (1 + kappa * D5) * in
      for (int s = 0; s < Ls; s++) {
        coeff.kappa[s] = 0.5 * (c_5[s] * (m_5 + 4.0) - 1.0) / (b_5[s] * (m_5 + 4.0) + 1.0); // 0.5 from gamma matrices
        // axpy
        coeff.a[s] = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        coeff.a[s] *= coeff.a[s] * static_cast<real>(a); // kappa_b * kappa_b * a
      }
    }

    void compute_coeff_m5inv_dwf()
    {
      kappa = 2.0 * (0.5 / (5.0 + m_5)); // 2  * kappa_5
      inv = 0.5 / (1.0 + std::pow(kappa, (int)Ls) * m_f);
    }

    void compute_coeff_m5inv_mobius(const Complex *b_5, const Complex *c_5)
    {
      // out = (1 + kappa * D5)^-1 * in = M5inv * in
      kappa = -(c_5[0].real() * (4.0 + m_5) - 1.0) / (b_5[0].real() * (4.0 + m_5) + 1.0); // kappa = kappa_b / kappa_c
      inv = 0.5 / (1.0 + std::pow(kappa, (int)Ls) * m_f);                                 // 0.5 from gamma matrices
      a *= pow(0.5 / (b_5[0].real() * (m_5 + 4.0) + 1.0), 2);                             // kappa_b * kappa_b * a
    }

    void compute_coeff_m5inv_zmobius(const Complex *b_5, const Complex *c_5)
    {
      // out = (1 + kappa * D5)^-1 * in = M5inv * in
      // Similar to mobius convention, but variadic across 5th dim
      complex<real> k = 1.0;
      for (int s = 0; s < Ls; s++) {
        coeff.kappa[s] = -(c_5[s] * (4.0 + m_5) - 1.0) / (b_5[s] * (4.0 + m_5) + 1.0);
        k *= coeff.kappa[s];
      }
      coeff.inv = static_cast<real>(0.5) / (static_cast<real>(1.0) + k * m_f);

      for (int s = 0; s < Ls; s++) { // axpy coefficients
        coeff.a[s] = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        coeff.a[s] *= coeff.a[s] * static_cast<real>(a);
      }
    }

    Dslash5Arg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f, double m_5,
               const Complex *b_5, const Complex *c_5, double a_) :
      kernel_param(dim3(in.VolumeCB() / in.X(4), in.X(4), in.SiteSubset())),
      out(out),
      in(in),
      x(x),
      nParity(in.SiteSubset()),
      volume_cb(in.VolumeCB()),
      volume_4d_cb(volume_cb / in.X(4)),
      Ls(in.X(4)),
      m_f(m_f),
      m_5(m_5),
      a(a_)
    {
      if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
      if (!in.isNative() || !out.isNative())
        errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      switch (type) {
      case Dslash5Type::DSLASH5_DWF: break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE: compute_coeff_mobius_pre(b_5, c_5); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB:
        compute_coeff_mobius_pre(b_5, c_5);
        compute_coeff_mobius(b_5, c_5);
        break;
      case Dslash5Type::DSLASH5_MOBIUS: compute_coeff_mobius(b_5, c_5); break;
      case Dslash5Type::M5_INV_DWF: compute_coeff_m5inv_dwf(); break;
      case Dslash5Type::M5_INV_MOBIUS_M5_PRE:
      case Dslash5Type::M5_PRE_MOBIUS_M5_INV:
        compute_coeff_mobius_pre(b_5, c_5);
        compute_coeff_m5inv_mobius(b_5, c_5);
        break;
      case Dslash5Type::M5_INV_MOBIUS:
      case Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG:
        compute_coeff_mobius(b_5, c_5);
        compute_coeff_m5inv_mobius(b_5, c_5);
        break;
      case Dslash5Type::M5_INV_ZMOBIUS: compute_coeff_m5inv_zmobius(b_5, c_5); break;
      default: errorQuda("Unknown Dslash5Type %d", static_cast<int>(type));
      }
    }
  };

  template <bool sync, bool dagger, bool shared, class Vector, class Arg, Dslash5Type type = Arg::type>
  __device__ __host__ inline Vector d5(const Arg &arg, const Vector &in, int parity, int x_cb, int s)
  {

    using real = typename Arg::real;
    constexpr bool is_variable = true;
    coeff_type<real, is_variable, Arg> coeff(arg);

    Vector out;

    if (mobius_m5::use_half_vector()) {
      // if using shared-memory caching then load spinor field for my site into cache
      typedef ColorSpinor<real, Arg::nColor, 4 / 2> HalfVector;
      SharedMemoryCache<HalfVector> cache(target::block_dim());

      { // forwards direction
        constexpr int proj_dir = dagger ? +1 : -1;
        if (shared) {
          if (sync) { cache.sync(); }
          cache.save(in.project(4, proj_dir));
          cache.sync();
        }
        const int fwd_s = (s + 1) % arg.Ls;
        const int fwd_idx = fwd_s * arg.volume_4d_cb + x_cb;
        HalfVector half_in;
        if (shared) {
          half_in = cache.load(threadIdx.x, fwd_s, parity);
        } else {
          Vector full_in = arg.in(fwd_idx, parity);
          half_in = full_in.project(4, proj_dir);
        }
        if (s == arg.Ls - 1) {
          out += (-arg.m_f * half_in).reconstruct(4, proj_dir);
        } else {
          out += half_in.reconstruct(4, proj_dir);
        }
      }

      { // backwards direction
        constexpr int proj_dir = dagger ? -1 : +1;
        if (shared) {
          cache.sync();
          cache.save(in.project(4, proj_dir));
          cache.sync();
        }
        const int back_s = (s + arg.Ls - 1) % arg.Ls;
        const int back_idx = back_s * arg.volume_4d_cb + x_cb;
        HalfVector half_in;
        if (shared) {
          half_in = cache.load(threadIdx.x, back_s, parity);
        } else {
          Vector full_in = arg.in(back_idx, parity);
          half_in = full_in.project(4, proj_dir);
        }
        if (s == 0) {
          out += (-arg.m_f * half_in).reconstruct(4, proj_dir);
        } else {
          out += half_in.reconstruct(4, proj_dir);
        }
      }

    } else { // use_half_vector

      // if using shared-memory caching then load spinor field for my site into cache
      SharedMemoryCache<Vector> cache(target::block_dim());
      if (shared) {
        if (sync) { cache.sync(); }
        cache.save(in);
        cache.sync();
      }

      { // forwards direction
        const int fwd_s = (s + 1) % arg.Ls;
        const int fwd_idx = fwd_s * arg.volume_4d_cb + x_cb;
        const Vector in = shared ? cache.load(threadIdx.x, fwd_s, parity) : arg.in(fwd_idx, parity);
        constexpr int proj_dir = dagger ? +1 : -1;
        if (s == arg.Ls - 1) {
          out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
        } else {
          out += in.project(4, proj_dir).reconstruct(4, proj_dir);
        }
      }

      { // backwards direction
        const int back_s = (s + arg.Ls - 1) % arg.Ls;
        const int back_idx = back_s * arg.volume_4d_cb + x_cb;
        const Vector in = shared ? cache.load(threadIdx.x, back_s, parity) : arg.in(back_idx, parity);
        constexpr int proj_dir = dagger ? -1 : +1;
        if (s == 0) {
          out += (-arg.m_f * in.project(4, proj_dir)).reconstruct(4, proj_dir);
        } else {
          out += in.project(4, proj_dir).reconstruct(4, proj_dir);
        }
      }
    } // use_half_vector

    if (type == Dslash5Type::DSLASH5_MOBIUS_PRE || type == Dslash5Type::M5_INV_MOBIUS_M5_PRE
        || type == Dslash5Type::M5_PRE_MOBIUS_M5_INV) {
      Vector diagonal = shared ? in : arg.in(s * arg.volume_4d_cb + x_cb, parity);
      out = coeff.alpha(s) * out + coeff.beta(s) * diagonal;
    } else if (type == Dslash5Type::DSLASH5_MOBIUS) {
      Vector diagonal = shared ? in : arg.in(s * arg.volume_4d_cb + x_cb, parity);
      out = coeff.kappa(s) * out + diagonal;
    }

    return out;
  }

  template <typename Arg> struct dslash5 {
    const Arg &arg;
    constexpr dslash5(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @brief Apply the D5 operator at given site
       @param[in] parity Parity we are on
       @param[in] x_b Checkerboarded 4-d space-time index
       @param[in] s Ls dimension coordinate
    */
    __device__ __host__ inline void operator()(int x_cb, int s, int parity)
    {
      using real = typename Arg::real;
      coeff_type<real, is_variable<Arg::type>::value, Arg> coeff(arg);
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      constexpr bool sync = false;
      constexpr bool shared = false;

      Vector out = d5<sync, Arg::dagger, shared, Vector, Arg>(arg, Vector(), parity, x_cb, s);

      if (Arg::xpay) {
        if (Arg::type == Dslash5Type::DSLASH5_DWF) {
          Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
          out = x + arg.a * out;
        } else if (Arg::type == Dslash5Type::DSLASH5_MOBIUS_PRE) {
          Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
          out = x + coeff.a(s) * out;
        } else if (Arg::type == Dslash5Type::DSLASH5_MOBIUS) {
          Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
          out = coeff.a(s) * x + out;
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

     @tparam shared Whether to use a shared memory scratch pad to
     store the input field acroos the Ls dimension to minimize global
     memory reads.
     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s_ Ls dimension coordinate
  */
  template <bool sync, bool dagger, bool shared, typename Vector, typename Arg>
  __device__ __host__ inline Vector constantInv(const Arg &arg, const Vector &in, int parity, int x_cb, int s_)
  {
    using real = typename Arg::real;
    const auto k = arg.kappa;
    const auto inv = arg.inv;

    // if using shared-memory caching then load spinor field for my site into cache
    SharedMemoryCache<Vector> cache(target::block_dim());
    if (shared) {
      // cache.save(arg.in(s_ * arg.volume_4d_cb + x_cb, parity));
      if (sync) { cache.sync(); }
      cache.save(in);
      cache.sync();
    }

    Vector out;

    for (int s = 0; s < arg.Ls; s++) {

      Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s * arg.volume_4d_cb + x_cb, parity);

      {
        int exp = s_ < s ? arg.Ls - s + s_ : s_ - s;
        real factorR = inv * fpow(k, exp) * (s_ < s ? -arg.m_f : static_cast<real>(1.0));
        constexpr int proj_dir = dagger ? -1 : +1;
        out += factorR * (in.project(4, proj_dir)).reconstruct(4, proj_dir);
      }

      {
        int exp = s_ > s ? arg.Ls - s_ + s : s - s_;
        real factorL = inv * fpow(k, exp) * (s_ > s ? -arg.m_f : static_cast<real>(1.0));
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

     @param[in] arg Argument struct containing any meta data and accessors
     @param[in] parity Parity we are on
     @param[in] x_b Checkerboarded 4-d space-time index
     @param[in] s_ Ls dimension coordinate
  */
  template <bool sync, bool dagger, bool shared, typename Vector, typename Arg>
  __device__ __host__ inline Vector variableInv(const Arg &arg, const Vector &in, int parity, int x_cb, int s_)
  {
    constexpr int nSpin = 4;
    using real = typename Arg::real;
    typedef ColorSpinor<real, Arg::nColor, nSpin / 2> HalfVector;
    coeff_type<real, is_variable<Arg::type>::value, Arg> coeff(arg);
    Vector out;

    if (mobius_m5::use_half_vector()) {
      SharedMemoryCache<HalfVector> cache(target::block_dim());

      { // first do R
        constexpr int proj_dir = dagger ? -1 : +1;

        if (shared) {
          if (sync) { cache.sync(); }
          cache.save(in.project(4, proj_dir));
          cache.sync();
        }

        int s = s_;
        auto R = coeff.inv();
        HalfVector r;
        for (int s_count = 0; s_count < arg.Ls; s_count++) {
          auto factorR = (s_ < s ? -arg.m_f * R : R);

          if (shared) {
            r += factorR * cache.load(threadIdx.x, s, parity);
          } else {
            Vector in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
            r += factorR * in.project(4, proj_dir);
          }

          R *= coeff.kappa(s);
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
        auto L = coeff.inv();
        HalfVector l;
        for (int s_count = 0; s_count < arg.Ls; s_count++) {
          auto factorL = (s_ > s ? -arg.m_f * L : L);

          if (shared) {
            l += factorL * cache.load(threadIdx.x, s, parity);
          } else {
            Vector in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
            l += factorL * in.project(4, proj_dir);
          }

          L *= coeff.kappa(s);
          s = (s + 1) % arg.Ls;
        }

        out += l.reconstruct(4, proj_dir);
      }
    } else { // use_half_vector
      SharedMemoryCache<Vector> cache(target::block_dim());
      if (shared) {
        if (sync) { cache.sync(); }
        cache.save(in);
        cache.sync();
      }

      { // first do R
        constexpr int proj_dir = dagger ? -1 : +1;

        int s = s_;
        auto R = coeff.inv();
        HalfVector r;
        for (int s_count = 0; s_count < arg.Ls; s_count++) {
          auto factorR = (s_ < s ? -arg.m_f * R : R);

          Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s * arg.volume_4d_cb + x_cb, parity);
          r += factorR * in.project(4, proj_dir);

          R *= coeff.kappa(s);
          s = (s + arg.Ls - 1) % arg.Ls;
        }

        out += r.reconstruct(4, proj_dir);
      }

      { // second do L
        constexpr int proj_dir = dagger ? +1 : -1;

        int s = s_;
        auto L = coeff.inv();
        HalfVector l;
        for (int s_count = 0; s_count < arg.Ls; s_count++) {
          auto factorL = (s_ > s ? -arg.m_f * L : L);

          Vector in = shared ? cache.load(threadIdx.x, s, parity) : arg.in(s * arg.volume_4d_cb + x_cb, parity);
          l += factorL * in.project(4, proj_dir);

          L *= coeff.kappa(s);
          s = (s + 1) % arg.Ls;
        }

        out += l.reconstruct(4, proj_dir);
      }
    } // use_half_vector

    return out;
  }

  /**
     @brief Functor for applying the M5 inverse operator
     @param[in] arg Argument struct containing any meta data and accessors
  */
  template <typename Arg> struct dslash5inv {
    const Arg &arg;
    constexpr dslash5inv(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @brief Apply the M5 inverse operator at a given site on the
       lattice.
       @param[in] parity Parity we are on
       @param[in] x_b Checkerboarded 4-d space-time index
       @param[in] s Ls dimension coordinate
    */
    __device__ __host__ inline void operator()(int x_cb, int s, int parity)
    {
      constexpr int nSpin = 4;
      using real = typename Arg::real;
      typedef ColorSpinor<real, Arg::nColor, nSpin> Vector;
      coeff_type<real, is_variable<Arg::type>::value, Arg> coeff(arg);

      Vector in = arg.in(s * arg.volume_4d_cb + x_cb, parity);
      Vector out;
      constexpr bool sync = false;
      if (mobius_m5::var_inverse()) { // zMobius, must call variableInv
        out = variableInv<sync, Arg::dagger, mobius_m5::shared()>(arg, in, parity, x_cb, s);
      } else {
        out = constantInv<sync, Arg::dagger, mobius_m5::shared()>(arg, in, parity, x_cb, s);
      }

      if (Arg::xpay) {
        Vector x = arg.x(s * arg.volume_4d_cb + x_cb, parity);
        out = x + coeff.a(s) * out;
      }

      arg.out(s * arg.volume_4d_cb + x_cb, parity) = out;
    }
  };

} // namespace quda
