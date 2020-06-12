#pragma once

template <typename store_t, bool is_fixed> struct SpinorNorm {
  using norm_t = float;
  norm_t *norm;
  unsigned int cb_norm_offset;

  SpinorNorm() : norm(nullptr), cb_norm_offset(0) {}

  SpinorNorm(const ColorSpinorField &x) : norm((norm_t *)x.Norm()), cb_norm_offset(x.NormBytes() / (2 * sizeof(norm_t)))
  {
  }

  SpinorNorm(const SpinorNorm &sn) : norm(sn.norm), cb_norm_offset(sn.cb_norm_offset) {}

  SpinorNorm &operator=(const SpinorNorm &src)
  {
    if (&src != this) {
      norm = src.norm;
      cb_norm_offset = src.cb_norm_offset;
    }
    return *this;
  }

  void set(const ColorSpinorField &x)
  {
    norm = (norm_t *)x.Norm();
    cb_norm_offset = x.NormBytes() / (2 * sizeof(norm_t));
  }

  __device__ inline norm_t load_norm(const int i, const int parity = 0) const { return norm[cb_norm_offset * parity + i]; }

  template <typename real, int n> __device__ inline norm_t store_norm(const vector_type<complex<real>, n> &v, int x, int parity)
  {
    norm_t max_[n];
    // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
    for (int i = 0; i < n; i++) max_[i] = fmaxf(fabsf((norm_t)v[i].real()), fabsf((norm_t)v[i].imag()));
      norm_t scale = 0.0;
#pragma unroll
      for (int i = 0; i < n; i++) scale = fmaxf(max_[i], scale);
      norm[x+parity*cb_norm_offset] = scale;

#ifdef __CUDA_ARCH__
      return __fdividef(fixedMaxValue<store_t>::value, scale);
#else
      return fixedMaxValue<store_t>::value / scale;
#endif
  }

  norm_t *Norm() { return norm; }
};

template <typename store_type_t> struct SpinorNorm<store_type_t, false> {
  using norm_t = float;
  SpinorNorm() {}
  SpinorNorm(const ColorSpinorField &x) {}
  SpinorNorm(const SpinorNorm &sn) {}
  SpinorNorm &operator=(const SpinorNorm &src) { return *this; }
  void set(const ColorSpinorField &x) {}
  __device__ inline norm_t load_norm(const int i, const int parity = 0) const { return 1.0; }
  template <typename real, int n> __device__ inline norm_t store_norm(const vector_type<real, n> &v, int x, int parity) { return 1.0; }
  void backup(char **norm_h, size_t norm_bytes) {}
  void restore(char **norm_h, size_t norm_bytes) {}
  norm_t *Norm() { return nullptr; }
};

/**
   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
   @param N Length of vector of RegType elements that this Spinor represents
*/
template <typename store_t, int N>
  struct Spinor : SpinorNorm<store_t, isFixed<store_t>::value> {
  using SN = SpinorNorm<store_t, isFixed<store_t>::value>;
  using Vector = typename VectorType<store_t, N>::type;
  store_t *spinor;
  int stride;
  unsigned int cb_offset;

public:
  Spinor() :
    SN(),
    spinor(nullptr),
    stride(0),
    cb_offset(0)
  { }

 Spinor(const ColorSpinorField &x) :
    SN(x),
    spinor(static_cast<store_t*>(const_cast<ColorSpinorField &>(x).V())),
    stride(x.Stride()),
    cb_offset(x.Bytes() / (2 * sizeof(store_t) * N))
  {
  }

  Spinor(const Spinor &st) :
    SN(st),
    spinor(st.spinor),
    stride(st.stride),
    cb_offset(st.cb_offset)
  {
  }

  Spinor &operator=(const Spinor &src)
  {
    if (&src != this) {
      SN::operator=(src);
      spinor = src.spinor;
      stride = src.stride;
      cb_offset = src.cb_offset;
    }
    return *this;
  }

  void set(const ColorSpinorField &x)
  {
    SN::set(x);
    spinor = static_cast<store_t*>(const_cast<ColorSpinorField &>(x).V());
    stride = x.Stride();
    cb_offset = x.Bytes() / (2 * sizeof(store_t) * N);
  }

  template <typename real, int n>
  __device__ inline void load(vector_type<complex<real>, n> &v, int x, int parity = 0) const
  {
    constexpr int len = 2 * n; // real-valued length
    float nrm = isFixed<store_t>::value ? SN::load_norm(x, parity) : 0.0;

    vector_type<real, len> v_;

    constexpr int M = len / N;
#pragma unroll
    for (int i=0; i<M; i++) {
      // first load from memory
      Vector vecTmp = vector_load<Vector>(spinor, parity * cb_offset + x + stride * i);
      // now copy into output and scale
#pragma unroll
      for (int j = 0; j < N; j++) copy_and_scale(v_[i * N + j], reinterpret_cast<store_t *>(&vecTmp)[j], nrm);
    }

    for (int i=0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
  }

  template <typename real, int n>
  __device__ __host__ inline void save(const vector_type<complex<real>, n> &v, int x, int parity = 0)
  {
    constexpr int len = 2 * n; // real-valued length
    vector_type<real, len> v_;

    if (isFixed<store_t>::value) {
      real scale_inv = SN::template store_norm(v, x, parity);
#pragma unroll
      for (int i = 0; i < n; i++) {
        v_[2 * i + 0] = scale_inv * v[i].real();
        v_[2 * i + 1] = scale_inv * v[i].imag();
      }
    } else {
#pragma unroll
      for (int i = 0; i < n; i++) {
        v_[2 * i + 0] = v[i].real();
        v_[2 * i + 1] = v[i].imag();
      }
    }

    constexpr int M = len / N;
#pragma unroll
    for (int i=0; i<M; i++) {
      Vector vecTmp;
      // first do scalar copy converting into storage type
#pragma unroll
      for (int j = 0; j < N; j++) copy_scaled(reinterpret_cast<store_t *>(&vecTmp)[j], v_[i * N + j]);
      // second do vectorized copy into memory
      vector_store(spinor, parity * cb_offset + x + stride * i, vecTmp);
    }
  }

};
