#pragma once

/**
   Checks that the types are set correctly.  The precision used in the
   RegType must match that of the InterType, and the ordering of the
   InterType must match that of the StoreType.  The only exception is
   when fixed precision is used, in which case, RegType can be a double
   and InterType can be single (with StoreType short or char).

   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
*/
template <typename RegType, typename InterType, typename StoreType> void checkTypes()
{
  const size_t reg_size = sizeof(((RegType *)0)->x);
  const size_t inter_size = sizeof(((InterType *)0)->x);
  const size_t store_size = sizeof(((StoreType *)0)->x);

  if (reg_size != inter_size && store_size != 2 && store_size != 1 && inter_size != 4)
    errorQuda("Precision of register (%lu) and intermediate (%lu) types must match\n", (unsigned long)reg_size,
        (unsigned long)inter_size);

  if (vec_length<InterType>::value != vec_length<StoreType>::value) {
    errorQuda("Vector lengths intermediate and register types must match\n");
  }

  if (vec_length<RegType>::value == 0) errorQuda("Vector type not supported\n");
  if (vec_length<InterType>::value == 0) errorQuda("Vector type not supported\n");
  if (vec_length<StoreType>::value == 0) errorQuda("Vector type not supported\n");
}

template <typename RegType, typename StoreType, bool is_fixed> struct SpinorNorm {
  typedef typename bridge_mapper<RegType, StoreType>::type InterType;
  float *norm;
  unsigned int cb_norm_offset;

  SpinorNorm() : norm(nullptr), cb_norm_offset(0) {}

  SpinorNorm(const ColorSpinorField &x) : norm((float *)x.Norm()), cb_norm_offset(x.NormBytes() / (2 * sizeof(float)))
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
    norm = (float *)x.Norm();
    cb_norm_offset = x.NormBytes() / (2 * sizeof(float));
  }

  __device__ inline float load_norm(const int i, const int parity = 0) const { return norm[cb_norm_offset * parity + i]; }

  template <int M> __device__ inline float store_norm(InterType x[M], int i, int parity)
  {
    float c[M/2];
#pragma unroll
    for (int j = 0; j < M/2; j++) c[j] = fmaxf(max_fabs(x[2*j+0]), max_fabs(x[2*j+1]));
#pragma unroll
    for (int j = 1; j < M/2; j++) c[0] = fmaxf(c[j], c[0]);
    norm[cb_norm_offset * parity + i] = c[0];
    return __fdividef(fixedMaxValue<StoreType>::value, c[0]);
  }

  // used to backup the field to the host
  void backup(char **norm_h, size_t norm_bytes)
  {
    if (norm_bytes > 0) {
      *norm_h = new char[norm_bytes];
      cudaMemcpy(*norm_h, norm, norm_bytes, cudaMemcpyDeviceToHost);
    }
    checkCudaError();
  }

  // restore the field from the host
  void restore(char **norm_h, size_t norm_bytes)
  {
    if (norm_bytes > 0) {
      cudaMemcpy(norm, *norm_h, norm_bytes, cudaMemcpyHostToDevice);
      delete[] * norm_h;
      *norm_h = 0;
    }
    checkCudaError();
  }

  float *Norm() { return norm; }
};

template <typename RegType, typename StoreType> struct SpinorNorm<RegType, StoreType, false> {
  typedef typename bridge_mapper<RegType, StoreType>::type InterType;
  SpinorNorm() {}
  SpinorNorm(const ColorSpinorField &x) {}
  SpinorNorm(const SpinorNorm &sn) {}
  SpinorNorm &operator=(const SpinorNorm &src) { return *this; }
  void set(const ColorSpinorField &x) {}
  __device__ inline float load_norm(const int i, const int parity = 0) const { return 1.0; }
  template <int M> __device__ inline float store_norm(InterType x[M], int i, int parity) { return 1.0; }
  void backup(char **norm_h, size_t norm_bytes) {}
  void restore(char **norm_h, size_t norm_bytes) {}
  float *Norm() { return nullptr; }
};

/**
   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
   @param N Length of vector of RegType elements that this Spinor represents
*/
template <typename RegType_, typename StoreType_, int N>
struct Spinor : SpinorNorm<RegType_, StoreType_, isFixed<StoreType_>::value> {
  typedef RegType_ RegType;
  typedef StoreType_ StoreType;
  typedef typename bridge_mapper<RegType,StoreType>::type InterType;
  typedef SpinorNorm<RegType, StoreType_, isFixed<StoreType>::value> SN;

  StoreType *spinor;
  int stride;
  unsigned int cb_offset;
#ifndef BLAS_SPINOR
  StoreType *ghost;
  int ghost_stride[4];
#endif

public:
  Spinor() :
    SN(),
    spinor(nullptr),
    stride(0),
    cb_offset(0)
#ifndef BLAS_SPINOR
    , ghost(nullptr)
#endif
  { }

  Spinor(const ColorSpinorField &x, int nFace = 1) :
    SN(x),
    spinor(static_cast<StoreType*>(const_cast<ColorSpinorField &>(x).V())),
    stride(x.Stride()),
    cb_offset(x.Bytes() / (2 * sizeof(StoreType)))
#ifndef BLAS_SPINOR
    , ghost(static_cast<StoreType*>(x.Ghost2()))
#endif
  {
    checkTypes<RegType, InterType, StoreType>();
#ifndef BLAS_SPINOR
    for (int d = 0; d < 4; d++) ghost_stride[d] = nFace * x.SurfaceCB(d);
#endif
  }

  Spinor(const Spinor &st) :
    SN(st),
    spinor(st.spinor),
    stride(st.stride),
    cb_offset(st.cb_offset)
#ifndef BLAS_SPINOR
    , ghost(st.ghost)
#endif
  {
#ifndef BLAS_SPINOR
    for (int d = 0; d < 4; d++) ghost_stride[d] = st.ghost_stride[d];
#endif
  }

  Spinor &operator=(const Spinor &src)
  {
    if (&src != this) {
      SN::operator=(src);
      spinor = src.spinor;
      stride = src.stride;
      cb_offset = src.cb_offset;
#ifndef BLAS_SPINOR
      ghost = src.ghost;
      for (int d = 0; d < 4; d++) ghost_stride[d] = src.ghost_stride[d];
#endif
    }
    return *this;
  }

  void set(const ColorSpinorField &x, int nFace = 1)
  {
    SN::set(x);
    spinor = static_cast<StoreType*>(const_cast<ColorSpinorField &>(x).V());
    stride = x.Stride();
    cb_offset = x.Bytes() / (2 * sizeof(StoreType));
#ifndef BLAS_SPINOR
    ghost = static_cast<StoreType*>(x.Ghost2());
    for (int d = 0; d < 4; d++) ghost_stride[d] = nFace * x.SurfaceCB(d);
#endif
    checkTypes<RegType, InterType, StoreType>();
  }

  __device__ inline void load(RegType x[], const int i, const int parity = 0) const
  {
    // load data into registers first using the storage order
    constexpr int M = (N * vec_length<RegType>::value) / vec_length<InterType>::value;
    InterType y[M];

    // fixed precision
    if (isFixed<StoreType>::value) {
      const float xN = SN::load_norm(i, parity);
#pragma unroll
      for (int j = 0; j < M; j++) copy_and_scale(y[j], spinor[cb_offset * parity + i + j * stride], xN);
    } else { // other types
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(y[j], spinor[cb_offset * parity + i + j * stride]);
    }

    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }

#ifndef BLAS_SPINOR
  /**
     Load the ghost spinor.  For Wilson fermions, we assume that the
     ghost is spin projected
  */
  __device__ inline void loadGhost(RegType x[], const int i, const int dim) const
  {
    // load data into registers first using the storage order
    const int Nspin = (N * vec_length<RegType>::value) / (3 * 2);
    // if Wilson, then load only half the number of components
    constexpr int M = ((N * vec_length<RegType>::value ) / vec_length<InterType>::value) / ((Nspin == 4) ? 2 : 1);

    InterType y[M];

    // fixed precision types (FIXME - these don't look correct?)
    if (isFixed<StoreType>::value) {
      float xN = SN::load_norm(i);
#pragma unroll
      for (int j = 0; j < M; j++) copy_and_scale(y[j], ghost[i + j * ghost_stride[dim]], xN);
    } else { // other types
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(y[j], ghost[i + j * ghost_stride[dim]]);
    }

    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }
#endif

  __device__ inline void save(RegType x[], int i, const int parity = 0)
  {
    constexpr int M = (N * vec_length<RegType>::value) / vec_length<InterType>::value;
    InterType y[M];
    convert<InterType, RegType>(y, x, M);

    if (isFixed<StoreType>::value) {
      float C = SN::template store_norm<M>(y, i, parity);
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(spinor[cb_offset * parity + i + j * stride], C * y[j]);
    } else {
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(spinor[cb_offset * parity + i + j * stride], y[j]);
    }
  }

  // used to backup the field to the host
  void backup(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes)
  {
    *spinor_h = new char[bytes];
    cudaMemcpy(*spinor_h, spinor, bytes, cudaMemcpyDeviceToHost);
    SN::backup(norm_h, norm_bytes);
    checkCudaError();
  }

  // restore the field from the host
  void restore(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes)
  {
    cudaMemcpy(spinor, *spinor_h, bytes, cudaMemcpyHostToDevice);
    SN::restore(norm_h, norm_bytes);
    delete[] * spinor_h;
    *spinor_h = 0;
    checkCudaError();
  }

  void *V()
  {
    return (void *)spinor;
  }

  QudaPrecision Precision() const
  {
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (sizeof(((StoreType *)0)->x) == sizeof(double))
      precision = QUDA_DOUBLE_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(float))
      precision = QUDA_SINGLE_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(short))
      precision = QUDA_HALF_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(char))
      precision = QUDA_QUARTER_PRECISION;
    else
      errorQuda("Unknown precision type\n");
    return precision;
  }

  int Stride() const { return stride; }
  int Bytes() const { return N * sizeof(RegType); }
};
