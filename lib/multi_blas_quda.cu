#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset

#include <tune_quda.h>
#include <blas_quda.h>
#include <color_spinor_field.h>

#include <jitify_helper.cuh>
#include <kernels/multi_blas_core.cuh>

namespace quda {

  namespace blas {

    qudaStream_t* getStream();

    template <int NXZ, typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW,
        typename Functor, typename T>
    class MultiBlas : public TunableVectorY
    {

  private:
    typedef typename scalar<FloatN>::type Float;
    typedef typename vector<Float, 2>::type Float2;
    static constexpr int NYW_max = max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor>();
    const int NYW;
    int max_warp_split;
    mutable int warp_split; // helper used to keep track of current warp splitting
    const int nParity;
    mutable MultiBlasArg<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor> arg;
    const coeff_array<T> &a, &b, &c;
    std::vector<ColorSpinorField *> &x, &y, &z, &w;

    // host pointers used for backing up fields when tuning
    // don't curry into the Spinors to minimize parameter size
    char *Y_h[NYW_max], *W_h[NYW_max], *Ynorm_h[NYW_max], *Wnorm_h[NYW_max];

    bool tuneSharedBytes() const { return false; }

    // for these streaming kernels, there is no need to tune the grid size, just use max
    unsigned int minGridSize() const { return maxGridSize(); }

  public:
    MultiBlas(SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Functor &f, const coeff_array<T> &a,
              const coeff_array<T> &b, const coeff_array<T> &c, std::vector<ColorSpinorField *> &x,
              std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z,
              std::vector<ColorSpinorField *> &w, int NYW, int length) :
      TunableVectorY(NYW),
      NYW(NYW),
      warp_split(1),
      nParity(x[0]->SiteSubset()),
      arg(X, Y, Z, W, f, NYW, length / nParity),
      a(a),
      b(b),
      c(c),
      x(x),
      y(y),
      z(z),
      w(w),
      Y_h(),
      W_h(),
      Ynorm_h(),
      Wnorm_h()
    {
      // heuristic for enabling if we need the warp-splitting optimization
      const int gpu_size = 2 * deviceProp.maxThreadsPerBlock * deviceProp.multiProcessorCount;
      switch (gpu_size / (x[0]->Length() * NYW)) {
      case 0: max_warp_split = 1; break; // we have plenty of work, no need to split
      case 1: max_warp_split = 2; break; // double the thread count
      case 2:                            // quadruple the thread count
      default: max_warp_split = 4;
      }
      max_warp_split = std::min(NXZ, max_warp_split); // ensure we only split if valid

      Amatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(a.data));
      Bmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(b.data));
      Cmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(c.data));

      strcpy(aux, x[0]->AuxString());
      if (x[0]->Precision() != y[0]->Precision()) {
        strcat(aux, ",");
        strcat(aux, y[0]->AuxString());
      }

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/multi_blas_core.cuh");
#endif
      }

      virtual ~MultiBlas() {}

      inline TuneKey tuneKey() const
      {
        char name[TuneKey::name_n];
        strcpy(name, num_to_string<NXZ>::value);
        strcat(name, std::to_string(NYW).c_str());
        strcat(name, typeid(arg.f).name());
        return TuneKey(x[0]->VolString(), name, aux);
      }

      inline void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        typedef typename scalar<FloatN>::type Float;
        typedef typename vector<Float, 2>::type Float2;

#ifdef JITIFY
        using namespace jitify::reflection;
        auto instance = program->kernel("quda::blas::multiBlasKernel")
                          .instantiate(Type<FloatN>(), M, NXZ, tp.aux.x, Type<decltype(arg)>());

        if (a.data) {
          Float2 A[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) A[NYW * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));

          auto Amatrix_d = instance.get_constant_ptr("quda::blas::Amatrix_d");
          cuMemcpyHtoDAsync(Amatrix_d, A, NXZ * NYW * sizeof(decltype(A[0])), stream);
        }

        if (b.data) {
          Float2 B[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) B[NYW * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));

          auto Bmatrix_d = instance.get_constant_ptr("quda::blas::Bmatrix_d");
          cuMemcpyHtoDAsync(Bmatrix_d, B, NXZ * NYW * sizeof(decltype(B[0])), stream);
        }

        if (c.data) {
          Float2 C[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) C[NYW * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));
          auto Cmatrix_d = instance.get_constant_ptr("quda::blas::Cmatrix_d");
          cuMemcpyHtoDAsync(Cmatrix_d, C, NXZ * NYW * sizeof(decltype(C[0])), stream);
        }

        tp.block.x *= tp.aux.x; // include warp-split factor
        jitify_error = instance.configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
        tp.block.x /= tp.aux.x; // restore block size
#else
        if (a.data) {
          Float2 A[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) A[NYW * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));
          cudaMemcpyToSymbolAsync(Amatrix_d, A, NXZ * NYW * sizeof(decltype(A[0])), 0, cudaMemcpyHostToDevice, stream);
        }

        if (b.data) {
          Float2 B[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) B[NYW * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));
          cudaMemcpyToSymbolAsync(Bmatrix_d, B, NXZ * NYW * sizeof(decltype(B[0])), 0, cudaMemcpyHostToDevice, stream);
        }

        if (c.data) {
          Float2 C[MAX_MATRIX_SIZE / sizeof(Float2)];
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++) C[NYW * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));
          cudaMemcpyToSymbolAsync(Cmatrix_d, C, NXZ * NYW * sizeof(decltype(C[0])), 0, cudaMemcpyHostToDevice, stream);
        }

        tp.block.x *= tp.aux.x; // include warp-split factor

        switch (tp.aux.x) {
        case 1: multiBlasKernel<FloatN, M, NXZ, 1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;
#ifdef WARP_SPLIT
        case 2: multiBlasKernel<FloatN, M, NXZ, 2><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;
        case 4: multiBlasKernel<FloatN, M, NXZ, 4><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;
#endif
        default: errorQuda("warp-split factor %d not instantiated", tp.aux.x);
        }

        tp.block.x /= tp.aux.x; // restore block size
#endif
      }

      void preTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (SpinorY::write) arg.Y[i].backup(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
          if (SpinorW::write) arg.W[i].backup(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
        }
      }

      void postTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (SpinorY::write) arg.Y[i].restore(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
          if (SpinorW::write) arg.W[i].restore(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
        }
      }

      bool advanceAux(TuneParam &param) const
      {
#ifdef WARP_SPLIT
        if (2 * param.aux.x <= max_warp_split) {
          param.aux.x *= 2;
          warp_split = param.aux.x;
          return true;
        } else {
          param.aux.x = 1;
          warp_split = param.aux.x;
          // reset the block dimension manually here to pick up the warp_split parameter
          resetBlockDim(param);
          return false;
        }
#else
        warp_split = 1;
        return false;
#endif
      }

      int blockStep() const { return deviceProp.warpSize / warp_split; }
      int blockMin() const { return deviceProp.warpSize / warp_split; }

      void initTuneParam(TuneParam &param) const
      {
        TunableVectorY::initTuneParam(param);
        param.grid.z = nParity;
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      void defaultTuneParam(TuneParam &param) const
      {
        TunableVectorY::defaultTuneParam(param);
        param.grid.z = nParity;
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      long long flops() const { return arg.f.flops() * vec_length<FloatN>::value * (long)arg.length * nParity * M; }

      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        return ((arg.f.streams() - 2) * x[0]->Bytes() + 2 * y[0]->Bytes());
      }

      int tuningIter() const { return 3; }
    };

    template <int NXZ_, typename RegType, typename StoreType, typename yType, int M,
              template <int, typename, typename> class Functor, typename write, typename T>
    void multiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                   std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                   std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w, int length)
    {
      typedef typename scalar<RegType>::type Float;
      typedef typename vector<Float, 2>::type Float2;
      typedef vector<Float, 2> vec2;

      const int NYW = y.size();
      // the below line enable NXZ = 128 for floating point types, which is invalid for fixed-point types
      constexpr int NXZ = isFixed<StoreType>::value && NXZ_ == 128 ? 64 : NXZ_;
      Functor<NXZ, Float2, RegType> f(NYW);
      constexpr int NYW_max = max_YW_size<NXZ, StoreType, yType, write, decltype(f)>();
      const int NYW_max_check
        = max_YW_size<write>(x.size(), x[0]->Precision(), y[0]->Precision(), f.use_z, f.use_w, false);

      if (!is_valid_NXZ(NXZ, false, x[0]->Precision() < QUDA_SINGLE_PRECISION))
        errorQuda("NXZ=%d is not a valid size ( MAX_MULTI_BLAS_N %d)", NXZ, MAX_MULTI_BLAS_N);
      if (NYW_max != NYW_max_check) errorQuda("Runtime %d and compile time %d limits disagree", NYW_max, NYW_max_check);
      if (NYW > NYW_max) errorQuda("NYW exceeds max size (%d > %d)", NYW, NYW_max);
      if (NXZ * NYW * sizeof(Float2) > MAX_MATRIX_SIZE)
        errorQuda("Coefficient matrix exceeds max size (%lu > %d)", NXZ * NYW * sizeof(Float2), MAX_MATRIX_SIZE);

      SpinorTexture<RegType, StoreType, M> X[NXZ];
      Spinor<RegType, yType, M, write::Y> Y[NYW_max];
      SpinorTexture<RegType, StoreType, M> Z[NXZ];
      Spinor<RegType, StoreType, M, write::W> W[NYW_max];

      for (int i = 0; i < NXZ; i++) {
        X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i]));
        Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));
      }
      for (int i = 0; i < NYW; i++) {
        Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i]));
        W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));
      }

      MultiBlas<NXZ, RegType, M, typename std::remove_reference<decltype(X[0])>::type,
                typename std::remove_reference<decltype(Y[0])>::type, typename std::remove_reference<decltype(Z[0])>::type,
                typename std::remove_reference<decltype(W[0])>::type, decltype(f), T>
        blas(X, Y, Z, W, f, a, b, c, x, y, z, w, NYW, length);
      blas.apply(*getStream());

      blas::bytes += blas.bytes();
      blas::flops += blas.flops();

      checkCudaError();
    }

    /**
       Driver for generic blas routine with four loads and two store.
    */
    template <int NXZ, template <int MXZ, typename Float, typename FloatN> class Functor, typename write, typename T>
    void uniMultiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                      CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                      CompositeColorSpinorField &w)
    {

      if (checkLocation(*x[0], *y[0], *z[0], *w[0]) == QUDA_CUDA_FIELD_LOCATION) {

        if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
#if defined(NSPIN4) || defined(NSPIN2) || defined(NSPIN1)
          const int M = 1;
          multiBlas<NXZ, double2, double2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

        } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION && x[0]->Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
            const int M = 1;
            multiBlas<NXZ, float4, float4, float4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (4 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif

          } else if (x[0]->Nspin() == 2 || x[0]->Nspin() == 1) {

#if defined(NSPIN2) || defined(NSPIN1)
            const int M = 1;
            multiBlas<NXZ, float2, float2, float2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (2 * M));
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

        } else if (y[0]->Precision() == QUDA_HALF_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
          if (x[0]->Ncolor() != 3) { errorQuda("nColor = %d is not supported", x[0]->Ncolor()); }
          if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
            const int M = 6;
            multiBlas<NXZ, float4, short4, short4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
          } else if (x[0]->Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 3;
            multiBlas<NXZ, float2, short2, short2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

        } else if (y[0]->Precision() == QUDA_QUARTER_PRECISION && x[0]->Precision() == QUDA_QUARTER_PRECISION) {

#if QUDA_PRECISION & 1
          if (x[0]->Ncolor() != 3) { errorQuda("nColor = %d is not supported", x[0]->Ncolor()); }
          if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
            const int M = 6;
            multiBlas<NXZ, float4, char4, char4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
          } else if (x[0]->Nspin() == 1) { // staggered
#if defined(NSPIN1)
            const int M = 3;
            multiBlas<NXZ, float2, char2, char2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
          } else {
            errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

        } else {

          errorQuda("Precision combination x=%d not supported\n", x[0]->Precision());
        }
      } else { // fields on the cpu
        errorQuda("Not implemented");
      }
    }

    /**
       Driver for generic blas routine with four loads and two store.
    */
    template <int NXZ, template <int MXZ, typename Float, typename FloatN> class Functor, typename write, typename T>
    void mixedMultiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
        CompositeColorSpinorField &w)
    {
      if (checkLocation(*x[0], *y[0], *z[0], *w[0]) == QUDA_CUDA_FIELD_LOCATION) {

        if (y[0]->Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
          if (x[0]->Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
            if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 12;
              multiBlas<NXZ, double2, float4, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            } else if (x[0]->Nspin() == 1) {

#if defined(NSPIN1)
              const int M = 3;
              multiBlas<NXZ, double2, float2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            }

#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

          } else if (x[0]->Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
            if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 12;
              multiBlas<NXZ, double2, short4, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif

            } else if (x[0]->Nspin() == 1) {

#if defined(NSPIN1)
              const int M = 3;
              multiBlas<NXZ, double2, short2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

          } else if (x[0]->Precision() == QUDA_QUARTER_PRECISION) {

#if QUDA_PRECISION & 1
            if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 12;
              multiBlas<NXZ, double2, char4, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif

            } else if (x[0]->Nspin() == 1) {

#if defined(NSPIN1)
              const int M = 3;
              multiBlas<NXZ, double2, char2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

          } else {
            errorQuda("Not implemented for this precision combination %d %d", x[0]->Precision(), y[0]->Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y[0]->Precision());
#endif

        } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION) {

#if (QUDA_PRECISION & 4)
          if (x[0]->Precision() == QUDA_HALF_PRECISION) {

#if (QUDA_PRECISION & 2)
            if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 6;
              multiBlas<NXZ, float4, short4, float4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif

            } else if (x[0]->Nspin() == 1) {

#if defined(NSPIN1)
              const int M = 3;
              multiBlas<NXZ, float2, short2, float2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            } else {
              errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
            }

#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y[0]->Precision());
#endif

          } else if (x[0]->Precision() == QUDA_QUARTER_PRECISION) {

#if (QUDA_PRECISION & 1)
            if (x[0]->Nspin() == 4) {
#if defined(NSPIN4)
              const int M = 6;
              multiBlas<NXZ, float4, char4, float4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif

            } else if (x[0]->Nspin() == 1) {

#if defined(NSPIN1)
              const int M = 3;
              multiBlas<NXZ, float2, char2, float2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
            } else {
              errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
            }

#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y[0]->Precision());
#endif

          } else {
            errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
          }
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, y[0]->Precision());
#endif
        } else {
          errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
        }
      } else { // fields on the cpu
        errorQuda("Not implemented");
      }
    }

    template <int NXZ, template <int MXZ, typename Float, typename FloatN> class Functor, typename write, typename T>
    void multiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                   CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                   CompositeColorSpinorField &w)
    {
      if (x[0]->Precision() != y[0]->Precision()) {
        mixedMultiBlas<NXZ, Functor, write>(a, b, c, x, y, x, w);
      } else {
        uniMultiBlas<NXZ, Functor, write>(a, b, c, x, y, x, w);
      }
    }

    template <template <int MXZ, typename Float, typename FloatN> class Functor, typename write, typename T>
    void multiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                   CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                   CompositeColorSpinorField &w)
    {
      // instantiate the loop unrolling template
      switch (x.size()) {
      // by default all powers of two <= 64 are instantiated
      case 1: multiBlas<1, Functor, write>(a, b, c, x, y, x, w); break;
      case 2: multiBlas<2, Functor, write>(a, b, c, x, y, x, w); break;
      case 4: multiBlas<4, Functor, write>(a, b, c, x, y, x, w); break;
      case 8: multiBlas<8, Functor, write>(a, b, c, x, y, x, w); break;
      case 16: multiBlas<16, Functor, write>(a, b, c, x, y, x, w); break;
      case 32: multiBlas<32, Functor, write>(a, b, c, x, y, x, w); break;
      case 64: multiBlas<64, Functor, write>(a, b, c, x, y, x, w); break;
      case 128: multiBlas<128, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 3
      case 3: multiBlas<3, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 5
      case 5: multiBlas<5, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 6
      case 6: multiBlas<6, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 7
      case 7: multiBlas<7, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 9
      case 9: multiBlas<9, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 10
      case 10: multiBlas<10, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 11
      case 11: multiBlas<11, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 12
      case 12: multiBlas<12, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 13
      case 13: multiBlas<13, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 14
      case 14: multiBlas<14, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 15
      case 15: multiBlas<15, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 17
      case 17: multiBlas<17, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 18
      case 18: multiBlas<18, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 19
      case 19: multiBlas<19, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 20
      case 20: multiBlas<20, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 21
      case 21: multiBlas<21, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 22
      case 22: multiBlas<22, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 23
      case 23: multiBlas<23, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 24
      case 24: multiBlas<24, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 25
      case 25: multiBlas<25, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 26
      case 26: multiBlas<26, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 27
      case 27: multiBlas<27, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 28
      case 28: multiBlas<28, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 29
      case 29: multiBlas<29, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 30
      case 30: multiBlas<30, Functor, write>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 31
      case 31: multiBlas<31, Functor, write>(a, b, c, x, y, x, w); break;
#endif // 31
#endif // 30
#endif // 29
#endif // 28
#endif // 27
#endif // 26
#endif // 25
#endif // 24
#endif // 23
#endif // 22
#endif // 21
#endif // 20
#endif // 19
#endif // 18
#endif // 17
#endif // 15
#endif // 14
#endif // 13
#endif // 12
#endif // 11
#endif // 10
#endif // 9
#endif // 7
#endif // 6
#endif // 5
#endif // 3
      default: errorQuda("x.size %lu greater than MAX_MULTI_BLAS_N %d", x.size(), MAX_MULTI_BLAS_N);
      }
    }

    using range = std::pair<size_t,size_t>;

    template <template <int MXZ, typename Float, typename FloatN> class Functor, typename T>
    void axpy_recurse(const T *a_, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                      const range &range_x, const range &range_y, int upper)
    {
      using write_ = write<0, 1, 0, 0>;
      // if greater than max single-kernel size, recurse
      if (y.size() > (size_t)max_YW_size<write_>(x.size(), x[0]->Precision(), y[0]->Precision(), false, false, false)) {
        // We need to split up 'a' carefully since it's row-major.
        T *tmpmajor = new T[x.size() * y.size()];
        T *tmpmajor0 = &tmpmajor[0];
        T *tmpmajor1 = &tmpmajor[x.size() * (y.size() / 2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());

        const unsigned int xlen = x.size();
        const unsigned int ylen0 = y.size()/2;
        const unsigned int ylen1 = y.size() - y.size()/2;

        int count = 0, count0 = 0, count1 = 0;
        for (unsigned int i = 0; i < xlen; i++)
        {
          for (unsigned int j = 0; j < ylen0; j++)
            tmpmajor0[count0++] = a_[count++];
          for (unsigned int j = 0; j < ylen1; j++)
            tmpmajor1[count1++] = a_[count++];
        }

        axpy_recurse<Functor>(tmpmajor0, x, y0, range_x, range(range_y.first, range_y.first + y0.size()), upper);
        axpy_recurse<Functor>(tmpmajor1, x, y1, range_x, range(range_y.first + y0.size(), range_y.second), upper);

        delete[] tmpmajor;
      } else {
        // if at the bottom of recursion,
        if (is_valid_NXZ(x.size(), false, x[0]->Precision() < QUDA_SINGLE_PRECISION)) {
          // since tile range is [first,second), e.g., [first,second-1], we need >= here
          // if upper triangular and upper-right tile corner is below diagonal return
          if (upper == 1 && range_y.first >= range_x.second) { return; }
          // if lower triangular and lower-left tile corner is above diagonal return
          if (upper == -1 && range_x.first >= range_y.second) { return; }

          // mark true since we will copy the "a" matrix into constant memory
          coeff_array<T> a(a_), b, c;
          multiBlas<Functor, write_>(a, b, c, x, y, x, y);
        } else {
          // split the problem in half and recurse
          const T *a0 = &a_[0];
          const T *a1 = &a_[(x.size() / 2) * y.size()];

          std::vector<ColorSpinorField *> x0(x.begin(), x.begin() + x.size() / 2);
          std::vector<ColorSpinorField *> x1(x.begin() + x.size() / 2, x.end());

          axpy_recurse<Functor>(a0, x0, y, range(range_x.first, range_x.first + x0.size()), range_y, upper);
          axpy_recurse<Functor>(a1, x1, y, range(range_x.first + x0.size(), range_x.second), range_y, upper);
        }
      } // end if (y.size() > max_YW_size())
    }

    void caxpy(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), 0);
    }

    void caxpy_U(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_U with non-square 'a' has not yet been implemented. Use block caxpy instead");
      }
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), 1);
    }

    void caxpy_L(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_L with non-square 'a' has not yet been implemented. Use block caxpy instead");
      }
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), -1);
    }

    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy(a, x.Components(), y.Components()); }

    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_U(a, x.Components(), y.Components()); }

    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_L(a, x.Components(), y.Components()); }

    void caxpyz_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
                        std::vector<ColorSpinorField*> &z, const range &range_x, const range &range_y,
                        int pass, int upper)
    {
      // if greater than max single-kernel size, recurse
      using write_ = write<0, 0, 0, 1>;

      if (y.size() > (size_t)max_YW_size<write_>(x.size(), x[0]->Precision(), y[0]->Precision(), false, true, false)) {
        // We need to split up 'a' carefully since it's row-major.
        Complex* tmpmajor = new Complex[x.size()*y.size()];
        Complex* tmpmajor0 = &tmpmajor[0];
        Complex* tmpmajor1 = &tmpmajor[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());

        std::vector<ColorSpinorField*> z0(z.begin(), z.begin() + z.size()/2);
        std::vector<ColorSpinorField*> z1(z.begin() + z.size()/2, z.end());

        const unsigned int xlen = x.size();
        const unsigned int ylen0 = y.size()/2;
        const unsigned int ylen1 = y.size() - y.size()/2;

        int count = 0, count0 = 0, count1 = 0;
        for (unsigned int i_ = 0; i_ < xlen; i_++)
        {
          for (unsigned int j = 0; j < ylen0; j++)
            tmpmajor0[count0++] = a_[count++];
          for (unsigned int j = 0; j < ylen1; j++)
            tmpmajor1[count1++] = a_[count++];
        }

        caxpyz_recurse(tmpmajor0, x, y0, z0, range_x, range(range_y.first, range_y.first + y0.size()), pass, upper);
        caxpyz_recurse(tmpmajor1, x, y1, z1, range_x, range(range_y.first + y0.size(), range_y.second), pass, upper);

        delete[] tmpmajor;
      } else {
        // if at bottom of recursion check where we are
        if (is_valid_NXZ(x.size(), false, x[0]->Precision() < QUDA_SINGLE_PRECISION)) {
          // check if tile straddles diagonal
          bool is_diagonal = (range_x.first < range_y.second) && (range_y.first < range_x.second);
          if (pass==1) {
            if (!is_diagonal) {
              // if upper triangular and upper-right tile corner is below diagonal return
              if (upper == 1 && range_y.first >= range_x.second) { return; }
              // if lower triangular and lower-left tile corner is above diagonal return
              if (upper == -1 && range_x.first >= range_y.second) { return; }
              caxpy(a_, x, z); return;  // off diagonal
            }
            return;
      	  } else {
            if (!is_diagonal) return; // We're on the first pass, so we only want to update the diagonal.
          }

          coeff_array<Complex> a(a_), b, c;
          multiBlas<multicaxpyz_, write_>(a, b, c, x, y, x, z);
        } else {
          // split the problem in half and recurse
          const Complex *a0 = &a_[0];
          const Complex *a1 = &a_[(x.size() / 2) * y.size()];

          std::vector<ColorSpinorField *> x0(x.begin(), x.begin() + x.size() / 2);
          std::vector<ColorSpinorField *> x1(x.begin() + x.size() / 2, x.end());

          caxpyz_recurse(a0, x0, y, z, range(range_x.first, range_x.first + x0.size()), range_y, pass, upper);
          caxpyz_recurse(a1, x1, y, z, range(range_x.first + x0.size(), range_x.second), range_y, pass, upper);
        }
      } // end if (y.size() > max_YW_size())
    }

    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 0);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 0);
    }

    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 1);
    }

    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, -1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, -1);
    }


    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz_U(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z) {
      caxpyz_L(a, x.Components(), y.Components(), z.Components());
    }

    void axpyBzpcx(const double *a_, std::vector<ColorSpinorField *> &x_, std::vector<ColorSpinorField *> &y_,
                   const double *b_, ColorSpinorField &z_, const double *c_)
    {
      using write_ = write<0, 1, 0, 1>;

      if (y_.size() <= (size_t)max_YW_size<write_>(1, z_.Precision(), y_[0]->Precision(), false, true, false)) {
        // swizzle order since we are writing to x_ and y_, but the
	// multi-blas only allow writing to y and w, and moreover the
	// block width of y and w must match, and x and z must match.
	std::vector<ColorSpinorField*> &y = y_;
	std::vector<ColorSpinorField*> &w = x_;

	// wrap a container around the third solo vector
	std::vector<ColorSpinorField*> x;
	x.push_back(&z_);

        coeff_array<double> a(a_), b(b_), c(c_);
        multiBlas<1, multi_axpyBzpcx_, write_>(a, b, c, x, y, x, w);
      } else {
        // split the problem in half and recurse
	const double *a0 = &a_[0];
	const double *b0 = &b_[0];
	const double *c0 = &c_[0];

	std::vector<ColorSpinorField*> x0(x_.begin(), x_.begin() + x_.size()/2);
	std::vector<ColorSpinorField*> y0(y_.begin(), y_.begin() + y_.size()/2);

	axpyBzpcx(a0, x0, y0, b0, z_, c0);

	const double *a1 = &a_[y_.size()/2];
	const double *b1 = &b_[y_.size()/2];
	const double *c1 = &c_[y_.size()/2];

	std::vector<ColorSpinorField*> x1(x_.begin() + x_.size()/2, x_.end());
	std::vector<ColorSpinorField*> y1(y_.begin() + y_.size()/2, y_.end());

	axpyBzpcx(a1, x1, y1, b1, z_, c1);
      }
    }

    void caxpyBxpz(const Complex *a_, std::vector<ColorSpinorField*> &x_, ColorSpinorField &y_,
		   const Complex *b_, ColorSpinorField &z_)
    {
      using write_ = write<0, 1, 0, 1>;

      if (is_valid_NXZ(x_.size(), false, x_[0]->Precision() < QUDA_SINGLE_PRECISION)) // only swizzle if we have to.
      {
        // swizzle order since we are writing to y_ and z_, but the
        // multi-blas only allow writing to y and w, and moreover the
        // block width of y and w must match, and x and z must match.
        // Also, wrap a container around them.
        std::vector<ColorSpinorField*> y;
        y.push_back(&y_);
        std::vector<ColorSpinorField*> w;
        w.push_back(&z_);

        // we're reading from x
        std::vector<ColorSpinorField*> &x = x_;

        coeff_array<Complex> a(a_), b(b_), c;
        multiBlas<multi_caxpyBxpz_, write_>(a, b, c, x, y, x, w);
      } else {
        // split the problem in half and recurse
        const Complex *a0 = &a_[0];
        const Complex *b0 = &b_[0];

        std::vector<ColorSpinorField*> x0(x_.begin(), x_.begin() + x_.size()/2);

        caxpyBxpz(a0, x0, y_, b0, z_);

        const Complex *a1 = &a_[x_.size()/2];
        const Complex *b1 = &b_[x_.size()/2];

        std::vector<ColorSpinorField*> x1(x_.begin() + x_.size()/2, x_.end());

        caxpyBxpz(a1, x1, y_, b1, z_);
      }
    }

    void axpy(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), 0);
    }

    void axpy_U(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_U with non-square 'a' has not yet been implemented. Use block axpy instead");
      }
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), 1);
    }

    void axpy_L(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_L with non-square 'a' has not yet been implemented. Use block axpy instead");
      }
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), -1);
    }

    // Composite field version
    void axpy(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy(a, x.Components(), y.Components()); }

    void axpy_U(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_U(a, x.Components(), y.Components()); }

    void axpy_L(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_L(a, x.Components(), y.Components()); }


  } // namespace blas

} // namespace quda
