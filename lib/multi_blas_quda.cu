#include <stdlib.h>
#include <stdio.h>
#include <cstring> // needed for memset
#include <typeinfo>

#include <tune_quda.h>
#include <blas_quda.h>
#include <color_spinor_field.h>

#include <jitify_helper.cuh>
#include <kernels/multi_blas_core.cuh>

namespace quda {

  namespace blas {

    cudaStream_t* getStream();

    template <int writeX, int writeY, int writeZ, int writeW>
    struct write {
      static constexpr int X = writeX;
      static constexpr int Y = writeY;
      static constexpr int Z = writeZ;
      static constexpr int W = writeW;
    };

    namespace detail
    {
      template <unsigned... digits> struct to_chars {
        static const char value[];
      };

      template <unsigned... digits> const char to_chars<digits...>::value[] = {('0' + digits)..., 0};

      template <unsigned rem, unsigned... digits> struct explode : explode<rem / 10, rem % 10, digits...> {
      };

      template <unsigned... digits> struct explode<0, digits...> : to_chars<digits...> {
      };
    } // namespace detail

    template <unsigned num> struct num_to_string : detail::explode<num / 10, num % 10> {
    };

    template <int NXZ, typename FloatN, int M, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW,
        typename Functor, typename T>
    class MultiBlas : public TunableVectorY
    {

  private:
      const int NYW;
      const int nParity;
      mutable MultiBlasArg<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor> arg;
      const coeff_array<T> &a, &b, &c;

      std::vector<ColorSpinorField *> &x, &y, &z, &w;

      // host pointers used for backing up fields when tuning
      // don't curry into the Spinors to minimize parameter size
      char *Y_h[MAX_MULTI_BLAS_N], *W_h[MAX_MULTI_BLAS_N], *Ynorm_h[MAX_MULTI_BLAS_N], *Wnorm_h[MAX_MULTI_BLAS_N];

      bool tuneSharedBytes() const { return false; }

  public:
      MultiBlas(SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Functor &f, const coeff_array<T> &a,
          const coeff_array<T> &b, const coeff_array<T> &c, std::vector<ColorSpinorField *> &x,
          std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
          int NYW, int length) :
          TunableVectorY(NYW),
          NYW(NYW),
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

      inline void apply(const cudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        typedef typename scalar<FloatN>::type Float;
        typedef typename vector<Float, 2>::type Float2;
#ifdef JITIFY
        using namespace jitify::reflection;
        auto instance
            = program->kernel("quda::blas::multiBlasKernel").instantiate(Type<FloatN>(), M, NXZ, Type<decltype(arg)>());

        // FIXME - if NXZ=1 no need to copy entire array
        // FIXME - do we really need strided access here?
        if (a.data && a.use_const) {
          Float2 A[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              A[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));

          auto Amatrix_d = instance.get_constant_ptr("quda::blas::Amatrix_d");
          cuMemcpyHtoDAsync(Amatrix_d, A, MAX_MATRIX_SIZE, *getStream());
        }

        if (b.data && b.use_const) {
          Float2 B[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              B[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));

          auto Bmatrix_d = instance.get_constant_ptr("quda::blas::Bmatrix_d");
          cuMemcpyHtoDAsync(Bmatrix_d, B, MAX_MATRIX_SIZE, *getStream());
        }

        if (c.data && c.use_const) {
          Float2 C[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              C[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));

          auto Cmatrix_d = instance.get_constant_ptr("quda::blas::Cmatrix_d");
          cuMemcpyHtoDAsync(Cmatrix_d, C, MAX_MATRIX_SIZE, *getStream());
        }

        jitify_error = instance.configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
        // FIXME - if NXZ=1 no need to copy entire array
        // FIXME - do we really need strided access here?
        if (a.data && a.use_const) {
          Float2 A[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              A[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));

          cudaMemcpyToSymbolAsync(Amatrix_d, A, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        }

        if (b.data && b.use_const) {
          Float2 B[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              B[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));

          cudaMemcpyToSymbolAsync(Bmatrix_d, B, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        }

        if (c.data && c.use_const) {
          Float2 C[MAX_MATRIX_SIZE / sizeof(Float2)];
          // since the kernel doesn't know the width of them matrix at compile
          // time we stride it and copy the padded matrix to GPU
          for (int i = 0; i < NXZ; i++)
            for (int j = 0; j < NYW; j++)
              C[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));

          cudaMemcpyToSymbolAsync(Cmatrix_d, C, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        }
#if CUDA_VERSION < 9000
        cudaMemcpyToSymbolAsync(arg_buffer, reinterpret_cast<char *>(&arg), sizeof(arg), 0, cudaMemcpyHostToDevice,
                                *getStream());
#endif
        multiBlasKernel<FloatN, M, NXZ><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
#endif
      }

      void preTune()
      {
        for (int i = 0; i < NYW; ++i) {
          arg.Y[i].backup(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
          arg.W[i].backup(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
        }
      }

      void postTune()
      {
        for (int i = 0; i < NYW; ++i) {
          arg.Y[i].restore(&Y_h[i], &Ynorm_h[i], y[i]->Bytes(), y[i]->NormBytes());
          arg.W[i].restore(&W_h[i], &Wnorm_h[i], w[i]->Bytes(), w[i]->NormBytes());
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        TunableVectorY::initTuneParam(param);
        param.grid.z = nParity;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        TunableVectorY::defaultTuneParam(param);
        param.grid.z = nParity;
      }

      long long flops() const { return arg.f.flops() * vec_length<FloatN>::value * (long)arg.length * nParity * M; }

      long long bytes() const
      {
        // the factor two here assumes we are reading and writing to the high precision vector
        return ((arg.f.streams() - 2) * x[0]->Bytes() + 2 * y[0]->Bytes());
      }

      int tuningIter() const { return 3; }
    };

    template <int NXZ, typename RegType, typename StoreType, typename yType, int M,
        template <int, typename, typename> class Functor, typename write, typename T>
    void multiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z,
        std::vector<ColorSpinorField *> &w, int length)
    {
      const int NYW = y.size();

      const int N = NXZ > NYW ? NXZ : NYW;
      if (N > MAX_MULTI_BLAS_N) errorQuda("Spinor vector length exceeds max size (%d > %d)", N, MAX_MULTI_BLAS_N);

      if (NXZ * NYW * sizeof(Complex) > MAX_MATRIX_SIZE)
        errorQuda("A matrix exceeds max size (%lu > %d)", NXZ * NYW * sizeof(Complex), MAX_MATRIX_SIZE);

      typedef typename scalar<RegType>::type Float;
      typedef typename vector<Float, 2>::type Float2;
      typedef vector<Float, 2> vec2;

      SpinorTexture<RegType, StoreType, M> X[NXZ];
      Spinor<RegType, yType, M, write::Y> Y[MAX_MULTI_BLAS_N];
      SpinorTexture<RegType, StoreType, M> Z[NXZ];
      Spinor<RegType, StoreType, M, write::W> W[MAX_MULTI_BLAS_N];

      for (int i = 0; i < NXZ; i++) {
        X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i]));
        Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));
      }
      for (int i = 0; i < NYW; i++) {
        Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i]));
        W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));
      }

      // if block caxpy is an 'outer product of caxpy' where 'x'

      Functor<NXZ, Float2, RegType> f(a, b, c, NYW);

      MultiBlas<NXZ, RegType, M, SpinorTexture<RegType, StoreType, M>, Spinor<RegType, yType, M, write::Y>,
                SpinorTexture<RegType, StoreType, M>, Spinor<RegType, StoreType, M, write::W>, decltype(f), T>
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
    void multiBlas(const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
        CompositeColorSpinorField &w)
    {

      if (checkLocation(*x[0], *y[0], *z[0], *w[0]) == QUDA_CUDA_FIELD_LOCATION) {

        if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
          const int M = 1;
          multiBlas<NXZ, double2, double2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
#else
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

        } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION && x[0]->Precision() == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
          if (x[0]->Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
            const int M = 1;
            multiBlas<NXZ, float4, float4, float4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (4 * M));
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif

          } else if (x[0]->Nspin() == 2 || x[0]->Nspin() == 1) {

#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
            const int M = 1;
            multiBlas<NXZ, float2, float2, float2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Length() / (2 * M));
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
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
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
            const int M = 6;
            multiBlas<NXZ, float4, short4, short4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
          } else if (x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
            const int M = 3;
            multiBlas<NXZ, float2, short2, short2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
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
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
            const int M = 6;
            multiBlas<NXZ, float4, char4, char4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
          } else if (x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
            const int M = 3;
            multiBlas<NXZ, float2, char2, char2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
            errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
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
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
              const int M = 12;
              multiBlas<NXZ, double2, float4, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
            } else if (x[0]->Nspin() == 1) {
#if defined(GPU_STAGGERED_DIRAC)
              const int M = 3;
              multiBlas<NXZ, double2, float2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
            }
#else
            errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif

          } else if (x[0]->Precision() == QUDA_HALF_PRECISION) {

#if QUDA_PRECISION & 2
            if (x[0]->Nspin() == 4) {
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
              const int M = 12;
              multiBlas<NXZ, double2, short4, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
            } else if (x[0]->Nspin() == 1) {
#if defined(GPU_STAGGERED_DIRAC)
              const int M = 3;
              multiBlas<NXZ, double2, short2, double2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
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
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
              const int M = 6;
              multiBlas<NXZ, float4, short4, float4, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif

            } else if (x[0]->Nspin() == 2 || x[0]->Nspin() == 1) {

#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_STAGGERED_DIRAC)
              const int M = 3;
              multiBlas<NXZ, float2, short2, float2, M, Functor, write>(a, b, c, x, y, z, w, x[0]->Volume());
#else
              errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
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
          errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, x[0]->Precision());
#endif
        } else {
          errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
        }
      } else { // fields on the cpu
        errorQuda("Not implemented");
      }
    }

    void caxpy_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
          int i_idx ,int j_idx, int upper) {

      if (y.size() > MAX_MULTI_BLAS_N) // if greater than max single-kernel size, recurse.
      {
        // We need to split up 'a' carefully since it's row-major.
        Complex* tmpmajor = new Complex[x.size()*y.size()];
        Complex* tmpmajor0 = &tmpmajor[0];
        Complex* tmpmajor1 = &tmpmajor[x.size()*(y.size()/2)];
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

        caxpy_recurse(tmpmajor0, x, y0, i_idx, 2*j_idx+0, upper);
        caxpy_recurse(tmpmajor1, x, y1, i_idx, 2*j_idx+1, upper);

        delete[] tmpmajor;
      }
      else
      {
        // if at the bottom of recursion,
        // return if on lower left for upper triangular,
        // return if on upper right for lower triangular.
        if (x.size() <= MAX_MULTI_BLAS_N) {
          if (upper == 1 && j_idx < i_idx) { return; }
          if (upper == -1 && j_idx > i_idx) { return; }
        }

        // mark true since we will copy the "a" matrix into constant memory
        coeff_array<Complex> a(a_, true), b, c;

        if (x[0]->Precision() == y[0]->Precision())
        {
          switch (x.size()) {
          case 1: multiBlas<1, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: multiBlas<2, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: multiBlas<3, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: multiBlas<4, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: multiBlas<5, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: multiBlas<6, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: multiBlas<7, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: multiBlas<8, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: multiBlas<9, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: multiBlas<10, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: multiBlas<11, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: multiBlas<12, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: multiBlas<13, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: multiBlas<14, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: multiBlas<15, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: multiBlas<16, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpy_recurse(a0, x0, y, 2*i_idx+0, j_idx, upper);
              caxpy_recurse(a1, x1, y, 2*i_idx+1, j_idx, upper);
              break;
          }
        }
        else // precisions don't agree.
        {
          switch (x.size()) {
          case 1: mixedMultiBlas<1, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: mixedMultiBlas<2, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: mixedMultiBlas<3, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: mixedMultiBlas<4, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: mixedMultiBlas<5, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: mixedMultiBlas<6, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: mixedMultiBlas<7, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: mixedMultiBlas<8, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: mixedMultiBlas<9, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: mixedMultiBlas<10, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: mixedMultiBlas<11, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: mixedMultiBlas<12, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: mixedMultiBlas<13, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: mixedMultiBlas<14, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: mixedMultiBlas<15, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: mixedMultiBlas<16, multicaxpy_, write<0, 1, 0, 0>>(a, b, c, x, y, x, y); break;
#endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpy_recurse(a0, x0, y, 2*i_idx+0, j_idx, upper);
              caxpy_recurse(a1, x1, y, 2*i_idx+1, j_idx, upper);
              break;
          }
        }
      } // end if (y.size() > MAX_MULTI_BLAS_N)
    }

    void caxpy(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      caxpy_recurse(a_, x, y, 0, 0, 0);
    }

    void caxpy_U(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_U with non-square 'a' has not yet been implemented. Use block caxpy instead.\n");
        return;
      }
      caxpy_recurse(a_, x, y, 0, 0, 1);
    }

    void caxpy_L(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_L with non-square 'a' has not yet been implemented. Use block caxpy instead.\n");
        return;
      }
      caxpy_recurse(a_, x, y, 0, 0, -1);
    }


    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy(a, x.Components(), y.Components()); }

    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_U(a, x.Components(), y.Components()); }

    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_L(a, x.Components(), y.Components()); }


    void caxpyz_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z, int i, int j, int pass, int upper) {

      if (y.size() > MAX_MULTI_BLAS_N) // if greater than max single-kernel size, recurse.
      {
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

        caxpyz_recurse(tmpmajor0, x, y0, z0, i, 2*j+0, pass, upper);
        caxpyz_recurse(tmpmajor1, x, y1, z1, i, 2*j+1, pass, upper);

        delete[] tmpmajor;
      }
      else
      {
      	// if at bottom of recursion check where we are
      	if (x.size() <= MAX_MULTI_BLAS_N) {
      	  if (pass==1) {
      	    if (i!=j)
            {
              if (upper == 1 && j < i) { return; } // upper right, don't need to update lower left.
              if (upper == -1 && i < j) { return; } // lower left, don't need to update upper right.
              caxpy(a_, x, z); return;  // off diagonal
            }
      	    return;
      	  } else {
      	    if (i!=j) return; // We're on the first pass, so we only want to update the diagonal.
      	  }
      	}

        // mark true since we will copy the "a" matrix into constant memory
        coeff_array<Complex> a(a_, true), b, c;

        if (x[0]->Precision() == y[0]->Precision())
        {
          switch (x.size()) {
          case 1: multiBlas<1, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: multiBlas<2, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: multiBlas<3, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: multiBlas<4, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: multiBlas<5, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: multiBlas<6, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: multiBlas<7, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: multiBlas<8, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: multiBlas<9, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: multiBlas<10, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: multiBlas<11, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: multiBlas<12, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: multiBlas<13, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: multiBlas<14, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: multiBlas<15, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: multiBlas<16, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpyz_recurse(a0, x0, y, z, 2*i+0, j, pass, upper);
              caxpyz_recurse(a1, x1, y, z, 2*i+1, j, pass, upper); // b/c we don't want to re-zero z.
              break;
          }
        }
        else // precisions don't agree.
        {
          switch (x.size()) {
          case 1: mixedMultiBlas<1, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: mixedMultiBlas<2, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: mixedMultiBlas<3, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: mixedMultiBlas<4, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: mixedMultiBlas<5, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: mixedMultiBlas<6, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: mixedMultiBlas<7, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: mixedMultiBlas<8, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: mixedMultiBlas<9, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: mixedMultiBlas<10, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: mixedMultiBlas<11, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: mixedMultiBlas<12, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: mixedMultiBlas<13, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: mixedMultiBlas<14, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: mixedMultiBlas<15, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: mixedMultiBlas<16, multicaxpyz_, write<0, 0, 0, 1>>(a, b, c, x, y, x, z); break;
#endif // 16
  #endif // 15
  #endif // 14
  #endif // 13
  #endif // 12
  #endif // 11
  #endif // 10
  #endif // 9
  #endif // 8
  #endif // 7
  #endif // 6
  #endif // 5
  #endif // 4
  #endif // 3
  #endif // 2
            default:
              // split the problem in half and recurse
              const Complex *a0 = &a_[0];
              const Complex *a1 = &a_[(x.size()/2)*y.size()];

              std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
              std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());

              caxpyz_recurse(a0, x0, y, z, 2*i+0, j, pass, upper);
              caxpyz_recurse(a1, x1, y, z, 2*i+1, j, pass, upper);
              break;
          }
        }
      } // end if (y.size() > MAX_MULTI_BLAS_N)
    }

    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, 0);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, 0);
    }

    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, 1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, 1);
    }

    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z) {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, 0, 0, 0, -1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, 0, 0, 1, -1);
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

    void axpyBzpcx(const double *a_, std::vector<ColorSpinorField*> &x_, std::vector<ColorSpinorField*> &y_,
		   const double *b_, ColorSpinorField &z_, const double *c_) {

      if (y_.size() <= MAX_MULTI_BLAS_N) {
	// swizzle order since we are writing to x_ and y_, but the
	// multi-blas only allow writing to y and w, and moreover the
	// block width of y and w must match, and x and z must match.
	std::vector<ColorSpinorField*> &y = y_;
	std::vector<ColorSpinorField*> &w = x_;

	// wrap a container around the third solo vector
	std::vector<ColorSpinorField*> x;
	x.push_back(&z_);

	// we will curry the parameter arrays into the functor
	coeff_array<double> a(a_,false), b(b_,false), c(c_,false);

	if (x[0]->Precision() != y[0]->Precision() ) {
          mixedMultiBlas<1, multi_axpyBzpcx_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w);
        } else {
          multiBlas<1, multi_axpyBzpcx_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w);
        }
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

      const int xsize = x_.size();
      if (xsize <= MAX_MULTI_BLAS_N) // only swizzle if we have to.
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

        // put a and b into constant space
        coeff_array<Complex> a(a_,true), b(b_,true), c;

        if (x[0]->Precision() != y[0]->Precision() )
        {
          switch(xsize)
          {
          case 1: mixedMultiBlas<1, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: mixedMultiBlas<2, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: mixedMultiBlas<3, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: mixedMultiBlas<4, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: mixedMultiBlas<5, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: mixedMultiBlas<6, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: mixedMultiBlas<7, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: mixedMultiBlas<8, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: mixedMultiBlas<9, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: mixedMultiBlas<10, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: mixedMultiBlas<11, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: mixedMultiBlas<12, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: mixedMultiBlas<13, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: mixedMultiBlas<14, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: mixedMultiBlas<15, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: mixedMultiBlas<16, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#endif // 16
#endif // 15
#endif // 14
#endif // 13
#endif // 12
#endif // 11
#endif // 10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
            default:
              // we can't hit the default, it ends up in the else below.
              break;
          }
        }
        else
        {
          switch(xsize)
          {
          case 1: multiBlas<1, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 2
          case 2: multiBlas<2, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 3
          case 3: multiBlas<3, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 4
          case 4: multiBlas<4, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 5
          case 5: multiBlas<5, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 6
          case 6: multiBlas<6, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 7
          case 7: multiBlas<7, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 8
          case 8: multiBlas<8, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 9
          case 9: multiBlas<9, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 10
          case 10: multiBlas<10, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 11
          case 11: multiBlas<11, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 12
          case 12: multiBlas<12, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 13
          case 13: multiBlas<13, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 14
          case 14: multiBlas<14, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 15
          case 15: multiBlas<15, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#if MAX_MULTI_BLAS_N >= 16
          case 16: multiBlas<16, multi_caxpyBxpz_, write<0, 1, 0, 1>>(a, b, c, x, y, x, w); break;
#endif // 16
#endif // 15
#endif // 14
#endif // 13
#endif // 12
#endif // 11
#endif // 10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
            default:
              // we can't hit the default, it ends up in the else below.
              break;
            }
        }
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

  } // namespace blas

} // namespace quda
