#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>
#include <uint_to_char.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/multi_reduce_core.cuh>

// work around for Fermi
#if (__COMPUTE_CAPABILITY__ < 300)
#undef MAX_MULTI_BLAS_N
#define MAX_MULTI_BLAS_N 2
#endif

namespace quda {

  namespace blas {

    cudaStream_t* getStream();
    cudaEvent_t* getReduceEvent();
    bool getFastReduce();

    template <int writeX, int writeY, int writeZ, int writeW>
    struct write {
      static constexpr int X = writeX;
      static constexpr int Y = writeY;
      static constexpr int Z = writeZ;
      static constexpr int W = writeW;
    };

    template <typename doubleN, typename ReduceType, typename FloatN, int M, int NXZ, typename Arg>
    void multiReduceLaunch(doubleN result[], Arg &arg, const TuneParam &tp, const cudaStream_t &stream, Tunable &tunable)
    {

      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

      if (getFastReduce() && !commAsyncReduction()) {
        // initialize the reduction values in 32-bit increments to INT_MIN
        constexpr int32_t words = sizeof(ReduceType) / sizeof(int32_t);
        void *h_reduce = getHostReduceBuffer();
        for (unsigned int i = 0; i < tp.grid.z * NXZ * arg.NYW * words; i++) {
          reinterpret_cast<int32_t *>(h_reduce)[i] = std::numeric_limits<int32_t>::min();
        }
      }

#ifdef WARP_MULTI_REDUCE
#error "Untested - should be reverified"
      // multiReduceKernel<ReduceType,FloatN,M,NXZ><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
#ifdef JITIFY
      using namespace jitify::reflection;
      tunable.jitifyError() = program->kernel("quda::blas::multiReduceKernel")
                                  .instantiate((int)tp.block.x, Type<ReduceType>(), Type<FloatN>(), M, NXZ, Type<Arg>())
                                  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                                  .launch(arg);
#else
#if CUDA_VERSION < 9000
      cudaMemcpyToSymbolAsync(arg_buffer, reinterpret_cast<char *>(&arg), sizeof(arg), 0, cudaMemcpyHostToDevice,
                              *getStream());
#endif
      LAUNCH_KERNEL_LOCAL_PARITY(multiReduceKernel, tp, stream, arg, ReduceType, FloatN, M, NXZ);
#endif
#endif

      if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64) || defined(__LP64__))
        if (deviceProp.canMapHostMemory) {
          if (getFastReduce()) {
            constexpr int32_t words = sizeof(ReduceType) / sizeof(int32_t);
            volatile int32_t *check = reinterpret_cast<int32_t *>(getHostReduceBuffer());
            int count = 0;
            for (unsigned int i = 0; i < tp.grid.z * NXZ * arg.NYW * words; i++) {
              // spin-wait until all values have been updated
              while (check[i] == std::numeric_limits<int32_t>::min()) {
                if (count++ % 10000 == 0) { // check error every 10000 iterations
                  // if there is an error in the kernel then we need to exit the spin-wait
                  if (cudaSuccess != cudaPeekAtLastError()) break;
                }
              }
            }
          } else {
            qudaEventRecord(*getReduceEvent(), stream);
            while (cudaSuccess != qudaEventQuery(*getReduceEvent())) {}
          }
        } else
#endif
        {
          qudaMemcpy(getHostReduceBuffer(), getMappedHostReduceBuffer(), tp.grid.z * sizeof(ReduceType) * NXZ * arg.NYW,
              cudaMemcpyDeviceToHost);
        }
      }

      // need to transpose for same order with vector thread reduction
      for (int i = 0; i < NXZ; i++) {
        for (int j = 0; j < arg.NYW; j++) {
          result[i * arg.NYW + j] = set(((ReduceType *)getHostReduceBuffer())[j * NXZ + i]);
          if (tp.grid.z == 2)
            sum(result[i * arg.NYW + j], ((ReduceType *)getHostReduceBuffer())[NXZ * arg.NYW + j * NXZ + i]);
        }
      }
    }

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

    template <int NXZ, typename doubleN, typename ReduceType, typename FloatN, int M, typename SpinorX,
        typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
    class MultiReduceCuda : public Tunable
    {

  private:
      const int NYW;
      int nParity;
      MultiReduceArg<NXZ, ReduceType, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer> arg;
      doubleN *result;

      std::vector<ColorSpinorField *> &x, &y, &z, &w;

      // don't curry into the Spinors to minimize parameter size
      char *Y_h[MAX_MULTI_BLAS_N], *W_h[MAX_MULTI_BLAS_N], *Ynorm_h[MAX_MULTI_BLAS_N], *Wnorm_h[MAX_MULTI_BLAS_N];

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      virtual bool advanceSharedBytes(TuneParam &param) const
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(param) ?
            sharedBytesPerThread() * nthreads :
            sharedBytesPerBlock(param);
        return false;
      }

      // we only launch thread blocks up to size 512 since the autoner
      // tuner favours smaller blocks and this helps with compile time
      unsigned int maxBlockSize(const TuneParam &param) const { return deviceProp.maxThreadsPerBlock / 2; }

  public:
      MultiReduceCuda(doubleN result[], SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Reducer &r,
          std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z,
          std::vector<ColorSpinorField *> &w, int NYW, int length) :
          NYW(NYW),
          nParity(x[0]->SiteSubset()),
          arg(X, Y, Z, W, r, NYW, length / nParity),
          x(x),
          y(y),
          z(z),
          w(w),
          result(result),
          Y_h(),
          W_h(),
          Ynorm_h(),
          Wnorm_h()
      {
        strcpy(aux, "policy_kernel,");
        strcat(aux, x[0]->AuxString());

        // since block dot product and block norm use the same functors, we need to distinguish them
        bool is_norm = false;
        if (NXZ == NYW) {
          is_norm = true;
          for (int i = 0; i < NXZ; i++) {
            if (x[i]->V() != y[i]->V() || x[i]->V() != z[i]->V() || x[i]->V() != w[i]->V()) {
              is_norm = false;
              break;
            }
          }
        }
        if (is_norm) strcat(aux, ",norm");

#ifdef JITIFY
        ::quda::create_jitify_program("kernels/multi_reduce_core.cuh");
#endif
      }

      inline TuneKey tuneKey() const
      {
        char name[TuneKey::name_n];
        strcpy(name, num_to_string<NXZ>::value);
        strcat(name, std::to_string(NYW).c_str());
        strcat(name, typeid(arg.r).name());
        return TuneKey(x[0]->VolString(), name, aux);
      }

      void apply(const cudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        multiReduceLaunch<doubleN, ReduceType, FloatN, M, NXZ>(result, arg, tp, stream, *this);
      }

      // Should these be NYW?
#ifdef WARP_MULTI_REDUCE
      /**
         @brief This is a specialized variant of the reducer that only
         assigns an individial warp within a thread block to a given row
         of the reduction.  It's typically slower than CTA-wide reductions
         and spreading the y dimension across blocks rather then within
         the blocks so left disabled.
      */
      bool advanceBlockDim(TuneParam &param) const
      {
        if (param.block.y < NYW) {
          param.block.y++;
          param.grid.y = (NYW + param.block.y - 1) / param.block.y;
          return true;
        } else {
          param.block.y = 1;
          param.grid.y = NYW;
          return false;
        }
      }
#endif

      bool advanceGridDim(TuneParam &param) const
      {
        bool rtn = Tunable::advanceGridDim(param);
        if (NYW > deviceProp.maxGridSize[1]) errorQuda("N=%d is greater than the maximum support grid size", NYW);
        return rtn;
      }

      void initTuneParam(TuneParam &param) const
      {
        Tunable::initTuneParam(param);
        param.block.y = 1;
        param.grid.y = NYW;
        param.grid.z = nParity;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        Tunable::defaultTuneParam(param);
        param.block.y = 1;
        param.grid.y = NYW;
        param.grid.z = nParity;
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

      long long flops() const
      {
        return NYW * NXZ * arg.r.flops() * vec_length<FloatN>::value * (long long)arg.length * nParity * M;
      }

      long long bytes() const
      {
        // this will be wrong when mixed precision is added
        return NYW * NXZ * arg.r.streams() * x[0]->Bytes();
      }

      int tuningIter() const { return 3; }
    };

    template <typename doubleN, typename ReduceType, typename RegType, typename StoreType, typename yType, int M, int NXZ,
        template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write, typename T>
    void multiReduce(doubleN result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z,
        std::vector<ColorSpinorField *> &w, int length)
    {

      const int NYW = y.size();

      memset(result, 0, NXZ * NYW * sizeof(doubleN));

      const int N_MAX = NXZ > NYW ? NXZ : NYW;
      const int N_MIN = NXZ < NYW ? NXZ : NYW;

      static_assert(MAX_MULTI_BLAS_N * MAX_MULTI_BLAS_N <= QUDA_MAX_MULTI_REDUCE,
          "MAX_MULTI_BLAS_N^2 exceeds maximum number of reductions");
      static_assert(MAX_MULTI_BLAS_N <= 16, "MAX_MULTI_BLAS_N exceeds maximum size 16");
      if (N_MAX > MAX_MULTI_BLAS_N)
        errorQuda("Spinor vector length exceeds max size (%d > %d)", N_MAX, MAX_MULTI_BLAS_N);

      if (NXZ * NYW * sizeof(Complex) > MAX_MATRIX_SIZE)
        errorQuda("A matrix exceeds max size (%lu > %d)", NXZ * NYW * sizeof(Complex), MAX_MATRIX_SIZE);

      for (int i = 0; i < N_MIN; i++) {
        checkSpinor(*x[i], *y[i]);
        checkSpinor(*x[i], *z[i]);
        checkSpinor(*x[i], *w[i]);
        if (!x[i]->isNative()) {
          warningQuda("Reductions on non-native fields are not supported\n");
          return;
        }
      }

      typedef typename scalar<RegType>::type Float;
      typedef typename vector<Float, 2>::type Float2;
      typedef vector<Float, 2> vec2;

#ifdef JITIFY
      // need to get constants pointer from jitify instance
      if (a.use_const || b.use_const || c.use_const)
        errorQuda("Constant memory buffer support not enabled with jitify yet");
#endif

      // FIXME - if NXZ=1 no need to copy entire array
      // FIXME - do we really need strided access here?
      if (a.data && a.use_const) {
        Float2 A[MAX_MATRIX_SIZE / sizeof(Float2)];
        // since the kernel doesn't know the width of them matrix at compile
        // time we stride it and copy the padded matrix to GPU
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) A[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(a.data[NYW * i + j]));

        cudaMemcpyToSymbolAsync(Amatrix_d, A, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        Amatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(a.data));
      }

      if (b.data && b.use_const) {
        Float2 B[MAX_MATRIX_SIZE / sizeof(Float2)];
        // since the kernel doesn't know the width of them matrix at compile
        // time we stride it and copy the padded matrix to GPU
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) B[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(b.data[NYW * i + j]));

        cudaMemcpyToSymbolAsync(Bmatrix_d, B, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        Bmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(b.data));
      }

      if (c.data && c.use_const) {
        Float2 C[MAX_MATRIX_SIZE / sizeof(Float2)];
        // since the kernel doesn't know the width of them matrix at compile
        // time we stride it and copy the padded matrix to GPU
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) C[MAX_MULTI_BLAS_N * i + j] = make_Float2<Float2>(Complex(c.data[NYW * i + j]));

        cudaMemcpyToSymbolAsync(Cmatrix_d, C, MAX_MATRIX_SIZE, 0, cudaMemcpyHostToDevice, *getStream());
        Cmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(c.data));
      }

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

      Reducer<NXZ, ReduceType, Float2, RegType> r(a, b, c, NYW);

      MultiReduceCuda<NXZ, doubleN, ReduceType, RegType, M, SpinorTexture<RegType, StoreType, M>,
                      Spinor<RegType, yType, M, write::Y>, SpinorTexture<RegType, StoreType, M>,
                      Spinor<RegType, StoreType, M, write::W>, decltype(r)>
        reduce(result, X, Y, Z, W, r, x, y, z, w, NYW, length);
      reduce.apply(*blas::getStream());

      blas::bytes += reduce.bytes();
      blas::flops += reduce.flops();

      checkCudaError();
    }

    /**
       Driver for multi-reduce with up to four vectors
    */
    template <int NXZ, typename doubleN, typename ReduceType,
        template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write,
        bool siteUnroll, typename T>
    void multiReduce(doubleN result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
        CompositeColorSpinorField &w)
    {
      const int NYW = y.size();

      int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

      QudaPrecision precision = checkPrecision(*x[0], *y[0], *z[0], *w[0]);

      if (precision == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
        if (x[0]->Nspin() == 4 || x[0]->Nspin() == 2) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_MULTIGRID)
          const int M = siteUnroll ? 12 : 1; // determines how much work per thread to do
          if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
          multiReduce<doubleN, ReduceType, double2, double2, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
          const int M = siteUnroll ? 3 : 1; // determines how much work per thread to do
          multiReduce<doubleN, ReduceType, double2, double2, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif

      } else if (precision == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = siteUnroll ? 6 : 1; // determines how much work per thread to do
          multiReduce<doubleN, ReduceType, float4, float4, float4, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (4 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC) || defined(GPU_MULTIGRID)
          const int M = siteUnroll ? 3 : 1;
          if (x[0]->Nspin() == 2 && siteUnroll) errorQuda("siteUnroll not supported for nSpin==2");
          multiReduce<doubleN, ReduceType, float2, float2, float2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif

      } else if (precision == QUDA_HALF_PRECISION) { // half precision

#if QUDA_PRECISION & 2
        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = 6;
          multiReduce<doubleN, ReduceType, float4, short4, short4, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
          const int M = 3;
          multiReduce<doubleN, ReduceType, float2, short2, short2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif

      } else if (precision == QUDA_QUARTER_PRECISION) { // quarter precision

#if QUDA_PRECISION & 1
        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = 6;
          multiReduce<doubleN, ReduceType, float4, char4, char4, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
          const int M = 3;
          multiReduce<doubleN, ReduceType, float2, char2, char2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif
      } else {
        errorQuda("Precision %d not supported\n", precision);
      }
    }

    /**
       Driver for multi-reduce with up to five vectors
    */
    template <int NXZ, typename doubleN, typename ReduceType,
        template <int MXZ, typename ReducerType, typename Float, typename FloatN> class Reducer, typename write,
        bool siteUnroll, typename T>
    void mixedMultiReduce(doubleN result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
        CompositeColorSpinorField &w)
    {
      const int NYW = y.size();

      checkPrecision(*x[0], *z[0]);
      checkPrecision(*y[0], *w[0]);

      assert(siteUnroll == true);
      int reduce_length = siteUnroll ? x[0]->RealLength() : x[0]->Length();

      if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_SINGLE_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = 12; // determines how much work per thread to do
          multiReduce<doubleN, ReduceType, double2, float4, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1) {
#ifdef GPU_STAGGERED_DIRAC
          const int M = 3; // determines how much work per thread to do
          multiReduce<doubleN, ReduceType, double2, float2, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = 6; // determines how much work per thread to do
          multiReduce<doubleN, ReduceType, double2, short4, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (4 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(GPU_STAGGERED_DIRAC)
          const int M = 3;
          multiReduce<doubleN, ReduceType, double2, short2, double2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, reduce_length / (2 * M));
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)
          const int M = 6;
          multiReduce<doubleN, ReduceType, float4, short4, float4, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#ifdef GPU_STAGGERED_DIRAC
          const int M = 3;
          multiReduce<doubleN, ReduceType, float2, short2, float2, M, NXZ, Reducer, write>(
              result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d fields", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else {
        errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
      }
    }

    template <int NXZ, typename doubleN, typename ReduceType,
        template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal, typename writeDiagonal,
        template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal,
        typename writeOffDiagonal, bool siteUnroll, typename T>
    void multiReduce(doubleN result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
        CompositeColorSpinorField &w, int i, int j)
    {

      if (x[0]->Precision() == y[0]->Precision()) {
        if (i == j) { // we are on the diagonal so invoke the diagonal reducer
          multiReduce<NXZ, doubleN, ReduceType, ReducerDiagonal, writeDiagonal, siteUnroll, T>(
              result, a, b, c, x, y, z, w);
        } else { // we are on the diagonal so invoke the off-diagonal reducer
          multiReduce<NXZ, doubleN, ReduceType, ReducerOffDiagonal, writeOffDiagonal, siteUnroll, T>(
              result, a, b, c, x, y, z, w);
        }
      } else {
        if (i == j) { // we are on the diagonal so invoke the diagonal reducer
          mixedMultiReduce<NXZ, doubleN, ReduceType, ReducerDiagonal, writeDiagonal, true, T>(
              result, a, b, c, x, y, z, w);
        } else { // we are on the diagonal so invoke the off-diagonal reducer
          mixedMultiReduce<NXZ, doubleN, ReduceType, ReducerOffDiagonal, writeOffDiagonal, true, T>(
              result, a, b, c, x, y, z, w);
        }
      }
    }

    void reDotProduct(double* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
#ifndef SSTEP
    errorQuda("S-step code not built\n");
#else
    switch(x.size()){
      case 1:
        multiReduce<1, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 2:
        multiReduce<2, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 3:
        multiReduce<3, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 4:
        multiReduce<4, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 5:
        multiReduce<5, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 6:
        multiReduce<6, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 7:
        multiReduce<7, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 8:
        multiReduce<8, double, QudaSumFloat, Dot, 0, 0, 0, 0, false>(
            result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      /*case 9:
        multiReduce<9,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 10:
        multiReduce<10,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 11:
        multiReduce<11,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 12:
        multiReduce<12,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 13:
        multiReduce<13,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 14:
        multiReduce<14,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 15:
        multiReduce<15,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;
      case 16:
        multiReduce<16,double,QudaSumFloat,Dot,0,0,0,0,false>
        (result, make_double2(0.0, 0.0), make_double2(0.0, 0.0), x, y, x, y);
        break;*/
      default:
        errorQuda("Unsupported vector size");
        break;
    }
#endif // SSTEP
    // do a single multi-node reduction only once we have computed all local dot products
    const int Nreduce = x.size()*y.size();
    reduceDoubleArray((double*)result, Nreduce);
  }


    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal, typename writeDiagonal,
	      template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal, typename writeOffDiagonal>
    void multiReduce_recurse(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			     std::vector<ColorSpinorField*>&z, std::vector<ColorSpinorField*>&w, int i_idx, int j_idx, bool hermitian, unsigned int tile_size) {

      if (y.size() > tile_size) // if greater than max single-kernel size, split and recurse
      {
        // Do the recurse first.
        Complex* result0 = &result[0];
        Complex* result1 = &result[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());
        std::vector<ColorSpinorField*> w0(w.begin(), w.begin() + w.size()/2);
        std::vector<ColorSpinorField*> w1(w.begin() + w.size()/2, w.end());
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x, y0, z, w0, i_idx, 2*j_idx+0, hermitian, tile_size);
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x, y1, z, w1, i_idx, 2*j_idx+1, hermitian, tile_size);
      }
      else
      {
        double2* cdot = new double2[x.size()*y.size()];

	// if at bottom of recursion, return if on lower left
	if (x.size() <= tile_size && hermitian) {
	  if (j_idx < i_idx) { return; }
	}

        coeff_array<Complex> a, b, c;

        if (x.size() <= tile_size) {
        switch(x.size()){ // COMMENT HERE FOR COMPILE TIME
        case 1:
          multiReduce<1, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 2
        case 2:
          multiReduce<2, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 3
        case 3:
          multiReduce<3, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 4
        case 4:
          multiReduce<4, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 5
        case 5:
          multiReduce<5, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 6
        case 6:
          multiReduce<6, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 7
        case 7:
          multiReduce<7, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 8
        case 8:
          multiReduce<8, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 9
	case 9:
          multiReduce<9, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 10
        case 10:
          multiReduce<10, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 11
        case 11:
          multiReduce<11, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 12
        case 12:
          multiReduce<12, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 13
        case 13:
          multiReduce<13, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 14
        case 14:
          multiReduce<14, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 15
        case 15:
          multiReduce<15, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 16
        case 16:
          multiReduce<16, double2, QudaSumFloat2, ReducerDiagonal, writeDiagonal, ReducerOffDiagonal, writeOffDiagonal, false>(
              cdot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#endif //16
#endif //15
#endif //14
#endif //13
#endif //12
#endif //11
#endif //10
#endif // 9
#endif // 8
#endif // 7
#endif // 6
#endif // 5
#endif // 4
#endif // 3
#endif // 2
	}
	} else {
          // split the problem and recurse. Splitting in x requires
          // memory reshuffling (unless y = 1).
          // Use a few temporary variables.

          Complex* tmpmajor = new Complex[x.size()*y.size()];
          Complex* result0 = &tmpmajor[0];
          Complex* result1 = &tmpmajor[(x.size()/2)*y.size()];
          std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
          std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());
          std::vector<ColorSpinorField*> z0(z.begin(), z.begin() + z.size()/2);
          std::vector<ColorSpinorField*> z1(z.begin() + z.size()/2, z.end());

          multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result0, x0, y, z0, w, 2*i_idx+0, j_idx, hermitian, tile_size);
          multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>(result1, x1, y, z1, w, 2*i_idx+1, j_idx, hermitian, tile_size);

          const unsigned int xlen0 = x.size()/2;
          const unsigned int xlen1 = x.size() - xlen0;
          const unsigned int ylen = y.size();

          // Copy back into result.
          int count = 0, count0 = 0, count1 = 0;
          for (unsigned int i = 0; i < ylen; i++)
          {
            for (unsigned int j = 0; j < xlen0; j++)
              result[count++] = result0[count0++];
            for (unsigned int j = 0; j < xlen1; j++)
              result[count++] = result1[count1++];
          }

          delete[] tmpmajor;
        }

	// we are at the leaf of the binary tree (e.g., we ran the kernel): perform the row-to-column-major transpose here.
        if (x.size() <= tile_size)
        {
          const unsigned int xlen = x.size();
          const unsigned int ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++)
              result[i*xlen+j] = Complex(cdot[j*ylen + i].x, cdot[j*ylen+i].y);
        }
        delete[] cdot;
      }
    }


    template <template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerDiagonal,
	      typename writeDiagonal,
	      template <int MXZ, typename ReducerType, typename Float, typename FloatN> class ReducerOffDiagonal,
	      typename writeOffDiagonal>
    class TileSizeTune : public Tunable {
      typedef std::vector<ColorSpinorField*> vec;
      Complex *result;
      vec &x, &y, &z, &w;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      unsigned int max_tile_size;

    public:
      TileSizeTune(Complex *result, vec &x, vec &y, vec &z, vec &w, bool hermitian, bool Anorm = false)
	: result(result), x(x), y(y), z(z), w(w), hermitian(hermitian), Anorm(Anorm), max_tile_size(1)
      {
      	strcpy(aux, "policy,");
      	strcat(aux, x[0]->AuxString());
      	strcat(aux, ",");
      	strcat(aux, y[0]->AuxString());
        if (hermitian) strcat(aux, ",hermitian");
        if (Anorm) strcat(aux, ",Anorm");
	strcat(aux,",n=");
	char size[8];
	u64toa(size, x.size());
	strcat(aux,size);
	strcat(aux,",m=");
	u64toa(size, y.size());
	strcat(aux,size);
        u64toa(size, MAX_MULTI_BLAS_N);
        strcat(aux, ",multi-blas-n=");
        strcat(aux, size);

        // before we do policy tuning we must ensure the kernel
        // constituents have been tuned since we can't do nested tuning
        // FIXME this will break if the kernels are destructive - which they aren't here
        if (getTuning() && getTuneCache().find(tuneKey()) == getTuneCache().end()) {
          disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

	  if ( x.size()==1 || y.size()==1 ) { // 1-d reduction

	    max_tile_size = std::min(MAX_MULTI_BLAS_N, (int)std::max(x.size(), y.size()));

	    // Make sure constituents are tuned.
	    for ( unsigned int tile_size=1; tile_size <= max_tile_size; tile_size++) {
	      multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
		(result, x, y, z, w, 0, 0, hermitian, tile_size);
	    }

	  } else { // 2-d reduction

	    // max_tile_size should be set to the largest power of 2 less than
	    // MAX_MULTI_BLAS_N, since we have a requirement that the
	    // tile size is a power of 2.
	    unsigned int max_count = 0;
	    unsigned int tile_size_tmp = MAX_MULTI_BLAS_N;
	    while (tile_size_tmp != 1) { tile_size_tmp = tile_size_tmp >> 1; max_count++; }
	    tile_size_tmp = 1;
	    for (unsigned int i = 0; i < max_count; i++) { tile_size_tmp = tile_size_tmp << 1; }
	    max_tile_size = tile_size_tmp;

	    // Make sure constituents are tuned.
	    for ( unsigned int tile_size=1; tile_size <= max_tile_size && tile_size <= x.size() &&
		    (tile_size <= y.size() || y.size()==1) ; tile_size*=2) {
	      multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
		(result, x, y, z, w, 0, 0, hermitian, tile_size);
	    }

            // also test case using a single kernel if both dimensions
            // are less than MAX_MULTI_BLAS_N
            if (x.size() <= MAX_MULTI_BLAS_N && y.size() <= MAX_MULTI_BLAS_N) {
	      multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
		(result, x, y, z, w, 0, 0, hermitian, MAX_MULTI_BLAS_N);
            }

          }

      	  enableProfileCount();
      	  setPolicyTuning(true);
        }
      }

      virtual ~TileSizeTune() { setPolicyTuning(false); }

      void apply(const cudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        // tp.aux.x is where the tile size is stored. "tp" is the tuning struct.
        // it contains blocksize, grid size, etc. Since we're only tuning
        // a policy, we don't care about those sizes. That's why we only
        // tune "aux.x", which is the tile size.
        multiReduce_recurse<ReducerDiagonal,writeDiagonal,ReducerOffDiagonal,writeOffDiagonal>
          (result, x, y, z, w, 0, 0, hermitian, tp.aux.x);
      }

      // aux.x is the tile size
      bool advanceAux(TuneParam &param) const
      {
	if ( x.size()==1 || y.size()==1 ) { // 1-d reduction

	  param.aux.x++;
	  if ( (unsigned int)param.aux.x <= max_tile_size ) {
	    return true;
	  } else {
	    param.aux.x = 1;
	    return false;
	  }

	} else { // 2-d reduction

	  if ( (unsigned int)(2*param.aux.x) <= max_tile_size &&
               (unsigned int)(2*param.aux.x) <= x.size() &&
	       (unsigned int)(2*param.aux.x) <= y.size() ) {
            param.aux.x *= 2; // only tune powers of two
	    return true;
	  } else if (x.size() <= MAX_MULTI_BLAS_N && y.size() <= MAX_MULTI_BLAS_N && param.aux.x < MAX_MULTI_BLAS_N) {
            // we've run out of power of two tiles to try, but before
            // we finish, try a single kernel if it fits
            param.aux.x = MAX_MULTI_BLAS_N;
            return true;
          } else {
	    param.aux.x = 1; // reset to the beginning (which we'd need for multi-dimensional tuning)
	    return false;
	  }

	}
      }

      bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const  {
      	Tunable::initTuneParam(param);
      	param.aux.x = 1; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
      }

      void defaultTuneParam(TuneParam &param) const  {
      	Tunable::defaultTuneParam(param); // default is max tile size
        // max_tile_size is MAX_MULTI_BLAS_N rounded down to the nearest power of 2.
      	param.aux.x = max_tile_size; param.aux.y = 0; param.aux.z = 0; param.aux.w = 0;
      }

      TuneKey tuneKey() const {
        return TuneKey(x[0]->VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return 0; } // FIXME
      long long bytes() const { return 0; } // FIXME

      void preTune() { } // FIXME - use write to determine what needs to be saved
      void postTune() { } // FIXME - use write to determine what needs to be saved
    };

    void cDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      // cDotProduct_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, false);
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce);

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++)
          result[j*ylen+i] = result_tmp[i*xlen + j];

      delete[] result_tmp;
    }

    void hDotProduct(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true, false); // last false is b/c L2 norm
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce); // FIXME - could optimize this for Hermiticity as well

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = j; i < ylen; i++) {
          result[j*ylen+i] = result_tmp[i*xlen + j];
          result[i*ylen+j] = conj(result_tmp[i*xlen + j]);
	}

      delete[] result_tmp;
    }

    // for (p, Ap) norms in CG which are Hermitian.
    void hDotProduct_Anorm(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y){
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block A-norm dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      TileSizeTune<Cdot,write<0,0,0,0>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true, true); // last true is b/c A norm
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce); // FIXME - could optimize this for Hermiticity as well

      // Switch from col-major to row-major
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = j; i < ylen; i++) {
          result[j*ylen+i] = result_tmp[i*xlen + j];
          result[i*ylen+j] = conj(result_tmp[i*xlen + j]);
  }

      delete[] result_tmp;
    }

    // takes the outer product of inner products between and y and copies y into z
    void cDotProductCopy(Complex* result, std::vector<ColorSpinorField*>& x, std::vector<ColorSpinorField*>& y,
			 std::vector<ColorSpinorField*>&z){

#if 0
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (y.size() != z.size()) errorQuda("Cannot copy input y of size %lu into z of size %lu\n", y.size(), z.size());

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      // When recursing, only the diagonal tiles will do the copy, the rest just do the outer product
      TileSizeTune<CdotCopy,write<0,0,0,1>,Cdot,write<0,0,0,0> > tile(result_tmp, x, y, x, y, true);
      tile.apply(0);

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce);

      // Switch from col-major to row-major.
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++)
          result[j*ylen+i] = result_tmp[i*xlen + j];

      delete[] result_tmp;
#else
      errorQuda("cDotProductCopy not enabled");
#endif
    }

   } // namespace blas

} // namespace quda
