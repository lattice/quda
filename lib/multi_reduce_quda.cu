#include <blas_quda.h>
#include <tune_quda.h>
#include <float_vector.h>
#include <color_spinor_field_order.h>
#include <uint_to_char.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/multi_reduce_core.cuh>

namespace quda {

  namespace blas {

    qudaStream_t* getStream();
    cudaEvent_t* getReduceEvent();
    bool getFastReduce();
    void initFastReduce(int words);
    void completeFastReduce(int32_t words);

    template <typename FloatN, int M, int NXZ, typename Arg, typename T>
    void multiReduceLaunch(T result[], Arg &arg, const TuneParam &tp, const qudaStream_t &stream, Tunable &tunable)
    {
      using reduce_t = typename Arg::Reducer::reduce_t;
      if (tp.grid.x > (unsigned int)deviceProp.maxGridSize[0])
        errorQuda("Grid size %d greater than maximum %d\n", tp.grid.x, deviceProp.maxGridSize[0]);

      const int32_t words = tp.grid.z * NXZ * arg.NYW * sizeof(reduce_t) / sizeof(int32_t);
      if (getFastReduce() && !commAsyncReduction()) initFastReduce(words);

#ifdef WARP_MULTI_REDUCE
#error "Untested - should be reverified"
      // multiReduceKernel<FloatN,M,NXZ><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#else
#ifdef JITIFY
      using namespace jitify::reflection;
      tunable.jitifyError() = program->kernel("quda::blas::multiReduceKernel")
                                  .instantiate((int)tp.block.x, Type<FloatN>(), M, NXZ, Type<Arg>())
                                  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                                  .launch(arg);
#else
      LAUNCH_KERNEL_REDUCE(multiReduceKernel, tunable, tp, stream, arg, FloatN, M, NXZ, Arg);
#endif
#endif

      if (!commAsyncReduction()) {
#if (defined(_MSC_VER) && defined(_WIN64) || defined(__LP64__))
        if (deviceProp.canMapHostMemory) {
          if (getFastReduce()) {
            completeFastReduce(words);
          } else {
            qudaEventRecord(*getReduceEvent(), stream);
            while (cudaSuccess != qudaEventQuery(*getReduceEvent())) {}
          }
        } else
#endif
        {
          qudaMemcpy(getHostReduceBuffer(), getMappedHostReduceBuffer(), tp.grid.z * sizeof(reduce_t) * NXZ * arg.NYW,
                     cudaMemcpyDeviceToHost);
        }
      }

      // need to transpose for same order with vector thread reduction
      auto buffer = (reduce_t *)getHostReduceBuffer();
      for (int i = 0; i < NXZ; i++) {
        for (int j = 0; j < arg.NYW; j++) {
          result[i * arg.NYW + j] = static_cast<T>(buffer[j * NXZ + i]);
          if (tp.grid.z == 2) result[i * arg.NYW + j] += static_cast<T>(buffer[NXZ * arg.NYW + j * NXZ + i]);
        }
      }
    }

    template <int NXZ, typename T, typename FloatN, int M, typename SpinorX,
              typename SpinorY, typename SpinorZ, typename SpinorW, typename Reducer>
    class MultiReduce : public Tunable
    {
      typedef typename scalar<FloatN>::type Float;
      static constexpr int NYW_max = max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer>();
      const int NYW;
      int nParity;
      MultiReduceArg<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Reducer> arg;
      T *result;

      std::vector<ColorSpinorField *> &x, &y, &z, &w;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      virtual bool advanceSharedBytes(TuneParam &param) const
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(param) ?
          sharedBytesPerThread() * nthreads : sharedBytesPerBlock(param);
        return false;
      }

      // we only launch thread blocks up to size 512 since the autotuner
      // tuner favours smaller blocks and this helps with compile time
      unsigned int maxBlockSize(const TuneParam &param) const { return 128; } // deviceProp.maxThreadsPerBlock / 2; }

    public:
      MultiReduce(T result[], SpinorX X[], SpinorY Y[], SpinorZ Z[], SpinorW W[], Reducer &r,
                  std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w, int NYW, int length) :
        NYW(NYW),
        nParity(x[0]->SiteSubset()),
        arg(X, Y, Z, W, r, NYW, length / nParity),
        x(x),
        y(y),
        z(z),
        w(w),
        result(result)
      {
        if (sizeof(arg) > MAX_MATRIX_SIZE)
          errorQuda("Kernel argument size %lu greater than maximum %d", sizeof(arg), MAX_MATRIX_SIZE);

        strcpy(aux, "policy_kernel,");
        strcat(aux, x[0]->AuxString());
        if (getFastReduce()) strcat(aux, ",fast_reduce");

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

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        multiReduceLaunch<FloatN, M, NXZ>(result, arg, tp, stream, *this);
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
          if (arg.r.write.Y) y[i]->backup();
          if (arg.r.write.W) w[i]->backup();
        }
      }

      void postTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (arg.r.write.Y) y[i]->restore();
          if (arg.r.write.W) w[i]->restore();
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

    template <typename RegType, typename StoreType, typename yType, int M, int NXZ,
              template <int MXZ, typename ReducerType, typename real> class Reducer, typename T>
    void multiReduce(T result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                     std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y, std::vector<ColorSpinorField *> &z,
                     std::vector<ColorSpinorField *> &w, int length)
    {
      const int NYW = y.size();
      const int N_MIN = NXZ < NYW ? NXZ : NYW;
      for (int i = 0; i < N_MIN; i++) {
        checkSpinor(*x[i], *y[i]);
        checkSpinor(*x[i], *z[i]);
        checkSpinor(*x[i], *w[i]);
        if (!x[i]->isNative()) {
          warningQuda("Reductions on non-native fields are not supported\n");
          return;
        }
      }

      using real = typename scalar<RegType>::type;
      Reducer<NXZ, device_reduce_t, real> r(NYW);

      constexpr int NYW_max = max_YW_size<NXZ, StoreType, yType, decltype(r)>();

      constexpr int scalar_width = decltype(r)::coeff_mul ? sizeof(typename decltype(r)::coeff_t) / sizeof(real) : 0;
      const int NYW_max_check
        = max_YW_size(x.size(), x[0]->Precision(), y[0]->Precision(), r.use_z, r.use_w, scalar_width, true);

      if (!is_valid_NXZ(NXZ, true))
        errorQuda("NXZ=%d is not a valid size ( MAX_MULTI_BLAS_N %d)", NXZ, MAX_MULTI_BLAS_N);
      if (NYW_max != NYW_max_check) errorQuda("Runtime %d and compile time %d limits disagree", NYW_max_check, NYW_max);
      if (NXZ * NYW > QUDA_MAX_MULTI_REDUCE)
        errorQuda("NXZ * NYW = %d exceeds maximum number of reductions %d", NXZ * NYW, QUDA_MAX_MULTI_REDUCE);
      if (NYW > NYW_max) errorQuda("NYW exceeds max size (%d > %d)", NYW, NYW_max);
      if (NXZ * NYW * scalar_width > MAX_MATRIX_SIZE)
        errorQuda("Coefficient matrix exceeds max size (%d > %d)", NXZ * NYW * scalar_width, MAX_MATRIX_SIZE);

      using coeff_t = typename decltype(r)::coeff_t;
      size_t len = MAX_MATRIX_SIZE / sizeof(coeff_t);
#ifdef JITIFY
      // need to get constants pointer from jitify instance
      if (a.data || b.data || c.data) errorQuda("Constant memory buffer support not enabled with jitify yet");
#endif

      if (a.data) {
        coeff_t A[len];
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) A[NYW * i + j] = coeff_t(a.data[NYW * i + j]);

        cudaMemcpyToSymbolAsync(Amatrix_d, A, NXZ * NYW * sizeof(decltype(A[0])), 0, cudaMemcpyHostToDevice,
                                *getStream());
        Amatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(a.data));
      }

      if (b.data) {
        coeff_t B[len];
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) B[NYW * i + j] = coeff_t(b.data[NYW * i + j]);

        cudaMemcpyToSymbolAsync(Bmatrix_d, B, NXZ * NYW * sizeof(decltype(B[0])), 0, cudaMemcpyHostToDevice,
                                *getStream());
        Bmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(b.data));
      }

      if (c.data) {
        coeff_t C[len];
        for (int i = 0; i < NXZ; i++)
          for (int j = 0; j < NYW; j++) C[NYW * i + j] = coeff_t(c.data[NYW * i + j]);

        cudaMemcpyToSymbolAsync(Cmatrix_d, C, NXZ * NYW * sizeof(decltype(C[0])), 0, cudaMemcpyHostToDevice,
                                *getStream());
        Cmatrix_h = reinterpret_cast<signed char *>(const_cast<T *>(c.data));
      }

      Spinor<RegType, StoreType, M> X[NXZ];
      Spinor<RegType, yType, M> Y[NYW_max];
      Spinor<RegType, StoreType, M> Z[NXZ];
      Spinor<RegType, StoreType, M> W[NYW_max];

      for (int i = 0; i < NXZ; i++) {
        X[i].set(*dynamic_cast<cudaColorSpinorField *>(x[i]));
        Z[i].set(*dynamic_cast<cudaColorSpinorField *>(z[i]));
      }
      for (int i = 0; i < NYW; i++) {
        Y[i].set(*dynamic_cast<cudaColorSpinorField *>(y[i]));
        W[i].set(*dynamic_cast<cudaColorSpinorField *>(w[i]));
      }

      MultiReduce<NXZ, T, RegType, M, typename std::remove_reference<decltype(X[0])>::type,
                  typename std::remove_reference<decltype(Y[0])>::type, typename std::remove_reference<decltype(Z[0])>::type,
                  typename std::remove_reference<decltype(W[0])>::type, decltype(r)>
        reduce(result, X, Y, Z, W, r, x, y, z, w, NYW, length);
      reduce.apply(*blas::getStream());

      blas::bytes += reduce.bytes();
      blas::flops += reduce.flops();

      checkCudaError();
    }

    /**
       Driver for multi-reduce with up to four vectors
    */
    template <int NXZ, template <int MXZ, typename ReducerType, typename real> class Reducer, typename T>
    void uniMultiReduce(T result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                        CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                        CompositeColorSpinorField &w)
    {
      QudaPrecision precision = checkPrecision(*x[0], *y[0], *z[0], *w[0]);

      if (precision == QUDA_DOUBLE_PRECISION) {

#if QUDA_PRECISION & 8
#if defined(NSPIN4) || defined(NSPIN2) || defined(NSPIN1)
        const int M = 1; // determines how much work per thread to do
        multiReduce<double2, double2, double2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Length() / (2 * M));
#else
        errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif

      } else if (precision == QUDA_SINGLE_PRECISION) {

#if QUDA_PRECISION & 4
#if defined(NSPIN4) || defined(NSPIN2) || defined(NSPIN1)
        const int M = 1; // determines how much work per thread to do
        multiReduce<float4, float4, float4, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Length() / (4 * M));
#else
        errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
#else
        errorQuda("QUDA_PRECISION=%d does not enable precision %d", QUDA_PRECISION, precision);
#endif

      } else if (precision == QUDA_HALF_PRECISION) { // half precision

#if QUDA_PRECISION & 2
        if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
          const int M = 6;
          multiReduce<float4, short4, short4, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#if defined(NSPIN1)
          const int M = 3;
          multiReduce<float2, short2, short2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
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
#if defined(NSPIN4)
          const int M = 6;
          multiReduce<float4, char4, char4, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#if defined(NSPIN1)
          const int M = 3;
          multiReduce<float2, char2, char2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
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
    template <int NXZ, template <int MXZ, typename ReducerType, typename real> class Reducer, typename T>
    void mixedMultiReduce(T result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                          CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                          CompositeColorSpinorField &w)
    {
      checkPrecision(*x[0], *z[0]);
      checkPrecision(*y[0], *w[0]);

      if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_SINGLE_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
          const int M = 12; // determines how much work per thread to do
          multiReduce<double2, float4, double2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else if (x[0]->Nspin() == 1) {
#if defined(NSPIN1)
          const int M = 3; // determines how much work per thread to do
          multiReduce<double2, float2, double2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d field", x[0]->Nspin());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else if (y[0]->Precision() == QUDA_DOUBLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
          const int M = 6; // determines how much work per thread to do
          multiReduce<double2, short4, double2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else if (x[0]->Nspin() == 1 || x[0]->Nspin() == 2) { // staggered
#if defined(NSPIN1)
          const int M = 3;
          multiReduce<double2, short2, double2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else if (y[0]->Precision() == QUDA_SINGLE_PRECISION && x[0]->Precision() == QUDA_HALF_PRECISION) {

        if (x[0]->Nspin() == 4) { // wilson
#if defined(NSPIN4)
          const int M = 6;
          multiReduce<float4, short4, float4, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else if (x[0]->Nspin() == 1) { // staggered
#if defined(NSPIN1)
          const int M = 3;
          multiReduce<float2, short2, float2, M, NXZ, Reducer>(result, a, b, c, x, y, z, w, x[0]->Volume());
#else
          errorQuda("blas has not been built for Nspin=%d order=%d fields", x[0]->Nspin(), x[0]->FieldOrder());
#endif
        } else {
          errorQuda("nSpin=%d is not supported\n", x[0]->Nspin());
        }

      } else {
        errorQuda("Precision combination x=%d y=%d not supported\n", x[0]->Precision(), y[0]->Precision());
      }
    }

    template <int NXZ,
              template <int MXZ, typename ReducerType, typename real> class ReducerDiagonal,
              template <int MXZ, typename ReducerType, typename real> class ReducerOffDiagonal, typename T>
    void multiReduce(T result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                     CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                     CompositeColorSpinorField &w, int i, int j)
    {
      if (x[0]->Precision() == y[0]->Precision()) {
        if (i == j) { // we are on the diagonal so invoke the diagonal reducer
          uniMultiReduce<NXZ, ReducerDiagonal, T>(result, a, b, c, x, y, z, w);
        } else { // we are on the diagonal so invoke the off-diagonal reducer
          uniMultiReduce<NXZ, ReducerOffDiagonal, T>(result, a, b, c, x, y, z, w);
        }
      } else {
        if (i == j) { // we are on the diagonal so invoke the diagonal reducer
          mixedMultiReduce<NXZ, ReducerDiagonal, T>(result, a, b, c, x, y, z, w);
        } else { // we are on the diagonal so invoke the off-diagonal reducer
          mixedMultiReduce<NXZ, ReducerOffDiagonal, T>(result, a, b, c, x, y, z, w);
        }
      }
    }

    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <int MXZ, typename ReducerType, typename real> class ReducerDiagonal,
              template <int MXZ, typename ReducerType, typename real> class ReducerOffDiagonal, typename T>
    void multiReduce_recurse(T *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                             std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w, int i_idx,
                             int j_idx, bool hermitian, uint2 tile_size)
    {
      if (y.size() > tile_size.y) { // if greater than max single-kernel size, split and recurse
        // Do the recurse first.
        T* result0 = &result[0];
        T* result1 = &result[x.size()*(y.size()/2)];
        std::vector<ColorSpinorField*> y0(y.begin(), y.begin() + y.size()/2);
        std::vector<ColorSpinorField*> y1(y.begin() + y.size()/2, y.end());
        std::vector<ColorSpinorField*> w0(w.begin(), w.begin() + w.size()/2);
        std::vector<ColorSpinorField*> w1(w.begin() + w.size()/2, w.end());
        multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result0, x, y0, z, w0, i_idx, 2*j_idx+0, hermitian, tile_size);
        multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result1, x, y1, z, w1, i_idx, 2*j_idx+1, hermitian, tile_size);
      } else {
        T* tmp_dot = new T[x.size()*y.size()];

	// if at bottom of recursion, return if on lower left
        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true) && hermitian) {
          if (j_idx < i_idx) { return; }
        }

        coeff_array<T> a, b, c;

        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true)) {
          switch (x.size()) {
          case 1:
            multiReduce<1, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
            break;
          case 2:
            multiReduce<2, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
            break;
          case 4:
            multiReduce<4, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
            break;
          case 8:
            multiReduce<8, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
            break;
          case 16:
            multiReduce<16, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
            break;
#if MAX_MULTI_BLAS_N >= 3
        case 3:
          multiReduce<3, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 5
        case 5:
          multiReduce<5, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 6
        case 6:
          multiReduce<6, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 7
        case 7:
          multiReduce<7, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 9
	case 9:
          multiReduce<9, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 10
        case 10:
          multiReduce<10, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 11
        case 11:
          multiReduce<11, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 12
        case 12:
          multiReduce<12, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 13
        case 13:
          multiReduce<13, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 14
        case 14:
          multiReduce<14, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 15
        case 15:
          multiReduce<15, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 17
        case 17:
          multiReduce<17, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 18
        case 18:
          multiReduce<18, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 19
        case 19:
          multiReduce<19, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 20
        case 20:
          multiReduce<20, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 21
        case 21:
          multiReduce<21, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 22
        case 22:
          multiReduce<22, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 23
        case 23:
          multiReduce<23, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 24
        case 24:
          multiReduce<24, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 25
        case 25:
          multiReduce<25, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 26
        case 26:
          multiReduce<26, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 27
        case 27:
          multiReduce<27, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 28
        case 28:
          multiReduce<28, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 29
        case 29:
          multiReduce<29, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 30
        case 30:
          multiReduce<30, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 31
        case 31:
          multiReduce<31, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#if MAX_MULTI_BLAS_N >= 32
        case 32:
          multiReduce<32, ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
          break;
#endif // 32
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
#endif //15
#endif //14
#endif //13
#endif //12
#endif //11
#endif //10
#endif // 9
#endif // 7
#endif // 6
#endif // 5
#endif // 3
        default: errorQuda("x.size %lu invalid (MAX_MULTI_BLAS_N = %d)", x.size(), MAX_MULTI_BLAS_N);
        }
        } else {
          // split the problem and recurse. Splitting in x requires
          // memory reshuffling (unless y = 1).
          // Use a few temporary variables.

          T* tmpmajor = new T[x.size()*y.size()];
          T* result0 = &tmpmajor[0];
          T* result1 = &tmpmajor[(x.size()/2)*y.size()];
          std::vector<ColorSpinorField*> x0(x.begin(), x.begin() + x.size()/2);
          std::vector<ColorSpinorField*> x1(x.begin() + x.size()/2, x.end());
          std::vector<ColorSpinorField*> z0(z.begin(), z.begin() + z.size()/2);
          std::vector<ColorSpinorField*> z1(z.begin() + z.size()/2, z.end());

          multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result0, x0, y, z0, w, 2*i_idx+0, j_idx, hermitian, tile_size);
          multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result1, x1, y, z1, w, 2*i_idx+1, j_idx, hermitian, tile_size);

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
        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true)) {
          const unsigned int xlen = x.size();
          const unsigned int ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++)
              result[i*xlen+j] = tmp_dot[j*ylen + i];
        }
        delete[] tmp_dot;
      }
    }

    template <template <int MXZ, typename ReducerType, typename real> class ReducerDiagonal,
              template <int MXZ, typename ReducerType, typename real> class ReducerOffDiagonal, typename T>
    class TileSizeTune : public Tunable
    {
      typedef std::vector<ColorSpinorField*> vec;
      T *result;
      vec &x, &y, &z, &w;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      int NYW_max;
      uint2 max_tile_size;

    public:
      TileSizeTune(T *result, vec &x, vec &y, vec &z, vec &w, int coeff_width, bool hermitian, bool Anorm = false,
                   bool nested_policy = false) :
        result(result),
        x(x),
        y(y),
        z(z),
        w(w),
        hermitian(hermitian),
        Anorm(Anorm)
      {
        NYW_max = max_YW_size(x.size(), x[0]->Precision(), y[0]->Precision(), false, false, coeff_width, true);
        max_tile_size = make_uint2(1, 1);

        strcpy(aux, nested_policy ? "nested_policy," : "policy,");
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
        if (!tuned()) {
          if (!nested_policy) disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

          // note the 1-d tuning is all redundent now that we call
          // multiReduce_recurse directly now for 1-d multi
          // reductions, but I'll keep this code here for now
          if (x.size() == 1) { // 1-d reduction

            max_tile_size = make_uint2(1, std::min(NYW_max, (int)y.size()));
            multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, max_tile_size);

          } else if (y.size() == 1) { // 1-d reduction

            max_tile_size = make_uint2(std::min((size_t)max_NXZ_power2(true), x.size()), 1);
            multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, max_tile_size);

          } else { // 2-d reduction

            // max_tile_size should be set to the largest power of 2,
            // since we have a requirement that the tile size is a
            // power of 2.
            // FIXME - we only do simple square tiling here
            max_tile_size = make_uint2(max_NXZ_power2(true), max_NXZ_power2(true));

            // Make sure constituents are tuned.
            for (unsigned int tile_size = 1;
                 tile_size <= max_tile_size.x && tile_size <= x.size() && (tile_size <= y.size() || y.size() == 1);
                 tile_size *= 2) {
              multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, make_uint2(tile_size, tile_size));
            }

            // also test case using a single kernel if both dimensions are less than max
            if (is_valid_NXZ(x.size(), true) && y.size() <= (unsigned int)NYW_max) {
              multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, make_uint2(x.size(), y.size()));
            }
          }

          if (!nested_policy) enableProfileCount();
          setPolicyTuning(true);
        }
      }

      virtual ~TileSizeTune() { setPolicyTuning(false); }

      void apply(const qudaStream_t &stream) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        // tp.aux.x is where the tile size is stored. "tp" is the tuning struct.
        // it contains blocksize, grid size, etc. Since we're only tuning
        // a policy, we don't care about those sizes. That's why we only
        // tune "aux.x", which is the tile size.
        multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, make_uint2(tp.aux.x, tp.aux.y));
      }

      // aux.x is the tile size
      bool advanceAux(TuneParam &param) const
      {
        // for 1-d reductions we don't do any tuning and just use the largest tile
        if (x.size() == 1 || y.size() == 1) {
          return false;
        } else { // 2-d reduction

          if ((unsigned int)(2 * param.aux.x) <= max_tile_size.x && (unsigned int)(2 * param.aux.y) <= max_tile_size.y
              && (unsigned int)(2 * param.aux.x) <= x.size() && (unsigned int)(2 * param.aux.y) <= y.size()) {
            // only tune powers of two
            param.aux.x *= 2;
            param.aux.y *= 2;
            return true;
          } else if (is_valid_NXZ(x.size(), true) && y.size() <= (size_t)NYW_max
                     && ((size_t)param.aux.x != x.size() || (size_t)param.aux.y != y.size())) {
            // we've run out of power of two tiles to try, but before
            // we finish, try a single kernel if it fits
            param.aux.x = x.size();
            param.aux.y = y.size();
            return true;
          } else {
            // reset to the beginning (which we'd need for multi-dimensional tuning)
            param.aux.x = 1;
            param.aux.y = 1;
            return false;
          }
        }
      }

      bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const  {
        Tunable::initTuneParam(param);
        if (x.size() == 1 || y.size() == 1) {
          param.aux.x = max_tile_size.x;
          param.aux.y = max_tile_size.y;
        } else { // only do non-trivial tuning for 2-d reductions
          param.aux.x = 1;
          param.aux.y = 1;
        }
        param.aux.z = 0;
        param.aux.w = 0;
      }

      void defaultTuneParam(TuneParam &param) const  {
        Tunable::defaultTuneParam(param); // default is max tile size
        param.aux.x = max_tile_size.x;
        param.aux.y = max_tile_size.y;
        param.aux.z = 0;
        param.aux.w = 0;
      }

      TuneKey tuneKey() const {
        return TuneKey(x[0]->VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return 0; } // FIXME
      long long bytes() const { return 0; } // FIXME

      void preTune() { } // FIXME - use write to determine what needs to be saved
      void postTune() { } // FIXME - use write to determine what needs to be saved
    };

    template <template <int MXZ, typename ReducerType, typename real> class ReducerDiagonal,
              template <int MXZ, typename ReducerType, typename real> class ReducerOffDiagonal, typename T>
    class TransposeTune : public Tunable
    {
      using TileTuner = TileSizeTune<ReducerDiagonal, ReducerOffDiagonal, T>;
      using vec = std::vector<ColorSpinorField *>;
      T *result;
      vec &x, &y, &z, &w;
      int coeff_width;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    public:
      TransposeTune(T *result, vec &x, vec &y, vec &z, vec &w, int coeff_width, bool hermitian, bool Anorm = false) :
        result(result),
        x(x),
        y(y),
        z(z),
        w(w),
        coeff_width(coeff_width),
        hermitian(hermitian),
        Anorm(Anorm)
      {
        strcpy(aux, "policy,");
        strcat(aux, x[0]->AuxString());
        strcat(aux, ",");
        strcat(aux, y[0]->AuxString());
        if (hermitian) strcat(aux, ",hermitian");
        if (Anorm) strcat(aux, ",Anorm");
        strcat(aux, ",n=");
        char size[8];
        u64toa(size, x.size());
        strcat(aux, size);
        strcat(aux, ",m=");
        u64toa(size, y.size());
        strcat(aux, size);
        u64toa(size, MAX_MULTI_BLAS_N);
        strcat(aux, ",multi-blas-n=");
        strcat(aux, size);

        // before we do policy tuning we must ensure the kernel
        // constituents have been tuned since we can't do nested tuning
        if (!tuned()) {
          disableProfileCount(); // purely for profiling reasons, don't want to profile tunings.

          // note the 1-d tuning is all redundent now that we call
          // multiReduce_recurse directly now for 1-d multi
          // reductions, but I'll keep this code here for now
          if (x.size() == 1) {
            TileTuner tile(result, x, y, z, w, coeff_width, hermitian, Anorm, true);
            tile.apply(0);
          } else if (y.size() == 1) {
            TileTuner tile(result, y, z, w, z, coeff_width, hermitian, Anorm, true);
            tile.apply(0);
          } else {

            { // tune regular inner product
              TileTuner tile(result, x, y, z, w, coeff_width, hermitian, Anorm, true);
              tile.apply(0);
            }

            { // tune transpose inner product
              TileTuner tile(result, y, z, w, z, coeff_width, hermitian, Anorm, true);
              tile.apply(0);
            }
          }

          enableProfileCount();
          setPolicyTuning(true);
        }
      }

      virtual ~TransposeTune() { setPolicyTuning(false); }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (tp.aux.x == 0) {
          TileTuner tile(result, x, y, z, w, coeff_width, hermitian, Anorm, true);
          tile.apply(stream);
        } else if (tp.aux.x == 1) {
          T *result_trans = new T[x.size() * y.size()];

          // swap (x<->y and w<-z> when doing transpose calculation)
          TileTuner tile(result_trans, y, x, w, z, coeff_width, hermitian, Anorm, true);
          tile.apply(stream);

          // tranpose the result if we are doing the transpose calculation
          const auto xlen = x.size();
          const auto ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++) result[i * xlen + j] = conj(result_trans[j * ylen + i]);

          delete[] result_trans;
        } else {
          errorQuda("Unexpected transpose parameter %d", tp.aux.x);
        }
      }

      bool advanceAux(TuneParam &param) const
      {
        if (x.size() == 1 || y.size() == 1) {
          return false;
        } else {
          if (param.aux.x == 0) {
            param.aux.x = 1;
            return true;
          } else {
            param.aux.x = 0;
            return false;
          }
        }
      }

      bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const
      {
        Tunable::initTuneParam(param);
        if (x.size() == 1)
          param.aux = make_int4(0, 0, 0, 0);
        else if (y.size() == 1)
          param.aux = make_int4(1, 0, 0, 0);
        else
          param.aux = make_int4(0, 0, 0, 0); // default is not to transpose
      }

      void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

      TuneKey tuneKey() const { return TuneKey(x[0]->VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 0; } // FIXME
      long long bytes() const { return 0; } // FIXME

      void preTune() {}  // FIXME - use write to determine what needs to be saved
      void postTune() {} // FIXME - use write to determine what needs to be saved
    };

    void reDotProduct(double *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      double *result_tmp = new double[x.size() * y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;
      int coeff_width = 0;

      if (x.size() == 1) {
        int NYW_max = max_YW_size(x.size(), x[0]->Precision(), y[0]->Precision(), false, false, coeff_width, true);
        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x[0]->Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<Dot, Dot>(result_tmp, x, y, x, y, 0, 0, false, max_tile_size);
      } else if (y.size() == 1) {

        double *result_trans = new double[x.size() * y.size()];

        // swap (x<->y and w<-z> when doing transpose calculation)
        int NXZ_max = max_YW_size(y.size(), y[0]->Precision(), x[0]->Precision(), false, false, coeff_width, true);
        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x[0]->Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<Dot, Dot>(result_trans, y, x, y, x, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++)
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = result_trans[j * ylen + i];

        delete[] result_trans;

      } else {
        TransposeTune<Dot, Dot, double> tile(result_tmp, x, y, x, y, coeff_width, false);
        tile.apply(0);
      }

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = x.size() * y.size();
      reduceDoubleArray(result_tmp, Nreduce);

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++) result[j * ylen + i] = result_tmp[i * xlen + j];

      delete[] result_tmp;
    }

    void cDotProduct(Complex *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      Complex *result_tmp = new Complex[x.size() * y.size()];
      for (unsigned int i = 0; i < x.size() * y.size(); i++) result_tmp[i] = 0.0;
      int coeff_width = 0;

      if (x.size() == 1) {
        int NYW_max = max_YW_size(x.size(), x[0]->Precision(), y[0]->Precision(), false, false, coeff_width, true);
        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x[0]->Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<Cdot, Cdot>(result_tmp, x, y, x, y, 0, 0, false, max_tile_size);
      } else if (y.size() == 1) {

        Complex *result_trans = new Complex[x.size() * y.size()];

        // swap (x<->y and w<-z> when doing transpose calculation)
        int NXZ_max = max_YW_size(y.size(), y[0]->Precision(), x[0]->Precision(), false, false, coeff_width, true);
        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x[0]->Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<Cdot, Cdot>(result_trans, y, x, y, x, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++)
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = conj(result_trans[j * ylen + i]);

        delete[] result_trans;

      } else {
        TransposeTune<Cdot, Cdot, Complex> trans(result_tmp, x, y, x, y, coeff_width, false);
        trans.apply(0);
      }

      // do a single multi-node reduction only once we have computed all local dot products
      const int Nreduce = 2*x.size()*y.size();
      reduceDoubleArray((double*)result_tmp, Nreduce);

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      const unsigned int xlen = x.size();
      const unsigned int ylen = y.size();
      for (unsigned int j = 0; j < xlen; j++)
        for (unsigned int i = 0; i < ylen; i++)
          result[j*ylen+i] = result_tmp[i*xlen + j];

      delete[] result_tmp;
    }

    void hDotProduct(Complex *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      int coeff_width = 0;
      TileSizeTune<Cdot, Cdot, Complex> tile(result_tmp, x, y, x, y, coeff_width, true, false); // last false is b/c L2 norm
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
    void hDotProduct_Anorm(Complex *result, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block A-norm dot product on non-square inputs");

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      int coeff_width = 0;
      TileSizeTune<Cdot, Cdot, Complex> tile(result_tmp, x, y, x, y, coeff_width, true, true); // last true is b/c A norm
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
      // FIXME - if this is enabled we need to ensure that use_w is
      // enabled above.  Also, I think this might break if the diagonal
      // write is different from the off-diagonal write
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (y.size() != z.size()) errorQuda("Cannot copy input y of size %lu into z of size %lu\n", y.size(), z.size());

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      int coeff_width = 0;
      // When recursing, only the diagonal tiles will do the copy, the rest just do the outer product
      TileSizeTune<double2, typename vector<device_reduce_t,2>::type,CdotCopy,Cdot,Complex> tile(result_tmp, x, y, x, y, coeff_width, true);
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
