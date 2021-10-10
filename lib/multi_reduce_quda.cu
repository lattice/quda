#include <blas_quda.h>
#include <uint_to_char.h>
#include <kernels/multi_reduce_core.cuh>
#include <tunable_reduction.h>

namespace quda {

  namespace blas {

    template <template <typename ...> class Reducer, typename store_t, typename y_store_t, int nSpin, typename T>
    class MultiReduce : public TunableMultiReduction<1>
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      const int NXZ;
      const int NYW;
      Reducer<device_reduce_t, real> r;
      const int nParity;
      const T &a, &b, &c;
      std::vector<ColorSpinorField *> &x, &y, &z, &w;
      host_reduce_t *result;
      QudaFieldLocation location;

      virtual bool advanceSharedBytes(TuneParam &param) const
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(param) ?
          sharedBytesPerThread() * nthreads : sharedBytesPerBlock(param);
        return false;
      }

    public:
      MultiReduce(const T &a, const T &b, const T &c, const ColorSpinorField &, const ColorSpinorField &,
                  std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                  std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w,
                  host_reduce_t *result) :
        TunableMultiReduction(*x[0], y.size()),
        NXZ(x.size()),
        NYW(y.size()),
        r(NXZ, NYW),
        nParity(x[0]->SiteSubset()),
        a(a),
        b(b),
        c(c),
        x(x),
        y(y),
        z(z),
        w(w),
        result(result),
        location(checkLocation(*x[0], *y[0], *z[0], *w[0]))
      {
        checkLength(*x[0], *y[0], *z[0], *w[0]);
        auto x_prec = checkPrecision(*x[0], *z[0], *w[0]);
        auto y_prec = y[0]->Precision();
        auto x_order = checkOrder(*x[0], *z[0], *w[0]);
        auto y_order = y[0]->FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        char aux2[TuneKey::aux_n];
        strcpy(aux2, aux);
        strcpy(aux, "policy_kernel,");
        strcat(aux, aux2);
        if (x_prec != y_prec) {
          strcat(aux, ",");
          strcat(aux, y[0]->AuxString());
        }

        char NXZ_str[16];
        char NYW_str[16];
        u32toa(NXZ_str, NXZ);
        u32toa(NYW_str, NYW);
        strcat(aux, ",Nxz=");
        strcat(aux, NXZ_str);
        strcat(aux, ",Nyw=");
        strcat(aux, NYW_str);

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

        apply(device::get_default_stream());

        blas::bytes += bytes();
        blas::flops += flops();
      }

      TuneKey tuneKey() const { return TuneKey(vol, typeid(r).name(), aux); }

      template <int NXZ> void compute(const qudaStream_t &stream)
      {
        staticCheck<NXZ, store_t, y_store_t, decltype(r)>(r, x, y);

        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value;
        if (site_unroll_check && (x[0]->Ncolor() != 3 || x[0]->Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x[0]->Nspin(), x[0]->Ncolor());

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(*x[0], *y[0], *z[0], *w[0]); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Reducer<device_reduce_t, device_real_t> r_(NXZ, NYW);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x[0]->Length() / M;

          MultiReduceArg<device_real_t, M, NXZ, device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, r_, NYW, length, nParity);

          std::vector<host_reduce_t> result_(NXZ * arg.NYW);

#if 0 // no parameters to set so far
          constexpr bool multi_1d = false;
          if (a.data) { set_param<multi_1d>(arg, 'a', a); }
          if (b.data) { set_param<multi_1d>(arg, 'b', b); }
          if (c.data) { set_param<multi_1d>(arg, 'c', c); }
#endif
          // we intentional do not do a global reduction in the launch, and defer until the entire "tile" is complete
          launch<MultiReduce_, host_reduce_t, comm_reduce_null<host_reduce_t>>(result_, tp, stream, arg);

          // need to transpose for same order with vector thread reduction
          for (int i = 0; i < NXZ; i++) {
            for (int j = 0; j < arg.NYW; j++) {
              result[i * arg.NYW + j] = result_[j * NXZ + i];
            }
          }

        } else {
          errorQuda("Only implemented for GPU fields");
        }
      }

      template <int n> std::enable_if_t<n!=1, void> instantiateLinear(const qudaStream_t &stream)
      {
        if (NXZ == n) compute<n>(stream);
        else instantiateLinear<n-1>(stream);
      }

      template <int n> std::enable_if_t<n==1, void> instantiateLinear(const qudaStream_t &stream)
      {
        compute<1>(stream);
      }

      template <int n> std::enable_if_t<n!=1, void> instantiatePow2(const qudaStream_t &stream)
      {
        if (NXZ == n) compute<n>(stream);
        else instantiatePow2<n/2>(stream);
      }

      template <int n> std::enable_if_t<n==1, void> instantiatePow2(const qudaStream_t &stream)
      {
        compute<1>(stream);
      }

      void apply(const qudaStream_t &stream)
      {
        constexpr int pow2_max = max_NXZ_power2<true, isFixed<store_t>::value>();
        if (NXZ <= pow2_max && is_power2(NXZ)) instantiatePow2<pow2_max>(stream);
        else if (NXZ <= MAX_MULTI_BLAS_N) instantiateLinear<MAX_MULTI_BLAS_N>(stream);
        else errorQuda("x.size %lu greater than MAX_MULTI_BLAS_N %d", x.size(), MAX_MULTI_BLAS_N);
      }

      void preTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (r.write.X) x[i]->backup();
          if (r.write.Y) y[i]->backup();
          if (r.write.Z) z[i]->backup();
          if (r.write.W) w[i]->backup();
        }
      }

      void postTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (r.write.X) x[i]->restore();
          if (r.write.Y) y[i]->restore();
          if (r.write.Z) z[i]->restore();
          if (r.write.W) w[i]->restore();
        }
      }

      long long flops() const
      {
        return NYW * NXZ * r.flops() * x[0]->Length();
      }

      long long bytes() const
      {
        // X and Z reads are repeated (and hopefully cached) across NYW
        // each Y and W read/write is done once
        return NYW * NXZ * (r.read.X + r.write.X) * x[0]->Bytes() +
          NYW * (r.read.Y + r.write.Y) * y[0]->Bytes() +
          NYW * NXZ * (r.read.Z + r.write.Z) * z[0]->Bytes() +
          NYW * (r.read.W + r.write.W) * w[0]->Bytes();
      }

      int tuningIter() const { return 3; }
    };

    template <template <typename ...> class ReducerDiagonal, template <typename ...> class ReducerOffDiagonal, typename T>
    void multiReduce(T result[], const coeff_array<T> &a, const coeff_array<T> &b, const coeff_array<T> &c,
                     CompositeColorSpinorField &x, CompositeColorSpinorField &y, CompositeColorSpinorField &z,
                     CompositeColorSpinorField &w, int i, int j)
    {
      if (i == j) { // we are on the diagonal so invoke the diagonal reducer
        using host_reduce_t = typename ReducerDiagonal<double, double>::reduce_t;
        instantiate<ReducerDiagonal, MultiReduce, true>(a, b, c, *x[0], *y[0], x, y, z, w, (host_reduce_t*)result);
      } else { // we are on the diagonal so invoke the off-diagonal reducer
        using host_reduce_t = typename ReducerOffDiagonal<double, double>::reduce_t;
        instantiate<ReducerOffDiagonal, MultiReduce, true>(a, b, c, *x[0], *y[0], x, y, z, w, (host_reduce_t*)result);
      }
    }

    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <typename ...> class ReducerDiagonal,
              template <typename ...> class ReducerOffDiagonal, typename T>
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
        multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result0, x, y0, z, w, i_idx, 2*j_idx+0, hermitian, tile_size);
        multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result1, x, y1, z, w, i_idx, 2*j_idx+1, hermitian, tile_size);
      } else {
        T* tmp_dot = new T[x.size()*y.size()];

	// if at bottom of recursion, return if on lower left
        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true) && hermitian) {
          if (j_idx < i_idx) { return; }
        }

        coeff_array<T> a, b, c;


        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true) && x.size() * y.size() <= (unsigned int)max_n_reduce()) {
          // problem will fit, so do the computation
          multiReduce<ReducerDiagonal, ReducerOffDiagonal>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);
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
          std::vector<ColorSpinorField*> w0(w.begin(), w.begin() + w.size()/2);
          std::vector<ColorSpinorField*> w1(w.begin() + w.size()/2, w.end());

          multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result0, x0, y, z0, w0, 2*i_idx+0, j_idx, hermitian, tile_size);
          multiReduce_recurse<ReducerDiagonal,ReducerOffDiagonal>(result1, x1, y, z1, w1, 2*i_idx+1, j_idx, hermitian, tile_size);

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
        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true) && x.size() * y.size() <= (unsigned int)max_n_reduce()) {
          const unsigned int xlen = x.size();
          const unsigned int ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++)
              result[i*xlen+j] = tmp_dot[j*ylen + i];
        }
        delete[] tmp_dot;
      }
    }

    template <template <typename ...> class ReducerDiagonal,
              template <typename ...> class ReducerOffDiagonal, typename T>
    class TileSizeTune : public Tunable
    {
      typedef std::vector<ColorSpinorField*> vec;
      T *result;
      vec &x, &y, &z, &w;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

      int NYW_max;
      uint2 max_tile_size;

    public:
      TileSizeTune(T *result, vec &x, vec &y, vec &z, vec &w, bool hermitian, bool Anorm = false,
                   bool nested_policy = false) :
        result(result),
        x(x),
        y(y),
        z(z),
        w(w),
        hermitian(hermitian),
        Anorm(Anorm)
      {
        NYW_max = std::min(
          (y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
           max_YW_size<ReducerDiagonal<device_reduce_t, double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
           max_YW_size<ReducerDiagonal<device_reduce_t, float>>(x.size(), x[0]->Precision(), y[0]->Precision())),
          (y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
           max_YW_size<ReducerOffDiagonal<device_reduce_t, double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
           max_YW_size<ReducerOffDiagonal<device_reduce_t, float>>(x.size(), x[0]->Precision(), y[0]->Precision()))
                               );

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

          // note the 1-d tuning is all redundant now that we call
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

        apply(device::get_default_stream());
      }

      virtual ~TileSizeTune() { setPolicyTuning(false); }

      void apply(const qudaStream_t &) {
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

    template <template <typename ...> class ReducerDiagonal,
              template <typename ...> class ReducerOffDiagonal, typename T>
    class TransposeTune : public Tunable
    {
      using TileTuner = TileSizeTune<ReducerDiagonal, ReducerOffDiagonal, T>;
      using vec = std::vector<ColorSpinorField *>;
      T *result;
      vec &x, &y;
      bool hermitian;
      bool Anorm;

      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    public:
      TransposeTune(T *result, vec &x, vec &y, bool hermitian, bool Anorm = false) :
        result(result),
        x(x),
        y(y),
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
            TileTuner tile(result, x, y, x, x, hermitian, Anorm, true);
          } else if (y.size() == 1) {
            TileTuner tile(result, y, x, y, y, hermitian, Anorm, true);
          } else {

            { // tune regular inner product
              TileTuner tile(result, x, y, x, x, hermitian, Anorm, true);
            }

            { // tune transpose inner product
              TileTuner tile(result, y, x, y, y, hermitian, Anorm, true);
            }
          }

          enableProfileCount();
          setPolicyTuning(true);
        }

        apply(device::get_default_stream());
      }

      virtual ~TransposeTune() { setPolicyTuning(false); }

      void apply(const qudaStream_t &)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (tp.aux.x == 0) {
          TileTuner(result, x, y, x, x, hermitian, Anorm, true);
        } else if (tp.aux.x == 1) {
          T *result_trans = new T[x.size() * y.size()];

          // swap (x<->y and w<-z> when doing transpose calculation)
          TileTuner(result_trans, y, x, y, y, hermitian, Anorm, true);

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

      if (x.size() == 1) {
        auto NYW_max = y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiDot<device_reduce_t, double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
          max_YW_size<multiDot<device_reduce_t, float>>(x.size(), x[0]->Precision(), y[0]->Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x[0]->Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<multiDot, multiDot>(result_tmp, x, y, x, x, 0, 0, false, max_tile_size);
      } else if (y.size() == 1 && x[0]->Precision() == y[0]->Precision()) {

        double *result_trans = new double[x.size() * y.size()];

        // swap (x<->y and w<-z> when doing transpose calculation)
        auto NXZ_max = x[0]->Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiDot<device_reduce_t, double>>(y.size(), y[0]->Precision(), x[0]->Precision()) :
          max_YW_size<multiDot<device_reduce_t, float>>(y.size(), y[0]->Precision(), x[0]->Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x[0]->Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<multiDot, multiDot>(result_trans, y, x, y, y, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++)
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = result_trans[j * ylen + i];

        delete[] result_trans;

      } else if (x[0]->Precision() == y[0]->Precision()) {
        TransposeTune<multiDot, multiDot, double>(result_tmp, x, y, false);
      } else {
        TileSizeTune<multiDot, multiDot, double>(result_tmp, x, y, x, x, false);
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

      if (x.size() == 1) {
        auto NYW_max = y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiCdot<device_reduce_t, double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
          max_YW_size<multiCdot<device_reduce_t, float>>(x.size(), x[0]->Precision(), y[0]->Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x[0]->Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<multiCdot, multiCdot>(result_tmp, x, y, x, x, 0, 0, false, max_tile_size);
      } else if (y.size() == 1 && x[0]->Precision() == y[0]->Precision()) {

        Complex *result_trans = new Complex[x.size() * y.size()];

        // swap (x<->y and w<-z> when doing transpose calculation)
        auto NXZ_max = x[0]->Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiCdot<device_reduce_t, double>>(y.size(), y[0]->Precision(), x[0]->Precision()) :
          max_YW_size<multiCdot<device_reduce_t, float>>(y.size(), y[0]->Precision(), x[0]->Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x[0]->Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<multiCdot, multiCdot>(result_trans, y, x, y, y, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++)
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = conj(result_trans[j * ylen + i]);

        delete[] result_trans;

      } else if (x[0]->Precision() == y[0]->Precision()) {
        TransposeTune<multiCdot, multiCdot, Complex>(result_tmp, x, y, false);
      } else {
        TileSizeTune<multiCdot, multiCdot, Complex>(result_tmp, x, y, x, x, false);
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

      TileSizeTune<multiCdot, multiCdot, Complex>(result_tmp, x, y, x, x, true, false); // last false is b/c L2 norm

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

      TileSizeTune<multiCdot, multiCdot, Complex>(result_tmp, x, y, x, x, true, true); // last true is b/c A norm

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
    void cDotProductCopy(Complex* , std::vector<ColorSpinorField*>&, std::vector<ColorSpinorField*>&,
			 std::vector<ColorSpinorField*>&)
    {
#if 0
      // FIXME - if this is enabled we need to ensure that use_w is
      // enabled above.  Also, I think this might break if the diagonal
      // write is different from the off-diagonal write
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (y.size() != z.size()) errorQuda("Cannot copy input y of size %lu into z of size %lu\n", y.size(), z.size());

      Complex* result_tmp = new Complex[x.size()*y.size()];
      for (unsigned int i = 0; i < x.size()*y.size(); i++) result_tmp[i] = 0.0;

      // When recursing, only the diagonal tiles will do the copy, the rest just do the outer product
      TileSizeTune<double2, typename vector<device_reduce_t,2>::type,multiCdotCopy,multiCdot,Complex>(result_tmp, x, y, x, y, true);

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
