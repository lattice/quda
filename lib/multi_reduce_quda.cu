#include <blas_quda.h>
#include <uint_to_char.h>
#include <kernels/multi_reduce_core.cuh>
#include <tunable_reduction.h>

namespace quda {

  namespace blas {

    template <template <typename ...> class Reducer, typename store_t, typename y_store_t, int nSpin,
              typename T>
    class MultiReduce : public TunableMultiReduction
    {
      using real = typename mapper<y_store_t>::type;
      using host_reduce_t = typename Reducer<double, real>::reduce_t;
      const int NXZ;
      const int NYW;
      Reducer<device_reduce_t, real> r;
      const int nParity;
      const T &a, &b, &c;
      cvector_ref<ColorSpinorField> &x, &y, &z, &w;
      T &result;
      QudaFieldLocation location;

      virtual bool advanceSharedBytes(TuneParam &param) const override
      {
        TuneParam next(param);
        advanceBlockDim(next); // to get next blockDim
        int nthreads = next.block.x * next.block.y * next.block.z;
        param.shared_bytes = sharedBytesPerThread() * nthreads > sharedBytesPerBlock(param) ?
          sharedBytesPerThread() * nthreads : sharedBytesPerBlock(param);
        return false;
      }

    public:
      template <typename Vx, typename Vy, typename Vz, typename Vw>
      MultiReduce(const T &a, const T &b, const T &c, const ColorSpinorField &x0, const ColorSpinorField &y0,
                  Vx &x, Vy &y, Vz &z, Vw &w, T &result) :
        TunableMultiReduction(x[0], 1u, y.size(), max_n_batch_block_multi_reduce()),
        NXZ(x.size()),
        NYW(y.size()),
        r(NXZ, NYW),
        nParity(x0.SiteSubset()),
        a(a),
        b(b),
        c(c),
        x(reinterpret_cast<cvector_ref<ColorSpinorField>&>(x)),
        y(reinterpret_cast<cvector_ref<ColorSpinorField>&>(y)),
        z(reinterpret_cast<cvector_ref<ColorSpinorField>&>(z)),
        w(reinterpret_cast<cvector_ref<ColorSpinorField>&>(w)),
        result(result),
        location(checkLocation(x[0], y[0], z[0], w[0]))
      {
        checkLength(x[0], y[0], z[0], w[0]);
        auto x_prec = checkPrecision(x[0], z[0], w[0]);
        auto y_prec = y0.Precision();
        auto x_order = checkOrder(x[0], z[0], w[0]);
        auto y_order = y0.FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        char aux2[TuneKey::aux_n];
        strcpy(aux2, aux);
        strcpy(aux, "policy_kernel,");
        strcat(aux, aux2);
        if (x_prec != y_prec) {
          strcat(aux, ",");
          strcat(aux, y0.AuxString().c_str());
        }

        char NXZ_str[16];
        char NYW_str[16];
        u32toa(NXZ_str, NXZ);
        u32toa(NYW_str, NYW);
        strcat(aux, ",Nxz=");
        strcat(aux, NXZ_str);
        strcat(aux, ",Nyw=");
        strcat(aux, NYW_str);

        char max_nyw_tile[8];
        u32toa(max_nyw_tile, max_n_batch_block_multi_reduce());
        strcat(aux, ",max_nyw_tile=");
        strcat(aux, max_nyw_tile);

        // since block dot product and block norm use the same functors, we need to distinguish them
        bool is_norm = false;
        if (NXZ == NYW) {
          is_norm = true;
          for (int i = 0; i < NXZ; i++) {
            if (x[i].V() != y[i].V() || x[i].V() != z[i].V() || x[i].V() != w[i].V()) {
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

      TuneKey tuneKey() const override { return TuneKey(vol, typeid(r).name(), aux); }

      template <int NXZ> void compute(const qudaStream_t &stream)
      {
        staticCheck<NXZ, store_t, y_store_t, decltype(r)>(r, x, y);

        auto &x0 = x[0];
        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value;
        if (site_unroll_check && (x0.Ncolor() != 3 || x0.Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x0.Nspin(), x0.Ncolor());

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(x[0], y[0], z[0], w[0]); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Reducer<device_reduce_t, device_real_t> r_(NXZ, NYW);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x0.Length() / M;

          MultiReduceArg<device_real_t, M, NXZ, device_store_t, N, device_y_store_t, Ny, decltype(r_)> arg(x, y, z, w, r_, NYW, length, nParity);

          std::vector<host_reduce_t> result_(NXZ * arg.NYW);

#if 0 // no parameters to set so far
          constexpr bool multi_1d = false;
          if (a.size()) { set_param<multi_1d>(arg, 'a', a); }
          if (b.size()) { set_param<multi_1d>(arg, 'b', b); }
          if (c.size()) { set_param<multi_1d>(arg, 'c', c); }
#endif
          launch<MultiReduce_>(result_, tp, stream, arg);

          // need to transpose for same order with vector thread reduction
          for (int i = 0; i < NXZ; i++) {
            for (int j = 0; j < arg.NYW; j++) {
              reinterpret_cast<host_reduce_t*>(result.data())[i * arg.NYW + j] = result_[j * NXZ + i];
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

      void apply(const qudaStream_t &stream) override
      {
        constexpr int pow2_max = max_NXZ_power2(true);
        if (NXZ <= pow2_max && is_power2(NXZ)) instantiatePow2<pow2_max>(stream);
        else if (NXZ <= MAX_MULTI_BLAS_N) instantiateLinear<MAX_MULTI_BLAS_N>(stream);
        else errorQuda("x.size %lu greater than MAX_MULTI_BLAS_N %d", x.size(), MAX_MULTI_BLAS_N);
      }

      void preTune() override
      {
        for (int i = 0; i < NYW; ++i) {
          if (r.write.X) x[i].backup();
          if (r.write.Y) y[i].backup();
          if (r.write.Z) z[i].backup();
          if (r.write.W) w[i].backup();
        }
      }

      void postTune() override
      {
        for (int i = 0; i < NYW; ++i) {
          if (r.write.X) x[i].restore();
          if (r.write.Y) y[i].restore();
          if (r.write.Z) z[i].restore();
          if (r.write.W) w[i].restore();
        }
      }

      long long flops() const override
      {
        return NYW * NXZ * r.flops() * x[0].Length();
      }

      long long bytes() const override
      {
        // X and Z reads are repeated (and hopefully cached) across NYW
        // each Y and W read/write is done once
        return NYW * NXZ * (r.read.X + r.write.X) * x[0].Bytes() +
          NYW * (r.read.Y + r.write.Y) * y[0].Bytes() +
          NYW * NXZ * (r.read.Z + r.write.Z) * z[0].Bytes() +
          NYW * (r.read.W + r.write.W) * w[0].Bytes();
      }
    };

    template <template <typename ...> class ReducerDiagonal, template <typename ...> class ReducerOffDiagonal,
              typename T, typename Vx, typename Vy, typename Vz, typename Vw>
    void multiReduce(std::vector<T> &result, const std::vector<T> &a, const std::vector<T> &b, const std::vector<T> &c,
                     Vx &x, Vy &y, Vz &z, Vw &w, int i, int j)
    {
      if (i == j) { // we are on the diagonal so invoke the diagonal reducer
        instantiate<ReducerDiagonal, MultiReduce, true>(a, b, c, x[0], y[0], x, y, z, w, result);
      } else { // we are on the diagonal so invoke the off-diagonal reducer
        instantiate<ReducerOffDiagonal, MultiReduce, true>(a, b, c, x[0], y[0], x, y, z, w, result);
      }
    }

    // This function does the outer product of dot products... in column major.
    // There's a function below called 'cDotProduct' that flips it to row major.
    template <template <typename ...> class reducer_diag,
              template <typename ...> class reducer_off, typename T, typename Vx, typename Vy, typename Vz, typename Vw>
    void multiReduce_recurse(std::vector<T> &result, Vx &x, Vy &y, Vz &z, Vw &w, int i_idx, int j_idx,
                             bool hermitian, uint2 tile_size)
    {
      if (y.size() > tile_size.y) { // if greater than max single-kernel size, split and recurse
        // Do the recurse first.
        auto result_ = std::make_pair( std::vector<T>((y.size() / 2) * x.size()),
                                       std::vector<T>(result.size() - (y.size() / 2) * x.size()) );
        auto y_ = bisect(y);
        multiReduce_recurse<reducer_diag, reducer_off>(result_.first, x, y_.first, z, w, i_idx, j_idx, hermitian, tile_size);
        multiReduce_recurse<reducer_diag, reducer_off>(result_.second, x, y_.second, z, w, i_idx, j_idx + y_.first.size(), hermitian, tile_size);

        result = join(result_);
      } else {

        if (x.size() <= tile_size.x && is_valid_NXZ(x.size(), true)) { // problem fits, so do the computation
          // if at bottom of recursion, return if on strict sub-diagonal
          if (hermitian && (j_idx + y.size() < i_idx + x.size())) return;

          std::vector<T> tmp_dot(x.size()*y.size());
          std::vector<T> a, b, c;

          multiReduce<reducer_diag, reducer_off>(tmp_dot, a, b, c, x, y, z, w, i_idx, j_idx);

          // perform the row-to-column-major transpose here.
          result = transpose(tmp_dot, x.size(), y.size());
        } else {
          // split the problem and recurse. Splitting in x requires
          // memory reshuffling (unless y = 1).

          auto result_ = std::make_pair( std::vector<T>((x.size() / 2) * y.size()),
                                         std::vector<T>(result.size() - (x.size() / 2) * y.size()) );
          auto x_ = bisect(x);
          auto z_ = bisect(z);
          auto w_ = bisect(w);

          multiReduce_recurse<reducer_diag, reducer_off>(result_.first, x_.first, y, z_.first, w_.first, i_idx, j_idx, hermitian, tile_size);
          multiReduce_recurse<reducer_diag, reducer_off>(result_.second, x_.second, y, z_.second, w_.second, i_idx + x_.first.size(), j_idx, hermitian, tile_size);

          result = join_row(result_, x.size() / 2, x.size() - x.size() / 2, y.size());
        }
      }
    }

    template <template <typename ...> class ReducerDiagonal,
              template <typename ...> class ReducerOffDiagonal, typename T,
              typename Vx, typename Vy, typename Vz = Vx, typename Vw = Vx>
    class TileSizeTune : public Tunable
    {
      std::vector<T> &result;
      Vx &x;
      Vy &y;
      Vz &z;
      Vw &w;
      bool hermitian;
      bool Anorm;

      int NYW_max;
      uint2 max_tile_size;

    public:
      TileSizeTune(std::vector<T> &result, Vx &x, Vy &y, Vz &z, Vw &w, bool hermitian, bool Anorm = false,
                   bool nested_policy = false) :
        result(result),
        x(x),
        y(y),
        z(z),
        w(w),
        hermitian(hermitian),
        Anorm(Anorm)
      {
        auto &x0 = x[0];
        auto &y0 = y[0];

        NYW_max = std::min(
          (y0.Precision() == QUDA_DOUBLE_PRECISION ?
           max_YW_size<ReducerDiagonal<device_reduce_t, double>>(x.size(), x0.Precision(), y0.Precision()) :
           max_YW_size<ReducerDiagonal<device_reduce_t, float>>(x.size(), x0.Precision(), y0.Precision())),
          (y0.Precision() == QUDA_DOUBLE_PRECISION ?
           max_YW_size<ReducerOffDiagonal<device_reduce_t, double>>(x.size(), x0.Precision(), y0.Precision()) :
           max_YW_size<ReducerOffDiagonal<device_reduce_t, float>>(x.size(), x0.Precision(), y0.Precision()))
                               );

        max_tile_size = make_uint2(1, 1);

        strcpy(aux, nested_policy ? "nested_policy," : "policy,");
        strcat(aux, x[0].AuxString().c_str());
      	strcat(aux, ",");
        strcat(aux, y[0].AuxString().c_str());
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

        char max_nyw_tile[8];
        u32toa(max_nyw_tile, max_n_batch_block_multi_reduce());
        strcat(aux, ",max_nyw_tile=");
        strcat(aux, max_nyw_tile);

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

      void apply(const qudaStream_t &) override {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        // tp.aux.x is where the tile size is stored. "tp" is the tuning struct.
        // it contains blocksize, grid size, etc. Since we're only tuning
        // a policy, we don't care about those sizes. That's why we only
        // tune "aux.x", which is the tile size.
        multiReduce_recurse<ReducerDiagonal, ReducerOffDiagonal>(result, x, y, z, w, 0, 0, hermitian, make_uint2(tp.aux.x, tp.aux.y));
      }

      // aux.x is the tile size
      bool advanceAux(TuneParam &param) const override
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

      bool advanceTuneParam(TuneParam &param) const override { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const override {
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

      void defaultTuneParam(TuneParam &param) const override {
        Tunable::defaultTuneParam(param); // default is max tile size
        param.aux.x = max_tile_size.x;
        param.aux.y = max_tile_size.y;
        param.aux.z = 0;
        param.aux.w = 0;
      }

      TuneKey tuneKey() const override {
        return TuneKey(x[0].VolString().c_str(), typeid(*this).name(), aux);
      }

      long long bytes() const override { return 0; } // FIXME

      void preTune() override { } // FIXME - use write to determine what needs to be saved
      void postTune() override { } // FIXME - use write to determine what needs to be saved
    };

    template <template <typename ...> class ReducerDiagonal,
              template <typename ...> class ReducerOffDiagonal, typename T, typename Vx, typename Vy>
    class TransposeTune : public Tunable
    {
      using TileTuner = TileSizeTune<ReducerDiagonal, ReducerOffDiagonal, T, Vx, Vy, Vx, Vx>;
      std::vector<T> &result;
      Vx &x;
      Vy &y;
      bool hermitian;
      bool Anorm;

    public:
      TransposeTune(std::vector<T> &result, Vx &x, Vy &y, bool hermitian, bool Anorm = false) :
        result(result),
        x(x),
        y(y),
        hermitian(hermitian),
        Anorm(Anorm)
      {
        strcpy(aux, "policy,");
        strcat(aux, x[0].AuxString().c_str());
        strcat(aux, ",");
        strcat(aux, y[0].AuxString().c_str());
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

        char max_nyw_tile[8];
        u32toa(max_nyw_tile, max_n_batch_block_multi_reduce());
        strcat(aux, ",max_nyw_tile=");
        strcat(aux, max_nyw_tile);

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

      void apply(const qudaStream_t &) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (tp.aux.x == 0) {
          TileTuner(result, x, y, x, x, hermitian, Anorm, true);
        } else if (tp.aux.x == 1) {
          std::vector<T> result_trans(x.size() * y.size());

          // swap (x<->y and w<-z> when doing transpose calculation)
          TileTuner(result_trans, y, x, y, y, hermitian, Anorm, true);

          // tranpose the result if we are doing the transpose calculation
          const auto xlen = x.size();
          const auto ylen = y.size();
          for (unsigned int j = 0; j < xlen; j++)
            for (unsigned int i = 0; i < ylen; i++) result[i * xlen + j] = conj(result_trans[j * ylen + i]);
        } else {
          errorQuda("Unexpected transpose parameter %d", static_cast<int>(tp.aux.x));
        }
      }

      bool advanceAux(TuneParam &param) const override
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

      bool advanceTuneParam(TuneParam &param) const override { return advanceAux(param); }

      void initTuneParam(TuneParam &param) const override
      {
        Tunable::initTuneParam(param);
        if (x.size() == 1)
          param.aux = make_int4(0, 0, 0, 0);
        else if (y.size() == 1)
          param.aux = make_int4(1, 0, 0, 0);
        else
          param.aux = make_int4(0, 0, 0, 0); // default is not to transpose
      }

      void defaultTuneParam(TuneParam &param) const override { initTuneParam(param); }

      TuneKey tuneKey() const override { return TuneKey(x[0].VolString().c_str(), typeid(*this).name(), aux); }

      long long bytes() const override { return 0; } // FIXME

      void preTune() override {}  // FIXME - use write to determine what needs to be saved
      void postTune() override {} // FIXME - use write to determine what needs to be saved
    };

    void reDotProduct(std::vector<double> &result, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const ColorSpinorField> &y)
    {
      auto &x0 = x[0];
      auto &y0 = y[0];

      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      std::vector<double> result_tmp(x.size() * y.size(), 0.0);

      if (x.size() == 1) {
        auto NYW_max = y0.Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiDot<device_reduce_t, double>>(x.size(), x0.Precision(), y0.Precision()) :
          max_YW_size<multiDot<device_reduce_t, float>>(x.size(), x0.Precision(), y0.Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x0.Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<multiDot, multiDot>(result_tmp, x, y, x, x, 0, 0, false, max_tile_size);
      } else if (y.size() == 1 && x0.Precision() == y0.Precision()) {

        std::vector<double> result_trans(x.size() * y.size());

        // swap (x<->y and w<-z> when doing transpose calculation)
        auto NXZ_max = x0.Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiDot<device_reduce_t, double>>(y.size(), y0.Precision(), x0.Precision()) :
          max_YW_size<multiDot<device_reduce_t, float>>(y.size(), y0.Precision(), x0.Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x0.Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<multiDot, multiDot>(result_trans, y, x, y, y, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++)
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = result_trans[j * ylen + i];

      } else if (x0.Precision() == y0.Precision()) {
        TransposeTune<multiDot, multiDot, double, decltype(x), decltype(y)>(result_tmp, x, y, false);
      } else {
        TileSizeTune<multiDot, multiDot, double, decltype(x), decltype(y)>(result_tmp, x, y, x, x, false);
      }

      // do a single multi-node reduction only once we have computed all local dot products
      comm_allreduce_sum(result_tmp);

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      result = transpose(result_tmp, y.size(), x.size());
    }

    void cDotProduct(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<const ColorSpinorField> &y)
    {
      auto &x0 = x[0];
      auto &y0 = y[0];

      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      std::vector<Complex> result_tmp(x.size() * y.size(), 0.0);

      if (x.size() == 1) {
        auto NYW_max = y0.Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiCdot<device_reduce_t, double>>(x.size(), x0.Precision(), y0.Precision()) :
          max_YW_size<multiCdot<device_reduce_t, float>>(x.size(), x0.Precision(), y0.Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NYW_max, (int)y.size(), x0.Ncolor() == 3 ? 32 : NYW_max} ));
        multiReduce_recurse<multiCdot, multiCdot>(result_tmp, x, y, x, x, 0, 0, false, max_tile_size);
      } else if (y.size() == 1 && x0.Precision() == y0.Precision()) {

        std::vector<Complex> result_trans(x.size() * y.size());

        // swap (x<->y and w<-z> when doing transpose calculation)
        auto NXZ_max = x0.Precision() == QUDA_DOUBLE_PRECISION ?
          max_YW_size<multiCdot<device_reduce_t, double>>(y.size(), y0.Precision(), x0.Precision()) :
          max_YW_size<multiCdot<device_reduce_t, float>>(y.size(), y0.Precision(), x0.Precision());

        // if fine-grid then we set max tile size to 32 to avoid unnecessary tuning
        uint2 max_tile_size = make_uint2(1, std::min( {NXZ_max, (int)x.size(), x0.Ncolor() == 3 ? 32 : NXZ_max} ));
        multiReduce_recurse<multiCdot, multiCdot>(result_trans, y, x, y, y, 0, 0, false, max_tile_size);

        // transpose the result if we are doing the transpose calculation
        const auto xlen = x.size();
        const auto ylen = y.size();
        for (unsigned int j = 0; j < xlen; j++) {
          for (unsigned int i = 0; i < ylen; i++) result_tmp[i * xlen + j] = conj(result_trans[j * ylen + i]);
        }
      } else if (x0.Precision() == y0.Precision()) {
        TransposeTune<multiCdot, multiCdot, Complex, decltype(x), decltype(y)>(result_tmp, x, y, false);
      } else {
        TileSizeTune<multiCdot, multiCdot, Complex, decltype(x), decltype(y)>(result_tmp, x, y, x, x, false);
      }

      // do a single multi-node reduction only once we have computed all local dot products
      comm_allreduce_sum(result_tmp);

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      result = transpose(result_tmp, y.size(), x.size());
    }

    void hDotProduct(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block dot product on non-square inputs");

      std::vector<Complex> result_tmp(x.size() * y.size(), 0.0);
      TileSizeTune<multiCdot, multiCdot, Complex, decltype(x), decltype(y)>(result_tmp, x, y, x, x, true, false); // last false is b/c L2 norm

      // do a single multi-node reduction only once we have computed all local dot products
      comm_allreduce_sum(result_tmp); // FIXME - could optimize this for Hermiticity as well

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      result = transpose(result_tmp, y.size(), x.size());

      // we have only computed result on upper block trinagular part
      // so copy over to lower block tringaular
      for (auto i = 0u; i < x.size(); i++)
        for (auto j = 0u; j < i; j++)
          result[i * y.size() + j] = conj(result[j * x.size() + i]);
    }

    // for (p, Ap) norms in CG which are Hermitian.
    void hDotProduct_Anorm(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                     cvector_ref<const ColorSpinorField> &y)
    {
      if (x.size() == 0 || y.size() == 0) errorQuda("vector.size() == 0");
      if (x.size() != y.size()) errorQuda("Cannot call Hermitian block A-norm dot product on non-square inputs");

      std::vector<Complex> result_tmp(x.size() * y.size(), 0.0);
      TileSizeTune<multiCdot, multiCdot, Complex, decltype(x), decltype(y)>(result_tmp, x, y, x, x, true, true); // last true is b/c A norm

      // do a single multi-node reduction only once we have computed all local dot products
      comm_allreduce_sum(result_tmp);

      // multiReduce_recurse returns a column-major matrix.
      // To be consistent with the multi-blas functions, we should
      // switch this to row-major.
      result = transpose(result_tmp, y.size(), x.size());

      // we have only computed result on upper block trinagular part
      // so copy over to lower block tringaular
      for (auto i = 0u; i < x.size(); i++)
        for (auto j = 0u; j < i; j++)
          result[i * y.size() + j] = conj(result[j * x.size() + i]);
    }

    void reDotProduct(double *result, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<double> result_(x.size() * y.size());
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<const ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      reDotProduct(result_, std::move(x_), std::move(y_));
      memcpy(result, result_.data(), x.size() * y.size() * sizeof(double));
    }

    void cDotProduct(Complex *result, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<Complex> result_(x.size() * y.size());
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<const ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      cDotProduct(result_, std::move(x_), std::move(y_));
      memcpy(result, result_.data(), x.size() * y.size() * sizeof(Complex));
    }

    void hDotProduct(Complex *result, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<Complex> result_(x.size() * y.size());
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<const ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      hDotProduct(result_, std::move(x_), std::move(y_));
      memcpy(result, result_.data(), x.size() * y.size() * sizeof(Complex));
    }

  } // namespace blas

} // namespace quda
