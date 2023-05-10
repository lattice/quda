#include <blas_quda.h>
#include <color_spinor_field.h>
#include <kernels/multi_blas_core.cuh>
#include <tunable_nd.h>

namespace quda {

  namespace blas {

    template <template <typename ...> class Functor, typename store_t, typename y_store_t, int nSpin, typename T>
    class MultiBlas : public TunableGridStrideKernel3D
    {
      using real = typename mapper<y_store_t>::type;
      const int NXZ;
      const int NYW;
      Functor<real> f;
      int max_warp_split;
      mutable int warp_split; // helper used to keep track of current warp splitting
      const int nParity;
      const T &a, &b, &c;
      cvector_ref<ColorSpinorField> &x, &y, &z, &w;

      bool tuneSharedBytes() const override { return false; }

      // for these streaming kernels, there is no need to tune the grid size, just use max
      unsigned int minGridSize() const override { return maxGridSize(); }

    public:
      template <typename Vx, typename Vy, typename Vz, typename Vw>
      MultiBlas(const T &a, const T &b, const T &c, const ColorSpinorField &x0, const ColorSpinorField &y0,
                Vx &x, Vy &y, Vz &z, Vw &w) :
        TunableGridStrideKernel3D(x0, y.size(), x0.SiteSubset()),
        NXZ(x.size()),
        NYW(y.size()),
        f(NXZ, NYW),
        warp_split(1),
        nParity(x0.SiteSubset()),
        a(a),
        b(b),
        c(c),
        x(reinterpret_cast<cvector_ref<ColorSpinorField>&>(x)),
        y(reinterpret_cast<cvector_ref<ColorSpinorField>&>(y)),
        z(reinterpret_cast<cvector_ref<ColorSpinorField>&>(z)),
        w(reinterpret_cast<cvector_ref<ColorSpinorField>&>(w))
      {
        checkLocation(x[0], y[0], z[0], w[0]);
        checkLength(x[0], y[0], z[0], w[0]);
        auto x_prec = checkPrecision(x[0], z[0], w[0]);
        auto y_prec = y0.Precision();
        auto x_order = checkOrder(x[0], z[0], w[0]);
        auto y_order = y0.FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        // heuristic for enabling if we need the warp-splitting optimization
        const int gpu_size = 2 * device::max_threads_per_block() * device::processor_count();
        switch (gpu_size / (x0.Length() * NYW)) {
        case 0: max_warp_split = 1; break; // we have plenty of work, no need to split
        case 1: max_warp_split = 2; break; // double the thread count
        case 2:                            // quadruple the thread count
        default: max_warp_split = 4;
        }
        max_warp_split = std::min(NXZ, max_warp_split); // ensure we only split if valid

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

#ifdef QUDA_FAST_COMPILE_REDUCE
        strcat(aux, ",fast_compile");
#endif

        apply(device::get_default_stream());

        blas::bytes += bytes();
        blas::flops += flops();
      }

      TuneKey tuneKey() const override { return TuneKey(vol, typeid(f).name(), aux); }

      template <typename Arg> void Launch(const TuneParam &tp, const qudaStream_t &stream, Arg &&arg)
      {
        constexpr bool multi_1d = Arg::Functor::multi_1d;
        if (a.size()) { set_param<multi_1d>(arg, 'a', a); }
        if (b.size()) { set_param<multi_1d>(arg, 'b', b); }
        if (c.size()) { set_param<multi_1d>(arg, 'c', c); }
        launch<MultiBlas_>(tp, stream, arg);
      }

      template <int NXZ> void compute(const qudaStream_t &stream)
      {
        staticCheck<NXZ, store_t, y_store_t, decltype(f)>(f, x, y);

        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value;
        if (site_unroll_check && (x[0].Ncolor() != 3 || x[0].Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x[0].Nspin(), x[0].Ncolor());

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(x[0], y[0], z[0], w[0]); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Functor<device_real_t> f_(NXZ, NYW);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x[0].Length() / (nParity * M);

          if (tp.aux.x > 1 && (length * tp.aux.x) % device::warp_size() != 0) {
            // if problem size isn't divisible by the warp size then we can't use warp splitting
            launchError() = QUDA_ERROR;
          } else {
            tp.block.x *= tp.aux.x; // include warp-split factor
            switch (tp.aux.x) {
            case 1:
              Launch(tp, stream, MultiBlasArg<1, device_real_t, M, NXZ, device_store_t, N,
                     device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
              break;
            case 2:
              if constexpr (enable_warp_split()) {
                Launch(tp, stream, MultiBlasArg<2, device_real_t, M, NXZ, device_store_t, N,
                       device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
                break;
              }
            case 4:
              if constexpr (enable_warp_split()) {
                Launch(tp, stream, MultiBlasArg<4, device_real_t, M, NXZ, device_store_t, N,
                       device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
                break;
              }
            default: errorQuda("warp-split factor %d not instantiated", static_cast<int>(tp.aux.x));
            }

            tp.block.x /= tp.aux.x; // restore block size
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

      // instantiate the loop unrolling template
      template <int NXZ_max> std::enable_if_t<NXZ_max!=1, void> instantiate(const qudaStream_t &stream)
      {
        // if multi-1d then constrain the templates to no larger than max-1d size
        constexpr auto max_nxz_pow2 = max_NXZ_power2(false, static_cast<QudaPrecision>(sizeof(y_store_t)));
        constexpr auto pow2_max = !decltype(f)::multi_1d ? max_nxz_pow2 : std::min(max_N_multi_1d_pow2(), max_nxz_pow2);
        constexpr auto linear_max = !decltype(f)::multi_1d ? MAX_MULTI_BLAS_N : std::min(max_N_multi_1d(), MAX_MULTI_BLAS_N);

        if (NXZ <= pow2_max && is_power2(NXZ)) instantiatePow2<pow2_max>(stream);
        else if (NXZ <= linear_max) instantiateLinear<linear_max>(stream);
        else errorQuda("x.size %lu greater than maximum supported size (pow2 = %d, linear = %d)", x.size(), pow2_max, linear_max);
      }

      template <int NXZ_max> std::enable_if_t<NXZ_max==1, void> instantiate(const qudaStream_t &stream)
      {
        compute<1>(stream);
      }

      void apply(const qudaStream_t &stream) override { instantiate<decltype(f)::NXZ_max>(stream); }

      void preTune() override
      {
        for (int i = 0; i < NYW; ++i) {
          if (f.write.Y) y[i].backup();
          if (f.write.W) w[i].backup();
        }
      }

      void postTune() override
      {
        for (int i = 0; i < NYW; ++i) {
          if (f.write.Y) y[i].restore();
          if (f.write.W) w[i].restore();
        }
      }

      bool advanceAux(TuneParam &param) const override
      {
        if (enable_warp_split()) {
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
        } else {
          warp_split = 1;
          return false;
        }
      }

      int blockStep() const override { return device::warp_size() / warp_split; }
      int blockMin() const override { return device::warp_size() / warp_split; }

      void initTuneParam(TuneParam &param) const override
      {
        TunableGridStrideKernel3D::initTuneParam(param);
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      void defaultTuneParam(TuneParam &param) const override
      {
        TunableGridStrideKernel3D::defaultTuneParam(param);
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      long long flops() const override
      {
        return NYW * NXZ * f.flops() * x[0].Length();
      }

      long long bytes() const override
      {
        // X and Z reads are repeated (and hopefully cached) across NYW
        // each Y and W read/write is done once
        return NYW * NXZ * (f.read.X + f.write.X) * x[0].Bytes() +
          NYW * (f.read.Y + f.write.Y) * y[0].Bytes() +
          NYW * NXZ * (f.read.Z + f.write.Z) * z[0].Bytes() +
          NYW * (f.read.W + f.write.W) * w[0].Bytes();
      }
    };

    using range = std::pair<size_t,size_t>;

    template <template <typename...> class Functor, typename T>
    void axpy_recurse(const std::vector<T> &a, cvector_ref<const ColorSpinorField> &x,
                      cvector_ref<ColorSpinorField> &y,
                      const range &range_x, const range &range_y, int upper)
    {
      if (a.size() != x.size() * y.size())
        errorQuda("coefficient size %lu does not match vector set %lu * %lu", a.size(), x.size(), y.size());

      // if greater than max single-kernel size, recurse
      size_t max_yw_size = y[0].Precision() == QUDA_DOUBLE_PRECISION ?
        max_YW_size<Functor<double>>(x.size(), x[0].Precision(), y[0].Precision()) :
        max_YW_size<Functor<float>>(x.size(), x[0].Precision(), y[0].Precision());

      if (y.size() > max_yw_size) {
        // We need to split up 'a' carefully since it's row-major.
        auto a_ = bisect_col(a, x.size(), y.size() / 2, y.size() - y.size() / 2);
        auto y_ = bisect(y);

        axpy_recurse<Functor>(a_.first, x, y_.first, range_x, range(range_y.first, range_y.first + y_.first.size()), upper);
        axpy_recurse<Functor>(a_.second, x, y_.second, range_x, range(range_y.first + y_.first.size(), range_y.second), upper);
      } else {
        // if at the bottom of recursion,
        if (is_valid_NXZ(x.size(), false, y[0].Precision())) {
          // since tile range is [first,second), e.g., [first,second-1], we need >= here
          // if upper triangular and upper-right tile corner is below diagonal return
          if (upper == 1 && range_y.first >= range_x.second) { return; }
          // if lower triangular and lower-left tile corner is above diagonal return
          if (upper == -1 && range_x.first >= range_y.second) { return; }

          // mark true since we will copy the "a" matrix into constant memory
          constexpr bool mixed = true;
          instantiate<Functor, MultiBlas, mixed>(a, std::vector<T>(), std::vector<T>(), x[0], y[0], x, y, x, x);
        } else {
          // split the problem in half and recurse
          auto x_ = bisect(x);
          auto a_ = bisect(a, y.size() * (x.size() / 2));

          axpy_recurse<Functor>(a_.first, x_.first, y, range(range_x.first, range_x.first + x_.first.size()), range_y, upper);
          axpy_recurse<Functor>(a_.second, x_.second, y, range(range_x.first + x_.first.size(), range_x.second), range_y, upper);
        }
      } // end if (y.size() > max_YW_size())
    }

    template <>
    void axpy<double>(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multiaxpy_>(a, x, y, range(0,x.size()), range(0,y.size()), 0);
    }

    template <>
    void axpy_U<double>(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block axpy_U with non-square 'a' (%lu != %lu) has not yet been implemented. Use block axpy instead",
                  x.size(), y.size());
      }
      axpy_recurse<multiaxpy_>(a, x, y, range(0, x.size()), range(0, y.size()), 1);
    }

    template <>
    void axpy_L<double>(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block axpy_L with non-square 'a' (%lu != %lu) has not yet been implemented. Use block axpy instead",
                  x.size(), y.size());
      }
      axpy_recurse<multiaxpy_>(a, x, y, range(0, x.size()), range(0, y.size()), -1);
    }

    template <>
    void axpy<Complex>(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multicaxpy_>(a, x, y, range(0,x.size()), range(0,y.size()), 0);
    }

    void caxpy(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      axpy(a, std::move(x), std::move(y));
    }

    template <>
    void axpy_U<Complex>(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size()) {
        errorQuda("An optimal block caxpy_U with non-square 'a' (%lu != %lu) has not yet been implemented. Use block caxpy instead",
                  x.size(), y.size());
      }
      axpy_recurse<multicaxpy_>(a, x, y, range(0,x.size()), range(0,y.size()), 1);
    }

    void caxpy_U(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      axpy_U(a, std::move(x), std::move(y));
    }

    template <>
    void axpy_L<Complex>(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size()) {
        errorQuda("An optimal block caxpy_L with non-square 'a' (%lu != %lu) has not yet been implemented. Use block caxpy instead",
                  x.size(), y.size());
      }
      axpy_recurse<multicaxpy_>(a, x, y, range(0,x.size()), range(0,y.size()), -1);
    }

    void caxpy_L(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y)
    {
      axpy_L(a, std::move(x), std::move(y));
    }

    template <template <typename...> class Functor, typename T>
    void axpyz_recurse(const std::vector<T> &a, cvector_ref<const ColorSpinorField> &x,
                       cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                       const range &range_x, const range &range_y, int pass, int upper)
    {
      if (a.size() != x.size() * y.size())
        errorQuda("coefficient size %lu does not match vector set %lu * %lu", a.size(), x.size(), y.size());

      // if greater than max single-kernel size, recurse
      size_t max_yw_size = y[0].Precision() == QUDA_DOUBLE_PRECISION ?
        max_YW_size<Functor<double>>(x.size(), x[0].Precision(), y[0].Precision()) :
        max_YW_size<Functor<float>>(x.size(), x[0].Precision(), y[0].Precision());

      if (y.size() > max_yw_size) {
        // We need to split up 'a' carefully since it's row-major.
        auto a_ = bisect_col(a, x.size(), y.size() / 2, y.size() - y.size() / 2);
        auto y_ = bisect(y);
        auto z_ = bisect(z);

        axpyz_recurse<Functor>(a_.first, x, y_.first, z_.first, range_x, range(range_y.first, range_y.first + y_.first.size()), pass, upper);
        axpyz_recurse<Functor>(a_.second, x, y_.second, z_.second, range_x, range(range_y.first + y_.first.size(), range_y.second), pass, upper);
      } else {
        // if at bottom of recursion check where we are
        if (is_valid_NXZ(x.size(), false, y[0].Precision())) {
          // check if tile straddles diagonal for L/U variants
          bool is_diagonal = (upper != 0) && (range_x.first < range_y.second) && (range_y.first < range_x.second);
          // check if tile is first to be updated for full matrices
          bool is_first = (upper == 0) && (range_x.first == 0);
          // whether to do axpyz
          bool do_axpyz = (upper != 0 && is_diagonal && pass == 0) || (upper == 0 && is_first);
          // whether to do axpy
          bool do_axpy = (upper != 0 && !is_diagonal && pass == 1) || (upper == 0 && !is_first);

          if (do_axpyz) {
            constexpr bool mixed = false;
            instantiate<Functor, MultiBlas, mixed>(a, std::vector<T>(), std::vector<T>(), x[0], y[0], x, y, x, z);
          } else if (do_axpy) {
            // if upper triangular and upper-right tile corner is below diagonal return
            if (upper == 1 && range_y.first >= range_x.second) { return; }
            // if lower triangular and lower-left tile corner is above diagonal return
            if (upper == -1 && range_x.first >= range_y.second) { return; }

            // off diagonal
            axpy(a, x, z);
          }
        } else {
          // split the problem in half and recurse
          auto x_ = bisect(x);
          auto a_ = bisect(a, y.size() * (x.size() / 2));

          axpyz_recurse<Functor>(a_.first, x_.first, y, z, range(range_x.first, range_x.first + x_.first.size()), range_y, pass, upper);
          axpyz_recurse<Functor>(a_.second, x_.second, y, z, range(range_x.first + x_.first.size(), range_x.second), range_y, pass, upper);
        }
      } // end if (y.size() > max_YW_size())
    }

    void axpyz(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      axpyz_recurse<multiaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 0);
    }

    void axpyz_U(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if (x.size() != y.size()) {
        errorQuda("An optimal block axpyz_U with non-square 'a' (%lu != %lu) has not yet been implemented. Use block axpyz instead",
                  x.size(), y.size());
      }
      // a is upper triangular.
      // first pass does the axpyz on the diagonal
      axpyz_recurse<multiaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 1);
      // second pass does axpy on the off diagonals
      axpyz_recurse<multiaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 1);
    }

    void axpyz_L(const std::vector<double> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if (x.size() != y.size()) {
        errorQuda("An optimal block axpyz_L with non-square 'a' (%lu != %lu) has not yet been implemented. Use block axpyz instead",
                  x.size(), y.size());
      }
      // a is upper triangular.
      // first pass does the axpyz on the diagonal
      axpyz_recurse<multiaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, -1);
      // second pass does axpy on the off diagonals
      axpyz_recurse<multiaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, -1);
    }

    void caxpyz(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      axpyz_recurse<multicaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 0);
    }

    void caxpyz_U(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if (x.size() != y.size()) {
        errorQuda("An optimal block caxpyz_U with non-square 'a' (%lu != %lu) has not yet been implemented. Use block caxpyz instead",
                  x.size(), y.size());
      }
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      axpyz_recurse<multicaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 1);
      // second pass does caxpy on the off diagonals
      axpyz_recurse<multicaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 1);
    }

    void axpyz_L(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z)
    {
      if (x.size() != y.size()) {
        errorQuda("An optimal block axpyz_U with non-square 'a' (%lu != %lu) has not yet been implemented. Use block axpyz instead",
                  x.size(), y.size());
      }
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      axpyz_recurse<multicaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, -1);
      // second pass does caxpy on the off diagonals
      axpyz_recurse<multicaxpyz_>(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, -1);
    }

    void axpyBzpcx(const std::vector<double> &a, cvector_ref<ColorSpinorField> &x_, cvector_ref<ColorSpinorField> &y_,
                   const std::vector<double> &b, ColorSpinorField &z_, const std::vector<double> &c)
    {
      if (y_.size() <= (size_t)max_N_multi_1d()) {
        // swizzle order since we are writing to x_ and y_, but the
	// multi-blas only allow writing to y and w, and moreover the
	// block width of y and w must match, and x and z must match.
	auto &y = y_;
	auto &w = x_;

	// wrap a container around the third solo vector
	cvector_ref<ColorSpinorField> x{z_};

        constexpr bool mixed = true;
        instantiate<multi_axpyBzpcx_, MultiBlas, mixed>(a, b, c, x[0], y[0], x, y, x, w);
      } else {
        // split the problem in half and recurse
        auto a_ = bisect(a);
        auto b_ = bisect(b);
        auto c_ = bisect(c);
        auto x = bisect(x_);
        auto y = bisect(y_);

	axpyBzpcx(a_.first, x.first, y.first, b_.first, z_, c_.first);
	axpyBzpcx(a_.second, x.second, y.second, b_.second, z_, c_.second);
      }
    }

    void caxpyBxpz(const std::vector<Complex> &a, cvector_ref<const ColorSpinorField> &x_, ColorSpinorField &y_,
                   const std::vector<Complex> &b, ColorSpinorField &z_)
    {
      if (x_.size() <= (size_t)max_N_multi_1d() && is_valid_NXZ(x_.size(), false, y_.Precision())) // only split if we have to.
      {
        // swizzle order since we are writing to y_ and z_, but the
        // multi-blas only allow writing to y and w, and moreover the
        // block width of y and w must match, and x and z must match.
        // Also, wrap a container around them.
        cvector_ref<ColorSpinorField> y{y_};
        cvector_ref<ColorSpinorField> w{z_};

        // we're reading from x
        auto &x = x_;

        constexpr bool mixed = true;
        instantiate<multi_caxpyBxpz_, MultiBlas, mixed>(a, b, std::vector<Complex>(), x[0], y[0], x, y, x, w);
      } else {
        // split the problem in half and recurse
        auto a_ = bisect(a);
        auto b_ = bisect(b);
        auto x = bisect(x_);

        caxpyBxpz(a_.first, x.first, y_, b_.first, z_);
        caxpyBxpz(a_.second, x.second, y_, b_.second, z_);
      }
    }

    // temporary wrappers
    void axpy(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<double> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(double));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      axpy(a_, x_, y_);
    }

    void axpy_U(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<double> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(double));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      axpy_U(a_, x_, y_);
    }

    void axpy_L(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      std::vector<double> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(double));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      axpy_L(a_, x_, y_);
    }

    void caxpy(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      std::vector<Complex> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(Complex));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      caxpy(a_, x_, y_);
    }

    void caxpy_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      std::vector<Complex> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(Complex));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      caxpy_U(a_, x_, y_);
    }

    void caxpy_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      std::vector<Complex> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(Complex));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      caxpy_L(a_, x_, y_);
    }

    void axpyz(const double *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
               std::vector<ColorSpinorField*> &z)
    {
      std::vector<double> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(double));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<const ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      vector_ref<ColorSpinorField> z_;
      for (auto &zi : z) z_.push_back(*zi);
      axpyz(a_, x_, y_, z_);
    }

    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
               std::vector<ColorSpinorField*> &z)
    {
      std::vector<Complex> a_(x.size() * y.size());
      memcpy(a_.data(), a, x.size() * y.size() * sizeof(Complex));
      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<const ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      vector_ref<ColorSpinorField> z_;
      for (auto &zi : z) z_.push_back(*zi);
      caxpyz(a_, x_, y_, z_);
    }

    void axpyBzpcx(const double *a, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                   const double *b, ColorSpinorField &z, const double *c)
    {
      std::vector<double> a_(x.size());
      memcpy(a_.data(), a, x.size() * sizeof(double));
      std::vector<double> b_(x.size());
      memcpy(b_.data(), b, x.size() * sizeof(double));
      std::vector<double> c_(x.size());
      memcpy(c_.data(), c, x.size() * sizeof(double));

      vector_ref<ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      vector_ref<ColorSpinorField> y_;
      for (auto &yi : y) y_.push_back(*yi);
      axpyBzpcx(a_, x_, y_, b_, z, c_);
    }

    void caxpyBxpz(const Complex *a, std::vector<ColorSpinorField*> &x, ColorSpinorField &y,
		   const Complex *b, ColorSpinorField &z)
    {
      std::vector<Complex> a_(x.size());
      memcpy(a_.data(), a, x.size() * sizeof(Complex));
      std::vector<Complex> b_(x.size());
      memcpy(b_.data(), b, x.size() * sizeof(Complex));

      vector_ref<const ColorSpinorField> x_;
      for (auto &xi : x) x_.push_back(*xi);
      caxpyBxpz(a_, x_, y, b_, z);
    }

    // Composite field version
    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y){ caxpy(a, x.Components(), y.Components()); }
    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_U(a, x.Components(), y.Components()); }
    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_L(a, x.Components(), y.Components()); }

    void axpy(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy(a, x.Components(), y.Components()); }
    void axpy_U(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_U(a, x.Components(), y.Components()); }
    void axpy_L(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_L(a, x.Components(), y.Components()); }

    void axpyz(const double *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      axpyz(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      caxpyz(a, x.Components(), y.Components(), z.Components());
    }

  } // namespace blas

} // namespace quda
