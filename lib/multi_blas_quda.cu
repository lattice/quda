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
      std::vector<ColorSpinorField *> &x, &y, &z, &w;

      bool tuneSharedBytes() const { return false; }

      // for these streaming kernels, there is no need to tune the grid size, just use max
      unsigned int minGridSize() const { return maxGridSize(); }

    public:
      MultiBlas(const T &a, const T &b, const T &c, const ColorSpinorField &, const ColorSpinorField &,
                std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                std::vector<ColorSpinorField *> &z, std::vector<ColorSpinorField *> &w) :
        TunableGridStrideKernel3D(*x[0], y.size(), x[0]->SiteSubset()),
        NXZ(x.size()),
        NYW(y.size()),
        f(NXZ, NYW),
        warp_split(1),
        nParity(x[0]->SiteSubset()),
        a(a),
        b(b),
        c(c),
        x(x),
        y(y),
        z(z),
        w(w)
      {
        checkLocation(*x[0], *y[0], *z[0], *w[0]);
        checkLength(*x[0], *y[0], *z[0], *w[0]);
        auto x_prec = checkPrecision(*x[0], *z[0], *w[0]);
        auto y_prec = y[0]->Precision();
        auto x_order = checkOrder(*x[0], *z[0], *w[0]);
        auto y_order = y[0]->FieldOrder();
        if (sizeof(store_t) != x_prec) errorQuda("Expected precision %lu but received %d", sizeof(store_t), x_prec);
        if (sizeof(y_store_t) != y_prec) errorQuda("Expected precision %lu but received %d", sizeof(y_store_t), y_prec);
        if (x_prec == y_prec && x_order != y_order) errorQuda("Orders %d %d do not match", x_order, y_order);

        // heuristic for enabling if we need the warp-splitting optimization
        const int gpu_size = 2 * device::max_threads_per_block() * device::processor_count();
        switch (gpu_size / (x[0]->Length() * NYW)) {
        case 0: max_warp_split = 1; break; // we have plenty of work, no need to split
        case 1: max_warp_split = 2; break; // double the thread count
        case 2:                            // quadruple the thread count
        default: max_warp_split = 4;
        }
        max_warp_split = std::min(NXZ, max_warp_split); // ensure we only split if valid

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

#ifdef QUDA_FAST_COMPILE_REDUCE
        strcat(aux, ",fast_compile");
#endif

        apply(device::get_default_stream());

        blas::bytes += bytes();
        blas::flops += flops();
      }

      TuneKey tuneKey() const { return TuneKey(vol, typeid(f).name(), aux); }

      template <typename Arg> void Launch(const TuneParam &tp, const qudaStream_t &stream, Arg &&arg)
      {
        constexpr bool multi_1d = Arg::Functor::multi_1d;
        if (a.data) { set_param<multi_1d>(arg, 'a', a); }
        if (b.data) { set_param<multi_1d>(arg, 'b', b); }
        if (c.data) { set_param<multi_1d>(arg, 'c', c); }
        launch<MultiBlas_>(tp, stream, arg);
      }

      template <int NXZ> void compute(const qudaStream_t &stream)
      {
        staticCheck<NXZ, store_t, y_store_t, decltype(f)>(f, x, y);

        constexpr bool site_unroll_check = !std::is_same<store_t, y_store_t>::value || isFixed<store_t>::value;
        if (site_unroll_check && (x[0]->Ncolor() != 3 || x[0]->Nspin() == 2))
          errorQuda("site unroll not supported for nSpin = %d nColor = %d", x[0]->Nspin(), x[0]->Ncolor());

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (location == QUDA_CUDA_FIELD_LOCATION) {
          if (site_unroll_check) checkNative(*x[0], *y[0], *z[0], *w[0]); // require native order when using site_unroll
          using device_store_t = typename device_type_mapper<store_t>::type;
          using device_y_store_t = typename device_type_mapper<y_store_t>::type;
          using device_real_t = typename mapper<device_y_store_t>::type;
          Functor<device_real_t> f_(NXZ, NYW);

          // redefine site_unroll with device_store types to ensure we have correct N/Ny/M values
          constexpr bool site_unroll = !std::is_same<device_store_t, device_y_store_t>::value || isFixed<device_store_t>::value;
          constexpr int N = n_vector<device_store_t, true, nSpin, site_unroll>();
          constexpr int Ny = n_vector<device_y_store_t, true, nSpin, site_unroll>();
          constexpr int M = site_unroll ? (nSpin == 4 ? 24 : 6) : N; // real numbers per thread
          const int length = x[0]->Length() / (nParity * M);

          tp.block.x *= tp.aux.x; // include warp-split factor

          switch (tp.aux.x) {
          case 1:
            Launch(tp, stream, MultiBlasArg<1, device_real_t, M, NXZ, device_store_t, N,
                   device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
            break;
#ifdef WARP_SPLIT
          case 2:
            Launch(tp, stream, MultiBlasArg<2, device_real_t, M, NXZ, device_store_t, N,
                   device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
            break;
          case 4:
            Launch(tp, stream, MultiBlasArg<4, device_real_t, M, NXZ, device_store_t, N,
                   device_y_store_t, Ny, decltype(f_)>(x, y, z, w, f_, NYW, length));
            break;
#endif
          default: errorQuda("warp-split factor %d not instantiated", tp.aux.x);
          }

          tp.block.x /= tp.aux.x; // restore block size
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
        constexpr int pow2_max = !decltype(f)::multi_1d ? max_NXZ_power2<false, isFixed<store_t>::value>() :
          std::min(max_N_multi_1d(), max_NXZ_power2<false, isFixed<store_t>::value>());
        constexpr int linear_max = !decltype(f)::multi_1d ? MAX_MULTI_BLAS_N : std::min(max_N_multi_1d(), MAX_MULTI_BLAS_N);

        if (NXZ <= pow2_max && is_power2(NXZ)) instantiatePow2<pow2_max>(stream);
        else if (NXZ <= linear_max) instantiateLinear<linear_max>(stream);
        else errorQuda("x.size %lu greater than maximum supported size (pow2 = %d, linear = %d)", x.size(), pow2_max, linear_max);
      }

      template <int NXZ_max> std::enable_if_t<NXZ_max==1, void> instantiate(const qudaStream_t &stream)
      {
        compute<1>(stream);
      }

      void apply(const qudaStream_t &stream) { instantiate<decltype(f)::NXZ_max>(stream); }

      void preTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (f.write.X) x[i]->backup();
          if (f.write.Y) y[i]->backup();
          if (f.write.Z) z[i]->backup();
          if (f.write.W) w[i]->backup();
        }
      }

      void postTune()
      {
        for (int i = 0; i < NYW; ++i) {
          if (f.write.X) x[i]->restore();
          if (f.write.Y) y[i]->restore();
          if (f.write.Z) z[i]->restore();
          if (f.write.W) w[i]->restore();
        }
      }

#ifdef WARP_SPLIT
      bool advanceAux(TuneParam &param) const
      {
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
      }
#else
      bool advanceAux(TuneParam &) const
      {
        warp_split = 1;
        return false;
      }
#endif

      int blockStep() const { return device::warp_size() / warp_split; }
      int blockMin() const { return device::warp_size() / warp_split; }

      void initTuneParam(TuneParam &param) const
      {
        TunableGridStrideKernel3D::initTuneParam(param);
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      void defaultTuneParam(TuneParam &param) const
      {
        TunableGridStrideKernel3D::defaultTuneParam(param);
        param.aux = make_int4(1, 0, 0, 0); // warp-split parameter
      }

      long long flops() const
      {
        return NYW * NXZ * f.flops() * x[0]->Length();
      }

      long long bytes() const
      {
        // X and Z reads are repeated (and hopefully cached) across NYW
        // each Y and W read/write is done once
        return NYW * NXZ * (f.read.X + f.write.X) * x[0]->Bytes() +
          NYW * (f.read.Y + f.write.Y) * y[0]->Bytes() +
          NYW * NXZ * (f.read.Z + f.write.Z) * z[0]->Bytes() +
          NYW * (f.read.W + f.write.W) * w[0]->Bytes();
      }

      int tuningIter() const { return 3; }
    };

    using range = std::pair<size_t,size_t>;

    template <template <typename...> class Functor, typename T>
    void axpy_recurse(const T *a_, std::vector<ColorSpinorField *> &x, std::vector<ColorSpinorField *> &y,
                      const range &range_x, const range &range_y, int upper, int coeff_width)
    {
      // if greater than max single-kernel size, recurse
      size_t max_yw_size = y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
        max_YW_size<Functor<double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
        max_YW_size<Functor<float>>(x.size(), x[0]->Precision(), y[0]->Precision());

      if (y.size() > max_yw_size) {
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

        axpy_recurse<Functor>(tmpmajor0, x, y0, range_x, range(range_y.first, range_y.first + y0.size()), upper, coeff_width);
        axpy_recurse<Functor>(tmpmajor1, x, y1, range_x, range(range_y.first + y0.size(), range_y.second), upper, coeff_width);

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
          constexpr bool mixed = true;
          instantiate<Functor, MultiBlas, mixed>(a, b, c, *x[0], *y[0], x, y, x, x);
        } else {
          // split the problem in half and recurse
          const T *a0 = &a_[0];
          const T *a1 = &a_[(x.size() / 2) * y.size()];

          std::vector<ColorSpinorField *> x0(x.begin(), x.begin() + x.size() / 2);
          std::vector<ColorSpinorField *> x1(x.begin() + x.size() / 2, x.end());

          axpy_recurse<Functor>(a0, x0, y, range(range_x.first, range_x.first + x0.size()), range_y, upper, coeff_width);
          axpy_recurse<Functor>(a1, x1, y, range(range_x.first + x0.size(), range_x.second), range_y, upper, coeff_width);
        }
      } // end if (y.size() > max_YW_size())
    }

    void caxpy(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), 0, 2);
    }

    void caxpy_U(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_U with non-square 'a' has not yet been implemented. Use block caxpy instead");
      }
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), 1, 2);
    }

    void caxpy_L(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y) {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block caxpy_L with non-square 'a' has not yet been implemented. Use block caxpy instead");
      }
      axpy_recurse<multicaxpy_>(a_, x, y, range(0,x.size()), range(0,y.size()), -1, 2);
    }

    void caxpy(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy(a, x.Components(), y.Components()); }

    void caxpy_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_U(a, x.Components(), y.Components()); }

    void caxpy_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y) { caxpy_L(a, x.Components(), y.Components()); }

    void caxpyz_recurse(const Complex *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y,
                        std::vector<ColorSpinorField*> &z, const range &range_x, const range &range_y,
                        int pass, int upper)
    {
      // if greater than max single-kernel size, recurse
      size_t max_yw_size = y[0]->Precision() == QUDA_DOUBLE_PRECISION ?
        max_YW_size<multicaxpyz_<double>>(x.size(), x[0]->Precision(), y[0]->Precision()) :
        max_YW_size<multicaxpyz_<float>>(x.size(), x[0]->Precision(), y[0]->Precision());

      if (y.size() > max_yw_size) {
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
          constexpr bool mixed = false;
          instantiate<multicaxpyz_, MultiBlas, mixed>(a, b, c, *x[0], *y[0], x, y, x, z);
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

    void caxpyz(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z)
    {
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 0);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 0);
    }

    void caxpyz_U(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z)
    {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, 1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, 1);
    }

    void caxpyz_L(const Complex *a, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y, std::vector<ColorSpinorField*> &z)
    {
      // a is upper triangular.
      // first pass does the caxpyz on the diagonal
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 0, -1);
      // second pass does caxpy on the off diagonals
      caxpyz_recurse(a, x, y, z, range(0, x.size()), range(0, y.size()), 1, -1);
    }


    void caxpyz(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      caxpyz(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_U(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      caxpyz_U(a, x.Components(), y.Components(), z.Components());
    }

    void caxpyz_L(const Complex *a, ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z)
    {
      caxpyz_L(a, x.Components(), y.Components(), z.Components());
    }

    void axpyBzpcx(const double *a_, std::vector<ColorSpinorField *> &x_, std::vector<ColorSpinorField *> &y_,
                   const double *b_, ColorSpinorField &z_, const double *c_)
    {
      if (y_.size() <= (size_t)max_N_multi_1d()) {
        // swizzle order since we are writing to x_ and y_, but the
	// multi-blas only allow writing to y and w, and moreover the
	// block width of y and w must match, and x and z must match.
	std::vector<ColorSpinorField*> &y = y_;
	std::vector<ColorSpinorField*> &w = x_;

	// wrap a container around the third solo vector
	std::vector<ColorSpinorField*> x;
	x.push_back(&z_);

        coeff_array<double> a(a_), b(b_), c(c_);
        constexpr bool mixed = true;
        instantiate<multi_axpyBzpcx_, MultiBlas, mixed>(a, b, c, *x[0], *y[0], x, y, x, w);
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
      if (x_.size() <= (size_t)max_N_multi_1d() &&
          is_valid_NXZ(x_.size(), false, x_[0]->Precision() < QUDA_SINGLE_PRECISION)) // only split if we have to.
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
        constexpr bool mixed = true;
        instantiate<multi_caxpyBxpz_, MultiBlas, mixed>(a, b, c, *x[0], *y[0], x, y, x, w);
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

    void axpy(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. false specifies the matrix is unstructured.
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), 0, 1);
    }

    void axpy_U(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. 1 indicates the matrix is upper-triangular,
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block axpy_U with non-square 'a' has not yet been implemented. Use block axpy instead");
      }
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), 1, 1);
    }

    void axpy_L(const double *a_, std::vector<ColorSpinorField*> &x, std::vector<ColorSpinorField*> &y)
    {
      // Enter a recursion.
      // Pass a, x, y. (0,0) indexes the tiles. -1 indicates the matrix is lower-triangular
      //                                         which lets us skip some tiles.
      if (x.size() != y.size())
      {
        errorQuda("An optimal block axpy_L with non-square 'a' has not yet been implemented. Use block axpy instead");
      }
      axpy_recurse<multiaxpy_>(a_, x, y, range(0, x.size()), range(0, y.size()), -1, 1);
    }

    // Composite field version
    void axpy(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy(a, x.Components(), y.Components()); }

    void axpy_U(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_U(a, x.Components(), y.Components()); }

    void axpy_L(const double *a, ColorSpinorField &x, ColorSpinorField &y) { axpy_L(a, x.Components(), y.Components()); }

  } // namespace blas

} // namespace quda
