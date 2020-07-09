#include <typeinfo>
#include <gauge_field.h>
#include <blas_cublas.h>
#include <blas_quda.h>
#include <tune_quda.h>

#include <jitify_helper.cuh>
#include <kernels/coarse_op_preconditioned.cuh>

#include <coarse_op_preconditioned_mma.cuh>

namespace quda {

  /**
     @brief Launcher for CPU instantiations of preconditioned coarse-link construction
  */
  template <QudaFieldLocation location, typename Arg>
  struct Launch {
    Launch(Arg &arg, CUresult &error, bool compute_max_only, TuneParam &tp, bool use_mma, const qudaStream_t &stream)
    {
      if (compute_max_only)
        CalculateYhatCPU<true, Arg>(arg);
      else
        CalculateYhatCPU<false, Arg>(arg);
    }
  };

  /**
     @brief Launcher for GPU instantiations of preconditioned coarse-link construction
  */
  template <typename Arg>
  struct Launch<QUDA_CUDA_FIELD_LOCATION, Arg> {
    Launch(Arg &arg, CUresult &error, bool compute_max_only, TuneParam &tp, bool use_mma, const qudaStream_t &stream)
    {
      if (compute_max_only) {
        if (!activeTuning()) {
          qudaMemsetAsync(arg.max_d, 0, sizeof(typename Arg::Float), stream);
        }
      }
#ifdef JITIFY
      if (use_mma) {
        errorQuda("MMA kernels haven't been jitify'ed.");
      } else {
        using namespace jitify::reflection;
        error = program->kernel("quda::CalculateYhatGPU")
          .instantiate(compute_max_only, Type<Arg>())
          .configure(tp.grid, tp.block, tp.shared_bytes, stream)
          .launch(arg);
      }
#else
      if (use_mma) {
        if (compute_max_only) {
          mma::launch_yhat_kernel<true>(arg, arg.Y.VolumeCB(), tp, stream);
        } else {
          mma::launch_yhat_kernel<false>(arg, arg.Y.VolumeCB(), tp, stream);
        }
      } else {
        if (compute_max_only) {
          CalculateYhatGPU<true, Arg><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          CalculateYhatGPU<false, Arg><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        }
      }
#endif
      if (compute_max_only) {
        if (!activeTuning()) { // only do copy once tuning is done
          qudaMemcpyAsync(arg.max_h, arg.max_d, sizeof(typename Arg::Float), cudaMemcpyDeviceToHost, stream);
          qudaStreamSynchronize(const_cast<qudaStream_t&>(stream));
        }
      }
    }
  };

  template <QudaFieldLocation location, typename Arg>
  class CalculateYhat : public TunableVectorYZ {

    using Float = typename Arg::Float;
    Arg &arg;
    const LatticeField &meta;
    const int n;

    bool compute_max_only;

    bool use_mma;

    long long flops() const { return 2l * arg.Y.VolumeCB() * 8 * n * n * (8*n-2); } // 8 from dir, 8 from complexity,
    long long bytes() const { return 2l * (arg.Xinv.Bytes() + 8*arg.Y.Bytes() + !compute_max_only * 8*arg.Yhat.Bytes()) * n; }

    unsigned int minThreads() const { return arg.Y.VolumeCB(); }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

    // all the tuning done is only in matrix tile size (Y/Z block.grid)
    int blockMin() const { return 8; }
    int blockStep() const { return 8; }
    unsigned int maxBlockSize(const TuneParam &param) const { return 8u; }

  public:
    CalculateYhat(Arg &arg, const LatticeField &meta, bool use_mma) :
      TunableVectorYZ(2 * arg.tile.M_tiles, 4 * arg.tile.N_tiles),
      arg(arg),
      meta(meta),
      n(arg.tile.n),
      compute_max_only(false),
      use_mma(use_mma)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/coarse_op_preconditioned.cuh");
#endif
        arg.max_d = static_cast<Float*>(pool_device_malloc(sizeof(Float)));
      }
      arg.max_h = static_cast<Float*>(pool_pinned_malloc(sizeof(Float)));
      strcpy(aux, compile_type_str(meta));
      strcat(aux, comm_dim_partitioned_string());
    }

    virtual ~CalculateYhat() {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        pool_device_free(arg.max_d);
      }
      pool_pinned_free(arg.max_h);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Launch<location, Arg>(arg, jitify_error, compute_max_only, tp, use_mma, stream);
    }

    /**
       Set if we're doing a max-only compute (fixed point only)
    */
    void setComputeMaxOnly(bool compute_max_only_) { compute_max_only = compute_max_only_; }

    bool advanceSharedBytes(TuneParam &param) const { return false; }

    bool advanceTuneParam(TuneParam &param) const {
      if (use_mma) {
        
        int max_aux;
        // clang-format off
        switch (arg.M) {
        case  48: max_aux = 2; break;
        case  64: max_aux = 6; break;
        case 128: max_aux = 7; break;
        case 192: max_aux = 4; break;
        default: errorQuda("Unsupported number of coarse dof %d\n", arg.M);
        }
        // clang-format on
        if (param.aux.x < max_aux) {
          param.aux.x++;
          return true;
        }
        return false;
      
      } else {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && meta.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
      }
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);
      if (compute_max_only) strcat(Aux, ",compute_max_only");
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        strcat(Aux, meta.MemType() == QUDA_MEMORY_MAPPED ? ",GPU-mapped" : ",GPU-device");
      } else if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        strcat(Aux, ",CPU");
        strcat(Aux, getOmpThreadStr());
      }
      if (use_mma) {
        strcat(Aux, ",MMA");
      }
      return TuneKey(meta.VolString(), typeid(*this).name(), Aux);
    }
  };

  /**
     @brief Calculate the preconditioned coarse-link field and the clover inverse.

     @param Yhat[out] Preconditioned coarse link field
     @param Xinv[out] Coarse clover inverse field
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
   */
  template<QudaFieldLocation location, typename storeFloat, typename Float, int N, QudaGaugeFieldOrder gOrder>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    // invert the clover matrix field
    const int n = X.Ncolor();
    if (X.Location() == QUDA_CUDA_FIELD_LOCATION && X.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      GaugeFieldParam param(X);
      // need to copy into AoS format for CUBLAS
      param.order = QUDA_MILC_GAUGE_ORDER;
      param.setPrecision( X.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : X.Precision() );
      cudaGaugeField X_(param);
      cudaGaugeField Xinv_(param);
      X_.copy(X);
      blas::flops += cublas::BatchInvertMatrix((void*)Xinv_.Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X.Location());

      if (Xinv.Precision() < QUDA_SINGLE_PRECISION) Xinv.Scale( Xinv_.abs_max() );

      Xinv.copy(Xinv_);

    } else if (X.Location() == QUDA_CPU_FIELD_LOCATION && X.Order() == QUDA_QDP_GAUGE_ORDER) {
      const cpuGaugeField *X_h = static_cast<const cpuGaugeField*>(&X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);
      blas::flops += cublas::BatchInvertMatrix(((void**)Xinv_h->Gauge_p())[0], ((void**)X_h->Gauge_p())[0], n, X_h->Volume(), X.Precision(), QUDA_CPU_FIELD_LOCATION);
    } else {
      errorQuda("Unsupported location=%d and order=%d", X.Location(), X.Order());
    }

    // now exchange Y halos of both forwards and backwards links for multi-process dslash
    const_cast<GaugeField&>(Y).exchangeGhost(QUDA_LINK_BIDIRECTIONAL);

    // compute the preconditioned links
    // Yhat_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
    // Yhat_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)
    {
      int xc_size[5];
      for (int i=0; i<4; i++) xc_size[i] = X.X()[i];
      xc_size[4] = 1;

      // TODO: use field order as the predicate.
      if (use_mma) {
      
        GaugeFieldParam param_Y(Y);
        GaugeFieldParam param_Yhat(Yhat);
        GaugeFieldParam param_Xinv(Xinv);

        constexpr QudaGaugeFieldOrder gOrder_milc = QUDA_MILC_GAUGE_ORDER;

        param_Y.order = gOrder_milc;
        param_Yhat.order = gOrder_milc;
        param_Xinv.order = gOrder_milc;

        // need to copy into AoS format for CUBLAS
        param_Y.setPrecision(X.Precision());
        param_Yhat.setPrecision(X.Precision());
        param_Xinv.setPrecision(X.Precision());

        cudaGaugeField Y_(param_Y);
        cudaGaugeField Yhat_(param_Yhat);
        cudaGaugeField Xinv_(param_Xinv);

        Xinv_.copy(Xinv);
        Y_.copy(Y);

        constexpr bool use_native_ghosts = true;
        // use spin-ignorant accessor to make multiplication simpler
        typedef typename gauge::FieldOrder<Float, N, 1, gOrder_milc, use_native_ghosts, storeFloat> gCoarse;
        typedef typename gauge::FieldOrder<Float, N, 1, gOrder_milc, use_native_ghosts, storeFloat> gPreconditionedCoarse;
        gCoarse yAccessor(Y_);
        gPreconditionedCoarse yHatAccessor(Yhat_);
        gCoarse xInvAccessor(Xinv_);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Xinv = %e\n", Xinv_.norm2(0));

        int comm_dim[4];
        for (int i = 0; i < 4; i++) comm_dim[i] = comm_dim_partitioned(i);

        using yHatArg = CalculateYhatArg<Float, gPreconditionedCoarse, gCoarse, N, 4, 2>;
        yHatArg arg(yHatAccessor, yAccessor, xInvAccessor, xc_size, comm_dim, 1);

        CalculateYhat<location, yHatArg> yHat(arg, Y_, use_mma);
        if (Yhat.Precision() == QUDA_HALF_PRECISION || Yhat.Precision() == QUDA_QUARTER_PRECISION) {
          yHat.setComputeMaxOnly(true);
          yHat.apply(0);

          double max_h_double = *arg.max_h;
          comm_allreduce_max(&max_h_double);
          *arg.max_h = static_cast<Float>(max_h_double);

          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Yhat Max = %e\n", *arg.max_h);

          Yhat_.Scale(*arg.max_h);
          arg.Yhat.resetScale(*arg.max_h);
        }
        yHat.setComputeMaxOnly(false);
        yHat.apply(0);

        Yhat.copy(Yhat_);

      } else {

        // use spin-ignorant accessor to make multiplication simpler
        typedef typename gauge::FieldOrder<Float,N,1,gOrder,true,storeFloat> gCoarse;
        typedef typename gauge::FieldOrder<Float,N,1,gOrder,true,storeFloat> gPreconditionedCoarse;
        gCoarse yAccessor(const_cast<GaugeField&>(Y));
        gPreconditionedCoarse yHatAccessor(const_cast<GaugeField&>(Yhat));
        gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Xinv = %e\n", Xinv.norm2(0));

        int comm_dim[4];
        for (int i=0; i<4; i++) comm_dim[i] = comm_dim_partitioned(i);
        typedef CalculateYhatArg<Float, gPreconditionedCoarse, gCoarse, N, 4, 2> yHatArg;
        yHatArg arg(yHatAccessor, yAccessor, xInvAccessor, xc_size, comm_dim, 1);

        CalculateYhat<location, yHatArg> yHat(arg, Y, use_mma);
        if (Yhat.Precision() == QUDA_HALF_PRECISION || Yhat.Precision() == QUDA_QUARTER_PRECISION) {
          yHat.setComputeMaxOnly(true);
          yHat.apply(0);

          double max_h_double = *arg.max_h;
          comm_allreduce_max(&max_h_double);
          *arg.max_h = static_cast<Float>(max_h_double);

          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Yhat Max = %e\n", *arg.max_h);

          Yhat.Scale(*arg.max_h);
          arg.Yhat.resetScale(*arg.max_h);
        }
        yHat.setComputeMaxOnly(false);
        yHat.apply(0);

      }

      if (getVerbosity() >= QUDA_VERBOSE)
        for (int d = 0; d < 8; d++)
          printfQuda("Yhat[%d] = %e (%e %e = %e x %e)\n", d, Yhat.norm2(d), Yhat.abs_max(d),
              Y.abs_max(d) * Xinv.abs_max(0), Y.abs_max(d), Xinv.abs_max(0));

    }

    // fill back in the bulk of Yhat so that the backward link is updated on the previous node
    // need to put this in the bulk of the previous node - but only send backwards the backwards
    // links to and not overwrite the forwards bulk
    Yhat.injectGhost(QUDA_LINK_BACKWARDS);

    // exchange forwards links for multi-process dslash dagger
    // need to put this in the ghost zone of the next node - but only send forwards the forwards
    // links and not overwrite the backwards ghost
    Yhat.exchangeGhost(QUDA_LINK_FORWARDS);
  }

  template <typename storeFloat, typename Float, int N>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<QUDA_CPU_FIELD_LOCATION,storeFloat,Float,N,gOrder>(Yhat, Xinv, Y, X, use_mma);
    } else {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;
      if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<QUDA_CUDA_FIELD_LOCATION,storeFloat,Float,N,gOrder>(Yhat, Xinv, Y, X, use_mma);
    }
  }

  // template on the number of coarse degrees of freedom
  template <typename storeFloat, typename Float>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma) {
    switch (Y.Ncolor()) {
    case 48: calculateYhat<storeFloat,Float,48>(Yhat, Xinv, Y, X, use_mma); break;
#ifdef NSPIN4
    // case 12: calculateYhat<storeFloat,Float,12>(Yhat, Xinv, Y, X, use_mma); break;
    case 64: calculateYhat<storeFloat,Float,64>(Yhat, Xinv, Y, X, use_mma); break;
#endif // NSPIN4
#ifdef NSPIN1
    case 128: calculateYhat<storeFloat,Float,128>(Yhat, Xinv, Y, X, use_mma); break;
    case 192: calculateYhat<storeFloat,Float,192>(Yhat, Xinv, Y, X, use_mma); break;
#endif // NSPIN1
    default: errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor()); break;
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma) {

#ifdef GPU_MULTIGRID
    QudaPrecision precision = checkPrecision(Xinv, Y, X);
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Yhat field......\n");

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (Yhat.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("Unsupported precision %d\n", Yhat.Precision());
      calculateYhat<double,double>(Yhat, Xinv, Y, X, use_mma);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      if (Yhat.Precision() == QUDA_SINGLE_PRECISION) {
        calculateYhat<float, float>(Yhat, Xinv, Y, X, use_mma);
      } else {
        errorQuda("Unsupported precision %d\n", precision);
      }
    } else if (precision == QUDA_HALF_PRECISION) {
      if (Yhat.Precision() == QUDA_HALF_PRECISION) {
        calculateYhat<short, float>(Yhat, Xinv, Y, X, use_mma);
      } else {
        errorQuda("Unsupported precision %d\n", precision);
      }
    } else {
      errorQuda("Unsupported precision %d\n", precision);
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing Yhat field\n");
#else
    errorQuda("Multigrid has not been built");
#endif
  }

} //namespace quda

