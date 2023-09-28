#include <typeinfo>
#include <gauge_field.h>
#include <blas_lapack.h>
#include <blas_quda.h>
#include <tunable_nd.h>
#include <kernels/coarse_op_preconditioned.cuh>
#include <coarse_op_preconditioned_mma_launch.h>
#include "multigrid.h"

namespace quda
{

  template <QudaFieldLocation location_template, typename Float_, typename PreconditionedGauge,
            typename Gauge, typename GaugeInv, int n, int M, int N, bool compute_max, bool use_mma>
  class CalculateYhat : public TunableKernel3D {
    using Float = Float_;
    using Arg = CalculateYhatArg<Float, PreconditionedGauge, Gauge, GaugeInv, n, M, N, compute_max>;
    Arg arg;
    GaugeField &Yhat;
    const GaugeField &Y;
    const GaugeField &Xinv;

    long long flops() const { return Y.Volume() * 8 * n * n * (8 * n - 2); } // 8 from dir, 8 from complexity,
    long long bytes() const { return 2l * (arg.Xinv.Bytes() + 8*arg.Y.Bytes() + !Arg::compute_max * 8*arg.Yhat.Bytes()) * n; }

    unsigned int minThreads() const { return Y.VolumeCB(); }

    // all the tuning done is only in matrix tile size (Y/Z block.grid)
    int blockMin() const { return 8; }
    int blockStep() const { return 8; }
    unsigned int maxBlockSize(const TuneParam &) const { return 8u; }
    bool tuneAuxDim() const { return use_mma; } // tune aux if doing mma

  public:
    CalculateYhat(GaugeField &Yhat, const GaugeField &Y, const GaugeField &Xinv) :
      TunableKernel3D(Y, 2 * arg.tile.M_tiles, 4 * arg.tile.N_tiles),
      arg(Yhat, Y, Xinv),
      Yhat(Yhat),
      Y(Y),
      Xinv(Xinv)
    {
      if (Arg::compute_max) {
        arg.max_h = static_cast<Float*>(pool_pinned_malloc(sizeof(Float)));
        if (location == QUDA_CUDA_FIELD_LOCATION) arg.max_d = static_cast<Float*>(pool_device_malloc(sizeof(Float)));
        arg.max = location == QUDA_CUDA_FIELD_LOCATION ? arg.max_d : arg.max_h;
      }

      if (location == QUDA_CUDA_FIELD_LOCATION) strcat(aux, Y.MemType() == QUDA_MEMORY_MAPPED ? ",GPU-mapped" : ",GPU-device");
      strcat(aux, comm_dim_partitioned_string());
      if constexpr (use_mma) {
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          strcat(aux, ",mma");
#ifdef QUDA_MMA_AVAILABLE
          strcat(aux, mma::mg_mma_dispatch_t<Float>::type::get_type_name().c_str());
#endif
        }
      }
      if (Arg::compute_max) strcat(aux, ",compute_max");

      apply(device::get_default_stream());

      if (Arg::compute_max) {
        comm_allreduce_max(*arg.max_h);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Yhat Max = %e\n", *arg.max_h);
        Yhat.Scale(*arg.max_h);
      }
    }

    ~CalculateYhat()
    {
      if (Arg::compute_max) {
        if (location == QUDA_CUDA_FIELD_LOCATION) pool_device_free(arg.max_d);
        pool_pinned_free(arg.max_h);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CUDA_FIELD_LOCATION && Arg::compute_max && !activeTuning()) {
        qudaMemsetAsync(arg.max_d, 0, sizeof(typename Arg::Float), stream);
      }

      if constexpr (location_template == QUDA_CUDA_FIELD_LOCATION) {
        if constexpr (use_mma) mma::launch_yhat_kernel(tp, stream, arg, *this);
        else launch_device<ComputeYhat>(tp, stream, arg);
      } else {
        launch_host<ComputeYhat>(tp, stream, arg);
      }

      if (location == QUDA_CUDA_FIELD_LOCATION && Arg::compute_max && !activeTuning()) { // only do copy once tuning is done
        qudaMemcpyAsync(arg.max_h, arg.max_d, sizeof(typename Arg::Float), qudaMemcpyDeviceToHost, stream);
        qudaStreamSynchronize(const_cast<qudaStream_t&>(stream));
      }
    }

    bool advanceSharedBytes(TuneParam &) const { return false; }

    bool advanceAux(TuneParam &param) const
    {
      if constexpr (use_mma) {
        constexpr bool query_max = true;
        int max = mma::launch_yhat_kernel<query_max>(param, device::get_default_stream(), arg, *this);
        if (param.aux.x < max) {
          param.aux.x++;
          return true;
        }
        return false;
      }
      return false;
    }

    bool advanceTuneParam(TuneParam &param) const
    {
      if (!use_mma) {
        if (location == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE)
          return TunableKernel3D::advanceTuneParam(param);
        else
          return false;
      } else {
        return false;
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableKernel3D::initTuneParam(param);
      param.aux = make_int4(0, 0, 0, 0);
    }

    void defaultTuneParam(TuneParam &param) const
    {
      TunableKernel3D::defaultTuneParam(param);
      param.aux = make_int4(0, 0, 0, 0);
    }
  };

  /**
     @brief Calculate the preconditioned coarse-link field and the clover inverse.
     @param Yhat[out] Preconditioned coarse link field
     @param Xinv[out] Coarse clover inverse field
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param use_mma[in] Whether or not use MMA (tensor core) to do the calculation
   */
  template <QudaFieldLocation location, typename storeFloat, typename Float, int N, QudaGaugeFieldOrder gOrder>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    constexpr QudaGaugeFieldOrder gOrder_milc = QUDA_MILC_GAUGE_ORDER;
    GaugeField *Xinv_aos = nullptr;

    // invert the clover matrix field
    const int n = X.Ncolor();

    if (X.Location() == QUDA_CUDA_FIELD_LOCATION) {

      auto create_gauge_copy = [](const GaugeField &X, bool copy_content) -> auto
      {
        GaugeField *output = nullptr;
        if (X.Order() == gOrder_milc && X.Precision() >= QUDA_SINGLE_PRECISION) {
          output = const_cast<GaugeField *>(&X);
        } else {
          GaugeFieldParam param(X);
          param.order = gOrder_milc;
          param.setPrecision(X.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : X.Precision());
          output = cudaGaugeField::Create(param);
          if (copy_content) output->copy(X);
        }
        return output;
      };

      GaugeField *X_aos = create_gauge_copy(X, true);
      Xinv_aos = create_gauge_copy(Xinv, false);

      blas::flops += invert((void *)Xinv_aos->Gauge_p(), (void *)X_aos->Gauge_p(), n, X_aos->Volume(),
                            X_aos->Precision(), X.Location());

      if (&Xinv != Xinv_aos) {
        if (Xinv.Precision() < QUDA_SINGLE_PRECISION) Xinv.Scale(Xinv_aos->abs_max());
        Xinv.copy(*Xinv_aos);
      }
      if (&X != X_aos) { delete X_aos; }

      if (!use_mma) { delete Xinv_aos; }

    } else if (X.Location() == QUDA_CPU_FIELD_LOCATION && X.Order() == QUDA_QDP_GAUGE_ORDER) {
      const cpuGaugeField *X_h = static_cast<const cpuGaugeField*>(&X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);
      blas::flops += invert(*(void**)Xinv_h->Gauge_p(), *(void**)X_h->Gauge_p(), n, X_h->Volume(), X.Precision(), X.Location());
    } else {
      errorQuda("Unsupported location=%d and order=%d", X.Location(), X.Order());
    }

    // now exchange Y halos of both forwards and backwards links for multi-process dslash
    const_cast<GaugeField&>(Y).exchangeGhost(QUDA_LINK_BIDIRECTIONAL);

    // compute the preconditioned links
    // Yhat_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
    // Yhat_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)
    {
      if (use_mma) {

        auto create_gauge_copy = [](const GaugeField &X, QudaGaugeFieldOrder order, bool copy_content) -> auto
        {
          GaugeField *output = nullptr;
          if (X.Order() == order) {
            output = const_cast<GaugeField *>(&X);
          } else {
            GaugeFieldParam param(X);
            param.order = order;
            // if we did the exchange on AoS order, then this zero initialize wouldn't be needed
            if (!copy_content) param.create = QUDA_ZERO_FIELD_CREATE;
            output = cudaGaugeField::Create(param);
            if (copy_content) output->copy(X);
          }
          return output;
        };

        GaugeField *Y_aos = create_gauge_copy(Y, gOrder_milc, true);
        GaugeField *Yhat_aos = create_gauge_copy(Yhat, gOrder_milc, false);

        constexpr bool use_native_ghosts = true;
        // use spin-ignorant accessor to make multiplication simpler
        using gCoarse = typename gauge::FieldOrder<Float, N, 1, gOrder_milc, use_native_ghosts, storeFloat>;
        using gPreconditionedCoarse = typename gauge::FieldOrder<Float, N, 1, gOrder_milc, use_native_ghosts, storeFloat>;
        // XXX: This doesn't work for double precision since hard-coded to single precision
        using gCoarseInv = gauge::FieldOrder<float, N, 1, gOrder_milc, use_native_ghosts, float>;

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Xinv = %e\n", Xinv_aos->norm2(0));

        if (Yhat.Precision() == QUDA_HALF_PRECISION || Yhat.Precision() == QUDA_QUARTER_PRECISION) {
          CalculateYhat<location, Float, gPreconditionedCoarse, gCoarse, gCoarseInv, N, 4, 2, true, true>
            (*Yhat_aos, *Y_aos, *Xinv_aos);
        }
        CalculateYhat<location, Float, gPreconditionedCoarse, gCoarse, gCoarseInv, N, 4, 2, false, true>
          (*Yhat_aos, *Y_aos, *Xinv_aos);

        if (&Y != Y_aos) { delete Y_aos; }

        if (&Yhat != Yhat_aos) {
          Yhat.copy(*Yhat_aos);
          delete Yhat_aos;
        }

        if (Xinv_aos != &Xinv) { delete Xinv_aos; }

      } else {

        // use spin-ignorant accessor to make multiplication simpler
        using gCoarse = typename gauge::FieldOrder<Float, N, 1, gOrder, true, storeFloat>;
        using gPreconditionedCoarse = typename gauge::FieldOrder<Float, N, 1, gOrder, true, storeFloat>;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Xinv = %e\n", Xinv.norm2(0));

        if (Yhat.Precision() == QUDA_HALF_PRECISION || Yhat.Precision() == QUDA_QUARTER_PRECISION) {
          CalculateYhat<location, Float, gPreconditionedCoarse, gCoarse, gCoarse, N, 4, 2, true, false>
            (Yhat, Y, Xinv);
        }
        CalculateYhat<location, Float, gPreconditionedCoarse, gCoarse, gCoarse, N, 4, 2, false, false>
          (Yhat, Y, Xinv);
      }

    }

    // fill back in the bulk of Yhat so that the backward link is updated on the previous node
    // need to put this in the bulk of the previous node - but only send backwards the backwards
    // links to and not overwrite the forwards bulk
    Yhat.injectGhost(QUDA_LINK_BACKWARDS);

    // exchange forwards links for multi-process dslash dagger
    // need to put this in the ghost zone of the next node - but only send forwards the forwards
    // links and not overwrite the backwards ghost
    Yhat.exchangeGhost(QUDA_LINK_FORWARDS);

    if (getVerbosity() >= QUDA_VERBOSE) {
      for (int d = 0; d < 8; d++)
        printfQuda("Yhat[%d] = %e (%e < %e x %e)\n", d, Yhat.norm2(d), Yhat.abs_max(d), Y.abs_max(d), Xinv.abs_max(0));
    }

  }

  template <typename storeFloat, typename Float, int N>
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<QUDA_CPU_FIELD_LOCATION, storeFloat, Float, N, gOrder>(Yhat, Xinv, Y, X, use_mma);
    } else {
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;
      // if (Y.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", Y.FieldOrder());
      calculateYhat<QUDA_CUDA_FIELD_LOCATION, storeFloat, Float, N, gOrder>(Yhat, Xinv, Y, X, use_mma);
    }
  }

  constexpr int Nc = @QUDA_MULTIGRID_NVEC@;

  template <>
  void calculateYhat<Nc>(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma)
  {
    if constexpr (is_enabled_multigrid()) {
      if (use_mma && Y.Location() == QUDA_CPU_FIELD_LOCATION)
        errorQuda("MG-MMA cannot be used with CPU location fields");
      QudaPrecision precision = checkPrecision(Xinv, Y, X, Yhat);
      logQuda(QUDA_SUMMARIZE, "Computing Yhat field......\n");

      if (precision == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) {
          if (use_mma) errorQuda("MG-MMA does not support double precision, yet.");
          calculateYhat<double, double, 2 * Nc>(Yhat, Xinv, Y, X, use_mma);
        } else {
          errorQuda("Double precision multigrid has not been enabled");
        }
      } else if (precision == QUDA_SINGLE_PRECISION) {
        calculateYhat<float, float, 2 * Nc>(Yhat, Xinv, Y, X, use_mma);
      } else if (precision == QUDA_HALF_PRECISION) {
        calculateYhat<short, float, 2 * Nc>(Yhat, Xinv, Y, X, use_mma);
      } else {
        errorQuda("Unsupported precision %d", precision);
      }

      logQuda(QUDA_SUMMARIZE, "....done computing Yhat field\n");
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
