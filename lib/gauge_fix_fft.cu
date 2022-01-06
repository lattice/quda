#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <unitarization_links.h>
#include <gauge_tools.h>

#include <FFT_Plans.h>
#include <instantiate.h>

#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <kernels/gauge_fix_fft.cuh>

namespace quda {

  template <typename Float>
  class GaugeFixFFTRotate : TunableKernel1D {
    template <int dir> using Arg = GaugeFixFFTRotateArg<Float, dir>;
    GaugeField &data;
    complex<Float> *tmp0;
    complex<Float> *tmp1;
    int dir;
    unsigned int minThreads() const { return data.Volume(); }

  public:
    GaugeFixFFTRotate(GaugeField &data) :
      TunableKernel1D(data),
      data(data),
      dir(0) {}

    void setDirection(int dir_, complex<Float> *data_in, complex<Float> *data_out)
    {
      dir = dir_;
      tmp0 = data_in;
      tmp1 = data_out;
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (dir) {
      case 0: launch<FFTrotate>(tp, stream, Arg<0>(data, tmp0, tmp1)); break;
      case 1: launch<FFTrotate>(tp, stream, Arg<1>(data, tmp0, tmp1)); break;
      default: errorQuda("Error in GaugeFixFFTRotate option");
      }
    }

    long long flops() const { return 0; }
    long long bytes() const { return 4 * sizeof(Float) * data.Volume(); }
  };

  template <typename Arg>
  class GaugeFixQuality : TunableReduction2D<> {
    Arg &arg;
    const GaugeField &meta;

  public:
    GaugeFixQuality(Arg &arg, const GaugeField &meta) :
      TunableReduction2D(meta),
      arg(arg),
      meta(meta) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<FixQualityFFT>(arg.result, tp, stream, arg);

      arg.result[0] /= static_cast<double>(3 * Arg::gauge_dir * meta.Volume());
      arg.result[1] /= static_cast<double>(3 * meta.Volume());
    }

    long long flops() const { return (36 * Arg::gauge_dir + 65) * meta.Volume(); }
    long long bytes() const
    { return (Arg::gauge_dir * meta.Bytes() / 4) + 12 * meta.Volume() * meta.Precision(); }
  };

  enum GaugeFixFFTKernel {
    KERNEL_SET_INVPSQ,
    KERNEL_NORMALIZE,
    KERNEL_GX,
    KERNEL_UEO
  };

  template <typename Arg> class GaugeFixerFFT : TunableKernel2D {
    Arg &arg;
    const GaugeField &field;
    GaugeFixFFTKernel type;
    char aux_tmp[TuneKey::aux_n];
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    GaugeFixerFFT(Arg &arg, const GaugeField &field) :
      TunableKernel2D(field, 2),
      arg(arg),
      field(field)
    {
      strcpy(aux_tmp, aux);
    }

    void set_type(GaugeFixFFTKernel type) {
      this->type = type;
      strcpy(aux, aux_tmp);
      switch (type) {
      case KERNEL_SET_INVPSQ: strcat(aux, ",set_invpsq"); break;
      case KERNEL_NORMALIZE: strcat(aux, ",normalize"); break;
      case KERNEL_GX: strcat(aux, ",gx"); break;
      case KERNEL_UEO:
        strcat(aux, ",ueo");
#ifdef GAUGEFIXING_DONT_USE_GX
        strcat(aux, "_new");
#endif
        break;
      default: errorQuda("Unknown kernel type %d", type);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (type) {
      case KERNEL_SET_INVPSQ: launch<set_invpsq>(tp, stream, arg); break;
      case KERNEL_NORMALIZE: launch<mult_norm_2d>(tp, stream, arg); break;
      case KERNEL_GX: launch<GX>(tp, stream, arg); break;
#ifdef GAUGEFIXING_DONT_USE_GX
      case KERNEL_UEO: launch<U_EO_NEW>(tp, stream, arg); break;
#else
      case KERNEL_UEO: launch<U_EO>(tp, stream, arg); break;
#endif //GAUGEFIXING_DONT_USE_GX
      default: errorQuda("Unexpected kernel type %d", type);
      }
    }

    void preTune()
    {
      switch (type) {
      case KERNEL_NORMALIZE: std::swap(arg.gx, arg.delta); break; // delta is irrelevant here, so use as backup
      case KERNEL_UEO: field.backup(); break;
      default: break;
      }
    }

    void postTune()
    {
      switch (type) {
      case KERNEL_NORMALIZE: std::swap(arg.gx, arg.delta); break;
      case KERNEL_UEO: field.restore(); break;
      default: break;
      }
    }

    long long flops() const
    {
      switch (type) {
      case KERNEL_SET_INVPSQ: return 2 * field.Volume();
      case KERNEL_NORMALIZE: return 2 * field.Volume();
      case KERNEL_GX: return (arg.elems == 6 ? 208 : 166) * field.Volume();
#ifdef GAUGEFIXING_DONT_USE_GX
      case KERNEL_UEO: return 2414 * field.Volume();
#else
      case KERNEL_UEO: return (arg.elems == 6 ? 1794 : 1536) * field.Volume();
#endif
      default: errorQuda("Unexpected kernel type %d", type); return 0;
      }
    }

    long long bytes() const
    {
      switch (type) {
      case KERNEL_SET_INVPSQ: return sizeof(typename Arg::Float) * field.Volume();
      case KERNEL_NORMALIZE: return 3 * sizeof(typename Arg::Float) * field.Volume();
      case KERNEL_GX: return 4 * arg.elems * field.Precision() * field.Volume();
#ifdef GAUGEFIXING_DONT_USE_GX
      case KERNEL_UEO: return field.Bytes() + (5 * 12 * sizeof(typename Arg::Float)) * field.Volume();
#else
      case KERNEL_UEO: return 26 * arg.elems * field.Precision() * field.Volume();
#endif
      default: errorQuda("Unexpected kernel type %d", type); return 0;
      }
    }
  };

  template <typename Float, QudaReconstructType recon, int gauge_dir>
  void gaugeFixingFFT(GaugeField& data, QudaGaugeFixParam &fix_param)
  {
    TimeProfile profileInternalGaugeFixFFT("InternalGaugeFixQudaFFT", false);
    
    QudaBoolean autotune = fix_param.fft_autotune;
    double alpha0 = fix_param.fft_alpha;
    double tolerance = fix_param.tolerance;
    QudaBoolean theta_condition = fix_param.theta_condition;
    int steps = fix_param.maxiter;
    int verbose_interval = fix_param.verbosity_interval;
    
    profileInternalGaugeFixFFT.TPSTART(QUDA_PROFILE_COMPUTE);

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if(autotune == QUDA_BOOLEAN_TRUE) printfQuda("\tAuto tune active: alpha will be adjusted as the algorithm progresses\n");
      else printfQuda("\tAuto tune not active: alpha will remain constant as the algorithm progresses\n");      
      printfQuda("\tAlpha parameter of the Steepest Descent Method: %e\n", alpha0);
      printfQuda("\tTolerance: %e\n", tolerance);
      printfQuda("\tStop criterion method: %s\n", theta_condition == QUDA_BOOLEAN_TRUE ? "Theta" : "Delta");
      printfQuda("\tMaximum number of iterations: %d\n", steps);
      printfQuda("\tPrint convergence results at every %d steps\n", verbose_interval);
    }
    
    unsigned int delta_pad = data.X()[0] * data.X()[1] * data.X()[2] * data.X()[3];
    int4 size = make_int4(data.X()[0], data.X()[1], data.X()[2], data.X()[3]);
    FFTPlanHandle plan_xy;
    FFTPlanHandle plan_zt;

    GaugeFixArg<Float, recon> arg(data, alpha0);
    SetPlanFFT2DMany(plan_zt, size, 0, data.Precision());     //for space and time ZT
    SetPlanFFT2DMany(plan_xy, size, 1, data.Precision());    //with space only XY

    GaugeFixFFTRotate<Float> GFRotate(data);

    GaugeFixerFFT<decltype(arg)> gfix(arg, data);
    gfix.set_type(KERNEL_SET_INVPSQ);
    gfix.apply(device::get_default_stream());

    GaugeFixQualityFFTArg<Float, recon, gauge_dir> argQ(data, arg.delta);
    GaugeFixQuality<decltype(argQ)> gfixquality(argQ, data);
    gfixquality.apply(device::get_default_stream());
    double action0 = argQ.getAction();
    if(getVerbosity() >= QUDA_SUMMARIZE) printf("Step: %05d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    double diff = 0.0;
    int iter = 0;
    for (iter = 0; iter < steps; iter++) {
      for (int k = 0; k < 6; k++) {
        //------------------------------------------------------------------------
        // Set a pointer do the element k in lattice volume
        // each element is stored with stride lattice volume
        // it uses gx as temporary array!!!!!!
        //------------------------------------------------------------------------
        complex<Float> *_array = arg.delta + k * delta_pad;
        //////  2D FFT + 2D FFT
        //------------------------------------------------------------------------
        // Perform FFT on xy plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_xy, _array, arg.gx, FFT_FORWARD);
        //------------------------------------------------------------------------
        // Rotate hypercube, xyzt -> ztxy
        //------------------------------------------------------------------------
        GFRotate.setDirection(0, arg.gx, _array);
        GFRotate.apply(device::get_default_stream());
        //------------------------------------------------------------------------
        // Perform FFT on zt plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_zt, _array, arg.gx, FFT_FORWARD);
        //------------------------------------------------------------------------
        // Normalize FFT and apply pmax^2/p^2
        //------------------------------------------------------------------------
        gfix.set_type(KERNEL_NORMALIZE);
        gfix.apply(device::get_default_stream());
        //------------------------------------------------------------------------
        // Perform IFFT on zt plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_zt, arg.gx, _array, FFT_INVERSE);
        //------------------------------------------------------------------------
        // Rotate hypercube, ztxy -> xyzt
        //------------------------------------------------------------------------
        GFRotate.setDirection(1, _array, arg.gx);
        GFRotate.apply(device::get_default_stream());
        //------------------------------------------------------------------------
        // Perform IFFT on xy plane
        //------------------------------------------------------------------------
        ApplyFFT(plan_xy, arg.gx, _array, FFT_INVERSE);
      }

#ifndef GAUGEFIXING_DONT_USE_GX
      //------------------------------------------------------------------------
      // Calculate g(x)
      // ------------------------------------------------------------------------
      // (using GX - else without using GX, gx will be created only
      // for plane rotation but with less size)
      gfix.set_type(KERNEL_GX);
      gfix.apply(device::get_default_stream());
#endif
      //------------------------------------------------------------------------
      // Apply gauge fix to current gauge field
      //------------------------------------------------------------------------
      gfix.set_type(KERNEL_UEO);
      gfix.apply(device::get_default_stream());

      //------------------------------------------------------------------------
      // Measure gauge quality and recalculate new Delta(x)
      //------------------------------------------------------------------------
      gfixquality.apply(device::get_default_stream());
      double action = argQ.getAction();
      diff = abs(action0 - action);
      if ((iter % verbose_interval) == (verbose_interval - 1) && getVerbosity() >= QUDA_SUMMARIZE)
        printf("Step: %05d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
      if ( autotune == QUDA_BOOLEAN_TRUE && ((action - action0) < -1e-14) ) {
        if ( arg.alpha > 0.01 ) {
          arg.alpha = 0.95 * arg.alpha;
          if(getVerbosity() >= QUDA_SUMMARIZE) printf("Changing alpha down -> %.4e\n", arg.alpha);
        }
      }
      //------------------------------------------------------------------------
      // Check gauge fix quality criterion
      //------------------------------------------------------------------------
      if ( theta_condition == QUDA_BOOLEAN_TRUE ) {   if ( argQ.getTheta() < tolerance ) break; }
      else { if ( diff < tolerance ) break; }

      action0 = action;
    }
    if ((iter % verbose_interval) != 0 && getVerbosity() >= QUDA_SUMMARIZE)
      printf("Step: %05d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter, argQ.getAction(), argQ.getTheta(), diff);
    
    // Reunitarize at end
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error,
                               reunit_allow_svd, reunit_svd_only,
                               svd_rel_error, svd_abs_error);
    int *num_failures_h = static_cast<int*>(mapped_malloc(sizeof(int)));
    int *num_failures_d = static_cast<int*>(get_mapped_device_pointer(num_failures_h));

    *num_failures_h = 0;
    unitarizeLinks(data, data, num_failures_d);
    if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);
    // end reunitarize

    arg.free();
    FFTDestroyPlan(plan_zt);
    FFTDestroyPlan(plan_xy);
    qudaDeviceSynchronize();
    profileInternalGaugeFixFFT.TPSTOP(QUDA_PROFILE_COMPUTE);


    double secs = profileInternalGaugeFixFFT.Last(QUDA_PROFILE_COMPUTE);
    double fftflop = 5.0 * (log2((double)( data.X()[0] * data.X()[1]) ) + log2( (double)(data.X()[2] * data.X()[3] )));
    fftflop *= (double)data.Volume();
    gfix.set_type(KERNEL_SET_INVPSQ);
    double gflops = gfix.flops() + gfixquality.flops();
    double gbytes = gfix.bytes() + gfixquality.bytes();
    gfix.set_type(KERNEL_NORMALIZE);
    double flop = gfix.flops() * recon / 2;
    double byte = gfix.bytes() * recon / 2;
    flop += (GFRotate.flops() + fftflop) * (recon / 2) * 2;
    byte += GFRotate.bytes() * (recon / 2) * 4;     //includes FFT reads, assuming 1 read and 1 write per site
#ifndef GAUGEFIXING_DONT_USE_GX
    gfix.set_type(KERNEL_GX);
    flop += gfix.flops();
    byte += gfix.bytes();
#endif
    gfix.set_type(KERNEL_UEO);
    flop += gfix.flops();
    byte += gfix.bytes();
    flop += gfixquality.flops();
    byte += gfixquality.bytes();
    gflops += flop * iter;
    gbytes += byte * iter;
    gflops += 4588.0 * data.Volume(); //Reunitarize at end
    gbytes += 2 * data.Bytes(); //Reunitarize at end
    
    gflops = (gflops * 1e-9) / (secs);
    gbytes = gbytes / (secs * 1e9);
    if (getVerbosity() > QUDA_SUMMARIZE)
      printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    
    host_free(num_failures_h);
  }

  template<typename Float, int nColors, QudaReconstructType recon> struct GaugeFixingFFT {
    GaugeFixingFFT(GaugeField& data, QudaGaugeFixParam &fix_param)
    {
      if (fix_param.gauge_dir == 4) {
	if (getVerbosity() > QUDA_SUMMARIZE) printfQuda("Starting Landau gauge fixing with FFTs\n");
        gaugeFixingFFT<Float, recon, 4>(data, fix_param);
      } else if (fix_param.gauge_dir == 3) {
	if (getVerbosity() > QUDA_SUMMARIZE) printfQuda("Starting Coulomb gauge fixing with FFTs\n");
        gaugeFixingFFT<Float, recon, 3>(data, fix_param);	
      } else {
	errorQuda("Unexpected gauge_dir = %d", fix_param.gauge_dir); 
      }
    }
  };

  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] data, quda gauge field
   * @param[in] fix_param Parameter struct defining the gauge fixing
   */
#if defined(GPU_GAUGE_ALG)
  void gaugeFixingFFT(GaugeField& data, QudaGaugeFixParam &fix_param)
  {
    if (comm_partitioned()) errorQuda("Gauge Fixing with FFTs in multi-GPU support NOT implemented yet!");
    instantiate<GaugeFixingFFT, ReconstructNo12>(data, fix_param);
  }
#else
  void gaugeFixingFFT(GaugeField&, QudaGaugeFixParam &)
  {
    errorQuda("Gauge fixing has bot been built");
  }
#endif

}
