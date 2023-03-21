#include <quda_internal.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <tunable_nd.h>
#include <kernels/gauge_fix_ovr.cuh>

namespace quda {

  struct ComputeBorderPoints : TunableKernel3D {
    const GaugeField &u;
    int **borderpoints;
    int nlinksfaces;
    unsigned int minThreads() const { return nlinksfaces; }

    ComputeBorderPoints(const GaugeField &u, int *borderpoints[2]) :
      TunableKernel3D(u, 2, 2),
      u(u),
      borderpoints(borderpoints),
      nlinksfaces(0)
    {
      for (int dir = 0; dir < 4; dir++) if (comm_dim_partitioned(dir)) nlinksfaces += u.LocalSurfaceCB(dir);
      apply(device::get_default_stream());
      qudaDeviceSynchronize();
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<BorderPointsCompute>(tp, stream, BorderIdArg(u, borderpoints));
    }

    long long bytes() const { return 2 * nlinksfaces * sizeof(int); }
  };

  /**
   * @brief Pre-calculate lattice border points used by the gauge
   * fixing with overrelaxation in multi-GPU implementation
   */
  void PreCalculateLatticeIndices(const GaugeField &u, int &threads, int *borderpoints[2])
  {
    ComputeBorderPoints(u, borderpoints);

    int nlinksfaces = 0;
    for (int dir = 0; dir < 4; dir++) if (comm_dim_partitioned(dir)) nlinksfaces += 2 * u.LocalSurfaceCB(dir);

    int size[2];
    for (int i = 0; i < 2; i++) {
      std::sort(borderpoints[i], borderpoints[i] + nlinksfaces);
      size[i] = std::unique(borderpoints[i], borderpoints[i] + nlinksfaces) - borderpoints[i];
    }
    if (size[0] == size[1]) threads = size[0];
    else errorQuda("BORDER: even and odd sizes does not match, not supported, %d:%d", size[0], size[1]);
  }

  /**
   * @brief Tunable object for the gauge fixing kernel
   */
  template<typename Float, QudaReconstructType recon, int gauge_dir>
  class GaugeFix : TunableKernel2D {
    GaugeField &u;
    double relax_boost;
    int *borderpoints[2];
    int parity;
    unsigned long threads;
    bool halo;

    bool advanceAux(TuneParam &param) const
    {
      param.aux.x = (param.aux.x + 1) % 6;
      if (!device::shared_memory_atomic_supported()) { // 1, 4 use shared memory atomics
	if(param.aux.x == 1 || param.aux.x == 4) param.aux.x++;
      }
      // mu must be contained in the block, types 0, 1, 2 have mu = 8 and 3, 4, 5 have mu = 4
      TunableKernel2D::resizeVector(param.aux.x < 3 ? 8 : 4);
      TunableKernel2D::resizeStep(param.aux.x < 3 ? 8 : 4);
      TunableKernel2D::initTuneParam(param);
      return param.aux.x == 0 ? false : true;
    }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const
    {
      switch (param.aux.x) {
      case 0: return 8 * param.block.x * 4 * sizeof(Float);
      case 1: return 8 * param.block.x * 4 * sizeof(Float) / 8;
      case 2: return 8 * param.block.x * 4 * sizeof(Float) / 8;
      case 3: return 4 * param.block.x * 4 * sizeof(Float);
      default: return 4 * param.block.x * sizeof(Float);
      }
    }

    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return threads; }

  public:
    GaugeFix(GaugeField &u, double relax_boost, int *borderpoints[2], bool halo, int threads) :
      TunableKernel2D(u, 8),
      u(u),
      relax_boost(relax_boost),
      borderpoints{borderpoints[0], borderpoints[1]},
      parity(0),
      halo(halo)
    {
      if (!halo) {
        this->threads = 1;
        for (int dir = 0; dir < 4; dir++) {
          auto border = comm_dim_partitioned(dir) ? u.R()[dir] + 1 : 0;
          this->threads *= u.X()[dir] - border * 2;
        }
        this->threads /= 2;

        if (this->threads == 0) errorQuda("Local volume is too small");
      } else {
        this->threads = threads;
      }
      strcat(aux, halo ? ",halo" : ",interior");
    }

    void setParity(const int par) { parity = par; }

    template <bool halo_, int type_> using Arg = GaugeFixArg<Float, recon, gauge_dir, halo_, type_>;

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (!halo) {
        switch (tp.aux.x) {
        case 0: launch<computeFix>(tp, stream, Arg<false, 0>(u, relax_boost, parity, borderpoints, threads)); break;
        case 1: launch<computeFix>(tp, stream, Arg<false, 1>(u, relax_boost, parity, borderpoints, threads)); break;
        case 2: launch<computeFix>(tp, stream, Arg<false, 2>(u, relax_boost, parity, borderpoints, threads)); break;
        case 3: launch<computeFix>(tp, stream, Arg<false, 3>(u, relax_boost, parity, borderpoints, threads)); break;
        case 4: launch<computeFix>(tp, stream, Arg<false, 4>(u, relax_boost, parity, borderpoints, threads)); break;
        case 5: launch<computeFix>(tp, stream, Arg<false, 5>(u, relax_boost, parity, borderpoints, threads)); break;
        default: errorQuda("Unexpected type = %u", tp.aux.x);
        }
      } else {
        switch (tp.aux.x) {
        case 0: launch<computeFix>(tp, stream, Arg<true, 0>(u, relax_boost, parity, borderpoints, threads)); break;
        case 1: launch<computeFix>(tp, stream, Arg<true, 1>(u, relax_boost, parity, borderpoints, threads)); break;
        case 2: launch<computeFix>(tp, stream, Arg<true, 2>(u, relax_boost, parity, borderpoints, threads)); break;
        case 3: launch<computeFix>(tp, stream, Arg<true, 3>(u, relax_boost, parity, borderpoints, threads)); break;
        case 4: launch<computeFix>(tp, stream, Arg<true, 4>(u, relax_boost, parity, borderpoints, threads)); break;
        case 5: launch<computeFix>(tp, stream, Arg<true, 5>(u, relax_boost, parity, borderpoints, threads)); break;
        default: errorQuda("Unexpected type = %u", tp.aux.x);
        }
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      param.aux.x = 0;
      TunableKernel2D::resizeVector(param.aux.x < 3 ? 8 : 4);
      TunableKernel2D::resizeStep(param.aux.x < 3 ? 8 : 4);
      TunableKernel2D::initTuneParam(param);
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    std::string paramString(const TuneParam &param) const
    {
      return std::string(TunableKernel2D::paramString(param)) + ", atomicadd=" + std::to_string(param.aux.x);
    }

    void preTune() { u.backup(); }
    void postTune() { u.restore(); }
    long long flops() const { return 3LL * (22 + 28 * gauge_dir + 224 * 3) * threads; }
    long long bytes() const { return 8LL * 2 * threads * u.Reconstruct() * sizeof(Float);  }
  };

  /**
   * @brief Tunable object for the gauge fixing quality kernel
   */
  template <typename Arg>
  class GaugeFixQuality : TunableReduction2D {
    Arg &arg;
    const GaugeField &meta;

  public:
    GaugeFixQuality(Arg &arg, const GaugeField &meta) :
      TunableReduction2D(meta),
      arg(arg),
      meta(meta)
    { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<FixQualityOVR>(arg.result, tp, stream, arg);

      arg.result[0] /= static_cast<double>(3 * Arg::gauge_dir * 2 * arg.threads.x * comm_size());
      arg.result[1] /= static_cast<double>(3 * 2 * arg.threads.x * comm_size());
    }

    long long flops() const { return (36LL * Arg::gauge_dir + 65LL) * meta.Volume(); }
    long long bytes() const { return 2LL * Arg::gauge_dir * meta.Volume() * meta.Reconstruct() * meta.Precision(); }
  };

  template <typename Float, QudaReconstructType recon, bool pack, bool top>
  class GaugeFixPacker : public TunableKernel1D {
    GaugeField &u;
    complex<Float> *array;
    int parity;
    int dim;
    long long bytes() const { return u.LocalSurfaceCB(dim) * sizeof(Float) * recon * 2; }
    unsigned int minThreads() const { return u.LocalSurfaceCB(dim); }

  public:
    GaugeFixPacker(GaugeField &u, complex<Float> *array, int parity, int dim, const qudaStream_t &stream) :
      TunableKernel1D(u),
      u(u),
      array(array),
      parity(parity),
      dim(dim)
    {
      strcat(aux, dim == 0 ? ",d=0" : dim == 1 ? ",d=1" : dim == 2 ? ",d=2" : ",d=3");
      apply(stream);
    }

    void apply(const qudaStream_t &stream)
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Packer>(tp, stream, GaugeFixPackArg<Float, recon, pack, top>(u, array, parity, dim));
    }
  };

  template <typename Float, QudaReconstructType recon, int gauge_dir>
  void gaugeFixingOVR(GaugeField &data,const int Nsteps, const int verbose_interval,
                      const double relax_boost, const double tolerance,
                      const int reunit_interval, const int stopWtheta)
  {
    TimeProfile profileInternalGaugeFixOVR("InternalGaugeFixQudaOVR", false);

    profileInternalGaugeFixOVR.TPSTART(QUDA_PROFILE_COMPUTE);
    double flop = 0;
    double byte = 0;

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("\tOverrelaxation boost parameter: %e\n", relax_boost);
      printfQuda("\tTolerance: %le\n", tolerance);
      printfQuda("\tStop criterion method: %s\n", stopWtheta ? "Theta" : "Delta");
      printfQuda("\tMaximum number of iterations: %d\n", Nsteps);
      printfQuda("\tReunitarize at every %d steps\n", reunit_interval);
      printfQuda("\tPrint convergence results at every %d steps\n", verbose_interval);
    }
    
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

    GaugeFixQualityOVRArg<Float, recon, gauge_dir> argQ(data);
    GaugeFixQuality<decltype(argQ)> GaugeFixQuality(argQ, data);

    void *send[4];
    void *recv[4];
    void *sendg[4];
    void *recvg[4];
    void *send_d[4];
    void *recv_d[4];
    void *sendg_d[4];
    void *recvg_d[4];
    void *hostbuffer_h[4];
    size_t offset[4];
    size_t bytes[4];
    // do the exchange
    MsgHandle *mh_recv_back[4];
    MsgHandle *mh_recv_fwd[4];
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_send_back[4];

    if (comm_partitioned()) {
      for (int d = 0; d < 4; d++) {
        if (!commDimPartitioned(d)) continue;
        offset[d] = data.LocalSurfaceCB(d) * recon;
        bytes[d] =  sizeof(Float) * offset[d];
        send_d[d] = device_malloc(bytes[d]);
        recv_d[d] = device_malloc(bytes[d]);
        sendg_d[d] = device_malloc(bytes[d]);
        recvg_d[d] = device_malloc(bytes[d]);
        hostbuffer_h[d] = (void*)pinned_malloc(4 * bytes[d]);
      }
      for (int d = 0; d < 4; d++) {
        if (!commDimPartitioned(d)) continue;
        recv[d] = hostbuffer_h[d];
        send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
        recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
        sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];
        mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
        mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
        mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
        mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
      }
    }

    int *borderpoints[2];
    int nlinksfaces = 0;
    int threads = 0;
    for (int dir = 0; dir < 4; dir++) if (comm_dim_partitioned(dir)) nlinksfaces += 2 * data.LocalSurfaceCB(dir);
    for (int i = 0; i < 2 && nlinksfaces; i++) { //even and odd ids
      borderpoints[i] = static_cast<int*>(managed_malloc(nlinksfaces * sizeof(int)));
      qudaMemset(borderpoints[i], 0, nlinksfaces * sizeof(int));
    }
    if (comm_partitioned()) PreCalculateLatticeIndices(data, threads, borderpoints);

    GaugeFixQuality.apply(device::get_default_stream());
    flop += (double)GaugeFixQuality.flops();
    byte += (double)GaugeFixQuality.bytes();
    double action0 = argQ.getAction();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\n", 0, argQ.getAction(), argQ.getTheta());

    *num_failures_h = 0;
    unitarizeLinks(data, data, num_failures_d);

    if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);

    GaugeFix<Float, recon, gauge_dir> gfixIntPoints(data, relax_boost, borderpoints, false, -1);
    GaugeFix<Float, recon, gauge_dir> gfixBorderPoints(data, relax_boost, borderpoints, true, threads);

    int iter = 0;
    for (iter = 0; iter < Nsteps; iter++) {
      for (int p = 0; p < 2; p++) {
        if (comm_partitioned()) {
          gfixBorderPoints.setParity(p); //compute border points
          gfixBorderPoints.apply(device::get_default_stream());
          flop += (double)gfixBorderPoints.flops();
          byte += (double)gfixBorderPoints.bytes();
        }

        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_start(mh_recv_back[d]);
          comm_start(mh_recv_fwd[d]);
        }
        //wait for the update to the halo points before start packing...
        qudaDeviceSynchronize();
        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          //extract top face
          GaugeFixPacker<Float, recon, true, true>
            (data, reinterpret_cast<complex<Float>*>(send_d[d]), p, d, device::get_stream(d));
          //extract bottom ghost
          GaugeFixPacker<Float, recon, true, false>
            (data, reinterpret_cast<complex<Float>*>(sendg_d[d]), 1 - p, d, device::get_stream(4 + d));
        }
        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          qudaMemcpyAsync(send[d], send_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(d));
          qudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(4 + d));
        }

        //compute interior points
        gfixIntPoints.setParity(p);
        gfixIntPoints.apply(device::get_default_stream());
        flop += (double)gfixIntPoints.flops();
        byte += (double)gfixIntPoints.bytes();

        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          qudaStreamSynchronize(device::get_stream(d));
          comm_start(mh_send_fwd[d]);
          qudaStreamSynchronize(device::get_stream(4 + d));
          comm_start(mh_send_back[d]);
        }
        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_recv_back[d]);
          qudaMemcpyAsync(recv_d[d], recv[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(d));
        }
        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_recv_fwd[d]);
          qudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(4 + d));
        }

        for (int d = 0; d < 4; d++) {
          if (!commDimPartitioned(d)) continue;
          GaugeFixPacker<Float, recon, false, false>
            (data, reinterpret_cast<complex<Float>*>(recv_d[d]), p, d, device::get_stream(d));
        }
        for (int d = 0; d < 4; d++ ) {
          if (!commDimPartitioned(d)) continue;
          GaugeFixPacker<Float, recon, false, true>
            (data, reinterpret_cast<complex<Float>*>(recvg_d[d]), 1 - p, d, device::get_stream(4 + d));
        }
        for (int d = 0; d < 4; d++ ) {
          if (!commDimPartitioned(d)) continue;
          comm_wait(mh_send_back[d]);
          comm_wait(mh_send_fwd[d]);
          qudaStreamSynchronize(device::get_stream(d));
          qudaStreamSynchronize(device::get_stream(4 + d));
        }
        qudaStreamSynchronize(device::get_default_stream());
      }

      if ((iter % reunit_interval) == (reunit_interval - 1)) {
        *num_failures_h = 0;
        unitarizeLinks(data, data, num_failures_d);
	if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);
        flop += 4588.0 * data.Volume();
        byte += 2 * data.Bytes();
      }
      GaugeFixQuality.apply(device::get_default_stream());
      flop += (double)GaugeFixQuality.flops();
      byte += (double)GaugeFixQuality.bytes();

      double action = argQ.getAction();
      double diff = abs(action0 - action);
      if ((iter % verbose_interval) == (verbose_interval - 1) && getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
      if (stopWtheta) {
        if (argQ.getTheta() < tolerance) break;
      } else {
        if ( diff < tolerance ) break;
      }
      action0 = action;
    }

    if ((iter % reunit_interval) != 0 )  {
      *num_failures_h = 0;
      unitarizeLinks(data, data, num_failures_d);
      if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);
      flop += 4588.0 * data.Volume();
      byte += 2 * data.Bytes();
    }

    if ((iter % verbose_interval) != 0 ) {
      GaugeFixQuality.apply(device::get_default_stream());
      flop += (double)GaugeFixQuality.flops();
      byte += (double)GaugeFixQuality.bytes();
      double action = argQ.getAction();
      double diff = abs(action0 - action);
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Step: %d\tAction: %.16e\ttheta: %.16e\tDelta: %.16e\n", iter + 1, argQ.getAction(), argQ.getTheta(), diff);
    }

    for (int i = 0; i < 2 && nlinksfaces; i++) managed_free(borderpoints[i]);
    host_free(num_failures_h);

    if ( comm_partitioned() ) {
      data.exchangeExtendedGhost(data.R(),false);
      for ( int d = 0; d < 4; d++ ) {
        if ( commDimPartitioned(d)) {
          comm_free(mh_send_fwd[d]);
          comm_free(mh_send_back[d]);
          comm_free(mh_recv_back[d]);
          comm_free(mh_recv_fwd[d]);
          device_free(send_d[d]);
          device_free(recv_d[d]);
          device_free(sendg_d[d]);
          device_free(recvg_d[d]);
          host_free(hostbuffer_h[d]);
        }
      }
    }

    qudaDeviceSynchronize();
    profileInternalGaugeFixOVR.TPSTOP(QUDA_PROFILE_COMPUTE);
    if (getVerbosity() >= QUDA_SUMMARIZE){
      double secs = profileInternalGaugeFixOVR.Last(QUDA_PROFILE_COMPUTE);
      double gflops = (flop * 1e-9) / (secs);
      double gbytes = byte / (secs * 1e9);
      printfQuda("Time: %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops * comm_size(), gbytes * comm_size());
    }
  }

  template <typename Float, int nColor, QudaReconstructType recon> struct GaugeFixingOVR {
  GaugeFixingOVR(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval,
                 const double relax_boost, const double tolerance, const int reunit_interval, const int stopWtheta)
    {
      if (gauge_dir == 4) {
	if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Starting Landau gauge fixing...\n");
        gaugeFixingOVR<Float, recon, 4>(data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
      } else if (gauge_dir == 3) {
	if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Starting Coulomb gauge fixing...\n");
        gaugeFixingOVR<Float, recon, 3>(data, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
      } else {
        errorQuda("Unexpected gauge_dir = %d", gauge_dir);
      }
    }
  };

  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] data, quda gauge field
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
   */
  void gaugeFixingOVR(GaugeField& data, const int gauge_dir, const int Nsteps, const int verbose_interval, const double relax_boost,
                      const double tolerance, const int reunit_interval, const int stopWtheta)
  {
    instantiate<GaugeFixingOVR>(data, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta);
  }

}   //namespace quda
