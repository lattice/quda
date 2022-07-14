#include <gauge_field.h>
#include <gauge_tools.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <kernels/gauge_polyakov_loop.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugeInsertTimeslice : public TunableKernel2D {
    GaugeField &u;
    const GaugeField &s;
    int timeslice;
    unsigned int minThreads() const { return s.LocalVolumeCB(); }

  public:
    GaugeInsertTimeslice(GaugeField &u, const GaugeField &s, int timeslice) :
      TunableKernel2D(u, 2),
      u(u),
      s(s),
      timeslice(timeslice)
    {
      strcat(aux, ",4d_vol=");
      strcat(aux, u.VolString());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeInsertTimesliceArg<Float, nColor, recon> arg(u, s, timeslice);
      launch<InsertTimeslice>(tp, stream, arg);
    }

    long long flops() const
    {
      // just a copy
      return 0;
    }

    long long bytes() const {
      // load timeslice, store into u
      return 2ll * s.Bytes();
    }
  };

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePolyakovLoopProduct : public TunableKernel2D {
    GaugeField &product_field;
    const GaugeField &u;
    unsigned int minThreads() const { return product_field.LocalVolumeCB(); }

  public:
    GaugePolyakovLoopProduct(const GaugeField &u, GaugeField &product_field) :
      TunableKernel2D(u, 2),
      product_field(product_field),
      u(u)
    {
      strcat(aux, ",4d_vol=");
      strcat(aux, u.VolString());
      strcat(aux, ",3d_vol=");
      strcat(aux, product_field.VolString());

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePolyakovLoopProductArg<Float, nColor, recon> arg(product_field, u);
      launch<PolyakovLoopProduct>(tp, stream, arg);
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      // multiplies for each loop
      return mat_mul_flops * u.Volume() / 4;
    }

    long long bytes() const {
      // links * one LatticeColorMatrix worth of data
      return u.Bytes() / 4 + product_field.Bytes();
    }
  };

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePolyakovLoopTrace : public TunableReduction2D {
    const GaugeField &u;
    using reduce_t = array<double, 2>;
    reduce_t &ploop;

  public:
    GaugePolyakovLoopTrace(const GaugeField &u, array<double, 2> &ploop) :
      TunableReduction2D(u),
      u(u),
      ploop(ploop)
    {
      if (u.Geometry() != QUDA_SCALAR_GEOMETRY && u.Geometry() != QUDA_VECTOR_GEOMETRY)
        errorQuda("Invalid geometry %d in Polyakov loop calculation", u.Geometry());
      strcat(aux, ",4d_vol=");
      strcat(aux, u.VolString());
      strcat(aux, ",geometry=");
      char aux2[3];
      u32toa(aux2, u.Geometry());
      strcat(aux, aux2);

      apply(device::get_default_stream());

      // 3-d volume normalization is done outside this class
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePolyakovLoopTraceArg<Float, nColor, recon> arg(u);
      launch<PolyakovLoopTrace>(ploop, tp, stream, arg);
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      int trace_direction = 3; // update for vector geometry, non-temporal loops
      // multiplies for each loop plus traces
      return mat_mul_flops * u.Volume() / u.Geometry() + 2 * Nc * u.Volume() / u.X()[trace_direction];
    }

    long long bytes() const {
      // links * one LatticeColorMatrix
      return u.Bytes() / u.Geometry();
    }
  };

  // to avoid multiple instantiations... FIXME doxygen
  void gaugeInsertTimeslice(GaugeField &u, GaugeField &s, int timeslice) {
    instantiate<GaugeInsertTimeslice>(u, s, timeslice);
  }

  void gaugePolyakovLoop(double ploop[2], const GaugeField& u, int dir, TimeProfile &profile) {

    if (dir != 3) errorQuda("Unsupported direction %d", dir);

    // output array
    array<double, 2> loop;

    // If the dir dimension isn't partitioned, we can just do a quick compute + reduce,
    // otherwise we need a gather workflow
    std::unique_ptr<GaugeField> condensed_field;

    // If the dir dimension isn't partitioned, we can just do a quick compute + reduce
    if (commDimPartitioned(dir)) {

      // Form a staging gauge field where each "t" slice corresponds to one rank in the "dir"
      // direction. Note that odd dimensions are carefully legal in the `t` dimension in this context
      // Note that we promote the precision to double for numerical stability
      profile.TPSTART(QUDA_PROFILE_INIT);
      GaugeFieldParam gParam(u);
      lat_dim_t x;
      for (int d = 0; d < 3; d++) x[d] = u.X()[d];
      x[3] = 1;
      gParam.x = x;
      gParam.create = QUDA_NULL_FIELD_CREATE;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.location = QUDA_CUDA_FIELD_LOCATION;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      gParam.setPrecision(QUDA_DOUBLE_PRECISION);

      std::unique_ptr<GaugeField> product_field = std::make_unique<cudaGaugeField>(gParam);
      GaugeField& product_field_ref = reinterpret_cast<GaugeField&>(*product_field.get());

      // Create the field we reduce into
      x[3] = comm_dim(3);
      gParam.x = x;
      gParam.create = QUDA_NULL_FIELD_CREATE;
      condensed_field = std::make_unique<cudaGaugeField>(gParam);
      GaugeField& condensed_field_ref = reinterpret_cast<GaugeField&>(*condensed_field.get());
      profile.TPSTOP(QUDA_PROFILE_INIT);

      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      // Compute my local timeslice
      instantiate<GaugePolyakovLoopProduct, ReconstructNo12>(u, product_field_ref);

      gaugeInsertTimeslice(condensed_field_ref, product_field_ref, comm_coord(3));
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

      // we can skip all of this if we're just doing a partition test
      if (comm_dim(dir) > 1) {
        // Send/gather other data; need to make this support direct GPU comms
        // I wonder if I can just use the strided send APIs?
        profile.TPSTART(QUDA_PROFILE_INIT);
        auto bytes = product_field_ref.TotalBytes();
        void* v_double_buffer[2];
        MsgHandle *recv_handle[2];
        MsgHandle *send_handle[2];
        for (int i = 0; i < 2; i++) {
          v_double_buffer[i] = pinned_malloc(bytes);
          // receive from in front of me, send behind
          recv_handle[i] = comm_declare_receive_relative(v_double_buffer[i], dir, 1, bytes);
          send_handle[i] = comm_declare_send_relative(v_double_buffer[i], dir, -1, bytes);
        }
        int SEND_BUFFER = 0;
        int RECV_BUFFER = 1;
        profile.TPSTOP(QUDA_PROFILE_INIT);

        // prepare first send
        profile.TPSTART(QUDA_PROFILE_D2H);
        product_field_ref.copy_to_buffer(v_double_buffer[SEND_BUFFER]);
        profile.TPSTOP(QUDA_PROFILE_D2H);

        // kick off
        for (int t = 1; t < comm_dim(3); t++) {
          profile.TPSTART(QUDA_PROFILE_COMMS);
          // post a receive from rank behind me
          comm_start(recv_handle[RECV_BUFFER]);
          comm_start(send_handle[SEND_BUFFER]);

          // wait
          comm_wait(recv_handle[RECV_BUFFER]);
          comm_wait(send_handle[SEND_BUFFER]);
          profile.TPSTOP(QUDA_PROFILE_COMMS);

          // copy the received buffer into our staging field
          profile.TPSTART(QUDA_PROFILE_H2D);
          product_field_ref.copy_from_buffer(v_double_buffer[RECV_BUFFER]);
          profile.TPSTOP(QUDA_PROFILE_H2D);

          // insert
          profile.TPSTART(QUDA_PROFILE_COMPUTE);
          gaugeInsertTimeslice(condensed_field_ref, product_field_ref, (comm_coord(3) + t) % comm_dim(3));
          profile.TPSTOP(QUDA_PROFILE_COMPUTE);

          // swap buffers
          SEND_BUFFER ^= 1;
          RECV_BUFFER ^= 1;
        }

        // clean up
        profile.TPSTART(QUDA_PROFILE_FREE);
        for (int i = 0; i < 2; i++) {
          comm_free(recv_handle[i]);
          comm_free(send_handle[i]);
          host_free(v_double_buffer[i]);
        }
        profile.TPSTOP(QUDA_PROFILE_FREE);
      }

    }

    const GaugeField& G = commDimPartitioned(dir) ? const_cast<const GaugeField&>(*condensed_field.get()) : u;

    // Trace over remaining bits
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugePolyakovLoopTrace, ReconstructNo12>(G, loop);
    // We normalize by the 3-d volume, times the 4-d communications dim to cancel out redundant counting
    long vol3d = u.Volume() * comm_dim(0) * comm_dim(1) * comm_dim(2) * comm_dim(3) / u.X()[3];
    ploop[0] = loop[0] / vol3d;
    ploop[1] = loop[1] / vol3d;
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

  }

} // namespace quda
