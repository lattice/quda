#include <quda_internal.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <pgauge_monte.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/pgauge_exchange.cuh>

namespace quda {

  static void *send_d[4];
  static void *recv_d[4];
  static void *sendg_d[4];
  static void *recvg_d[4];
  static void *hostbuffer_h[4];
  static MsgHandle *mh_recv_back[4];
  static MsgHandle *mh_recv_fwd[4];
  static MsgHandle *mh_send_fwd[4];
  static MsgHandle *mh_send_back[4];
  static int *X;
  static bool init = false;

  /**
   * @brief Release all allocated memory used to exchange data between nodes
   */
  void PGaugeExchangeFree()
  {
    if (comm_partitioned()) {
      if (init) {
        for (int d = 0; d < 4; d++ ) {
          if (commDimPartitioned(d)) {
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
        host_free(X);
        init = false;
      }
    }
  }

  template<typename Float, int nColor, QudaReconstructType recon>
  struct PGaugeExchanger : TunableKernel1D {
    GaugeField &U;
    GaugeFixUnPackArg<Float, recon> arg;
    const char *dim_str[4] = { "0", "1", "2", "3" };
    unsigned int minThreads() const override { return arg.threads.x; }

    PGaugeExchanger(GaugeField& U, const int dir, const int parity) :
      TunableKernel1D(U),
      U(U),
      arg(U)
    {
      if (init) {
        for (int d = 0; d < 4; d++) {
          if (X[d] != U.X()[d]) {
            PGaugeExchangeFree();
            printfQuda("PGaugeExchange needs to be reinitialized...\n");
            break;
          }
        }
      }

      size_t bytes[4];
      void *send[4];
      void *recv[4];
      void *sendg[4];
      void *recvg[4];
      for (int d = 0; d < 4; d++) {
        if (!commDimPartitioned(d)) continue;
        bytes[d] = sizeof(Float) * U.SurfaceCB(d) * recon;
      }

      if (!init) {
        X = (int*)safe_malloc(4 * sizeof(int));
        for (int d = 0; d < 4; d++) X[d] = U.X()[d];

        for (int d = 0; d < 4; d++ ) {
          if (!commDimPartitioned(d)) continue;
          // store both parities and directions in each
          send_d[d] = device_malloc(bytes[d]);
          recv_d[d] = device_malloc(bytes[d]);
          sendg_d[d] = device_malloc(bytes[d]);
          recvg_d[d] = device_malloc(bytes[d]);
          hostbuffer_h[d] = (void*)pinned_malloc(4 * bytes[d]);
          recv[d] = hostbuffer_h[d];
          send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
          recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
          sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];

          mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
          mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
          mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
          mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
        }
        init = true;
      } else {
        for (int d = 0; d < 4; d++ ) {
          if (!commDimPartitioned(d)) continue;
          recv[d] = hostbuffer_h[d];
          send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
          recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
          sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];
        }
      }

      qudaDeviceSynchronize();
      for (int d = 0; d < 4; d++) {
        if (!commDimPartitioned(d)) continue;
        comm_start(mh_recv_back[d]);
        comm_start(mh_recv_fwd[d]);

        arg.threads.x = U.SurfaceCB(d);
        arg.parity = parity;
        arg.face = d;
        arg.dir = dir;

        //extract top face
        arg.pack = true;
        arg.array = reinterpret_cast<complex<Float>*>(send_d[d]);
        arg.borderid = X[d] - U.R()[d] - 1;
        apply(device::get_stream(0));

        //extract bottom
        arg.array = reinterpret_cast<complex<Float>*>(sendg_d[d]);
        arg.borderid = U.R()[d];
        apply(device::get_stream(1));

        qudaMemcpyAsync(send[d], send_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(0));
        qudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], qudaMemcpyDeviceToHost, device::get_stream(1));

        qudaStreamSynchronize(device::get_stream(0));
        comm_start(mh_send_fwd[d]);

        qudaStreamSynchronize(device::get_stream(1));
        comm_start(mh_send_back[d]);

        comm_wait(mh_recv_back[d]);
        qudaMemcpyAsync(recv_d[d], recv[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(0));

        // insert
        arg.pack = false;
        arg.array = reinterpret_cast<complex<Float>*>(recv_d[d]);
        arg.borderid = U.R()[d] - 1;
        apply(device::get_stream(0));

        comm_wait(mh_recv_fwd[d]);
        qudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], qudaMemcpyHostToDevice, device::get_stream(1));

        arg.array = reinterpret_cast<complex<Float>*>(recvg_d[d]);
        arg.borderid = X[d] - U.R()[d];
        apply(device::get_stream(1));

        comm_wait(mh_send_back[d]);
        comm_wait(mh_send_fwd[d]);
        qudaStreamSynchronize(device::get_stream(0));
        qudaStreamSynchronize(device::get_stream(1));
      }
      qudaDeviceSynchronize();
    }

    TuneKey tuneKey() const override
    {
      std::string aux2 = std::string(aux) + ",dim=" + dim_str[arg.face] + ",geo_dir=" + dim_str[arg.dir] +
        (arg.pack ? ",extract" : ",insert");
      return TuneKey(vol, typeid(*this).name(), aux2.c_str());
    }

    void apply(const qudaStream_t &stream) override
    {
      auto tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<unpacker>(tp, stream, arg);
    }

    long long bytes() const override { return 2 * U.SurfaceCB(arg.face) * U.Reconstruct() * U.Precision(); }
  };

  void PGaugeExchange(GaugeField& U, const int dir, const int parity)
  {
    if (comm_partitioned()) instantiate<PGaugeExchanger>(U, dir, parity);
  }

}
