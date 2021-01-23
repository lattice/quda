#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <comm_quda.h>
#include <pgauge_monte.h>
#include <instantiate.h>

namespace quda {

  template <typename Float, QudaReconstructType recon>
  struct GaugeFixUnPackArg {
    int X[4]; // grid dimensions
    using Gauge = typename gauge_mapper<Float, recon>::type;
    Gauge dataOr;
    int size;
    complex<Float> *array;
    int parity;
    int face;
    int dir;
    int borderid;
    GaugeFixUnPackArg(GaugeField & data)
      : dataOr(data)
    {
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
    }
  };

  template <int NElems, typename Float, bool pack, typename Arg>
  __global__ void Kernel_UnPack(Arg arg)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= arg.size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    switch ( arg.face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = arg.borderid;
      xodd = (arg.borderid + x[2] + x[3] + arg.parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = arg.borderid;
      xodd = (arg.borderid  + x[2] + x[3] + arg.parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = arg.borderid;
      xodd = (arg.borderid  + x[1] + x[3] + arg.parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = arg.borderid;
      xodd = (arg.borderid  + x[1] + x[2] + arg.parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }

    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Complex;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[NElems];
    Complex data[9];

    if (pack) {
      arg.dataOr.load(data, id, arg.dir, arg.parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < NElems / 2; ++i ) arg.array[idx + arg.size * i] = Complex(tmp[2*i+0], tmp[2*i+1]);
    } else {
      for ( int i = 0; i < NElems / 2; ++i ) {
        tmp[2*i+0] = arg.array[idx + arg.size * i].real();
        tmp[2*i+1] = arg.array[idx + arg.size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, arg.dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, arg.dir, arg.parity);
    }
  }

  static void *send_d[4];
  static void *recv_d[4];
  static void *sendg_d[4];
  static void *recvg_d[4];
  static void *hostbuffer_h[4];
  static qudaStream_t GFStream[2];
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
    if ( comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3) ) {
      if (init) {
        cudaStreamDestroy(GFStream[0]);
        cudaStreamDestroy(GFStream[1]);
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

  template<typename Float, int nColor, QudaReconstructType recon> struct PGaugeExchanger {
    PGaugeExchanger(GaugeField& data, const int dir, const int parity)
    {
      if (init) {
        for (int d = 0; d < 4; d++) {
          if (X[d] != data.X()[d]) {
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
        bytes[d] =  sizeof(Float) * data.SurfaceCB(d) * recon;
      }

      if (!init) {
        X = (int*)safe_malloc(4 * sizeof(int));
        for (int d = 0; d < 4; d++) X[d] = data.X()[d];

        cudaStreamCreate(&GFStream[0]);
        cudaStreamCreate(&GFStream[1]);
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

      GaugeFixUnPackArg<Float, recon> arg(data);

      qudaDeviceSynchronize();
      for (int d = 0; d < 4; d++) {
        if ( !commDimPartitioned(d)) continue;
        comm_start(mh_recv_back[d]);
        comm_start(mh_recv_fwd[d]);

        TuneParam tp;
        tp.block = make_uint3(128, 1, 1);
        tp.grid = make_uint3((data.SurfaceCB(d) + tp.block.x - 1) / tp.block.x, 1, 1);

        arg.size = data.SurfaceCB(d);
        arg.parity = parity;
        arg.face = d;
        arg.dir = dir;

        //extract top face
        arg.array = reinterpret_cast<complex<Float>*>(send_d[d]); 
        arg.borderid = X[d] - data.R()[d] - 1;
        qudaLaunchKernel(Kernel_UnPack<recon, Float, true, decltype(arg)>, tp, GFStream[0], arg);

        //extract bottom
        arg.array = reinterpret_cast<complex<Float>*>(sendg_d[d]);
        arg.borderid = data.R()[d];
        qudaLaunchKernel(Kernel_UnPack<recon, Float, true, decltype(arg)>, tp, GFStream[1], arg);

        qudaMemcpyAsync(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[0]);
        qudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[1]);

        qudaStreamSynchronize(GFStream[0]);
        comm_start(mh_send_fwd[d]);

        qudaStreamSynchronize(GFStream[1]);
        comm_start(mh_send_back[d]);

        comm_wait(mh_recv_back[d]);
        qudaMemcpyAsync(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice, GFStream[0]);

        arg.array = reinterpret_cast<complex<Float>*>(recv_d[d]);
        arg.borderid = data.R()[d] - 1;
        qudaLaunchKernel(Kernel_UnPack<recon, Float, false, decltype(arg)>, tp, GFStream[0], arg);

        comm_wait(mh_recv_fwd[d]);
        qudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice, GFStream[1]);

        arg.array = reinterpret_cast<complex<Float>*>(recvg_d[d]);
        arg.borderid = X[d] - data.R()[d];
        qudaLaunchKernel(Kernel_UnPack<recon, Float, false, decltype(arg)>, tp, GFStream[1], arg);

        comm_wait(mh_send_back[d]);
        comm_wait(mh_send_fwd[d]);
        qudaStreamSynchronize(GFStream[0]);
        qudaStreamSynchronize(GFStream[1]);
      }
      qudaDeviceSynchronize();
    }
  };

  void PGaugeExchange(GaugeField& data, const int dir, const int parity)
  {
#ifdef GPU_GAUGE_ALG
    if ( comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3) ) {
      instantiate<PGaugeExchanger>(data, dir, parity);
    }
#else
    errorQuda("Pure gauge code has not been built");
#endif
  }
}
