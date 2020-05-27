#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub_helper.cuh>
#include <launch_kernel.cuh>
#include <comm_quda.h>


namespace quda {

#ifdef GPU_GAUGE_ALG

#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, arg, parity, ...)     \
  if ( tp.block.z == 0 ) { \
    switch ( tp.block.x ) {             \
    case 256:                \
      kernel<0, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 512:                \
      kernel<0, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 768:                \
      kernel<0, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 1024:               \
      kernel<0, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    default:                \
      errorQuda("%s not implemented for %d threads", # kernel, tp.block.x); \
    } \
  } \
  else{ \
    switch ( tp.block.x ) {             \
    case 256:                \
      kernel<1, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 512:                \
      kernel<1, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 768:                \
      kernel<1, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    case 1024:               \
      kernel<1, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>> (arg, parity);   \
      break;                \
    default:                \
      errorQuda("%s not implemented for %d threads", # kernel, tp.block.x); \
    } \
  }


  template <typename Gauge>
  struct GaugeFixUnPackArg {
    int X[4]; // grid dimensions
    Gauge dataOr;
    GaugeFixUnPackArg(Gauge & dataOr, cudaGaugeField & data)
      : dataOr(dataOr) {
      for ( int dir = 0; dir < 4; ++dir ) X[dir] = data.X()[dir];
    }
  };


  template<int NElems, typename Float, typename Gauge, bool pack>
  __global__ void Kernel_UnPack(int size, GaugeFixUnPackArg<Gauge> arg, \
                                complex<Float> *array, int parity, int face, int dir, int borderid){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= size ) return;
    int X[4];
    for ( int dr = 0; dr < 4; ++dr ) X[dr] = arg.X[dr];
    int x[4];
    int za, xodd;
    switch ( face ) {
    case 0: //X FACE
      za = idx / ( X[1] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[0] = borderid;
      xodd = (borderid + x[2] + x[3] + parity) & 1;
      x[1] = (2 * idx + xodd)  - za * X[1];
      break;
    case 1: //Y FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[2];
      x[2] = za - x[3] * X[2];
      x[1] = borderid;
      xodd = (borderid  + x[2] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 2: //Z FACE
      za = idx / ( X[0] / 2);
      x[3] = za / X[1];
      x[1] = za - x[3] * X[1];
      x[2] = borderid;
      xodd = (borderid  + x[1] + x[3] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    case 3: //T FACE
      za = idx / ( X[0] / 2);
      x[2] = za / X[1];
      x[1] = za - x[2] * X[1];
      x[3] = borderid;
      xodd = (borderid  + x[1] + x[2] + parity) & 1;
      x[0] = (2 * idx + xodd)  - za * X[0];
      break;
    }
    int id = (((x[3] * X[2] + x[2]) * X[1] + x[1]) * X[0] + x[0]) >> 1;
    typedef complex<Float> Complex;
    typedef typename mapper<Float>::type RegType;
    RegType tmp[NElems];
    Complex data[9];
    if ( pack ) {
      arg.dataOr.load(data, id, dir, parity);
      arg.dataOr.reconstruct.Pack(tmp, data, id);
      for ( int i = 0; i < NElems / 2; ++i ) array[idx + size * i] = Complex(tmp[2*i+0], tmp[2*i+1]);
    }
    else{
      for ( int i = 0; i < NElems / 2; ++i ) {
        tmp[2*i+0] = array[idx + size * i].real();
        tmp[2*i+1] = array[idx + size * i].imag();
      }
      arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0, arg.dataOr.X, arg.dataOr.R);
      arg.dataOr.save(data, id, dir, parity);
    }
  }

#ifdef MULTI_GPU
  static void *send[4];
  static void *recv[4];
  static void *sendg[4];
  static void *recvg[4];
  static void *send_d[4];
  static void *recv_d[4];
  static void *sendg_d[4];
  static void *recvg_d[4];
  static void *hostbuffer_h[4];
  static qudaStream_t GFStream[2];
  static size_t offset[4];
  static size_t bytes[4];
  static size_t faceVolume[4];
  static size_t faceVolumeCB[4];
  // do the exchange
  static MsgHandle *mh_recv_back[4];
  static MsgHandle *mh_recv_fwd[4];
  static MsgHandle *mh_send_fwd[4];
  static MsgHandle *mh_send_back[4];
  static int X[4];
  static dim3 block[4];
  static dim3 grid[4];
  static bool notinitialized = true;
#endif // MULTI_GPU

  /**
   * @brief Release all allocated memory used to exchange data between nodes
   */
  void PGaugeExchangeFree(){
#ifdef MULTI_GPU
    if ( comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3) ) {
      cudaStreamDestroy(GFStream[0]);
      cudaStreamDestroy(GFStream[1]);
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
        #ifndef GPU_COMMS
          host_free(hostbuffer_h[d]);
        #endif
        }
      }
      notinitialized = true;
    }
#endif
  }


  template<typename Float, int NElems, typename Gauge>
  void PGaugeExchange( Gauge dataOr,  cudaGaugeField& data, const int dir, const int parity) {


#ifdef MULTI_GPU
    if ( notinitialized == false ) {
      for ( int d = 0; d < 4; d++ ) {
        if ( X[d] != data.X()[d] ) {
          PGaugeExchangeFree();
          notinitialized = true;
          printfQuda("PGaugeExchange needs to be reinitialized...\n");
          break;
        }
      }
    }
    if ( notinitialized ) {
      for ( int d = 0; d < 4; d++ ) {
        X[d] = data.X()[d];
      }
      for ( int i = 0; i < 4; i++ ) {
        faceVolume[i] = 1;
        for ( int j = 0; j < 4; j++ ) {
          if ( i == j ) continue;
          faceVolume[i] *= X[j];
        }
        faceVolumeCB[i] = faceVolume[i] / 2;
      }

      cudaStreamCreate(&GFStream[0]);
      cudaStreamCreate(&GFStream[1]);
      for ( int d = 0; d < 4; d++ ) {
        if ( !commDimPartitioned(d)) continue;
        // store both parities and directions in each
        offset[d] = faceVolumeCB[d] * NElems;
        bytes[d] =  sizeof(Float) * offset[d];
        send_d[d] = device_malloc(bytes[d]);
        recv_d[d] = device_malloc(bytes[d]);
        sendg_d[d] = device_malloc(bytes[d]);
        recvg_d[d] = device_malloc(bytes[d]);
      #ifndef GPU_COMMS
        hostbuffer_h[d] = (void*)pinned_malloc(4 * bytes[d]);
      #endif
        block[d] = make_uint3(128, 1, 1);
        grid[d] = make_uint3((faceVolumeCB[d] + block[d].x - 1) / block[d].x, 1, 1);
      }

      for ( int d = 0; d < 4; d++ ) {
        if ( !commDimPartitioned(d)) continue;
      #ifdef GPU_COMMS
        recv[d] = recv_d[d];
        send[d] = send_d[d];
        recvg[d] = recvg_d[d];
        sendg[d] = sendg_d[d];
      #else
        recv[d] = hostbuffer_h[d];
        send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
        recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3 * bytes[d];
        sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2 * bytes[d];
      #endif
        // look into storing these for later
        mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
        mh_recv_fwd[d]  = comm_declare_receive_relative(recvg[d], d, +1, bytes[d]);
        mh_send_back[d] = comm_declare_send_relative(sendg[d], d, -1, bytes[d]);
        mh_send_fwd[d]  = comm_declare_send_relative(send[d], d, +1, bytes[d]);
      }
      notinitialized = false;
    }
    GaugeFixUnPackArg<Gauge> dataexarg(dataOr, data);


    for ( int d = 0; d < 4; d++ ) {
      if ( !commDimPartitioned(d)) continue;
      comm_start(mh_recv_back[d]);
      comm_start(mh_recv_fwd[d]);

      //extract top face
      Kernel_UnPack<NElems, Float, Gauge, true> <<< grid[d], block[d], 0, GFStream[0] >>>
	(faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(send_d[d]), parity, d, dir, X[d] -  data.R()[d] - 1);
      //extract bottom
      Kernel_UnPack<NElems, Float, Gauge, true> <<< grid[d], block[d], 0, GFStream[1] >>>
	(faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(sendg_d[d]), parity, d, dir, data.R()[d]);

    #ifndef GPU_COMMS
      cudaMemcpyAsync(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[0]);
      cudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[1]);
    #endif
      qudaStreamSynchronize(GFStream[0]);
      comm_start(mh_send_fwd[d]);

      qudaStreamSynchronize(GFStream[1]);
      comm_start(mh_send_back[d]);

    #ifndef GPU_COMMS
      comm_wait(mh_recv_back[d]);
      cudaMemcpyAsync(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice, GFStream[0]);
    #endif
      #ifdef GPU_COMMS
      comm_wait(mh_recv_back[d]);
      #endif
      Kernel_UnPack<NElems, Float, Gauge, false> <<< grid[d], block[d], 0, GFStream[0] >>>
	(faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recv_d[d]), parity, d, dir, data.R()[d] - 1);

    #ifndef GPU_COMMS
      comm_wait(mh_recv_fwd[d]);
      cudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice, GFStream[1]);
    #endif

      #ifdef GPU_COMMS
      comm_wait(mh_recv_fwd[d]);
      #endif
      Kernel_UnPack<NElems, Float, Gauge, false> <<< grid[d], block[d], 0, GFStream[1] >>>
	(faceVolumeCB[d], dataexarg, reinterpret_cast<complex<Float>*>(recvg_d[d]), parity, d, dir, X[d] - data.R()[d]);

      comm_wait(mh_send_back[d]);
      comm_wait(mh_send_fwd[d]);
      qudaStreamSynchronize(GFStream[0]);
      qudaStreamSynchronize(GFStream[1]);
    }
    checkCudaError();
    qudaDeviceSynchronize();
  #endif

  }


  template<typename Float>
  void PGaugeExchange( cudaGaugeField& data, const int dir, const int parity) {


    // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
    // Need to fix this!!
    if ( data.isNative() ) {
      if ( data.Reconstruct() == QUDA_RECONSTRUCT_NO ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	PGaugeExchange<Float, 18>(G(data), data, dir, parity);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_12 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
        PGaugeExchange<Float, 12>(G(data), data, dir, parity);
      } else if ( data.Reconstruct() == QUDA_RECONSTRUCT_8 ) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
        PGaugeExchange<Float, 8>(G(data), data, dir, parity);
      } else {
        errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
      }
    } else {
      errorQuda("Invalid Gauge Order\n");
    }
  }

#endif // GPU_GAUGE_ALG

  void PGaugeExchange( cudaGaugeField& data, const int dir, const int parity) {

#ifdef GPU_GAUGE_ALG
#ifdef MULTI_GPU
    if ( comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3) ) {
      if ( data.Precision() == QUDA_HALF_PRECISION ) {
        errorQuda("Half precision not supported\n");
      }
      if ( data.Precision() == QUDA_SINGLE_PRECISION ) {
        PGaugeExchange<float> (data, dir, parity);
      } else if ( data.Precision() == QUDA_DOUBLE_PRECISION ) {
        PGaugeExchange<double>(data, dir, parity);
      } else {
        errorQuda("Precision %d not supported", data.Precision());
      }
    }
#endif
#else
    errorQuda("Pure gauge code has not been built");
#endif
  }
}
