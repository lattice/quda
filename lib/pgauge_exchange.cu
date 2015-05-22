#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>

#include <hisq_links_quda.h> //reunit gauge links!!!!!

#include <comm_quda.h>


#define BORDER_RADIUS 2

namespace quda {


static int numParams = 18;

#define LAUNCH_KERNEL_GAUGEFIX(kernel, tp, stream, arg, parity, ...)     \
  if(tp.block.z==0){\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<0, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:                \
    kernel<0, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:                \
    kernel<0, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<0, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }\
  else{\
  switch (tp.block.x) {             \
  case 256:                \
    kernel<1, 32,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 512:                \
    kernel<1, 64,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 768:                \
    kernel<1, 96,__VA_ARGS__>           \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  case 1024:               \
    kernel<1, 128,__VA_ARGS__>            \
      <<< tp.grid.x, tp.block.x, tp.shared_bytes, stream >>>(arg, parity);   \
    break;                \
  default:                \
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }\
  }







template <typename Gauge>
struct GaugeFixUnPackArg {
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  GaugeFixUnPackArg(Gauge &dataOr, cudaGaugeField &data)
    : dataOr(dataOr) {
/*#ifdef MULTI_GPU
    if(comm_size() == 1){
      for(int dir=0; dir<4; ++dir) border[dir] = 0;
      for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
    }
    else{
      for(int dir=0; dir<4; ++dir){
        if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
        else border[dir] = 0;
      }
      for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
    }
#else*/
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
//#endif
  }
};


template<int NElems, typename Float, typename Gauge, bool pack>
__global__ void Kernel_UnPack(int size, GaugeFixUnPackArg<Gauge> arg, \
  typename ComplexTypeId<Float>::Type *array, int parity, int face, int dir, int borderid){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size) return;
  int X[4]; 
  for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];
  int x[4];
  int za, xodd;
  switch(face){
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
  int id = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  typedef typename ComplexTypeId<Float>::Type Cmplx;
  typedef typename mapper<Float>::type RegType;
  RegType tmp[NElems];
  RegType data[18];
  if(pack){
    arg.dataOr.load(data, id, dir, parity);
    arg.dataOr.reconstruct.Pack(tmp, data, id);
    for(int i=0; i<NElems/2; ++i) array[idx + size * i] = ((Cmplx*)tmp)[i];
  }
else{
    for(int i=0; i<NElems/2; ++i) ((Cmplx*)tmp)[i] = array[idx + size * i];
    arg.dataOr.reconstruct.Unpack(data, tmp, id, dir, 0);
    arg.dataOr.save(data, id, dir, parity);
  }
}

static  void *send[4];
static  void *recv[4];
static  void *sendg[4];
static  void *recvg[4];
static  void *send_d[4];
static  void *recv_d[4];
static  void *sendg_d[4];
static  void *recvg_d[4];
static  void *hostbuffer_h[4];
static  cudaStream_t GFStream[2];
static  size_t offset[4];
static  size_t bytes[4];
static  size_t faceVolume[4];
static  size_t faceVolumeCB[4];
  // do the exchange
static  MsgHandle *mh_recv_back[4];
static  MsgHandle *mh_recv_fwd[4];
static  MsgHandle *mh_send_fwd[4];
static  MsgHandle *mh_send_back[4];
static  int X[4];
static  dim3 block[4];
static  dim3 grid[4];
static bool notinitialized = true;



void PGaugeExchangeFree(){
#ifdef MULTI_GPU
  if(comm_size() > 1){
    cudaStreamDestroy(GFStream[0]);
    cudaStreamDestroy(GFStream[1]);
    for (int d=0; d<4; d++) {
      if (commDimPartitioned(d)) {
        comm_free(mh_send_fwd[d]);
        comm_free(mh_send_back[d]);
        comm_free(mh_recv_back[d]);
        comm_free(mh_recv_fwd[d]);
        device_free(send_d[d]);
        device_free(recv_d[d]);
        device_free(sendg_d[d]);
        device_free(recvg_d[d]);
        #ifndef GPU_COMMS
        free(hostbuffer_h[d]);
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
  if(notinitialized == false){
    for (int d=0; d<4; d++){
    if(X[d] != data.X()[d]){
      PGaugeExchangeFree();
      notinitialized = true;
      printfQuda("PGaugeExchange needs to be reinitialized...\n");
      break;
    }
    }
  }
  if(notinitialized){
    for (int d=0; d<4; d++){
        X[d] = data.X()[d];
    }
    for (int i=0; i<4; i++) {
      faceVolume[i] = 1;
      for (int j=0; j<4; j++) {
        if (i==j) continue;
        faceVolume[i] *= X[j];
      }
      faceVolumeCB[i] = faceVolume[i]/2;
    }

    cudaStreamCreate(&GFStream[0]);
    cudaStreamCreate(&GFStream[1]);
    for (int d=0; d<4; d++) {
      if (!commDimPartitioned(d)) continue;
      // store both parities and directions in each
      offset[d] = faceVolumeCB[d] * NElems;
      bytes[d] =  sizeof(Float) * offset[d];
      send_d[d] = device_malloc(bytes[d]);
      recv_d[d] = device_malloc(bytes[d]);
      sendg_d[d] = device_malloc(bytes[d]);
      recvg_d[d] = device_malloc(bytes[d]);
      #ifndef GPU_COMMS
      hostbuffer_h[d] = (void*)malloc(4*bytes[d]);
      #endif
      block[d] = make_uint3(128, 1, 1);
      grid[d] = make_uint3((faceVolumeCB[d] + block[d].x - 1) / block[d].x, 1, 1);
    }

    for (int d=0; d<4; d++) {
      if (!commDimPartitioned(d)) continue;
      #ifdef GPU_COMMS
      recv[d] = recv_d[d];
      send[d] = send_d[d];
      recvg[d] = recvg_d[d];
      sendg[d] = sendg_d[d];
      #else
      recv[d] = hostbuffer_h[d];
      send[d] = static_cast<char*>(hostbuffer_h[d]) + bytes[d];
      recvg[d] = static_cast<char*>(hostbuffer_h[d]) + 3*bytes[d];
      sendg[d] = static_cast<char*>(hostbuffer_h[d]) + 2*bytes[d];      
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


    for (int d=0; d<4; d++) {
      if (!commDimPartitioned(d)) continue;
      comm_start(mh_recv_back[d]);  
      comm_start(mh_recv_fwd[d]); 
   
      //extract top face
      Kernel_UnPack<NElems, Float, Gauge, true><<<grid[d], block[d], 0, GFStream[0]>>>(\
        faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(send_d[d]), parity, d, dir, X[d]-3);
      //extract bottom
      Kernel_UnPack<NElems, Float, Gauge, true><<<grid[d], block[d], 0, GFStream[1]>>>(\
        faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(sendg_d[d]), parity, d, dir, 2);
      
    #ifndef GPU_COMMS
      cudaMemcpyAsync(send[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[0]);
      cudaMemcpyAsync(sendg[d], sendg_d[d], bytes[d], cudaMemcpyDeviceToHost, GFStream[1]);
    #endif
      cudaStreamSynchronize(GFStream[0]);
      comm_start(mh_send_fwd[d]);

      cudaStreamSynchronize(GFStream[1]);
      comm_start(mh_send_back[d]);
 
    #ifndef GPU_COMMS
      comm_wait(mh_recv_back[d]);
      cudaMemcpyAsync(recv_d[d], recv[d], bytes[d], cudaMemcpyHostToDevice, GFStream[0]);
    #endif
      #ifdef GPU_COMMS
      comm_wait(mh_recv_back[d]);
      #endif
      Kernel_UnPack<NElems, Float, Gauge, false><<<grid[d], block[d], 0, GFStream[0]>>>(\
        faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(recv_d[d]), parity, d, dir, 1);

    #ifndef GPU_COMMS
      comm_wait(mh_recv_fwd[d]);
      cudaMemcpyAsync(recvg_d[d], recvg[d], bytes[d], cudaMemcpyHostToDevice, GFStream[1]);
    #endif

      #ifdef GPU_COMMS
      comm_wait(mh_recv_fwd[d]);
      #endif
      Kernel_UnPack<NElems, Float, Gauge, false><<<grid[d], block[d], 0, GFStream[1]>>>(\
        faceVolumeCB[d], dataexarg, reinterpret_cast<typename ComplexTypeId<Float>::Type*>(recvg_d[d]), parity, d, dir, X[d] - 2); 

      comm_wait(mh_send_back[d]);
      comm_wait(mh_send_fwd[d]);
      cudaStreamSynchronize(GFStream[0]);
      cudaStreamSynchronize(GFStream[1]);
  }
  checkCudaError();
  cudaDeviceSynchronize();
  #endif

}


template<typename Float>
void PGaugeExchange( cudaGaugeField& data, const int dir, const int parity) {


  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      numParams = 18;
      PGaugeExchange<Float, 18>(FloatNOrder<Float, 18, 2, 18>(data), data, dir, parity);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      numParams = 12;
      PGaugeExchange<Float, 12>(FloatNOrder<Float, 18, 2, 12>(data), data, dir, parity);
    
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      numParams = 8;
      PGaugeExchange<Float, 8>(FloatNOrder<Float, 18, 2,  8>(data), data, dir, parity);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      numParams = 18;
      PGaugeExchange<Float, 18>(FloatNOrder<Float, 18, 4, 18>(data), data, dir, parity);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      numParams = 12;
      PGaugeExchange<Float, 12>(FloatNOrder<Float, 18, 4, 12>(data), data, dir, parity);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      numParams = 8;
      PGaugeExchange<Float, 8>(FloatNOrder<Float, 18, 4,  8>(data), data, dir, parity);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}

  void PGaugeExchange( cudaGaugeField& data, const int dir, const int parity) {

#ifdef MULTI_GPU
  if(comm_size() > 1){
    if(data.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported\n");
    }
    if (data.Precision() == QUDA_SINGLE_PRECISION) {
      PGaugeExchange<float> (data, dir, parity);
    } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
      PGaugeExchange<double>(data, dir, parity);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
  }
#endif
  }
}