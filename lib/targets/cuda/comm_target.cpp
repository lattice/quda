#include <comm_quda.h>
#include <quda_api.h>
#include <quda_cuda_api.h>
#include <algorithm>
#include <shmem_helper.cuh>

namespace quda
{

#define CHECK_CUDA_ERROR(func)                                                                                         \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

  bool comm_peer2peer_possible(int local_gpuid, int neighbor_gpuid)
  {
    int canAccessPeer[2];
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer[0], local_gpuid, neighbor_gpuid));
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, local_gpuid));

    // require symmetric peer-to-peer access to enable peer-to-peer
    return canAccessPeer[0] && canAccessPeer[1];
  }

  int comm_peer2peer_performance(int local_gpuid, int neighbor_gpuid)
  {
    int accessRank[2] = {};
    if (comm_peer2peer_possible(local_gpuid, neighbor_gpuid)) {
      CHECK_CUDA_ERROR(
        cudaDeviceGetP2PAttribute(&accessRank[0], cudaDevP2PAttrPerformanceRank, local_gpuid, neighbor_gpuid));
      CHECK_CUDA_ERROR(
        cudaDeviceGetP2PAttribute(&accessRank[1], cudaDevP2PAttrPerformanceRank, neighbor_gpuid, local_gpuid));
    }

    // return the slowest direction of access (lower is faster)
    return std::max(accessRank[0], accessRank[1]);
  }

  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
#ifndef NVSHMEM_COMMS
    // handles for obtained ghost pointers
    cudaIpcMemHandle_t remote_handle[QUDA_MAX_DIM][2];

  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; ++dir) {
      MsgHandle *sendHandle = nullptr;
      MsgHandle *receiveHandle = nullptr;
      int disp = (dir == 1) ? +1 : -1;

      // first set up receive
      if (comm_peer2peer_enabled(1 - dir, dim)) {
        receiveHandle = comm_declare_receive_relative(&remote_handle[dim][1 - dir], dim, -disp, sizeof(remote_handle));
      }
      // now send
      cudaIpcMemHandle_t local_handle;
      if (comm_peer2peer_enabled(dir, dim)) {
        CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&local_handle, local));
        sendHandle = comm_declare_send_relative(&local_handle, dim, disp, sizeof(local_handle));
      }
      if (receiveHandle) comm_start(receiveHandle);
      if (sendHandle) comm_start(sendHandle);

      if (receiveHandle) comm_wait(receiveHandle);
      if (sendHandle) comm_wait(sendHandle);

      if (sendHandle) comm_free(sendHandle);
      if (receiveHandle) comm_free(receiveHandle);
    }
  }
#endif

  // open the remote memory handles and set the send ghost pointers
  for (int dim = 0; dim < 4; ++dim) {
#ifndef NVSHMEM_COMMS
    // TODO: We maybe can force loopback comms to use the IB path here
    if (comm_dim(dim) == 1) continue;
#endif
    // even if comm_dim(2) == 2, we might not have p2p enabled in both directions, so check this
    const int num_dir = (comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim) && comm_peer2peer_enabled(1, dim)) ? 1 : 2;
    for (int dir = 0; dir < num_dir; dir++) {
      remote[dim][dir] = nullptr;
#ifndef NVSHMEM_COMMS
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(&remote[dim][dir], remote_handle[dim][dir], cudaIpcMemLazyEnablePeerAccess));
#else
      remote[dim][dir] = nvshmem_ptr(static_cast<char *>(local), comm_neighbor_rank(dir, dim));
#endif
    }
    if (num_dir == 1) remote[dim][1] = remote[dim][0];
  }
}

#ifndef NVSHMEM_COMMS
void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote)
{
  for (int dim = 0; dim < 4; ++dim) {

    if (comm_dim(dim) == 1) continue;
    const int num_dir = (comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim) && comm_peer2peer_enabled(1, dim)) ? 1 : 2;

    if (comm_peer2peer_enabled(1, dim)) {
      // only close this handle if it doesn't alias the back ghost
      if (num_dir == 2 && remote[dim][1]) CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(remote[dim][1]));
    }

    if (comm_peer2peer_enabled(0, dim)) {
      if (remote[dim][0]) CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(remote[dim][0]));
    }
  } // iterate over dim
}
#else
void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &) { }
#endif

void comm_create_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
{
  // handles for obtained events
  cudaIpcEventHandle_t ipcRemoteEventHandle[QUDA_MAX_DIM][2];

  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; ++dir) {
      MsgHandle *sendHandle = nullptr;
      MsgHandle *receiveHandle = nullptr;
      int disp = (dir == 1) ? +1 : -1;

      // first set up receive
      if (comm_peer2peer_enabled(1 - dir, dim)) {
        receiveHandle = comm_declare_receive_relative(&ipcRemoteEventHandle[dim][1 - dir], dim, -disp,
                                                      sizeof(ipcRemoteEventHandle[dim][1 - dir]));
      }

      cudaIpcEventHandle_t handle;

      // now send
      if (comm_peer2peer_enabled(dir, dim)) {
        cudaEvent_t event;
        CHECK_CUDA_ERROR(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
        local[dim][dir].event = reinterpret_cast<void *>(event);
        CHECK_CUDA_ERROR(cudaIpcGetEventHandle(&handle, event));
        sendHandle = comm_declare_send_relative(&handle, dim, disp, sizeof(handle));
      } else {
        local[dim][dir].event = nullptr;
      }

      if (receiveHandle) comm_start(receiveHandle);
      if (sendHandle) comm_start(sendHandle);

      if (receiveHandle) comm_wait(receiveHandle);
      if (sendHandle) comm_wait(sendHandle);

      if (sendHandle) comm_free(sendHandle);
      if (receiveHandle) comm_free(receiveHandle);
    }
  }

  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; ++dir) {
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      cudaEvent_t event;
      CHECK_CUDA_ERROR(cudaIpcOpenEventHandle(&event, ipcRemoteEventHandle[dim][dir]));
      remote[dim][dir].event = reinterpret_cast<void *>(event);
    }
  }
}

void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &, array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
{
  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; dir++) {
      cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(local[dim][dir].event);
      if (comm_peer2peer_enabled(dir, dim)) CHECK_CUDA_ERROR(cudaEventDestroy(event));
    }
  } // iterate over dim
}

} // namespace quda
