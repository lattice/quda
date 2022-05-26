#include <comm_quda.h>
#include <quda_api.h>
#include <quda_hip_api.h>
#include <algorithm>
#include <hip/hip_runtime_api.h>

namespace quda
{

#define CHECK_HIP_ERROR(func) target::hip::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

  bool comm_peer2peer_possible(int local_gpuid, int neighbor_gpuid)
  {
    int canAccessPeer[2];
    CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccessPeer[0], local_gpuid, neighbor_gpuid));
    CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, local_gpuid));

    // require symmetric peer-to-peer access to enable peer-to-peer
    return canAccessPeer[0] && canAccessPeer[1];
  }

  int comm_peer2peer_performance(int local_gpuid, int neighbor_gpuid)
  {
    int accessRank[2] = {};
    if (comm_peer2peer_possible(local_gpuid, neighbor_gpuid)) {
      CHECK_HIP_ERROR(hipDeviceGetP2PAttribute(&accessRank[0], hipDevP2PAttrPerformanceRank, local_gpuid, neighbor_gpuid));
      CHECK_HIP_ERROR(hipDeviceGetP2PAttribute(&accessRank[1], hipDevP2PAttrPerformanceRank, neighbor_gpuid, local_gpuid));
    }

    // return the slowest direction of access (lower is faster)
    return std::max(accessRank[0], accessRank[1]);
  }

  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
    // handles for obtained ghost pointers
    hipIpcMemHandle_t remote_handle[QUDA_MAX_DIM][2];

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
        hipIpcMemHandle_t local_handle;
        if (comm_peer2peer_enabled(dir, dim)) {
          CHECK_HIP_ERROR(hipIpcGetMemHandle(&local_handle, local));
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

    // open the remote memory handles and set the send ghost pointers
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      // even if comm_dim(2) == 2, we might not have p2p enabled in both directions, so check this
      const int num_dir
        = (comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim) && comm_peer2peer_enabled(1, dim)) ? 1 : 2;
      for (int dir = 0; dir < num_dir; ++dir) {
        remote[dim][dir] = nullptr;
        if (!comm_peer2peer_enabled(dir, dim)) continue;
        CHECK_HIP_ERROR(hipIpcOpenMemHandle(&remote[dim][dir], remote_handle[dim][dir], hipIpcMemLazyEnablePeerAccess));
      }
      if (num_dir == 1) remote[dim][1] = remote[dim][0];
    }
  }

  void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote)
  {
    for (int dim = 0; dim < 4; ++dim) {

      if (comm_dim(dim) == 1) continue;
      const int num_dir
        = (comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim) && comm_peer2peer_enabled(1, dim)) ? 1 : 2;

      if (comm_peer2peer_enabled(1, dim)) {
        // only close this handle if it doesn't alias the back ghost
        if (num_dir == 2 && remote[dim][1]) CHECK_HIP_ERROR(hipIpcCloseMemHandle(remote[dim][1]));
      }

      if (comm_peer2peer_enabled(0, dim)) {
        if (remote[dim][0]) CHECK_HIP_ERROR(hipIpcCloseMemHandle(remote[dim][0]));
      }
    } // iterate over dim
  }

  void comm_create_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                  array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
  {
    // handles for obtained events
    hipIpcEventHandle_t ipcRemoteEventHandle[QUDA_MAX_DIM][2];

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

        hipIpcEventHandle_t handle;

        // now send
        if (comm_peer2peer_enabled(dir, dim)) {
          hipEvent_t event;
          CHECK_HIP_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming | hipEventInterprocess));
          local[dim][dir].event = event;
          CHECK_HIP_ERROR(hipIpcGetEventHandle(&handle, event));
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
        hipEvent_t event = nullptr;
        CHECK_HIP_ERROR(hipIpcOpenEventHandle(&event, ipcRemoteEventHandle[dim][dir]));
        remote[dim][dir].event = reinterpret_cast<void *>(event);
      }
    }
  }

  void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &, array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
  {
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      for (int dir = 0; dir < 2; dir++) {
        hipEvent_t &event = reinterpret_cast<hipEvent_t &>(local[dim][dir].event);
        if (comm_peer2peer_enabled(dir, dim)) CHECK_HIP_ERROR(hipEventDestroy(event));
      }
    } // iterate over dim
  }

} // namespace quda
