#include <comm_quda.h>
#include <lattice_field.h>
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

  // Forward (dir 0) and backward (dir 1) neighbours share ONE IPC mapping (alias)
  // only for a size-2, non-C-star dim with P2P enabled both ways; otherwise each
  // direction is a distinct mapping.  create and destroy MUST use this identical
  // predicate -- a size-2 C-star dim opens two mappings, so teardown must close both.
  inline int comm_neighbor_p2p_num_dir(int dim)
  {
    return (!comm_dim_cstar(dim) && comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim)
            && comm_peer2peer_enabled(1, dim)) ?
      1 :
      2;
  }

  void comm_create_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
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
      const int num_dir = comm_neighbor_p2p_num_dir(dim);
      for (int dir = 0; dir < num_dir; ++dir) {
        remote[dim][dir] = nullptr;
        if (!comm_peer2peer_enabled(dir, dim)) continue;
        CHECK_HIP_ERROR(hipIpcOpenMemHandle(&remote[dim][dir], remote_handle[dim][dir], hipIpcMemLazyEnablePeerAccess));
      }
      if (num_dir == 1) remote[dim][1] = remote[dim][0];
    }
  }

  void comm_destroy_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote)
  {
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      // Same predicate as create: num_dir==1 means [dim][1] aliases [dim][0] (close
      // once); a size-2 C-star dim gives num_dir==2 (two distinct handles, close both).
      const int num_dir = comm_neighbor_p2p_num_dir(dim);
      for (int dir = 0; dir < num_dir; ++dir) {
        if (!comm_peer2peer_enabled(dir, dim)) continue;
        if (!remote[dim][dir]) continue;
        CHECK_HIP_ERROR(hipIpcCloseMemHandle(remote[dim][dir]));
        remote[dim][dir] = nullptr;
      }
      if (num_dir == 1) remote[dim][1] = nullptr; // aliased to [dim][0], already closed
    }
  }

  // HIP keeps the cudaIPC/event protocol and never builds with NVSHMEM, so the
  // NVSHMEM-transport neighbor pointer functions are no-ops (defined for symmetry
  // with the unified comm API).
  void comm_create_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *)
  {
    for (int dim = 0; dim < 4; ++dim)
      for (int dir = 0; dir < 2; ++dir) remote[dim][dir] = nullptr;
  }

  void comm_destroy_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &) { }

  // Stream-gated signalling is not wired on the HIP backend
  // (comm::p2p_signal_supported(STREAM_GATED) == false), so the resolver never
  // selects it and these are no-ops.
  void comm_create_stream_gated_comms() { }
  void comm_destroy_stream_gated_comms() { }

  // REMOTE_WRITE (direct-store halo packing) is safety-gated only under MNNVL, where a
  // receiver-side flush guard is required.  HIP has no MNNVL path, so -- as on non-MNNVL
  // CUDA -- the policy is always offered (its bit-1 enable is still honoured by the tuner).
  bool comm_p2p_remote_write_supported() { return true; }

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

  void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                   array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
  {
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      for (int dir = 0; dir < 2; dir++) {
        if (!comm_peer2peer_enabled(dir, dim)) continue;
        // First close our imported view of the neighbour's event (the
        // counterpart to hipIpcOpenEventHandle in
        // comm_create_neighbor_event).  Without this, every IPC reset
        // leaks the imported handle.
        hipEvent_t &remote_event = reinterpret_cast<hipEvent_t &>(remote[dim][dir].event);
        if (remote_event) {
          CHECK_HIP_ERROR(hipEventDestroy(remote_event));
          remote[dim][dir].event = nullptr;
        }
        // Then destroy the local interprocess event we exported.
        hipEvent_t &event = reinterpret_cast<hipEvent_t &>(local[dim][dir].event);
        if (event) {
          CHECK_HIP_ERROR(hipEventDestroy(event));
          local[dim][dir].event = nullptr;
        }
      }
    } // iterate over dim
  }

  // -------------------------------------------------------------------------
  // comm_p2p_* (HIP target backend): identical shape to the CUDA target since
  // the bodies use only platform-neutral qudaEvent / comm wrappers. See
  // lib/targets/cuda/comm_target.cpp for the rationale.

  void comm_p2p_signal_send_done(FieldKind, int buf, int dim, int dir, const qudaStream_t &stream)
  {
    qudaEventRecord(LatticeField::ipcCopyEvent[buf][dim][dir], stream);
    comm_start(LatticeField::mh_send_p2p[buf][dim][dir]);
  }

  int comm_p2p_query_send_drained(FieldKind, int buf, int dim, int dir)
  {
    return comm_query(LatticeField::mh_send_p2p[buf][dim][dir]);
  }

  void comm_p2p_wait_send_drained(FieldKind, int buf, int dim, int dir)
  {
    comm_wait(LatticeField::mh_send_p2p[buf][dim][dir]);
    qudaEventSynchronize(LatticeField::ipcCopyEvent[buf][dim][dir]);
  }

  int comm_p2p_query_recv_signal(FieldKind, int buf, int dim, int dir)
  {
    return comm_query(LatticeField::mh_recv_p2p[buf][dim][dir]);
  }

  void comm_p2p_wait_recv_signal(FieldKind, int buf, int dim, int dir)
  {
    comm_wait(LatticeField::mh_recv_p2p[buf][dim][dir]);
    qudaEventSynchronize(LatticeField::ipcRemoteCopyEvent[buf][dim][dir]);
  }

  // Stream-mem-op signalling primitives (Phase 5).  HIP equivalent is
  // hipStreamWriteValue64 / hipStreamWaitValue64 but we have no HIP test
  // machine yet; left as errorQuda stubs so the QUDA_P2P_STREAM_GATED
  // sub-policy errors loudly on HIP if it's ever selected; the tuner will
  // never pick it because the body never returns successfully.
  void comm_p2p_stream_signal_send_done(FieldKind, int, int, int, const qudaStream_t &)
  {
    errorQuda("comm_p2p_stream_signal_send_done not yet implemented on the HIP target");
  }

  void comm_p2p_stream_wait_recv_signal(FieldKind, int, int, int, const qudaStream_t &)
  {
    errorQuda("comm_p2p_stream_wait_recv_signal not yet implemented on the HIP target");
  }

  // ============================================================================
  // Unified P2P signal API (Phase B/C of TransportContext refactor).  Dispatches
  // to the per-kind implementations.  STREAM_GATED calls will errorQuda until a
  // HIP implementation lands -- so the HIP backend's p2p_signal_supported()
  // (when added) must return false for STREAM_GATED to keep the tuner from
  // picking it.
  // ============================================================================

  void comm_p2p_signal_send_done(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream,
                                 QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC: comm_p2p_signal_send_done(kind, buf, dim, dir, stream); return;
    case QudaP2PSignal::STREAM_GATED: comm_p2p_stream_signal_send_done(kind, buf, dim, dir, stream); return;
    }
    errorQuda("comm_p2p_signal_send_done: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

  void comm_p2p_wait_recv_signal(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream,
                                 QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC:
      (void)stream;
      comm_p2p_wait_recv_signal(kind, buf, dim, dir);
      return;
    case QudaP2PSignal::STREAM_GATED: comm_p2p_stream_wait_recv_signal(kind, buf, dim, dir, stream); return;
    }
    errorQuda("comm_p2p_wait_recv_signal: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

  void comm_p2p_wait_send_drained(FieldKind kind, int buf, int dim, int dir, QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC: comm_p2p_wait_send_drained(kind, buf, dim, dir); return;
    case QudaP2PSignal::STREAM_GATED:
      // STREAM_GATED has no separate "send drained" concept: stream ordering
      // guarantees that subsequent ops on the same stream observe the prior
      // hipStreamWriteValue64 having completed.
      return;
    }
    errorQuda("comm_p2p_wait_send_drained: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

} // namespace quda
