#include <comm_quda.h>
#include <comm_target.h>
#include <malloc_quda.h>
#include <lattice_field.h>
#include <quda_api.h>
#include <quda_cuda_api.h>
#include <algorithm>
#include <shmem_helper.cuh>
#ifdef QUDA_MNNVL
#include <cuda.h>
#include <nvml.h>
#endif


namespace quda
{

#define CHECK_CUDA_ERROR(func)                                                                                         \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

#ifdef QUDA_MNNVL
  namespace comm_target
  {
    unsigned int get_fabric_clique_id()
    {
      int dev_id = -1;
      if (cudaGetDevice(&dev_id) != cudaSuccess) return 0;
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, dev_id) != cudaSuccess) return 0;
      char pciBusId[32];
      snprintf(pciBusId, sizeof(pciBusId), "%08x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
      if (nvmlInit() != NVML_SUCCESS) return 0;
      nvmlDevice_t dev;
      if (nvmlDeviceGetHandleByPciBusId(pciBusId, &dev) != NVML_SUCCESS) return 0;
      nvmlGpuFabricInfo_t info = {};
      if (nvmlDeviceGetGpuFabricInfo(dev, &info) != NVML_SUCCESS) return 0;
      if (info.status != NVML_SUCCESS) return 0;
      return info.cliqueId;
    }

    size_t fabric_handle_size() { return sizeof(CUmemFabricHandle); }

    void *open_fabric_probe(void *out_handle)
    {
      void *probe = device_comm_buffer_malloc(1);
      CUmemFabricHandle local_fh = get_p2p_fabric_handle(probe);
      memcpy(out_handle, &local_fh, sizeof(CUmemFabricHandle));
      return probe;
    }

    bool try_import_fabric_handle(const void *peer_handle)
    {
      CUmemFabricHandle fh;
      memcpy(&fh, peer_handle, sizeof(CUmemFabricHandle));
      CUmemGenericAllocationHandle h;
      CUresult err = cuMemImportFromShareableHandle(&h, &fh, CU_MEM_HANDLE_TYPE_FABRIC);
      if (err != CUDA_SUCCESS) return false;
      return cuMemRelease(h) == CUDA_SUCCESS;
    }

    void close_fabric_probe(void *probe) { device_comm_buffer_free(probe); }
  } // namespace comm_target
#endif // QUDA_MNNVL

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

  // Forward (dir 0) and backward (dir 1) neighbours share ONE P2P mapping (alias)
  // only for a size-2, non-C-star dim with P2P enabled both ways; otherwise each
  // direction is a distinct mapping.  create and destroy MUST use this identical
  // predicate -- a size-2 C-star dim opens two mappings, so teardown must free both.
  inline int comm_neighbor_p2p_num_dir(int dim)
  {
    return (!comm_dim_cstar(dim) && comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim)
            && comm_peer2peer_enabled(1, dim)) ?
      1 :
      2;
  }

  // QUDA-owned P2P neighbour mapping.  Exchanges a SINGLE fabric handle (MNNVL)
  // or cudaIPC handle (non-IMEX) for the LOCAL contiguous P2P recv buffer and
  // imports the peer's, so P2P writes target a single-allocation, RDMA-capable
  // buffer.  Compiled in ALL builds INCLUDING NVSHMEM: under NVSHMEM the
  // symmetric heap does not give QUDA a reusable single-handle export, so P2P
  // must own its own DeviceCommBuffer.  The NVSHMEM-transport remote pointer is
  // handled separately by comm_create_neighbor_memory_shmem.
#ifdef QUDA_MNNVL
  struct FabricExport {
    CUmemFabricHandle handle;
    uint64_t size;
    uint64_t generation;
  };

  struct RemoteFabricMapping {
    size_t size;
    uint64_t generation;
  };

  // Imported peer VMM pointer -> exact mapping metadata.  Recording this at
  // import avoids querying VMM address ranges during teardown and makes VA
  // reuse across allocation generations visible in debug logs.
  static std::map<void *, RemoteFabricMapping> p2p_remote_mappings;
#endif

  void comm_create_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
#ifdef QUDA_MNNVL
    // MNNVL build: exchange CUmemFabricHandle (from VMM allocator) via MPI.
    // The peer side will cuMemImportFromShareableHandle + cuMemAddressReserve
    // + cuMemMap + cuMemSetAccess to map the remote buffer into our address
    // space.
    FabricExport remote_export[QUDA_MAX_DIM][2] = {};

    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      for (int dir = 0; dir < 2; ++dir) {
        MsgHandle *sendHandle = nullptr;
        MsgHandle *receiveHandle = nullptr;
        int disp = (dir == 1) ? +1 : -1;

        if (comm_peer2peer_enabled(1 - dir, dim)) {
          receiveHandle = comm_declare_receive_relative(&remote_export[dim][1 - dir], dim, -disp,
                                                        sizeof(FabricExport));
        }
        FabricExport local_export = {};
        if (comm_peer2peer_enabled(dir, dim)) {
          local_export.handle = get_p2p_fabric_handle(local);
          local_export.size = get_p2p_buffer_size(local);
          local_export.generation = get_p2p_buffer_generation(local);
          sendHandle = comm_declare_send_relative(&local_export, dim, disp, sizeof(local_export));
        }
        if (receiveHandle) comm_start(receiveHandle);
        if (sendHandle) comm_start(sendHandle);

        if (receiveHandle) comm_wait(receiveHandle);
        if (sendHandle) comm_wait(sendHandle);

        if (sendHandle) comm_free(sendHandle);
        if (receiveHandle) comm_free(receiveHandle);
      }
    }
#else
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
        receiveHandle = comm_declare_receive_relative(&remote_handle[dim][1 - dir], dim, -disp,
                                                      sizeof(remote_handle[dim][1 - dir]));
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
    // TODO: We maybe can force loopback comms to use the IB path here
    if (comm_dim(dim) == 1) continue;
    const int num_dir = comm_neighbor_p2p_num_dir(dim);
    for (int dir = 0; dir < num_dir; dir++) {
      remote[dim][dir] = nullptr;
#ifdef QUDA_MNNVL
      // Import peer's fabric handle, reserve a local VA range, map.
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      CUmemGenericAllocationHandle h;
      const auto &peer = remote_export[dim][dir];
      CUmemFabricHandle peer_handle = peer.handle;
      CUresult err = cuMemImportFromShareableHandle(&h, &peer_handle, CU_MEM_HANDLE_TYPE_FABRIC);
      if (err != CUDA_SUCCESS) errorQuda("cuMemImportFromShareableHandle FABRIC failed for (dim=%d, dir=%d)", dim, dir);

      if (peer.size == 0) errorQuda("Peer exported a zero-sized P2P buffer (dim=%d dir=%d)", dim, dir);
      const size_t local_size = get_p2p_buffer_size(local);
      if (peer.size != local_size)
        errorQuda("P2P buffer size mismatch: local=%zu peer=%lu (dim=%d dir=%d)", local_size,
                  (unsigned long)peer.size, dim, dir);
      size_t map_size = peer.size;

      CUdeviceptr peer_ptr = 0;
      err = cuMemAddressReserve(&peer_ptr, map_size, 0, 0, 0);
      if (err != CUDA_SUCCESS) errorQuda("cuMemAddressReserve (peer) failed");
      err = cuMemMap(peer_ptr, map_size, 0, h, 0);
      if (err != CUDA_SUCCESS) errorQuda("cuMemMap (peer) failed");

      int local_dev;
      CHECK_CUDA_ERROR(cudaGetDevice(&local_dev));
      CUmemAccessDesc acc = {};
      acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      acc.location.id = local_dev;
      acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      err = cuMemSetAccess(peer_ptr, map_size, &acc, 1);
      if (err != CUDA_SUCCESS) errorQuda("cuMemSetAccess (peer) failed");
      err = cuMemRelease(h);
      if (err != CUDA_SUCCESS) errorQuda("cuMemRelease (peer import) failed (dim=%d dir=%d)", dim, dir);
      remote[dim][dir] = (void *)peer_ptr;
      auto inserted
        = p2p_remote_mappings.emplace((void *)peer_ptr, RemoteFabricMapping {map_size, peer.generation});
      if (!inserted.second) errorQuda("Duplicate imported P2P mapping address %p", (void *)peer_ptr);
      logQuda(QUDA_DEBUG_VERBOSE,
              "MNNVL P2P import: local=%p peer=%p size=%zu generation=%lu dim=%d dir=%d\n", local,
              (void *)peer_ptr, map_size, (unsigned long)peer.generation, dim, dir);
#else
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(&remote[dim][dir], remote_handle[dim][dir], cudaIpcMemLazyEnablePeerAccess));
#endif
    }
    if (num_dir == 1) remote[dim][1] = remote[dim][0];
  }
}

  // NVSHMEM-transport neighbour pointer: nvshmem_ptr of the peer's SYMMETRIC
  // recv buffer (the Shmem direct-NVLink put destination).  No-op when not
  // built with NVSHMEM.  Kept strictly separate from the P2P import above so a
  // single pointer family never means both "NVSHMEM symmetric remote" and
  // "QUDA P2P import".
  void comm_create_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
#ifdef NVSHMEM_COMMS
    // With NVSHMEM disabled at runtime, `local` is a DeviceCommBuffer (not a
    // symmetric-heap pointer), so nvshmem_ptr() on it is invalid; the symmetric
    // remote pointer is unused on the non-shmem transport anyway.  No-op.
    if (!comm_nvshmem_enabled()) {
      for (int dim = 0; dim < 4; ++dim)
        for (int dir = 0; dir < 2; ++dir) remote[dim][dir] = nullptr;
      return;
    }
    for (int dim = 0; dim < 4; ++dim) {
      const int num_dir = comm_neighbor_p2p_num_dir(dim);
      for (int dir = 0; dir < num_dir; dir++) {
        remote[dim][dir] = nvshmem_ptr(static_cast<char *>(local), comm_neighbor_rank(dir, dim));
      }
      if (num_dir == 1) remote[dim][1] = remote[dim][0];
    }
#else
    // not an NVSHMEM build -- the symmetric remote pointer is unused.
    for (int dim = 0; dim < 4; ++dim)
      for (int dir = 0; dir < 2; ++dir) remote[dim][dir] = nullptr;
#endif
  }

// Tear down a QUDA-owned P2P neighbour mapping created by
// comm_create_neighbor_memory_p2p.  Symmetric with create: compiled in ALL
// builds (including NVSHMEM) so the P2P imports made under NVSHMEM are released.
#ifdef QUDA_MNNVL
void comm_destroy_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote)
{
  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    // Same predicate as create: num_dir==1 means [dim][1] aliases [dim][0] (one
    // mapping, unmap once); a size-2 C-star dim gives num_dir==2 (two mappings).
    const int num_dir = comm_neighbor_p2p_num_dir(dim);
    for (int dir = 0; dir < num_dir; ++dir) {
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      void *ptr = remote[dim][dir];
      if (!ptr) continue;
      // Unmap/free using the size recorded at import time, then drop the registry
      // entry.  A non-null import with no recorded size is an invariant failure.
      auto it = p2p_remote_mappings.find(ptr);
      if (it == p2p_remote_mappings.end())
        errorQuda("comm_destroy_neighbor_memory_p2p: no recorded size for imported ptr %p (dim=%d dir=%d)", ptr, dim,
                  dir);
      const size_t map_size = it->second.size;
      logQuda(QUDA_DEBUG_VERBOSE, "MNNVL P2P unmap: peer=%p size=%zu generation=%lu dim=%d dir=%d\n", ptr,
              map_size, (unsigned long)it->second.generation, dim, dir);
      CUresult err = cuMemUnmap((CUdeviceptr)ptr, map_size);
      if (err != CUDA_SUCCESS) errorQuda("cuMemUnmap (peer) failed (dim=%d dir=%d)", dim, dir);
      err = cuMemAddressFree((CUdeviceptr)ptr, map_size);
      if (err != CUDA_SUCCESS) errorQuda("cuMemAddressFree (peer) failed (dim=%d dir=%d)", dim, dir);
      p2p_remote_mappings.erase(it);
      remote[dim][dir] = nullptr;
    }

    if (comm_peer2peer_enabled(0, dim)) {
      if (remote[dim][0]) CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(remote[dim][0]));
    }
  }
  stream_gated_comms_init = true;
}
#else
void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &) { }
#endif
    return;
  }

  qudaDeviceSynchronize();
  comm_barrier();
  for (int b = 0; b < 2; b++) {
    comm_destroy_neighbor_memory_p2p(flag_buffer_remote_d[b]);
    if (flag_buffer_d[b]) {
      device_comm_buffer_free(flag_buffer_d[b]);
      flag_buffer_d[b] = nullptr;
    }
  }
  qudaDeviceSynchronize();
  comm_barrier();
  stream_gated_comms_init = false;

#ifdef QUDA_MNNVL
  // This runs right after destroyIPCComms() (ghost P2P imports) + the flag-buffer
  // teardown above, so every QUDA P2P import for this communicator is released:
  // the registry must be empty.  A leftover entry is a leaked VMM mapping.
  if (!p2p_remote_mappings.empty())
    errorQuda("comm_destroy_stream_gated_comms: %zu imported P2P mapping(s) still outstanding",
              p2p_remote_mappings.size());
#endif
}

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

void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                 array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
{
  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; dir++) {
      if (!comm_peer2peer_enabled(dir, dim)) continue;
      // First close our imported view of the neighbour's event (the
      // counterpart to cudaIpcOpenEventHandle in
      // comm_create_neighbor_event).  Without this, every IPC reset
      // leaks the imported handle.
      cudaEvent_t &remote_event = reinterpret_cast<cudaEvent_t &>(remote[dim][dir].event);
      if (remote_event) {
        CHECK_CUDA_ERROR(cudaEventDestroy(remote_event));
        remote[dim][dir].event = nullptr;
      }
      // Then destroy the local interprocess event we exported.
      cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(local[dim][dir].event);
      if (event) {
        CHECK_CUDA_ERROR(cudaEventDestroy(event));
        local[dim][dir].event = nullptr;
      }
    }
  } // iterate over dim
}

} // namespace quda
