#include <comm_quda.h>
#include <quda_api.h>
#include <quda_cuda_api.h>
#include <util_quda.h>
#include <algorithm>
#include <shmem_helper.cuh>
#include <cstring>

namespace quda
{

#define CU_CHECK(cmd)                                                          \
  do {                                                                         \
    CUresult e = cmd;                                                          \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *errstring;                                                   \
      cuGetErrorName(e, &errstring);                                           \
      fprintf(stderr, "Failed: Cuda Error %s: %d '%s'\n", __FILE__, __LINE__,  \
              errstring);                                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

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

  static CUresult try_map_device_ptr_from_handle(void **ptr, CUmemGenericAllocationHandle handle, size_t bytes, int device)
  {
    // reserve / map its allocation (we could also map into one huge range)
    CUdeviceptr dptr = 0;
    CUresult result = cuMemAddressReserve(&dptr, bytes, 0, 0, 0);
    if (result != CUDA_SUCCESS) return result;

    result = cuMemMap(dptr, bytes, 0, handle, 0);
    if (result != CUDA_SUCCESS) {
      cuMemAddressFree(dptr, bytes);
      return result;
    }

    // cuMemMap does not make the address accessible, we need to make it
    // accessible explicitly
    CUmemAccessDesc access_desc{};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = (int)device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    result = cuMemSetAccess(dptr, bytes, &access_desc, 1);
    if (result != CUDA_SUCCESS) {
      cuMemUnmap(dptr, bytes);
      cuMemAddressFree(dptr, bytes);
      return result;
    }

    *ptr = reinterpret_cast<void *>(dptr);
    return CUDA_SUCCESS;
  }

  void *map_device_ptr_from_handle(CUmemGenericAllocationHandle handle, size_t bytes, int device)
  {
    void *ptr = nullptr;
    CU_CHECK(try_map_device_ptr_from_handle(&ptr, handle, bytes, device));
    return ptr;
  }

  static void validate_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, const char *mapping,
                                       bool release_invalid);

  void comm_peer2peer_validate_nvshmem()
  {
#ifdef NVSHMEM_COMMS
    if (!comm_nvshmem_enabled()) return;

    void *probe = nvshmem_malloc(1);
    if (!probe) errorQuda("Failed to allocate NVSHMEM peer-reachability probe");

    array_2d<void *, QUDA_MAX_DIM, 2> remote {};
#ifdef MNNVL_COMMS
    int cross_clique_links = 0;
#endif
    for (int dim = 0; dim < 4; ++dim) {
      for (int dir = 0; dir < 2; ++dir) {
        if (comm_dim(dim) <= 1) continue;
        void *neighbor = nvshmem_ptr(probe, comm_neighbor_rank(dir, dim));
#ifdef MNNVL_COMMS
        if (!neighbor) cross_clique_links++;
#endif
        if (comm_peer2peer_enabled(dir, dim)) remote[dim][dir] = neighbor;
      }
    }

    validate_neighbor_memory(remote, "NVSHMEM", false);
    comm_peer2peer_finalize_validation();
    comm_peer2peer_report_validated();
    nvshmem_free(probe);
#ifdef MNNVL_COMMS
    comm_allreduce_int(cross_clique_links);
    if (cross_clique_links)
      warningQuda("You are working with MNNVL=ON, NVSHMEM=ON and enabled, using cross-clique communications "
                  "(neighbors without direct NVSHMEM mappings). This mode has known issues with MPI communications "
                  "across clique boundaries. A current workaround is to disable GDR with QUDA_ENABLE_GDR=0.");
#endif
#endif
  }

  void *comm_create_local_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, size_t bytes)
  {
    // Allocate local memory handle
    CUdevice device;
    CU_CHECK(cuCtxGetDevice(&device));

    int devAttr = -1;
    CU_CHECK(cuDeviceGetAttribute(&devAttr, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
    if (devAttr == 0) {
        printf("Device %d does not support the CUDA VMM APis, fabric cannot be used", (int)device);
    }
    CU_CHECK(cuDeviceGetAttribute(&devAttr, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    if (devAttr == 0) {
        printf("Device %d does not support the CUDA fabric handles, fabric cannot be used", (int)device);
    }

    size_t granularity = 0;
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)device;
    CU_CHECK(cuDeviceGetAttribute(&devAttr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
    if (devAttr == 0) {
      fprintf(stderr, "Device %d does not support GPUDirect RDMA for CUDA VMM allocations\n", (int)device);
      exit(1);
    }
    prop.allocFlags.gpuDirectRDMACapable = 1;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // get the padded size from minimum granularity
    size_t sz_align = (bytes + granularity - 1) & ~(granularity - 1);

    // each rank creates a handle that is exchanged among all ranks
    CUmemGenericAllocationHandle this_handle = 0;
    CU_CHECK(cuMemCreate(&this_handle, sz_align, &prop, 0));

    CUmemFabricHandle shandle;
    CU_CHECK(cuMemExportToShareableHandle(&shandle, this_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    // handles for obtained ghost pointers
    CUmemFabricHandle remote_handle[QUDA_MAX_DIM][2];

    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      for (int dir = 0; dir < 2; ++dir) {
        MsgHandle *sendHandle = nullptr;
        MsgHandle *receiveHandle = nullptr;
        int disp = (dir == 1) ? +1 : -1;

        // first set up receive
        if (comm_peer2peer_enabled(1 - dir, dim)) {
          receiveHandle = comm_declare_receive_relative(&remote_handle[dim][1 - dir], dim, -disp, sizeof(CUmemFabricHandle));
        }
        // now send
        if (comm_peer2peer_enabled(dir, dim)) {
          sendHandle = comm_declare_send_relative(&shandle, dim, disp, sizeof(CUmemFabricHandle));
        }
        if (receiveHandle) comm_start(receiveHandle);
        if (sendHandle) comm_start(sendHandle);

        if (receiveHandle) comm_wait(receiveHandle);
        if (sendHandle) comm_wait(sendHandle);

        if (sendHandle) comm_free(sendHandle);
        if (receiveHandle) comm_free(receiveHandle);
      }
    }

    void *local_ptr = map_device_ptr_from_handle(this_handle, sz_align, device);

    // open the remote memory handles and set the send ghost pointers
    for (int dim = 0; dim < 4; ++dim) {
      // TODO: We maybe can force loopback comms to use the IB path here
      if (comm_dim(dim) == 1) continue;
      // even if comm_dim(2) == 2, we might not have p2p enabled in both directions, so check this
      const int num_dir = (comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim) && comm_peer2peer_enabled(1, dim)) ? 1 : 2;
      for (int dir = 0; dir < num_dir; dir++) {
        remote[dim][dir] = nullptr;
        if (!comm_peer2peer_enabled(dir, dim)) continue;
        CUmemGenericAllocationHandle handle = 0;
        CUresult result = cuMemImportFromShareableHandle(&handle, &remote_handle[dim][dir], CU_MEM_HANDLE_TYPE_FABRIC);
        if (result == CUDA_SUCCESS) {
          (void)try_map_device_ptr_from_handle(&remote[dim][dir], handle, sz_align, device);
          CU_CHECK(cuMemRelease(handle));
        }
      }
      if (num_dir == 1) remote[dim][1] = remote[dim][0];
    }
    validate_neighbor_memory(remote, "Fabric", true);
    CU_CHECK(cuMemRelease(this_handle));

    return local_ptr;
  }

  void free_cu_ptr(void *ptr) {
    CUdeviceptr alloc_ptr;
    size_t sz;
    // get the original allocation from the pointer
    CU_CHECK(cuMemGetAddressRange(&alloc_ptr, &sz, (CUdeviceptr)ptr));
    // release the data
    CU_CHECK(cuMemUnmap(alloc_ptr, sz));
    CU_CHECK(cuMemAddressFree(alloc_ptr, sz));
  }

  void comm_destroy_local_neighbor_memory(void *local, array_2d<void *, QUDA_MAX_DIM, 2> &remote)
  {
    if (local) {
      free_cu_ptr(local);
    }
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      if (remote[dim][0]) free_cu_ptr(remote[dim][0]);
      if (remote[dim][1] && remote[dim][1] != remote[dim][0]) free_cu_ptr(remote[dim][1]);
      remote[dim][0] = nullptr;
      remote[dim][1] = nullptr;
    } // iterate over dim
  }

  static void validate_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, const char *mapping,
                                       bool release_invalid)
  {
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;

      int local_access[2] = {};
      int neighbor_access[2] = {};
      MsgHandle *send_handle[2] = {};
      MsgHandle *recv_handle[2] = {};

      for (int dir = 0; dir < 2; ++dir) {
        int disp = dir == 0 ? -1 : +1;
        local_access[dir] = comm_peer2peer_enabled(dir, dim) && remote[dim][dir] != nullptr;
        recv_handle[dir] = comm_declare_receive_relative(&neighbor_access[dir], dim, disp, sizeof(int));
        send_handle[dir] = comm_declare_send_relative(&local_access[dir], dim, disp, sizeof(int));
      }

      for (int dir = 0; dir < 2; ++dir) comm_start(recv_handle[dir]);
      for (int dir = 0; dir < 2; ++dir) comm_start(send_handle[dir]);
      for (int dir = 0; dir < 2; ++dir) comm_wait(recv_handle[dir]);
      for (int dir = 0; dir < 2; ++dir) comm_wait(send_handle[dir]);

      bool valid[2];
      for (int dir = 0; dir < 2; ++dir) valid[dir] = local_access[dir] && neighbor_access[dir];

      if (release_invalid) {
        if (remote[dim][0] == remote[dim][1]) {
          if (!valid[0] && !valid[1] && remote[dim][0]) free_cu_ptr(remote[dim][0]);
        } else {
          if (!valid[0] && remote[dim][0]) free_cu_ptr(remote[dim][0]);
          if (!valid[1] && remote[dim][1]) free_cu_ptr(remote[dim][1]);
        }
      }

      for (int dir = 0; dir < 2; ++dir) {
        comm_free(recv_handle[dir]);
        comm_free(send_handle[dir]);

        if (!valid[dir]) {
          if (comm_peer2peer_enabled(dir, dim)) {
            warningQuda("Disabling peer-to-peer access to rank %d in dir=%d, dim=%d because the %s remote mapping "
                        "is not bidirectional",
                        comm_neighbor_rank(dir, dim), dir, dim, mapping);
            comm_peer2peer_disable(dir, dim);
          }
          remote[dim][dir] = nullptr;
        }
      }
    }
  }

  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
    const bool use_nvshmem = comm_nvshmem_enabled();

    if (use_nvshmem) {
#ifdef NVSHMEM_COMMS
      for (int dim = 0; dim < 4; ++dim) {
        for (int dir = 0; dir < 2; ++dir) {
          remote[dim][dir] = nullptr;
          if (comm_dim(dim) > 1 && comm_peer2peer_enabled(dir, dim))
            remote[dim][dir] = nvshmem_ptr(static_cast<char *>(local), comm_neighbor_rank(dir, dim));
        }
      }
      validate_neighbor_memory(remote, "NVSHMEM", false);
      return;
#else
      errorQuda("NVSHMEM peer mapping requested in a build without NVSHMEM support");
#endif
    }

    // handles for obtained ghost pointers
    cudaIpcMemHandle_t remote_handle[QUDA_MAX_DIM][2];

    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      for (int dir = 0; dir < 2; ++dir) {
        MsgHandle *sendHandle = nullptr;
        MsgHandle *receiveHandle = nullptr;
        int disp = (dir == 1) ? +1 : -1;

        if (comm_peer2peer_enabled(1 - dir, dim)) {
          receiveHandle = comm_declare_receive_relative(&remote_handle[dim][1 - dir], dim, -disp,
                                                        sizeof(remote_handle[dim][1 - dir]));
        }
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

    // Open the remote IPC handles.
    for (int dim = 0; dim < 4; ++dim) {
      if (comm_dim(dim) == 1) continue;
      // even if comm_dim(dim) == 2, we might not have p2p enabled in both directions, so check this
      const int num_dir = (!comm_dim_cstar(dim) && comm_dim(dim) == 2 && comm_peer2peer_enabled(0, dim)
                           && comm_peer2peer_enabled(1, dim)) ?
        1 :
        2;
      for (int dir = 0; dir < num_dir; dir++) {
        remote[dim][dir] = nullptr;
        if (comm_peer2peer_enabled(dir, dim)) {
          CHECK_CUDA_ERROR(
            cudaIpcOpenMemHandle(&remote[dim][dir], remote_handle[dim][dir], cudaIpcMemLazyEnablePeerAccess));
        }
      }
      if (num_dir == 1) remote[dim][1] = remote[dim][0];
    }
  }

void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote)
{
#ifdef NVSHMEM_COMMS
  if (comm_nvshmem_enabled()) return;
#endif
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

void comm_create_local_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
{
  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; ++dir) {
      if (comm_peer2peer_enabled(dir, dim)) {
        cudaEvent_t event;
        CHECK_CUDA_ERROR(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
        local[dim][dir].event = reinterpret_cast<void *>(event);
      } else {
        local[dim][dir].event = nullptr;
      }
    }
  }
}

void comm_destroy_local_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
{
  for (int dim = 0; dim < 4; ++dim) {
    if (comm_dim(dim) == 1) continue;
    for (int dir = 0; dir < 2; dir++) {
      cudaEvent_t &event = reinterpret_cast<cudaEvent_t &>(local[dim][dir].event);
      if (comm_peer2peer_enabled(dir, dim)) CHECK_CUDA_ERROR(cudaEventDestroy(event));
    }
  } // iterate over dim
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
