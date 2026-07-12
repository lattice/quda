#include <comm_quda.h>
#include <comm_target.h>
#include <malloc_quda.h>
#include <lattice_field.h>
#include <quda_api.h>
#include <quda_cuda_api.h>
#include <algorithm>
#include <cstring>
#include <cstdlib> // getenv/atoi for QUDA_P2P_FORCE_FLUSH override
#include <map>
#include <vector>
#include <shmem_helper.cuh>
#include <malloc_target.h> // get_p2p_fabric_handle / get_p2p_buffer_size / _generation (CUDA/MNNVL)
#ifdef QUDA_MNNVL
#include <cuda.h>
#endif

namespace quda
{

#define CHECK_CUDA_ERROR(func)                                                                                         \
  target::cuda::set_runtime_error(func, #func, __func__, __FILE__, __STRINGIFY__(__LINE__));

  // Wrap a CUDA driver (cu*) call: on failure abort via set_driver_error, which
  // reports the CUresult name plus file:line.  Mirrors CHECK_CUDA_ERROR for the
  // runtime API so we don't repeat `if (err != CUDA_SUCCESS) errorQuda(...)` at
  // every driver call.  The argument is evaluated exactly once.  Sites that add
  // per-direction (dim/dir) context keep an explicit errorQuda instead.
#define CHECK_CU_DRIVER(call) target::cuda::set_driver_error(call, #call, __func__, __FILE__, __STRINGIFY__(__LINE__));

  namespace comm_target
  {
#ifdef QUDA_MNNVL
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
      if (err != CUDA_SUCCESS) {
        // A failed import means "peer not reachable over the fabric" -- the caller
        // downgrades this link to MPI.  This probe must stay NON-fatal so the
        // symmetric capability exchange can decide the transport, so we never
        // abort here.  But we distinguish the expected not-reachable/not-permitted
        // codes from an unexpected one (e.g. CUDA_ERROR_NOT_SUPPORTED, which means
        // fabric/IMEX is unavailable altogether): surface the latter with a warning
        // rather than silently masking a misconfiguration as a benign fallback.
        // (The expected-code set is best-effort; an unexpected code only produces a
        // diagnostic warning, never a behavior change.)
        const char *name = nullptr;
        cuGetErrorName(err, &name);
        if (err != CUDA_ERROR_NOT_PERMITTED && err != CUDA_ERROR_INVALID_VALUE)
          warningQuda("try_import_fabric_handle: unexpected import error %s -- treating peer as unreachable",
                      name ? name : "UNKNOWN");
        else
          logQuda(QUDA_DEBUG_VERBOSE, "try_import_fabric_handle: peer unreachable (%s)\n", name ? name : "UNKNOWN");
        cudaGetLastError(); // clear any latched runtime error state
        return false;
      }
      return cuMemRelease(h) == CUDA_SUCCESS;
    }

    void close_fabric_probe(void *probe) { device_comm_buffer_free(probe); }
#else
    // Non-MNNVL build: fabric P2P is unavailable.  These stubs let target-agnostic
    // code call the facade under `if constexpr (comm_build_is_mnnvl())` (both
    // branches must compile) while never executing at runtime.
    size_t fabric_handle_size() { return 0; }
    void *open_fabric_probe(void *) { return nullptr; }
    bool try_import_fabric_handle(const void *) { return false; }
    void close_fabric_probe(void *) { }
#endif // QUDA_MNNVL
  }    // namespace comm_target

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
          receiveHandle = comm_declare_receive_relative(&remote_export[dim][1 - dir], dim, -disp, sizeof(FabricExport));
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
        receiveHandle
          = comm_declare_receive_relative(&remote_handle[dim][1 - dir], dim, -disp, sizeof(remote_handle[dim][1 - dir]));
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
        errorQuda("P2P buffer size mismatch: local=%zu peer=%lu (dim=%d dir=%d)", local_size, (unsigned long)peer.size,
                  dim, dir);
      size_t map_size = peer.size;

      CUdeviceptr peer_ptr = 0;
      CHECK_CU_DRIVER(cuMemAddressReserve(&peer_ptr, map_size, 0, 0, 0));
      CHECK_CU_DRIVER(cuMemMap(peer_ptr, map_size, 0, h, 0));

      int local_dev;
      CHECK_CUDA_ERROR(cudaGetDevice(&local_dev));
      CUmemAccessDesc acc = {};
      acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      acc.location.id = local_dev;
      acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CHECK_CU_DRIVER(cuMemSetAccess(peer_ptr, map_size, &acc, 1));
      err = cuMemRelease(h);
      if (err != CUDA_SUCCESS) errorQuda("cuMemRelease (peer import) failed (dim=%d dir=%d)", dim, dir);
      remote[dim][dir] = (void *)peer_ptr;
      auto inserted = p2p_remote_mappings.emplace((void *)peer_ptr, RemoteFabricMapping {map_size, peer.generation});
      if (!inserted.second) errorQuda("Duplicate imported P2P mapping address %p", (void *)peer_ptr);
      logQuda(QUDA_DEBUG_VERBOSE, "MNNVL P2P import: local=%p peer=%p size=%zu generation=%lu dim=%d dir=%d\n", local,
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
#ifdef NVSHMEM_COMMS
  void comm_create_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
#else
  // `local` is unused without NVSHMEM -- leave it unnamed so -Werror=unused-parameter is happy.
  void comm_create_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *)
#endif
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
        logQuda(QUDA_DEBUG_VERBOSE, "MNNVL P2P unmap: peer=%p size=%zu generation=%lu dim=%d dir=%d\n", ptr, map_size,
                (unsigned long)it->second.generation, dim, dir);
        CUresult err = cuMemUnmap((CUdeviceptr)ptr, map_size);
        if (err != CUDA_SUCCESS) errorQuda("cuMemUnmap (peer) failed (dim=%d dir=%d)", dim, dir);
        err = cuMemAddressFree((CUdeviceptr)ptr, map_size);
        if (err != CUDA_SUCCESS) errorQuda("cuMemAddressFree (peer) failed (dim=%d dir=%d)", dim, dir);
        p2p_remote_mappings.erase(it);
        remote[dim][dir] = nullptr;
      }
      if (num_dir == 1) remote[dim][1] = nullptr; // aliased to [dim][0], already freed
    }
  }
#else
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
        CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(remote[dim][dir]));
        remote[dim][dir] = nullptr;
      }
      if (num_dir == 1) remote[dim][1] = nullptr; // aliased to [dim][0], already closed
    }
  }
#endif

  // The NVSHMEM-transport remote pointer is owned by NVSHMEM (nvshmem_ptr); there
  // is nothing for QUDA to unmap.  No-op (defined for symmetry with create).
  void comm_destroy_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &) { }

  // ---------------------------------------------------------------------------
  // Stream-gated signal-slot ("flag") buffer lifecycle.  Allocated from
  // LatticeField::createIPCComms() and torn down from freeGhostBuffer(), so it
  // tracks the communicator and is recreated after a push_communicator() switch;
  // idempotent via stream_gated_comms_init.  Created only when stream-gating is
  // the resolved transport.  Kept OUT of the per-field ghost-buffer realloc churn
  // so the (constant-size) buffer's address is stable and its handle is opened
  // exactly once -- re-opening a same-address handle after free+realloc is what
  // CUDA 12.x rejects with "invalid argument".  Compiled in all builds (incl.
  // NVSHMEM): stream-gating can be the resolved transport regardless of NVSHMEM.
  namespace
  {
    bool stream_gated_comms_init = false;

    // Whether the device advertises CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
    // i.e. whether CU_STREAM_WAIT_VALUE_FLUSH is accepted by the driver.  GB200
    // reports 0 and rejects the flag (CUDA_ERROR_NOT_SUPPORTED), so the receiver
    // wait must stay plain GEQ there.  Set in comm_create_stream_gated_comms().
    bool stream_gated_remote_flush_supported = false;

    // Whether the REMOTE_WRITE P2P policy is safe to offer (see
    // comm_p2p_remote_write_supported).  Default true (correct for non-MNNVL,
    // which needs no flush); on MNNVL builds it is tied to the remote-write flush
    // capability in comm_create_stream_gated_comms() below.
    bool remote_write_p2p_supported = true;

    // Stream-mem-op signalling state.  Only this TU (the CUDA stream-gated
    // backend) touches it.  Forward-only protocol; slots dimensioned per
    // FieldKind so COLOR_SPINOR and GAUGE signals never alias.
    // Layout in flag_buffer_d[buf]: a uint64_t array of size
    // N_FIELD_KINDS * QUDA_MAX_DIM * 2, slot (kind,dim,dir) at byte offset
    // ((int(kind) * QUDA_MAX_DIM + dim) * 2 + dir) * sizeof(uint64_t).
    array<void *, 2> flag_buffer_d = {};                            // local slot buffer, per bufferIndex
    array_3d<void *, 2, QUDA_MAX_DIM, 2> flag_buffer_remote_d = {}; // peer's flag_buffer_d, cudaIPC-mapped
    // Monotonic host-side counters [kind][buf][dim][dir]: sender increments before
    // writing the peer's slot, receiver before waiting on its local slot.
    uint64_t flag_send_counter[N_FIELD_KINDS][2][QUDA_MAX_DIM][2] = {};
    uint64_t flag_recv_counter[N_FIELD_KINDS][2][QUDA_MAX_DIM][2] = {};
  } // namespace

  void comm_create_stream_gated_comms()
  {
    if (stream_gated_comms_init) return;
    if (comm::p2p_signal() != QudaP2PSignal::STREAM_GATED) return;

    bool has_p2p_neighbor = false;
    for (int dim = 0; dim < 4; ++dim)
      for (int dir = 0; dir < 2; ++dir) has_p2p_neighbor = has_p2p_neighbor || comm_peer2peer_enabled(dir, dim);
    if (!has_p2p_neighbor) return;

    CUdevice device;
    CHECK_CU_DRIVER(cuCtxGetDevice(&device));
    int supported = 0;
    CUresult err = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device);
    if (err != CUDA_SUCCESS || !supported) errorQuda("STREAM_GATED requires 64-bit CUDA stream memory operations");

    // Defaults hold on non-MNNVL builds (the block below is compiled out): the
    // REMOTE_WRITE policy stays offered (gated only by QUDA_ENABLE_P2P bit 1 in the
    // tuner) and the flush is never armed -- that path is correct without a flush.
    stream_gated_remote_flush_supported = false;
    remote_write_p2p_supported = true;

#ifdef QUDA_MNNVL
    // On MNNVL a REMOTE_WRITE halo lands in the peer's imported fabric-VMM mapping
    // as SM stores over many transactions; the doorbell can be observed before the
    // last data transaction commits, corrupting an iterative solve.  The only guard
    // is a receiver-side CU_STREAM_WAIT_VALUE_FLUSH.  Flush and remote-write are a
    // SINGLE knob here: remote-write is safe iff the flush is armed, and the flush
    // is only useful when remote-write is on.  Enable the pair iff bit 1 is set AND
    // (the device can flush OR the user forces it with QUDA_P2P_FORCE_FLUSH=1).
    //
    // Raw device capability: does the driver accept CU_STREAM_WAIT_VALUE_FLUSH
    // (CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES)?  GB200 reports 0 and rejects
    // the flag (CUDA_ERROR_NOT_SUPPORTED).
    bool flush_capable = false;
    err = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device);
    if (err == CUDA_SUCCESS && supported) flush_capable = true;

    const bool remote_write_requested = comm_peer2peer_enabled_global() & 2; // QUDA_ENABLE_P2P bit 1
    const char *force_env = getenv("QUDA_P2P_FORCE_FLUSH");
    const bool force_flush = force_env && atoi(force_env) == 1;
    if (remote_write_requested && !flush_capable) {
      if (force_flush)
        warningQuda("STREAM_GATED: remote-write requested but device does not report remote-write flush; "
                    "forcing CU_STREAM_WAIT_VALUE_FLUSH (QUDA_P2P_FORCE_FLUSH=1) -- the driver may reject it.");
      else
        warningQuda("STREAM_GATED: remote-write requested (QUDA_ENABLE_P2P bit 1) but device does not report "
                    "remote-write flush; disabling remote-write. Set QUDA_P2P_FORCE_FLUSH=1 to force it.");
    }
    const bool use_remote_write = remote_write_requested && (flush_capable || force_flush);
    stream_gated_remote_flush_supported = use_remote_write; // arm flush iff remote-write is enabled
    remote_write_p2p_supported = use_remote_write;          // offer the REMOTE_WRITE policy iff so
#endif

    const size_t bytes = N_FIELD_KINDS * QUDA_MAX_DIM * 2 * sizeof(uint64_t);
    // Re-zero the host signal counters in lock-step with the (re)zeroed device slots
    // below.  On a communicator switch push_communicator() -> freeGhostBuffer() frees this
    // buffer and a later createIPCComms() re-allocates it against the new communicator's
    // neighbours; the expected/written counter values must restart from 0 to stay paired
    // with the fresh, zeroed slots (otherwise the GEQ wait carries stale counts across the
    // switch and can pass prematurely or hang).  All ranks reach here collectively.
    std::memset(flag_send_counter, 0, sizeof(flag_send_counter));
    std::memset(flag_recv_counter, 0, sizeof(flag_recv_counter));
    for (int b = 0; b < 2; b++) {
      flag_buffer_d[b] = device_comm_buffer_malloc(bytes);
      qudaMemset(flag_buffer_d[b], 0, bytes);
      comm_create_neighbor_memory_p2p(flag_buffer_remote_d[b], flag_buffer_d[b]);
    }

    // Invariant: with stream-gating active, every enabled P2P direction must have a
    // valid local slot buffer and remote slot mapping.  A null here would make the
    // signal/wait primitives no-ops and silently expose stale halos.
    for (int b = 0; b < 2; b++) {
      if (!flag_buffer_d[b]) errorQuda("stream-gated flag buffer null after creation (buf=%d)", b);
      for (int dim = 0; dim < 4; ++dim) {
        if (comm_dim(dim) == 1) continue;
        for (int dir = 0; dir < 2; ++dir) {
          if (!comm_peer2peer_enabled(dir, dim)) continue;
          if (!flag_buffer_remote_d[b][dim][dir])
            errorQuda("stream-gated remote flag mapping null for enabled P2P dir (buf=%d dim=%d dir=%d)", b, dim, dir);
        }
      }
    }
    stream_gated_comms_init = true;
  }

  bool comm_p2p_remote_write_supported() { return remote_write_p2p_supported; }

  void comm_destroy_stream_gated_comms()
  {
    if (!stream_gated_comms_init) {
#ifdef QUDA_MNNVL
      // freeGhostBuffer() calls this immediately after destroyIPCComms(), even on
      // communicators with no P2P neighbours.  The remote-map registry must be
      // empty regardless of whether flag buffers were needed.
      if (!p2p_remote_mappings.empty())
        errorQuda("comm_destroy_stream_gated_comms: %zu imported P2P mapping(s) remain without flag-buffer state",
                  p2p_remote_mappings.size());
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

  // ---------------------------------------------------------------------------
  // comm_p2p_* (CUDA target backend): IPC-event + MPI-doorbell. Bodies are
  // 1:1 with the direct calls color_spinor_field.cpp and gauge_field.cpp made
  // before this refactor; FieldKind is accepted but unused here (the stream-gated
  // backend uses it for per-kind slot dimensioning so COLOR_SPINOR and GAUGE
  // signals never alias).

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

  // ---------------------------------------------------------------------------
  // Stream-mem-op signalling primitives.
  // Forward-only protocol: sender writes a monotonic uint64_t counter to peer's
  // slot via cuStreamWriteValue64; receiver stream-waits via cuStreamWaitValue64
  // with GEQ and CU_STREAM_WAIT_VALUE_FLUSH so the preceding remote halo write is
  // visible to downstream device work.  Slots are dimensioned per FieldKind so
  // COLOR_SPINOR and GAUGE signals never alias; the protocol is one-way (no
  // acknowledgement back-edge).

  namespace
  {
    inline CUdeviceptr slot_dev_addr(void *base, FieldKind kind, int dim, int dir)
    {
      return reinterpret_cast<CUdeviceptr>(reinterpret_cast<uint64_t *>(base)
                                           + (static_cast<int>(kind) * QUDA_MAX_DIM + dim) * 2 + dir);
    }
  } // anonymous namespace

  void comm_p2p_stream_signal_send_done(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream)
  {
    if (!comm_peer2peer_enabled(dir, dim)) return;
    // P2P is enabled for this direction and stream-gating is the resolved transport,
    // so the remote flag mapping must exist -- a null means lifecycle setup was
    // skipped, which would silently drop the halo signal.
    if (!flag_buffer_remote_d[buf][dim][dir])
      errorQuda("stream-gated signal: remote flag mapping null (buf=%d dim=%d dir=%d)", buf, dim, dir);

    // Peer-at-(dim, dir) sees us as their (dim, 1-dir) neighbour, so we write
    // their slot at (kind, dim, 1-dir).
    CUdeviceptr peer = slot_dev_addr(flag_buffer_remote_d[buf][dim][dir], kind, dim, 1 - dir);
    uint64_t value = ++flag_send_counter[static_cast<int>(kind)][buf][dim][dir];

    logQuda(QUDA_DEBUG_VERBOSE,
            "comm_p2p_stream_signal_send_done: k=%d buf=%d dim=%d dir=%d -> "
            "peer_slot(kind=%d,dim=%d,dir=%d) val=%lu\n",
            static_cast<int>(kind), buf, dim, dir, static_cast<int>(kind), dim, 1 - dir, (unsigned long)value);

    CHECK_CU_DRIVER(cuStreamWriteValue64(target::cuda::get_stream(stream), peer, value, CU_STREAM_WRITE_VALUE_DEFAULT));
  }

  void comm_p2p_stream_wait_recv_signal(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream)
  {
    if (!comm_peer2peer_enabled(dir, dim)) return;
    // As above: an enabled P2P direction under stream-gating must have a local slot
    // buffer; a null would make the wait a no-op and expose a stale halo.
    if (!flag_buffer_d[buf])
      errorQuda("stream-gated wait: local flag buffer null (buf=%d dim=%d dir=%d)", buf, dim, dir);

    CUdeviceptr local = slot_dev_addr(flag_buffer_d[buf], kind, dim, dir);
    uint64_t expected = ++flag_recv_counter[static_cast<int>(kind)][buf][dim][dir];

    logQuda(QUDA_DEBUG_VERBOSE,
            "comm_p2p_stream_wait_recv_signal: k=%d buf=%d dim=%d dir=%d "
            "local_slot(kind=%d,dim=%d,dir=%d) expected=%lu\n",
            static_cast<int>(kind), buf, dim, dir, static_cast<int>(kind), dim, dir, (unsigned long)expected);

    // Receiver-side ordering for REMOTE_WRITE halos: add CU_STREAM_WAIT_VALUE_FLUSH
    // so the wait drains outstanding remote writes to us before it is satisfied.
    // Only where the device supports it (stream_gated_remote_flush_supported); on
    // devices that reject the flag (GB200, CUDA_ERROR_NOT_SUPPORTED) remote-write is
    // not offered as a policy (comm_p2p_remote_write_supported), so plain GEQ is
    // correct here.
    unsigned int wait_flags = CU_STREAM_WAIT_VALUE_GEQ;
    if (stream_gated_remote_flush_supported) wait_flags |= CU_STREAM_WAIT_VALUE_FLUSH;
    CHECK_CU_DRIVER(cuStreamWaitValue64(target::cuda::get_stream(stream), local, expected, wait_flags));
  }

  // ============================================================================
  // Unified P2P signal API (Phase B of the TransportContext refactor).
  // These overloads take a QudaP2PSignal and dispatch to the appropriate
  // per-kind implementation defined above.  Subsequent phases will migrate the
  // dslash + gauge call sites to use these signatures, after which the per-kind
  // functions can become file-static.
  // ============================================================================

  namespace
  {
    // Defensive guard for the enum-taking dispatchers: a caller may pass
    // STREAM_GATED explicitly, so make sure stream-gating is actually the
    // resolved P2P transport before dispatching to the stream-mem-op leaves.
    // The resolved transport is init-verified/device-supported (see
    // comm_create_stream_gated_comms, which errors if 64-bit stream mem ops are
    // unavailable), so a mismatch here means a caller picked the wrong signal.
    inline void assert_stream_gated_resolved(const char *who)
    {
      if (comm::p2p_signal() != QudaP2PSignal::STREAM_GATED)
        errorQuda("%s: STREAM_GATED signalling requested but it is not the resolved P2P transport", who);
    }
  } // anonymous namespace

  void comm_p2p_signal_send_done(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream,
                                 QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC: comm_p2p_signal_send_done(kind, buf, dim, dir, stream); return;
    case QudaP2PSignal::STREAM_GATED:
      assert_stream_gated_resolved(__func__);
      comm_p2p_stream_signal_send_done(kind, buf, dim, dir, stream);
      return;
    }
    errorQuda("comm_p2p_signal_send_done: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

  void comm_p2p_wait_recv_signal(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream,
                                 QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC:
      // Legacy path is host-blocking (synchronizes on ipcRemoteCopyEvent).  For
      // stream-ordered REMOTE_IPC behavior under this unified API we'd want
      // cudaStreamWaitEvent instead -- but no existing call site uses that
      // pattern, and current callers that need host-blocking will migrate to
      // the legacy 4-arg comm_p2p_wait_recv_signal directly.  For now, dispatch
      // to host-blocking (matches semantics of the existing 4-arg variant);
      // callers that pass a stream and care about stream-ordering on REMOTE_IPC
      // should use stream wait events explicitly.
      (void)stream;
      comm_p2p_wait_recv_signal(kind, buf, dim, dir);
      return;
    case QudaP2PSignal::STREAM_GATED:
      assert_stream_gated_resolved(__func__);
      comm_p2p_stream_wait_recv_signal(kind, buf, dim, dir, stream);
      return;
    }
    errorQuda("comm_p2p_wait_recv_signal: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

  void comm_p2p_wait_send_drained(FieldKind kind, int buf, int dim, int dir, QudaP2PSignal signal)
  {
    switch (signal) {
    case QudaP2PSignal::REMOTE_IPC: comm_p2p_wait_send_drained(kind, buf, dim, dir); return;
    case QudaP2PSignal::STREAM_GATED:
      assert_stream_gated_resolved(__func__);
      // STREAM_GATED has no separate "send drained" concept: stream ordering
      // guarantees that any subsequent op enqueued on the same stream will
      // observe the prior cuStreamWriteValue64 completed.  Nothing to wait for.
      return;
    }
    errorQuda("comm_p2p_wait_send_drained: unknown QudaP2PSignal %d", static_cast<int>(signal));
  }

} // namespace quda
