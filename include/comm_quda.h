#pragma once
#include <cstdint>
#include <vector>
#include <quda_constants.h>
#include <quda_api.h>
#include <array.h>

#ifdef __cplusplus
extern "C" {
#endif

/* defined in quda.h; redefining here to avoid circular references */
typedef int (*QudaCommsMap)(const int *coords, void *fdata);

#ifdef __cplusplus
}
#endif

/** Maximum length in bytes of the host string */
#define QUDA_MAX_HOSTNAME_STRING 128

namespace quda
{

  typedef struct MsgHandle_s MsgHandle;
  typedef struct Topology_s Topology;

  char *comm_hostname(void);
  double comm_drand(void);
  Topology *comm_create_topology(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data);
  void comm_destroy_topology(Topology *topo);
  int comm_ndim(const Topology *topo);
  const int *comm_dims(const Topology *topo);
  const int *comm_coords(const Topology *topo);
  const int *comm_coords_from_rank(const Topology *topo, int rank);
  int comm_rank_from_coords(const Topology *topo, const int *coords);
  int comm_rank_displaced(const Topology *topo, const int displacement[]);
  void comm_set_default_topology(Topology *topo);
  Topology *comm_default_topology(void);

  // routines related to direct peer-2-peer access
  void comm_set_neighbor_ranks(Topology *topo = NULL);
  int comm_neighbor_rank(int dir, int dim);

  /**
     Return the number of processes in the dimension dim
     @param dim Dimension which we are querying
     @return Length of process dimensions
  */
  int comm_dim(int dim);

  /**
     Return whether the dimension dim is a C* dimension or not
     @param dim Dimension which we are querying
     @return C* dimension or nor
  */
  bool comm_dim_cstar(int dim);

  /**
     Return the global number of processes in the dimension dim
     @param dim Dimension which we are querying
     @return Length of process dimensions
  */
  int comm_dim_global(int dim);

  /**
     Return the coordinate of this process in the dimension dim
     @param dim Dimension which we are querying
     @return Coordinate of this process
  */
  int comm_coord(int dim);

  /**
     Return the global coordinates of this process in the dimension dim
     @param dim Dimension which we are querying
     @return Coordinate of this process
  */
  int comm_coord_global(int dim);

  /**
   * Declare a message handle for sending `nbytes` to the `rank` with `tag`.
   */
  MsgHandle *comm_declare_send_rank(void *buffer, int rank, int tag, size_t nbytes);

  /**
   * Declare a message handle for receiving `nbytes` from the `rank` with `tag`.
   */
  MsgHandle *comm_declare_recv_rank(void *buffer, int rank, int tag, size_t nbytes);

  /**
     Create a persistent message handler for a relative send.  This
     should not be called directly, and instead the helper macro
     (without the trailing underscore) should be called instead.
     @param buffer Buffer from which message will be sent
     @param dim Dimension in which message will be sent
     @param dir Direction in which messaged with be sent (0 - backwards, 1 forwards)
     @param nbytes Size of message in bytes
  */
  MsgHandle *comm_declare_send_relative_(const char *func, const char *file, int line, void *buffer, int dim, int dir,
                                         size_t nbytes);

#define comm_declare_send_relative(buffer, dim, dir, nbytes)                                                           \
  comm_declare_send_relative_(__func__, __FILE__, __LINE__, buffer, dim, dir, nbytes)

  /**
     Create a persistent message handler for a relative send.  This
     should not be called directly, and instead the helper macro
     (without the trailing underscore) should be called instead.
     @param buffer Buffer into which message will be received
     @param dim Dimension from message will be received
     @param dir Direction from messaged with be recived (0 - backwards, 1 forwards)
     @param nbytes Size of message in bytes
  */
  MsgHandle *comm_declare_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                            int dir, size_t nbytes);

#define comm_declare_receive_relative(buffer, dim, dir, nbytes)                                                        \
  comm_declare_receive_relative_(__func__, __FILE__, __LINE__, buffer, dim, dir, nbytes)

  /**
     Create a persistent strided message handler for a relative send.
     This should not be called directly, and instead the helper macro
     (without the trailing underscore) should be called instead.
     @param buffer Buffer from which message will be sent
     @param dim Dimension in which message will be sent
     @param dir Direction in which messaged with be sent (0 - backwards, 1 forwards)
     @param blksize Size of block in bytes
     @param nblocks Number of blocks
     @param stride Stride between blocks in bytes
  */
  MsgHandle *comm_declare_strided_send_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                                 int dir, size_t blksize, int nblocks, size_t stride);

#define comm_declare_strided_send_relative(buffer, dim, dir, blksize, nblocks, stride)                                 \
  comm_declare_strided_send_relative_(__func__, __FILE__, __LINE__, buffer, dim, dir, blksize, nblocks, stride)

  /**
     Create a persistent strided message handler for a relative receive
     This should not be called directly, and instead the helper macro
     (without the trailing underscore) should be called instead.
     @param buffer Buffer into which message will be received
     @param dim Dimension from message will be received
     @param dir Direction from messaged with be recived (0 - backwards, 1 forwards)
     @param blksize Size of block in bytes
     @param nblocks Number of blocks
     @param stride Stride between blocks in bytes
  */
  MsgHandle *comm_declare_strided_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                                    int dir, size_t blksize, int nblocks, size_t stride);

#define comm_declare_strided_receive_relative(buffer, dim, dir, blksize, nblocks, stride)                              \
  comm_declare_strided_receive_relative_(__func__, __FILE__, __LINE__, buffer, dim, dir, blksize, nblocks, stride)

  void comm_finalize(void);
  void comm_dim_partitioned_set(int dim);
  int comm_dim_partitioned(int dim);

  /**
     @brief Loop over comm_dim_partitioned(dim) for all comms dimensions
     @return Whether any communications dimensions are partitioned
  */
  int comm_partitioned();

  /**
     @brief Create the topology and partition strings that are used in tuneKeys
  */
  void comm_set_tunekey_string();

  /**
     @brief Return a string that defines the comm partitioning (used as a tuneKey)
     @param comm_dim_override Optional override for partitioning
     @return String specifying comm partitioning
  */
  const char *comm_dim_partitioned_string(const int *comm_dim_override = 0);

  /**
     @brief Return a string that defines the comm topology (for use as a tuneKey)
     @return String specifying comm topology
  */
  const char *comm_dim_topology_string();

  /**
     @brief Return a string that defines the P2P/GDR environment
     variable configuration (for use as a tuneKey to enable unique
     policies).
     @return String specifying comm config
  */
  const char *comm_config_string();

  /**
     @brief Initialize the communications, implemented in comm_single.cpp, comm_qmp.cpp, and comm_mpi.cpp
  */
  void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data,
                 bool user_set_comm_handle = false, void *user_comm = nullptr);

  /**
     @brief Initialize the communications common to all communications abstractions
  */
  void comm_init_common(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data);

  /**
     @return Rank id of this process
  */
  int comm_rank(void);

  /**
     @return the default rank id of this process.
     This doesn't go through the communicator route, so it can be called without initializing the communicator stack.
  */
  int comm_rank_global(void);

  /**
     @return Number of processes
  */
  size_t comm_size(void);

  /**
     @return GPU id associated with this process
  */
  int comm_gpuid(void);

  /**
     @return Whether are doing determinisitic multi-process reductions or not
  */
  bool comm_deterministic_reduce();

  /**
     @brief Gather all hostnames
     @param[out] hostname_recv_buf char array of length
     QUDA_MAX_HOSTNAME_STRING*comm_size() that will be filled in GPU ids for all processes.
     Each hostname is in rank order, with QUDA_MAX_HOSTNAME_STRING bytes for each.
  */
  void comm_gather_hostname(char *hostname_recv_buf);

  /**
     @brief Gather all GPU ids
     @param[out] gpuid_recv_buf int array of length comm_size() that
     will be filled in GPU ids for all processes (in rank order).
  */
  void comm_gather_gpuid(int *gpuid_recv_buf);

#ifdef QUDA_MNNVL
  /**
     @brief Gather every rank's local GPU NVML fabric clique ID into clique_recv_buf
     (length comm_size()).  Used by comm_peer2peer_init to detect MNNVL reachability
     across hostnames: ranks with the same clique ID can do NVLink fabric P2P.
   */
  void comm_gather_clique_id(unsigned int *clique_recv_buf);

  /**
     @brief Allgather a CUmemFabricHandle (opaque, size handle_size bytes).
     Used by the fabric-reachability probe at communicator init to discover
     which peer ranks can actually be reached via cuMemImportFromShareableHandle
     (NVML's cliqueId is uninformative on some MNNVL systems, e.g. Ptyche
     where it returns a constant sentinel).
   */
  void comm_gather_fabric_handle(void *send_handle, void *recv_buf, size_t handle_size);
#endif

  /**
     Enabled peer-to-peer communication.
     @param hostname_buf Array that holds all process hostnames
  */
  void comm_peer2peer_init(const char *hostname_recv_buf);

  /**
     @brief Query if peer-to-peer communication is possible between two GPUs
     @param[in] local_gpuid GPU associated with this process
     @param[in] neighbor_gpuid GPU associated with neighboring process
     (assumed on same node)
     @return True/false if peer-to-peer is possible
  */
  bool comm_peer2peer_possible(int local_gpuid, int neighbor_gpuid);

  /**
     @brief Query the performance of peer-to-peer communication between two GPUs
     @param[in] local_gpuid GPU associated with this process
     @param[in] neighbor_gpuid GPU associated with neighboring process
     (assumed on same node)
     @return Relative performance ranking between this pair of GPUs
  */
  int comm_peer2peer_performance(int local_gpuid, int neighbor_gpuid);

  /**
     @brief QUDA-owned P2P exchange of local memory addresses between
     logically neighboring processes.  Exchanges a SINGLE fabric (MNNVL) or
     cudaIPC (non-IMEX) handle for the local contiguous P2P buffer and imports
     the peer's, so P2P writes target a single-allocation, RDMA-capable buffer.
     Compiled in ALL builds (including NVSHMEM) -- under NVSHMEM the symmetric
     heap does not give a reusable single-handle export, so P2P owns its own
     DeviceCommBuffer.  Only defined between peer-to-peer-enabled devices.
     @param[out] remote Array of remote memory pointers to neighboring pointers
     @param[in] local The process-local memory pointer to be exchanged
  */
  void comm_create_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local);

  /**
     @brief NVSHMEM-transport neighbor pointer (nvshmem_ptr of the peer's
     symmetric recv buffer): the Shmem direct-NVLink put destination.  No-op
     when not built with NVSHMEM.  Kept strictly separate from the P2P import so
     a single pointer family never means both "NVSHMEM symmetric remote" and
     "QUDA P2P import".
     @param[out] remote Array of remote memory pointers to neighboring pointers
     @param[in] local The process-local symmetric buffer
  */
  void comm_create_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local);

  /**
     @brief Deallocate the QUDA-owned P2P remote addresses created by
     comm_create_neighbor_memory_p2p.
     @param[in] remote Array of remote memory pointers to neighboring pointers
  */
  void comm_destroy_neighbor_memory_p2p(array_2d<void *, QUDA_MAX_DIM, 2> &remote);

  /**
     @brief Tear down the NVSHMEM-transport neighbor pointers (no-op: owned by
     NVSHMEM).
     @param[in] remote Array of remote memory pointers to neighboring pointers
  */
  void comm_destroy_neighbor_memory_shmem(array_2d<void *, QUDA_MAX_DIM, 2> &remote);

  /**
     @brief Allocate and IPC-exchange the stream-mem-op signal-slot ("flag")
     buffers used by the stream-gated P2P signalling path.  No-op unless
     stream-gating is the resolved transport (comm::p2p_signal() ==
     STREAM_GATED), so events-only / HIP builds never allocate it.  The flag
     buffer is constant-size and is created from createIPCComms() (once per
     communicator, idempotent).  It is deliberately kept out of the per-field
     ghost-buffer create/destroy churn -- stable across ordinary ghost-buffer
     resizing -- so its address does not move and its IPC handle never needs
     re-opening (which CUDA rejects on same-address reuse).
  */
  void comm_create_stream_gated_comms();

  /**
     @brief Whether the REMOTE_WRITE P2P policy (the packing kernel stores halos
     directly into the peer's buffer) is safe to offer to the dslash policy
     tuner on this build/device.  On MNNVL/fabric builds remote-write halos land
     in the peer's imported VMM buffer across multiple transactions and the
     doorbell can be observed before the last data transaction commits; the only
     safe guard is a receiver-side flush (CU_STREAM_WAIT_VALUE_FLUSH).  GB200
     reports CAN_FLUSH_REMOTE_WRITES=0 and rejects that flag (CUDA_ERROR_NOT_
     SUPPORTED), so remote-write is dropped from the policy list there and the
     copy-engine path is used instead.  Always true on non-MNNVL builds (that
     path is correct without a flush).  Resolved at P2P setup
     (comm_create_stream_gated_comms).
  */
  bool comm_p2p_remote_write_supported();

  /**
     @brief Tear down the stream-gated signal-slot buffers created by
     comm_create_stream_gated_comms.  Called from freeGhostBuffer(), so the
     buffers are destroyed (and later recreated) across communicator changes.
  */
  void comm_destroy_stream_gated_comms();

  /**
     @brief Create unique events shared between each logical pair of
     neighboring processes, e.g., the event in the forwards direction
     in a given dimension on a given process aliases the event in the
     backward direction in the same dimension, and is unique
     between that process pair. This exchange is only defined between
     devices that are peer-to-peer enabled.
     @param[out] remote Array of remote events to neighboring processes
     @param[in] local Array of local event to neighboring processes
   */
  void comm_create_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                  array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local);

  /**
     @brief Destroy the coupled events
     @param[out] remote Array of remote events to neighboring processes
     @param[in] local Array of local event to neighboring processes
   */
  void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                   array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local);

  /**
     @brief Returns true if any peer-to-peer capability is present on
     this system (regardless of whether it has been disabled or not.  We
     use this, for example, to determine if we need to allocate pinned
     device memory or not.
  */
  bool comm_peer2peer_present();

  /**
     Query what peer-to-peer communication is enabled globally
     @return 2-bit number reporting 1 for copy engine, 2 for remote writes
  */
  int comm_peer2peer_enabled_global();

  /**
     Query if peer-to-peer communication is enabled
     @param dir Direction (0 - backwards, 1 forwards)
     @param dim Dimension (0-3)
     @return Whether peer-to-peer is enabled
  */
  bool comm_peer2peer_enabled(int dir, int dim);

  /**
     @brief Enable / disable peer-to-peer communication: used for dslash
     policies that do not presently support peer-to-peer communication
     @param[in] enable Boolean flag to enable / disable peer-to-peer communication
  */
  void comm_enable_peer2peer(bool enable);

  /**
     Query if intra-node (non-peer-to-peer) communication is enabled
     in a given dimension and direction
     @param dir Direction (0 - backwards, 1 forwards)
     @param dim Dimension (0-3)
     @return Whether intra-node communication is enabled
  */
  bool comm_intranode_enabled(int dir, int dim);

  /**
     @brief Enable / disable intra-node (non-peer-to-peer)
     communication
     @param[in] enable Boolean flag to enable / disable intra-node
     (non peer-to-peer) communication
  */
  void comm_enable_intranode(bool enable);

  /**
     @brief Query if GPU Direct RDMA communication is enabled (global setting)
  */
  bool comm_gdr_enabled();

  /**
     @brief Return if zero-copy policy kernels have been enabled.  By
     default kernels that read their communication halos directly from
     host memory are disabled to reduce tuning time, since on
     PCIe-based architectures, these kernels underperform and can take
     excessive tuning time.  They can be enabled with the environment
     variable QUDA_ENABLE_ZERO_COPY=1
     @return Return if zero-copy policy halos are enabled
   */
  bool comm_zero_copy_enabled();

  /**
     @brief Query if NVSHMEM communication is enabled (global setting)
  */
  bool comm_nvshmem_enabled();

  /**
      @brief Query if GPU Direct RDMA communication is blacklisted for this GPU
  */
  bool comm_gdr_blacklist();

  /**
     Create a persistent message handler for a relative send
     @param buffer Buffer from which message will be sent
     @param dim Dimension in which message will be sent
     @param dir Direction in which messaged with be sent (0 - backwards, 1 forwards)
     @param nbytes Size of message in bytes
  */
  MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes);

  /**
     Create a persistent message handler for a relative receive
     @param buffer Buffer into which message will be received
     @param dim Dimension from message will be received
     @param dir Direction from messaged with be recived (0 - backwards, 1 forwards)
     @param nbytes Size of message in bytes
  */
  MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes);

  /**
     Create a persistent strided message handler for a displaced send
     @param buffer Buffer from which message will be sent
     @param displacement Array of offsets specifying the relative node to which we are sending
     @param blksize Size of block in bytes
     @param nblocks Number of blocks
     @param stride Stride between blocks in bytes
  */
  MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                 size_t stride);

  /**
     Create a persistent strided message handler for a displaced receive
     @param buffer Buffer into which message will be received
     @param displacement Array of offsets specifying the relative node from which we are receiving
     @param blksize Size of block in bytes
     @param nblocks Number of blocks
     @param stride Stride between blocks in bytes
  */
  MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                    size_t stride);

  void comm_free(MsgHandle *&mh);
  void comm_start(MsgHandle *mh);
  void comm_wait(MsgHandle *mh);
  int comm_query(MsgHandle *mh);

  /**
     Backend-substitutable layer for peer-to-peer halo signalling.

     Wraps the per-(buf, dim, dir) "I'm done writing your buffer" /
     "wait for peer to be done writing mine" sequence.  Dispatches on the
     resolved QudaP2PSignal: REMOTE_IPC uses cudaIpcEvent + MPI doorbell;
     STREAM_GATED uses cuStreamWriteValue64 / cuStreamWaitValue64 on
     fabric-mapped signal slots.  Keyed by FieldKind so COLOR_SPINOR and
     GAUGE signals never alias.
  */
  enum class FieldKind { COLOR_SPINOR, GAUGE };
  static constexpr int N_FIELD_KINDS = 2; // keep in sync with FieldKind

  /**
     P2P signalling mechanism (used by the comm_p2p_signal_* unified API
     below to dispatch to the appropriate per-kind implementation inside the
     backend).

     - REMOTE_IPC  : cudaIPC event record/wait + MPI doorbell.  Single-node
                     only -- the IPC event handle does not cross the MNNVL
                     fabric.  HIP backend uses hipIpc analogues.
     - STREAM_GATED: cuStreamWriteValue64 / cuStreamWaitValue64 on a slot in
                     a peer-mapped flag buffer.  Works cross-clique within an
                     MNNVL NVLink fabric.  HIP requires hipStreamWriteValue64
                     (not confirmed yet -- backend allow-list filters this).
  */
  enum class QudaP2PSignal {
    REMOTE_IPC,
    STREAM_GATED
  };

  namespace comm
  {
    /** Does the active backend (and build) support the given P2P signalling kind?
        Backend-provided allow-list: CUDA non-MNNVL supports both; CUDA MNNVL supports
        STREAM_GATED only; HIP supports REMOTE_IPC only.  Implementation lives in
        lib/targets/<backend>/p2p_signal_defaults.cpp. */
    bool p2p_signal_supported(QudaP2PSignal kind);

    /** Backend/build default signalling kind when QUDA_P2P_TRANSPORT is unset.
        Policy: prefer REMOTE_IPC (events) -- matches develop behaviour, least
        surprise -- but clamp to the supported set, so CUDA-MNNVL (where
        REMOTE_IPC is unsupported) falls back to STREAM_GATED.  Implemented
        per-backend in lib/targets/<backend>/p2p_signal_defaults.cpp. */
    QudaP2PSignal p2p_signal_default();

    /** Resolve the active P2P signalling transport for this run (cached).
        Reads the QUDA_P2P_TRANSPORT env var (string, case-insensitive:
        "stream_gated" | "events"); if set it must be supported by this
        backend/build or we errorQuda (an explicit request is never silently
        substituted).  If unset, returns p2p_signal_default().  This is the
        single source of truth for both the autotuned (dslash) and the
        non-autotuned (gauge) P2P consumers. */
    QudaP2PSignal p2p_signal();
  } // namespace comm

  /** Sender: P2P write into peer's recv buffer is complete; peer may now read. */
  void comm_p2p_signal_send_done(FieldKind kind, int buf, int dim, int dir, const qudaStream_t &stream);

  /** Sender: non-blocking query — has the local send been drained? */
  int comm_p2p_query_send_drained(FieldKind kind, int buf, int dim, int dir);

  /** Sender: block host until local send is drained AND own copy event has fired. */
  void comm_p2p_wait_send_drained(FieldKind kind, int buf, int dim, int dir);

  /** Receiver: non-blocking query — has the peer signalled completion? */
  int comm_p2p_query_recv_signal(FieldKind kind, int buf, int dim, int dir);

  /** Receiver: block host until peer signal arrives AND peer's copy event has fired. */
  void comm_p2p_wait_recv_signal(FieldKind kind, int buf, int dim, int dir);

  // Stream-mem-op signalling primitives (Phase 5).  Forward-only protocol:
  // sender writes a monotonic uint64_t counter to the peer's slot via
  // cuStreamWriteValue64; receiver stream-waits via cuStreamWaitValue64 with
  // GEQ and, where supported by the device, a remote-write visibility flush.
  // No MPI doorbell, no IPC event, no host poll.  Slot storage lives
  // in the CUDA backend TU (lib/targets/cuda/comm_target.cpp).  This is the
  // protocol the future MNNVL fabric backend will use; on the current setup
  // it operates on cuMemAlloc-backed cudaIPC-shared memory.

  /**
     Unified P2P signal API.  Dispatches internally on QudaP2PSignal to the
     appropriate per-kind implementation inside the backend (event-based via
     cudaIPC + MPI doorbell, or stream-mem-op via cuStreamWriteValue64 /
     WaitValue64).  The per-kind functions are no longer part of the public
     header -- the stream-mem-op variants are file-private to the CUDA
     backend's comm_target.cpp; the event-based variants remain externally
     visible (above) for the few remaining direct callers but new code should
     use the QudaP2PSignal overloads.
  */
  void comm_p2p_signal_send_done(FieldKind kind, int buf, int dim, int dir,
                                  const qudaStream_t &stream, QudaP2PSignal signal);
  void comm_p2p_wait_recv_signal(FieldKind kind, int buf, int dim, int dir,
                                  const qudaStream_t &stream, QudaP2PSignal signal);
  void comm_p2p_wait_send_drained(FieldKind kind, int buf, int dim, int dir,
                                   QudaP2PSignal signal);

  template <typename T> void comm_allreduce_sum(T &v);
  template <typename T> void comm_allreduce_max(T &v);
  template <typename T> void comm_allreduce_min(T &v);

  void comm_allreduce_int(int &data);
  void comm_allreduce_xor(uint64_t &data);

  /**
     @brief Broadcast from the root rank
     @param[in,out] data The data to be read from on the root rank, and
     written to on all other ranks
     @param[in] nbytes The size in bytes of data to be broadcast
     @param[in] root The process that will be broadcasting
  */
  void comm_broadcast(void *data, size_t nbytes, int root = 0);

  /**
     @brief Multi-process barrier that applies to the present
     communicator
   */
  void comm_barrier(void);

  /**
     @brief Multi-process barrier that is global regardless of the
     present communicator
   */
  void comm_barrier_global(void);

  void comm_abort(int status);
  void comm_abort_(int status);

  int commDim(int);
  int commCoords(int);
  int commDimPartitioned(int dir);
  void commDimPartitionedSet(int dir);

  /**
   * @brief Reset the comm dim partioned array to zero,
   * @details This should only be needed for automated testing
   * when different partitioning is applied within a single run.
   */
  void commDimPartitionedReset();
  bool commGlobalReduction();
  void commGlobalReductionPush(bool global_reduce);
  void commGlobalReductionPop();

  bool commAsyncReduction();
  void commAsyncReductionSet(bool global_reduce);

} // namespace quda
