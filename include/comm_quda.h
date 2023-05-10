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
     Return the coording of this process in the dimension dim
     @param dim Dimension which we are querying
     @return Coordinate of this process
  */
  int comm_coord(int dim);

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
     @brief Symmetric exchange of local memory addresses between
     logically neighboring processes on the lattice.  The remote
     addresses that are returned are directly addressable by the local
     process and can be read or written to by a kernel, or can be
     copied to and from.  This exchange is only defined between
     devices that are peer-to-peer enabled.
     @param[out] remote Array of remote memory pointers to neighboring
     pointers
     @param[in] local The process-local memory pointer to be exchanged
     from this process
  */
  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local);

  /**
     @brief Deallocate the remote addresses to logically neighboring
     processes on the on the lattice.
     @param[in] remote Array of remote memory pointers to neighboring
     pointers
  */
  void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote);

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

  void comm_barrier(void);
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
