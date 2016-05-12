#ifndef _COMM_QUDA_H
#define _COMM_QUDA_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct MsgHandle_s MsgHandle;
  typedef struct Topology_s Topology;

  /* defined in quda.h; redefining here to avoid circular references */ 
  typedef int (*QudaCommsMap)(const int *coords, void *fdata);

  /* implemented in comm_common.cpp */

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
  void comm_set_neighbor_ranks(Topology *topo=NULL);
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
     Create a persistent message handler for a relative send.  This
     should not be called directly, and instead the helper macro
     (without the trailing underscore) should be called instead.
     @param buffer Buffer from which message will be sent
     @param dim Dimension in which message will be sent
     @param dir Direction in which messaged with be sent (0 - backwards, 1 forwards)
     @param nbytes Size of message in bytes
  */
  MsgHandle *comm_declare_send_relative_(const char *func, const char *file, int line,
					 void *buffer, int dim, int dir, size_t nbytes);

#define comm_declare_send_relative(buffer, dim, dir, nbytes)		\
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
  MsgHandle *comm_declare_receive_relative_(const char *func, const char *file, int line,
					    void *buffer, int dim, int dir, size_t nbytes);

#define comm_declare_receive_relative(buffer, dim, dir, nbytes)		\
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
  MsgHandle *comm_declare_strided_send_relative_(const char *func, const char *file, int line,
						 void *buffer, int dim, int dir,
						 size_t blksize, int nblocks, size_t stride);

#define comm_declare_strided_send_relative(buffer, dim, dir, blksize, nblocks, stride) \
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
  MsgHandle *comm_declare_strided_receive_relative_(const char *func, const char *file, int line,
						    void *buffer, int dim, int dir,
						    size_t blksize, int nblocks, size_t stride);

#define comm_declare_strided_receive_relative(buffer, dim, dir, blksize, nblocks, stride) \
  comm_declare_strided_receive_relative_(__func__, __FILE__, __LINE__, buffer, dim, dir, blksize, nblocks, stride)

  void comm_finalize(void);
  void comm_dim_partitioned_set(int dim);
  int comm_dim_partitioned(int dim);

  /**
     @brief Loop over comm_dim_partitioned(dim) for all comms dimensions
     @return Whether any communications dimensions are partitioned
   */
  int comm_partitioned();

  /* implemented in comm_single.cpp, comm_qmp.cpp, and comm_mpi.cpp */

  void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data);
  int comm_rank(void);
  int comm_size(void);
  int comm_gpuid(void);

  /**
     Enabled peer-to-peer communication.
     @param hostname_buf Array that holds all process hostnames
   */
  void comm_peer2peer_init(const char *hostname_recv_buf);

  /**
     Query if peer-to-peer communication is enabled
     @param dir Direction (0 - backwards, 1 forwards)
     @param dim Dimension (0-3)
     @return Whether peer-to-peer is enabled
  */
  bool comm_peer2peer_enabled(int dir, int dim);

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
  MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[], 
						 size_t blksize, int nblocks, size_t stride);

  /**
     Create a persistent strided message handler for a displaced receive
     @param buffer Buffer into which message will be received
     @param displacement Array of offsets specifying the relative node from which we are receiving
     @param blksize Size of block in bytes
     @param nblocks Number of blocks
     @param stride Stride between blocks in bytes
  */
  MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						    size_t blksize, int nblocks, size_t stride);

  void comm_free(MsgHandle *mh);
  void comm_start(MsgHandle *mh);
  void comm_wait(MsgHandle *mh);
  int comm_query(MsgHandle *mh);
  void comm_allreduce(double* data);
  void comm_allreduce_max(double* data);
  void comm_allreduce_array(double* data, size_t size);
  void comm_allreduce_int(int* data);
  void comm_broadcast(void *data, size_t nbytes);
  void comm_barrier(void);
  void comm_abort(int status);

#ifdef __cplusplus
}
#endif

#endif /* _COMM_QUDA_H */
