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
  int comm_dim(int dim);
  int comm_coord(int dim);
  MsgHandle *comm_declare_send_relative(void *buffer, int dim, int dir, size_t nbytes);
  MsgHandle *comm_declare_receive_relative(void *buffer, int dim, int dir, size_t nbytes);
  void comm_finalize(void);
  void comm_dim_partitioned_set(int dim);
  int comm_dim_partitioned(int dim);


  /* implemented in comm_single.cpp, comm_qmp.cpp, and comm_mpi.cpp */

  void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data);
  int comm_rank(void);
  int comm_size(void);
  int comm_gpuid(void);
  MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes);
  MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes);
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
