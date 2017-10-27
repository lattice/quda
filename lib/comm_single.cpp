/**
 * Dummy communications layer for single-GPU backend.
 */

#include <stdlib.h>
#include <string.h>
#include <csignal>
#include <comm_quda.h>

static char partition_string[16] = ",comm=0000";
static char topology_string[16] = ",topo=1111";

void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);
}

int comm_rank(void) { return 0; }

int comm_size(void) { return 1; }

int comm_gpuid(void) { return 0; }

void comm_gather_hostname(char *hostname_recv_buf) {
  strncpy(hostname_recv_buf, comm_hostname(), 128);
}

void comm_gather_gpuid(int *gpuid_recv_buf) {
  gpuid_recv_buf[0] = comm_gpuid();
}

MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{ return NULL; }

MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{ return NULL; }

MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[],
					       size_t blksize, int nblocks, size_t stride)
{ return NULL; }

MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{ return NULL; }

void comm_free(MsgHandle *mh) {}

void comm_start(MsgHandle *mh) {}

void comm_wait(MsgHandle *mh) {}

int comm_query(MsgHandle *mh) { return 1; }

void comm_allreduce(double* data) {}

void comm_allreduce_max(double* data) {}

void comm_allreduce_array(double* data, size_t size) {}

void comm_allreduce_int(int* data) {}

void comm_allreduce_xor(uint64_t *data) {}

void comm_broadcast(void *data, size_t nbytes) {}

void comm_barrier(void) {}

void comm_abort(int status) {
#ifdef HOST_DEBUG
  raise(SIGINT);
#endif
  exit(status);
}

const char* comm_dim_partitioned_string() {
  return partition_string;
}

const char* comm_dim_topology_string() {
  return topology_string;
}
