/**
 * Dummy communications layer for single-GPU backend.
 */

#include <stdlib.h>
#include <string.h>
#include <comm_quda.h>

void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  comm_init_common(ndim, dims, rank_from_coords, map_data);
}

int comm_rank(void) { return 0; }

size_t comm_size(void) { return 1; }

void comm_gather_hostname(char *hostname_recv_buf) {
  strncpy(hostname_recv_buf, comm_hostname(), 128);
}

void comm_gather_gpuid(int *gpuid_recv_buf) {
  gpuid_recv_buf[0] = comm_gpuid();
}

MsgHandle *comm_declare_send_displaced(void *, const int [], size_t)
{ return NULL; }

MsgHandle *comm_declare_receive_displaced(void *, const int [], size_t)
{ return NULL; }

MsgHandle *comm_declare_strided_send_displaced(void *, const int [], size_t, int, size_t)
{ return NULL; }

MsgHandle *comm_declare_strided_receive_displaced(void *, const int [], size_t, int, size_t)
{ return NULL; }

void comm_free(MsgHandle *&) {}

void comm_start(MsgHandle *) {}

void comm_wait(MsgHandle *) {}

int comm_query(MsgHandle *) { return 1; }

void comm_allreduce(double*) {}

void comm_allreduce_max(double*) {}

void comm_allreduce_min(double*) {}

void comm_allreduce_array(double*, size_t) {}

void comm_allreduce_max_array(double*, size_t) {}

void comm_allreduce_int(int*) {}

void comm_allreduce_xor(uint64_t *) {}

void comm_broadcast(void *, size_t) {}

void comm_barrier(void) {}

void comm_abort_(int status) {
  exit(status);
}
