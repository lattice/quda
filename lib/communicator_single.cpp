/**
 * Dummy communications layer for single-GPU backend.
 */

#include <stdlib.h>
#include <string.h>

#include <communicator_quda.h>

Communicator::Communicator(int nDim, const int *commDims, QudaCommsMap rank_from_coords, void *map_data,
                           bool user_set_comm_handle, void *user_comm)
{
  comm_init(nDim, commDims, rank_from_coords, map_data);
}

Communicator::Communicator(Communicator &other, const int *comm_split)
{
  constexpr int nDim = 4;

  quda::CommKey comm_dims_split;

  quda::CommKey comm_key_split;
  quda::CommKey comm_color_split;

  for (int d = 0; d < nDim; d++) {
    assert(other.comm_dim(d) % comm_split[d] == 0);
    comm_dims_split[d] = other.comm_dim(d) / comm_split[d];
    comm_key_split[d] = other.comm_coord(d) % comm_dims_split[d];
    comm_color_split[d] = other.comm_coord(d) / comm_dims_split[d];
  }

  QudaCommsMap func = lex_rank_from_coords_dim_t;
  comm_init(nDim, comm_dims_split.data(), func, comm_dims_split.data());

  printf("Creating a split communicator for a single build, which doesn't really make sense.\n");
}

Communicator::~Communicator() { comm_finalize(); }

void Communicator::comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  for (int d = 0; d < ndim; d++) {
    if (dims[d] > 1) errorQuda("Grid dimension grid[%d] = %d greater than 1", d, dims[d]);
  }
  comm_init_common(ndim, dims, rank_from_coords, map_data);
}

int Communicator::comm_rank(void) { return 0; }

int Communicator::comm_size(void) { return 1; }

void Communicator::comm_gather_hostname(char *hostname_recv_buf) { strncpy(hostname_recv_buf, comm_hostname(), 128); }

void Communicator::comm_gather_gpuid(int *gpuid_recv_buf) { gpuid_recv_buf[0] = comm_gpuid(); }

MsgHandle *Communicator::comm_declare_send_rank(void *buffer, int rank, int tag, size_t nbytes) { return nullptr; }

MsgHandle *Communicator::comm_declare_recv_rank(void *buffer, int rank, int tag, size_t nbytes) { return nullptr; }

MsgHandle *Communicator::comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  return nullptr;
}

MsgHandle *Communicator::comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  return nullptr;
}

MsgHandle *Communicator::comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize,
                                                             int nblocks, size_t stride)
{
  return nullptr;
}

MsgHandle *Communicator::comm_declare_strided_receive_displaced(void *buffer, const int displacement[], size_t blksize,
                                                                int nblocks, size_t stride)
{
  return nullptr;
}

void Communicator::comm_free(MsgHandle *&mh) {}

void Communicator::comm_start(MsgHandle *mh) {}

void Communicator::comm_wait(MsgHandle *mh) {}

int Communicator::comm_query(MsgHandle *mh) { return 1; }

void Communicator::comm_allreduce(double *data) {}

void Communicator::comm_allreduce_max(double *data) {}

void Communicator::comm_allreduce_min(double *data) {}

void Communicator::comm_allreduce_array(double *data, size_t size) {}

void Communicator::comm_allreduce_max_array(double *data, size_t size) {}

void Communicator::comm_allreduce_int(int *data) {}

void Communicator::comm_allreduce_xor(uint64_t *data) {}

void Communicator::comm_broadcast(void *data, size_t nbytes) {}

void Communicator::comm_barrier(void) {}

void Communicator::comm_abort_(int status) { exit(status); }

int Communicator::comm_rank_global() { return 0; }
