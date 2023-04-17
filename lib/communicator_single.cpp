/**
 * Dummy communications layer for single-GPU backend.
 */

#include <stdlib.h>
#include <string.h>

#include <communicator_quda.h>

namespace quda
{

  Communicator::Communicator(int nDim, const int *commDims, QudaCommsMap rank_from_coords, void *map_data, bool, void *)
  {
    comm_init(nDim, commDims, rank_from_coords, map_data);
    globalReduce.push(true);
  }

  Communicator::Communicator(Communicator &other, const int *comm_split) : globalReduce(other.globalReduce)
  {
    constexpr int nDim = 4;

    CommKey comm_dims_split;
    CommKey comm_key_split;
    CommKey comm_color_split;

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

  size_t Communicator::comm_size(void) { return 1; }

  void Communicator::comm_gather_hostname(char *hostname_recv_buf) { strncpy(hostname_recv_buf, comm_hostname(), QUDA_MAX_HOSTNAME_STRING); }

  void Communicator::comm_gather_gpuid(int *gpuid_recv_buf) { gpuid_recv_buf[0] = comm_gpuid(); }

  MsgHandle *Communicator::comm_declare_send_rank(void *, int, int, size_t) { return nullptr; }

  MsgHandle *Communicator::comm_declare_recv_rank(void *, int, int, size_t) { return nullptr; }

  MsgHandle *Communicator::comm_declare_send_displaced(void *, const int[], size_t) { return nullptr; }

  MsgHandle *Communicator::comm_declare_receive_displaced(void *, const int[], size_t) { return nullptr; }

  MsgHandle *Communicator::comm_declare_strided_send_displaced(void *, const int[], size_t, int, size_t)
  {
    return nullptr;
  }

  MsgHandle *Communicator::comm_declare_strided_receive_displaced(void *, const int[], size_t, int, size_t)
  {
    return nullptr;
  }

  void Communicator::comm_free(MsgHandle *&) { }

  void Communicator::comm_start(MsgHandle *) { }

  void Communicator::comm_wait(MsgHandle *) { }

  int Communicator::comm_query(MsgHandle *) { return 1; }

  void Communicator::comm_allreduce_sum_array(double *, size_t) { }

  void Communicator::comm_allreduce_max_array(deviation_t<double> *, size_t) { }

  void Communicator::comm_allreduce_max_array(double *, size_t) { }

  void Communicator::comm_allreduce_min_array(double *, size_t) { }

  void Communicator::comm_allreduce_int(int &) { }

  void Communicator::comm_allreduce_xor(uint64_t &) { }

  void Communicator::comm_broadcast(void *, size_t, int) { }

  void Communicator::comm_barrier(void) { }

  void Communicator::comm_abort_(int status) { exit(status); }

  int Communicator::comm_rank_global() { return 0; }

} // namespace quda
