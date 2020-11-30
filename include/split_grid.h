#pragma once

#include <quda.h>
#include <comm_quda.h>
#include <communicator_quda.h>

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>

int comm_rank_from_coords(const int *coords);

namespace quda
{

  template <class Field>
  void inline split_field(Field &collect_field, std::vector<Field *> &v_base_field, const CommKey &comm_key,
                          QudaPCType pc_type = QUDA_4D_PC)
  {
    CommKey full_dim = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};
    CommKey full_idx = {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)};

    int rank = comm_rank();
    int total_rank = product(full_dim);

    auto grid_dim = full_dim / comm_key;  // Communicator grid.
    auto block_dim = full_dim / grid_dim; // The full field needs to be partitioned according to the communicator grid.

    int n_replicates = product(comm_key);
    std::vector<void *> v_send_buffer_h(n_replicates, nullptr);
    std::vector<MsgHandle *> v_mh_send(n_replicates, nullptr);

    int n_fields = v_base_field.size();
    if (n_fields == 0) { errorQuda("Empty vector!"); }

    const auto meta = v_base_field[0];

    // Send cycles
    for (int i = 0; i < n_replicates; i++) {
      auto grid_idx = coordinate_from_index(i, comm_key);
      auto block_idx = full_idx / block_dim;

      auto dst_idx = grid_idx * grid_dim + block_idx;

      int dst_rank = comm_rank_from_coords(dst_idx.data());
      int tag = rank * total_rank + dst_rank; // tag = src_rank * total_rank + dst_rank

      size_t bytes = meta->TotalBytes();

      v_send_buffer_h[i] = pinned_malloc(bytes);

      v_base_field[i % n_fields]->copy_to_buffer(v_send_buffer_h[i]);

      v_mh_send[i] = comm_declare_send_rank(v_send_buffer_h[i], dst_rank, tag, bytes);
      comm_start(v_mh_send[i]);
    }

    using param_type = typename Field::param_type;

    param_type param(*meta);
    Field *buffer_field = Field::Create(param);

    CommKey thread_dim = {meta->full_dim(0), meta->full_dim(1), meta->full_dim(2), meta->full_dim(3)};

    // Receive cycles
    for (int i = 0; i < n_replicates; i++) {
      auto thread_idx = coordinate_from_index(i, comm_key);
      auto src_idx = (full_idx % grid_dim) * block_dim + thread_idx;

      int src_rank = comm_rank_from_coords(src_idx.data());
      int tag = src_rank * total_rank + rank;

      size_t bytes = buffer_field->TotalBytes();

      void *recv_buffer_h = pinned_malloc(bytes);

      auto mh_recv = comm_declare_recv_rank(recv_buffer_h, src_rank, tag, bytes);

      comm_start(mh_recv);
      comm_wait(mh_recv);

      buffer_field->copy_from_buffer(recv_buffer_h);

      comm_free(mh_recv);
      host_free(recv_buffer_h);

      auto offset = thread_idx * thread_dim;

      quda::copyFieldOffset(collect_field, *buffer_field, offset, pc_type);
    }

    delete buffer_field;

    comm_barrier();

    for (auto &p : v_send_buffer_h) {
      if (p) { host_free(p); }
    };
    for (auto &p : v_mh_send) {
      if (p) { comm_free(p); }
    };
  }

  template <class Field>
  void inline join_field(std::vector<Field *> &v_base_field, const Field &collect_field, const CommKey &comm_key,
                         QudaPCType pc_type = QUDA_4D_PC)
  {
    CommKey full_dim = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};
    CommKey full_idx = {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)};

    int rank = comm_rank();
    int total_rank = product(full_dim);

    auto grid_dim = full_dim / comm_key;  // Communicator grid.
    auto block_dim = full_dim / grid_dim; // The full field needs to be partitioned according to the communicator grid.

    int n_replicates = product(comm_key);
    std::vector<void *> v_send_buffer_h(n_replicates, nullptr);
    std::vector<MsgHandle *> v_mh_send(n_replicates, nullptr);

    int n_fields = v_base_field.size();
    if (n_fields == 0) { errorQuda("Empty vector!"); }

    const auto &meta = *(v_base_field[0]);

    using param_type = typename Field::param_type;

    param_type param(meta);
    Field *buffer_field = Field::Create(param);

    CommKey thread_dim = {meta.full_dim(0), meta.full_dim(1), meta.full_dim(2), meta.full_dim(3)};

    // Send cycles
    for (int i = 0; i < n_replicates; i++) {

      auto thread_idx = coordinate_from_index(i, comm_key);
      auto dst_idx = (full_idx % grid_dim) * block_dim + thread_idx;

      int dst_rank = comm_rank_from_coords(dst_idx.data());
      int tag = rank * total_rank + dst_rank;

      size_t bytes = meta.TotalBytes();

      auto offset = thread_idx * thread_dim;
      quda::copyFieldOffset(*buffer_field, collect_field, offset, pc_type);

      v_send_buffer_h[i] = pinned_malloc(bytes);
      buffer_field->copy_to_buffer(v_send_buffer_h[i]);

      v_mh_send[i] = comm_declare_send_rank(v_send_buffer_h[i], dst_rank, tag, bytes);

      comm_start(v_mh_send[i]);
    }

    // Receive cycles
    for (int i = 0; i < n_replicates; i++) {

      auto grid_idx = coordinate_from_index(i, comm_key);
      auto block_idx = full_idx / block_dim;

      auto src_idx = grid_idx * grid_dim + block_idx;

      int src_rank = comm_rank_from_coords(src_idx.data());
      int tag = src_rank * total_rank + rank;

      size_t bytes = buffer_field->TotalBytes();

      void *recv_buffer_h = pinned_malloc(bytes);

      auto mh_recv = comm_declare_recv_rank(recv_buffer_h, src_rank, tag, bytes);

      comm_start(mh_recv);
      comm_wait(mh_recv);

      v_base_field[i % n_fields]->copy_from_buffer(recv_buffer_h);

      comm_free(mh_recv);
      host_free(recv_buffer_h);
    }

    delete buffer_field;

    comm_barrier();

    for (auto &p : v_send_buffer_h) { host_free(p); };
    for (auto &p : v_mh_send) { comm_free(p); };
  }

} // namespace quda
