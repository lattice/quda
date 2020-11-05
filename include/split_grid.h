#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h> // convenient quark field container

#include <communicator_quda.h>

#include <gauge_field.h>
#include <color_spinor_field.h>

int comm_rank_from_coords(const int *coords);

namespace quda
{

  void inline copyOffsetField(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    copyOffsetGauge(out, in, offset);
  }

  void inline copyOffsetField(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    copyOffsetColorSpinor(out, in, offset);
  }

  int inline product(const CommKey &input) { return input[0] * input[1] * input[2] * input[3]; }

  CommKey inline operator+(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey sum;
    for (int d = 0; d < nDim; d++) { sum[d] = lhs[d] + rhs[d]; }
    return sum;
  }

  CommKey inline operator*(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey product;
    for (int d = 0; d < nDim; d++) { product[d] = lhs[d] * rhs[d]; }
    return product;
  }

  CommKey inline operator/(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey quotient;
    for (int d = 0; d < nDim; d++) { quotient[d] = lhs[d] / rhs[d]; }
    return quotient;
  }

  CommKey inline operator%(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey mod;
    for (int d = 0; d < nDim; d++) { mod[d] = lhs[d] % rhs[d]; }
    return mod;
  }

  CommKey inline coordinate_from_index(int index, CommKey dim)
  {
    CommKey coord;
    for (int d = 0; d < nDim; d++) {
      coord[d] = index % dim[d];
      index /= dim[d];
    }
    return coord;
  }

  int inline index_from_coordinate(CommKey coord, CommKey dim)
  {
    return ((coord[3] * dim[2] + coord[2]) * dim[1] + coord[1]) * dim[0] + coord[0];
  }

  void inline print(const CommKey &comm_key)
  {
    printf("(%3d,%3d,%3d,%3d)", comm_key[0], comm_key[1], comm_key[2], comm_key[3]);
  }

  auto inline get_data(GaugeField *f) { return f->Gauge_p(); }

  auto inline get_data(ColorSpinorField *f) { return f->V(); }

  /**
    Here we implement the field to/from buffer routines depending on whether the field
    is a gauge field or a colorspinor field.
  */

  void inline cpu_field_to_buffer(void *buffer, GaugeField &f)
  {
    if (f.Order() == QUDA_QDP_GAUGE_ORDER || f.Order() == QUDA_QDPJIT_GAUGE_ORDER) {
      void **p = static_cast<void **>(f.Gauge_p());
      int dbytes = f.Bytes() / 4;
      static_assert(sizeof(char) == 1, "Assuming sizeof(char) == 1");
      char *dst_buffer = reinterpret_cast<char *>(buffer);
      for (int d = 0; d < 4; d++) { std::memcpy(&dst_buffer[d * dbytes], p[d], dbytes); }
    } else if (f.Order() == QUDA_CPS_WILSON_GAUGE_ORDER || f.Order() == QUDA_MILC_GAUGE_ORDER
               || f.Order() == QUDA_MILC_SITE_GAUGE_ORDER || f.Order() == QUDA_BQCD_GAUGE_ORDER
               || f.Order() == QUDA_TIFR_GAUGE_ORDER || f.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      void *p = f.Gauge_p();
      int bytes = f.Bytes();
      std::memcpy(buffer, p, bytes);
    } else {
      errorQuda("Unsupported order = %d\n", f.Order());
    }
  }

  void inline cpu_buffer_to_field(GaugeField &f, const void *buffer)
  {
    if (f.Order() == QUDA_QDP_GAUGE_ORDER || f.Order() == QUDA_QDPJIT_GAUGE_ORDER) {
      void **p = static_cast<void **>(f.Gauge_p());
      int dbytes = f.Bytes() / 4;
      static_assert(sizeof(char) == 1, "Assuming sizeof(char) == 1");
      const char *dst_buffer = reinterpret_cast<const char *>(buffer);
      for (int d = 0; d < 4; d++) { std::memcpy(p[d], &dst_buffer[d * dbytes], dbytes); }
    } else if (f.Order() == QUDA_CPS_WILSON_GAUGE_ORDER || f.Order() == QUDA_MILC_GAUGE_ORDER
               || f.Order() == QUDA_MILC_SITE_GAUGE_ORDER || f.Order() == QUDA_BQCD_GAUGE_ORDER
               || f.Order() == QUDA_TIFR_GAUGE_ORDER || f.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      void *p = f.Gauge_p();
      int bytes = f.Bytes();
      std::memcpy(p, buffer, bytes);
    } else {
      errorQuda("Unsupported order = %d\n", f.Order());
    }
  }

  void inline cpu_field_to_buffer(void *buffer, const ColorSpinorField &f)
  {
    int bytes = f.Bytes();
    std::memcpy(buffer, f.V(), bytes);
  }

  void inline cpu_buffer_to_field(ColorSpinorField &f, const void *buffer)
  {
    int bytes = f.Bytes();
    std::memcpy(f.V(), buffer, bytes);
  }

  template <class F> struct param_mapper {
  };

  template <> struct param_mapper<GaugeField> {
    using type = GaugeFieldParam;
  };

  template <> struct param_mapper<ColorSpinorField> {
    using type = ColorSpinorParam;
  };

  template <class Field>
  void inline split_field(Field &collect_field, std::vector<Field *> &v_base_field, const CommKey &comm_key)
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

      // TODO: For now only the non-GDR communication is implemented.
      // THIS IS A COMMENT: printf("rank %4d -> rank %4d: tag = %4d\n", comm_rank(), dst_rank, tag);

      size_t bytes = meta->Bytes();

      v_send_buffer_h[i] = pinned_malloc(bytes);
      if (meta->Location() == QUDA_CPU_FIELD_LOCATION) {
        cpu_field_to_buffer(v_send_buffer_h[i], *v_base_field[i % n_fields]);
      } else {
        qudaMemcpy(v_send_buffer_h[i], get_data(v_base_field[i % n_fields]), bytes, cudaMemcpyDeviceToHost);
      }
      v_mh_send[i] = comm_declare_send_rank(v_send_buffer_h[i], dst_rank, tag, bytes);

      comm_start(v_mh_send[i]);
    }

    using param_type = typename param_mapper<Field>::type;

    param_type param(*meta);
    Field *buffer_field = Field::Create(param);

    const int *X = meta->X();
    CommKey thread_dim = {X[0], X[1], X[2], X[3]};

    // Receive cycles
    for (int i = 0; i < n_replicates; i++) {
      auto thread_idx = coordinate_from_index(i, comm_key);
      auto src_idx = (full_idx % grid_dim) * block_dim + thread_idx;

      int src_rank = comm_rank_from_coords(src_idx.data());
      int tag = src_rank * total_rank + rank;

      // TODO: For now only the cpu-gpu communication is implemented.
      // THIS IS A COMMENT: printf("rank %4d <- rank %4d: tag = %4d\n", comm_rank(), src_rank, tag);

      size_t bytes = buffer_field->Bytes();

      void *recv_buffer_h = pinned_malloc(bytes);

      auto mh_recv = comm_declare_recv_rank(recv_buffer_h, src_rank, tag, bytes);

      comm_start(mh_recv);
      comm_wait(mh_recv);

      if (meta->Location() == QUDA_CPU_FIELD_LOCATION) {
        cpu_buffer_to_field(*buffer_field, recv_buffer_h);
      } else {
        qudaMemcpy(get_data(buffer_field), recv_buffer_h, bytes, cudaMemcpyHostToDevice);
      }

      comm_free(mh_recv);
      host_free(recv_buffer_h);

      auto offset = thread_idx * thread_dim;

      quda::copyOffsetField(collect_field, *buffer_field, offset.data());
    }

    delete buffer_field;

    comm_barrier();

    for (auto &p : v_send_buffer_h) { host_free(p); };
    for (auto &p : v_mh_send) { comm_free(p); };
  }

  template <class Field>
  void inline join_field(std::vector<Field *> &v_base_field, const Field &collect_field, const CommKey &comm_key)
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

    using param_type = typename param_mapper<Field>::type;

    param_type param(meta);
    Field *buffer_field = Field::Create(param);

    const int *X = meta.X();
    CommKey thread_dim = {X[0], X[1], X[2], X[3]};

    // Send cycles
    for (int i = 0; i < n_replicates; i++) {

      auto thread_idx = coordinate_from_index(i, comm_key);
      auto dst_idx = (full_idx % grid_dim) * block_dim + thread_idx;

      int dst_rank = comm_rank_from_coords(dst_idx.data());
      int tag = rank * total_rank + dst_rank;

      // TODO: For now only the cpu-gpu communication is implemented.
      // THIS IS A COMMENT: printf("rank %4d -> rank %4d: tag = %4d\n", comm_rank(), dst_rank, tag);

      size_t bytes = meta.Bytes();

      auto offset = thread_idx * thread_dim;
      quda::copyOffsetField(*buffer_field, collect_field, offset.data());

      v_send_buffer_h[i] = pinned_malloc(bytes);
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        cpu_field_to_buffer(v_send_buffer_h[i], *buffer_field);
      } else {
        qudaMemcpy(v_send_buffer_h[i], get_data(buffer_field), bytes, cudaMemcpyDeviceToHost);
      }
      v_mh_send[i] = comm_declare_send_rank(v_send_buffer_h[i], dst_rank, tag, bytes);

      comm_start(v_mh_send[i]);
    }

    // Receive cycles
    for (int i = 0; i < n_replicates; i++) {

      auto grid_idx = coordinate_from_index(i, comm_key);
      auto block_idx = full_idx / block_dim;

      auto src_idx = grid_idx * grid_dim + block_idx;

      int src_rank = comm_rank_from_coords(src_idx.data());
      int tag = src_rank * total_rank + rank; // tag = src_rank * total_rank + dst_rank

      // TODO: For now only the cpu-gpu communication is implemented.
      // THIS IS A COMMENT: printf("rank %4d <- rank %4d: tag = %4d\n", comm_rank(), src_rank, tag);

      size_t bytes = buffer_field->Bytes();

      void *recv_buffer_h = pinned_malloc(bytes);

      auto mh_recv = comm_declare_recv_rank(recv_buffer_h, src_rank, tag, bytes);

      comm_start(mh_recv);
      comm_wait(mh_recv);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        cpu_buffer_to_field(*v_base_field[i % n_fields], recv_buffer_h);
      } else {
        qudaMemcpy(get_data(v_base_field[i % n_fields]), recv_buffer_h, bytes, cudaMemcpyHostToDevice);
      }

      comm_free(mh_recv);
      host_free(recv_buffer_h);
    }

    delete buffer_field;

    comm_barrier();

    for (auto &p : v_send_buffer_h) { host_free(p); };
    for (auto &p : v_mh_send) { comm_free(p); };
  }

} // namespace quda
