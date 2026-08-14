#pragma once

#include <cstdlib>
#include <cstring>

#include <quda.h>
#include <comm_quda.h>
#include <communicator_quda.h>

#include <malloc_quda.h>
#include <quda_api.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <reference_wrapper_helper.h>

namespace quda
{

  int comm_rank_from_coords(const int *coords);

  /**
    @brief Whether the reshuffle may hand device pointers straight to MPI, skipping the host
    bounce entirely.

    Three conditions, all necessary:

    - `QUDA_ENABLE_SPLIT_GDR` is not explicitly "0". This is a runtime escape hatch rather than a
      feature switch: this is a header, so reverting the transport by rebuilding costs most of the
      library, and being able to flip it inside a running allocation is the point. Enabling GDR is
      also documented to interact with MPI over IPC handle ownership (communicator.cpp, where
      comm_gdr_enabled() bumps enable_peer_to_peer to suppress the conflicting policies) -- if this
      transport ever hangs rather than answering wrongly, that is the first thing to rule out.

    - The field is device resident. A host-resident field has nothing to gain, and its pointer would
      be wrong for an RDMA send.

    - comm_gdr_enabled().

    NOTE ON THE LAST CONDITION. comm_gdr_enabled() is a *user assertion about their MPI* -- it only
    reads QUDA_ENABLE_GDR (communicator_stack.cpp) -- not a capability probe. Setting
    QUDA_ENABLE_GDR=1 without a GPU-aware MPI will fault here exactly as it already does in the
    Dslash GDR policies, which hand a device pointer to this same comm_declare_* family
    (lattice_field.cpp:394). That is deliberate: this inherits QUDA's existing contract rather than
    inventing a second one.
  */
  inline bool split_use_device_comms(QudaFieldLocation location)
  {
    static const bool enabled = []() {
      const char *env = getenv("QUDA_ENABLE_SPLIT_GDR");
      return !(env && strcmp(env, "0") == 0);
    }();
    return enabled && location == QUDA_CUDA_FIELD_LOCATION && comm_gdr_enabled();
  }

  /**
    @brief Allocate one staging buffer for the reshuffle: device memory from QUDA's pooled allocator
    when the transport can RDMA out of it, pinned host memory otherwise.

    pool_device_malloc and NOT device_comms_pinned_malloc. The latter is what the ghost buffers use
    and is the obvious thing to reach for, but its CUDA and HIP implementations differ -- HIP rounds
    up to a 2 MiB boundary and short-circuits when peer-to-peer is absent (hip/malloc.cpp:351,241),
    CUDA does neither -- which would make the split-grid device footprint target-dependent. An RDMA
    send needs a device pointer and nothing more; the comms-pinned variant exists for the NVSHMEM
    symmetric heap and P2P IPC mapping, and split-grid already refuses to run under NVSHMEM
    (communicator_stack.cpp).
  */
  inline void *split_buffer_malloc(size_t bytes, bool device)
  {
    return device ? pool_device_malloc(bytes) : host_pinned_malloc(bytes);
  }

  /**
    @brief Ceiling on the staging footprint of a single reshuffle, in bytes. Override with
    QUDA_SPLIT_MAX_STAGING_MIB (default 256 MiB).

    Pre-posting every receive needs n_replicates buffers live at once, and n_replicates * bytes is
    NOT small for every caller. The colour-spinor reshuffles are fine -- at production that is
    9 x 17.9 MB -- but UpdateSplitGauge splits a gauge field with n_fields == 1, where the base
    field is a few hundred MB per rank, so pre-posting all of it would add several GB to the HBM
    high-water on a configuration that already runs out of device memory at P >= 6. Measured at
    three nodes: +188 MiB, essentially all of it the gauge.

    So the pre-post is conditional, and the fallback is the original algorithm -- post all sends,
    then one receive at a time -- rather than a partially-windowed one. That is deliberate: with a
    sliding window the completion of rank A's window can depend on rank B having advanced to a
    later window, and proving that graph acyclic across every split key is not worth the risk when
    the only caller that trips the ceiling (the gauge) is cached after the first solve and does not
    care about its latency.

    Note this bounds the RECEIVE staging. join_field also holds n_replicates send buffers, one per
    slice, which is unchanged in count from the original -- only their location moved from pinned
    host to device.
  */
  inline size_t split_max_staging_bytes()
  {
    static const size_t cap = []() {
      size_t mib = 256;
      const char *env = getenv("QUDA_SPLIT_MAX_STAGING_MIB");
      if (env) {
        char *end = nullptr;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && v > 0) { mib = v; }
      }
      return mib * 1024ul * 1024ul;
    }();
    return cap;
  }

  /**
    @brief Whether UpdateSplitGauge should return its pool residue to the driver when it finishes a
    rebuild. Default on; set QUDA_FLUSH_POOL_AFTER_SPLIT_GAUGE=0 to disable.

    pool::device_free_ puts a freed block in deviceCache and never calls cudaFree, so when a rebuild
    finishes the cache is holding two things that nothing afterwards has a use for: the split_field
    staging for the two precise link splits, and the whole full-grid tower, which setupGaugeFields
    deletes after GaugeBundleBackup has copied it.

    DEFAULT ON. Turn it off for any workload that rebuilds far more often than it solves -- gauge
    generation driving split multi-src solves would be the case to watch.
  */
  inline bool split_flush_pool_after_gauge()
  {
    static const bool enabled = []() {
      const char *env = getenv("QUDA_FLUSH_POOL_AFTER_SPLIT_GAUGE");
      return !(env && strcmp(env, "0") == 0);
    }();
    return enabled;
  }

  /**
    @brief Whether the split gauge tower may point its eigensolver tier at its own precise field
    instead of building a separate one (gauge_backup.h's alias_eigensolver). DEFAULT OFF -- set
    QUDA_SPLIT_ALIAS_EIG=1 to enable. Enable this if no deflation occurs within a sub-grid to
    skip allocating sub-grid fields for the eigensolver.
  */
  inline bool split_alias_eigensolver()
  {
    static const bool enabled = []() {
      const char *env = getenv("QUDA_SPLIT_ALIAS_EIG");
      return env && strcmp(env, "1") == 0;
    }();
    return enabled;
  }

  inline void split_buffer_free(void *ptr, bool device)
  {
    if (!ptr) return;
    if (device) {
      pool_device_free(ptr);
    } else {
      host_free(ptr);
    }
  }

  /**
    @brief Whether the reshuffle may skip its staging copies where the field's own allocation is
    already what MPI has to send (split_field) or what copyFieldOffset can write into (join_field).
    Default on; set QUDA_SPLIT_ZERO_COPY=0 to disable.
  */
  inline bool split_zero_copy_enabled()
  {
    static const bool enabled = []() {
      const char *env = getenv("QUDA_SPLIT_ZERO_COPY");
      return !(env && strcmp(env, "0") == 0);
    }();
    return enabled;
  }

  /**
    @brief The device pointer MPI can send TotalBytes() from directly, or nullptr when this field
    shape needs a staging copy first.

    split_field used to allocate a staging buffer per distinct source field and fill it with
    copy_to_buffer purely so that comm_declare_send_rank had a pointer. For the shapes below that
    copy is the identity: copy_to_buffer on a device-resident field is a single qudaMemcpy out of
    data() with no reordering, so the field's own allocation already IS the buffer. Sending from it
    is bit-identical, not merely equivalent.
  */
  inline void *split_send_pointer(const GaugeField &f)
  {
    return f.is_pointer_array(f.Order()) ? nullptr : f.data();
  }

  inline void *split_send_pointer(const ColorSpinorField &f) { return f.data(); }

  inline void *split_send_pointer(const CloverField &) { return nullptr; }

  /**
    @brief Whether copyFieldOffset may write straight into a staging buffer through a
    QUDA_REFERENCE_FIELD_CREATE view of it, letting join_field drop its scratch buffer_field and
    the device-to-device copy that drained it.
  */
  inline bool split_can_reference(const ColorSpinorField &) { return true; }

  inline bool split_can_reference(const GaugeField &) { return false; }

  inline bool split_can_reference(const CloverField &) { return false; }

  /**
    @brief Point a field param at an external allocation, so the Field built from it is a view.

    Only the colour-spinor overload is ever reached -- split_can_reference() is the gate, and it is
    false for the other two. The others exist so that the templates stay instantiable for every
    Field type, and they abort rather than silently doing the wrong thing if that gate is ever
    loosened without reading the note above.
  */
  inline void split_param_reference(ColorSpinorParam &param, void *ptr)
  {
    param.v = ptr;
    param.create = QUDA_REFERENCE_FIELD_CREATE;
  }

  inline void split_param_reference(GaugeFieldParam &, void *)
  {
    errorQuda("A reference-created GaugeField exchanges ghosts from its constructor; see split_can_reference()");
  }

  inline void split_param_reference(CloverFieldParam &, void *)
  {
    errorQuda("CloverField::copy_to_buffer concatenates two allocations; see split_can_reference()");
  }

  /**
    @brief Retire everything the reshuffle has enqueued on the default stream. No-op on the host
    transport.

    With host staging, copy_to_buffer was a device-to-host copy and copy_from_buffer a host-to-device
    copy, and cudaMemcpy in either of those directions is synchronous with respect to the host: the
    call did not return until the bytes were there. comm_start could therefore be issued on the next
    line and the data was guaranteed present.

    With device staging both become device-to-device, and cudaMemcpy D2D is explicitly documented as
    NOT host-synchronising -- it is enqueued on the default stream and returns. cuMemcpyDtoD and the
    HIP equivalents behave the same way. So comm_start could hand MPI a buffer that the copy (and, in
    join_field, the copyFieldOffset kernel feeding it) had not yet written, and the RDMA or IPC engine
    would ship whatever was there. Symmetrically on the receive side: a shared receive buffer could be
    handed to the next MPI receive, or returned to the pool, while an unretired D2D read of it was
    still outstanding.

    qudaDeviceSynchronize and not a stream synchronise: every operation involved (qudaMemcpy_ and
    CopyFieldOffset::apply) targets device::get_default_stream(), so the two are equivalent here, and
    the device-wide form stays correct if that ever stops being true. The cost is a sync per
    reshuffle against a call that already ends in a comm_barrier.
  */
  inline void split_sync_device(bool device)
  {
    if (device) { qudaDeviceSynchronize(); }
  }

  template <class Field>
  void inline split_field(Field &collect_field, cvector_ref<Field> &v_base_field, const CommKey &comm_key,
                          QudaPCType pc_type = QUDA_4D_PC)
  {
    CommKey comm_grid_dim = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};
    CommKey comm_grid_idx = {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)};

    int rank = comm_rank();
    int total_rank = product(comm_grid_dim);

    /**
      The term partition in the variable names and comments can mean two things:
      - The processor grid (with dimension comm_grid_dim) is divided into (sub)partitions.
      - For the collecting field, on each processor it contains several partitions, each partition is a copy of
        the base field.
      The term partition_dim means the number of partitions in each direction, and (unsurprisingly) partition_dim
      is the same for the above two meanings, i.e. if I divide the overall processor grid by 3 in one direction,
      the collect field will be 3 times fatter compared to the base field, in that direction.

      In this file the term *_dim and *_idx are all arrays of 4 int's - one can simplify them as 1d-int to understand
      things and the extension to 4d is trivial.
    */

    auto processor_dim = comm_grid_dim / comm_key; // How many processors are there in a processor grid sub-partition?
    auto partition_dim
      = comm_grid_dim / processor_dim; // How many such sub-partitions are there? partition_dim == comm_key

    int n_replicates = product(comm_key);

    int n_fields = v_base_field.size();
    if (n_fields == 0) { errorQuda("split_field: input field vec has zero size."); }

    const auto &meta = v_base_field[0];
    const size_t bytes = meta.TotalBytes();
    const bool device = split_use_device_comms(meta.Location());

    using param_type = typename Field::param_type;
    param_type param(meta);

    // Unpack straight out of the receive buffer where the Field type allows a view of it.
    const bool unpack_in_place = device && split_zero_copy_enabled() && split_can_reference(meta);
    Field buffer_field = unpack_in_place ? Field() : Field(param);

    CommKey field_dim = {meta.full_dim(0), meta.full_dim(1), meta.full_dim(2), meta.full_dim(3)};

    // Post EVERY receive before any send. Declaring a receive only after the previous one had
    // completed -- as this used to -- serialises the whole exchange: under a rendezvous protocol the
    // sender cannot move data until its match is posted, so each message waited on the previous
    // message's transfer, its copy and its kernel. Nothing here is ordered, so post it all and let
    // the network overlap it.
    //
    // Unless that would cost too much memory, in which case fall back to the original
    // one-at-a-time receive -- see split_max_staging_bytes().
    const bool prepost = static_cast<size_t>(n_replicates) * bytes <= split_max_staging_bytes();
    const int n_recv_buffers = prepost ? n_replicates : 1;

    std::vector<void *> v_recv_buffer(n_recv_buffers, nullptr);
    std::vector<MsgHandle *> v_mh_recv(n_replicates, nullptr);

    // Where does replicate i's data come from? Needed both to pre-post and to post late.
    auto recv_peer = [&](int i) {
      auto partition_idx
        = coordinate_from_index(i, comm_key); // Here this means which partition of the field we are working on.
      auto src_idx
        = (comm_grid_idx % processor_dim) * partition_dim + partition_idx; // And where does this partition comes from?
      return comm_rank_from_coords(src_idx.data());
    };

    for (int i = 0; i < n_recv_buffers; i++) { v_recv_buffer[i] = split_buffer_malloc(bytes, device); }

    // Retire the stream BEFORE arming any receive, not just before the sends.
    split_sync_device(device);

    if (prepost) {
      for (int i = 0; i < n_replicates; i++) {
        int src_rank = recv_peer(i);
        int tag = src_rank * total_rank + rank;
        v_mh_recv[i] = comm_declare_recv_rank(v_recv_buffer[i], src_rank, tag, bytes);
        comm_start(v_mh_recv[i]);
      }
    }

    // One staging buffer per DISTINCT source field, not one per replicate.
    const int n_send_buffers = n_fields < n_replicates ? n_fields : n_replicates;

    // ... and no staging buffer at all where the field can be sent from its own pointer.
    std::vector<void *> v_send_buffer(n_send_buffers, nullptr);
    std::vector<void *> v_send_from(n_send_buffers, nullptr);
    std::vector<MsgHandle *> v_mh_send(n_replicates, nullptr);

    for (int i = 0; i < n_send_buffers; i++) {
      // Only on the device transport. With pinned host staging copy_to_buffer is a real
      // device-to-host copy and the field pointer would be the wrong address space, so
      // QUDA_ENABLE_SPLIT_GDR=0 keeps exactly the behaviour it had.
      v_send_from[i]
        = (device && split_zero_copy_enabled()) ? split_send_pointer(v_base_field[i]) : nullptr;
      if (!v_send_from[i]) {
        v_send_buffer[i] = split_buffer_malloc(bytes, device);
        v_base_field[i].copy_to_buffer(v_send_buffer[i]);
        v_send_from[i] = v_send_buffer[i];
      }
    }

    // The send buffers are only filled once the stream drains.
    split_sync_device(device);

    for (int i = 0; i < n_replicates; i++) {
      auto partition_idx = coordinate_from_index(i, comm_key); // Which partition to send to?
      auto processor_idx = comm_grid_idx / partition_dim;      // Which processor in that partition to send to?

      auto dst_idx = partition_idx * processor_dim + processor_idx;

      int dst_rank = ::quda::comm_rank_from_coords(dst_idx.data());
      int tag = rank * total_rank + dst_rank; // tag = src_rank * total_rank + dst_rank

      v_mh_send[i] = comm_declare_send_rank(v_send_from[i % n_fields], dst_rank, tag, bytes);
      comm_start(v_mh_send[i]);
    }

    // Unpack in replicate order. Each copyFieldOffset writes a disjoint region of collect_field
    // (offset = partition_idx * field_dim), so completion order would not change the result -- but
    // wait-any is deferred, along with the guard it needs for the n_fields < n_replicates case.
    for (int i = 0; i < n_replicates; i++) {
      const int b = prepost ? i : 0;

      if (!prepost) { // post it now, into the single shared buffer
        int src_rank = recv_peer(i);
        int tag = src_rank * total_rank + rank;
        v_mh_recv[i] = comm_declare_recv_rank(v_recv_buffer[b], src_rank, tag, bytes);
        comm_start(v_mh_recv[i]);
      }

      comm_wait(v_mh_recv[i]);

      auto partition_idx = coordinate_from_index(i, comm_key);
      auto offset = partition_idx * field_dim;

      if (unpack_in_place) {
        param_type recv_param(param);
        split_param_reference(recv_param, v_recv_buffer[b]);
        Field recv_field(recv_param);
        quda::copyFieldOffset(collect_field, recv_field, offset, pc_type);
      } else {
        buffer_field.copy_from_buffer(v_recv_buffer[b]);
        quda::copyFieldOffset(collect_field, buffer_field, offset, pc_type);
      }

      // Without a pre-post every replicate lands in the SAME receive buffer, so the next iteration's
      // MPI receive would overwrite it while this iteration's read of it is still queued -- the
      // copy_from_buffer, or, in place, the copyFieldOffset kernel reading the view. Retire it here.
      // (With a pre-post each buffer is written once and read once, and the sync below suffices.)
      if (!prepost) { split_sync_device(device); }
    }

    // Retire the reads out of the receive buffers before pool_device_free hands them to anyone else.
    split_sync_device(device);

    // The sends are never waited on, so this barrier is load-bearing: it is the proxy for "every
    // rank has drained its receives, therefore my sends have delivered, therefore the buffers and
    // handles are safe to release". Removing it requires waiting on the sends first. With
    // pool_device_free the buffer re-enters circulation immediately, so dropping the barrier
    // without that wait would let the allocator hand an in-flight send buffer to someone else.
    // It is what covers the zero-copy sends too: it is the point after which the caller may write
    // v_base_field again, because until it returns MPI may still be reading out of it.
    comm_barrier();

    for (int i = 0; i < n_replicates; i++) {
      if (v_mh_recv[i]) { comm_free(v_mh_recv[i]); }
      if (v_mh_send[i]) { comm_free(v_mh_send[i]); }
    }
    for (auto &p : v_recv_buffer) { split_buffer_free(p, device); }
    for (auto &p : v_send_buffer) { split_buffer_free(p, device); }
  }

  template <class Field>
  void inline join_field(cvector_ref<Field> &v_base_field, const Field &collect_field, const CommKey &comm_key,
                         QudaPCType pc_type = QUDA_4D_PC)
  {
    CommKey comm_grid_dim = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};
    CommKey comm_grid_idx = {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)};

    int rank = comm_rank();
    int total_rank = product(comm_grid_dim);

    auto processor_dim = comm_grid_dim / comm_key; // Communicator grid.
    auto partition_dim
      = comm_grid_dim / processor_dim; // The full field needs to be partitioned according to the communicator grid.

    int n_replicates = product(comm_key);

    int n_fields = v_base_field.size();
    if (n_fields == 0) { errorQuda("join_field: output field vec has zero size."); }

    const auto &meta = v_base_field[0];
    const size_t bytes = meta.TotalBytes();
    const bool device = split_use_device_comms(meta.Location());

    using param_type = typename Field::param_type;

    param_type param(meta);

    // Pack straight into the send buffer where the Field type allows a view of it.
    const bool pack_in_place = device && split_zero_copy_enabled() && split_can_reference(meta);
    Field buffer_field = pack_in_place ? Field() : Field(param);

    CommKey field_dim = {meta.full_dim(0), meta.full_dim(1), meta.full_dim(2), meta.full_dim(3)};

    // Post every receive first, memory permitting -- see the note in split_field.
    const bool prepost = static_cast<size_t>(n_replicates) * bytes <= split_max_staging_bytes();
    const int n_recv_buffers = prepost ? n_replicates : 1;

    std::vector<void *> v_recv_buffer(n_recv_buffers, nullptr);
    std::vector<MsgHandle *> v_mh_recv(n_replicates, nullptr);

    auto recv_peer = [&](int i) {
      auto partition_idx = coordinate_from_index(i, comm_key);
      auto processor_idx = comm_grid_idx / partition_dim;
      auto src_idx = partition_idx * processor_dim + processor_idx;
      return comm_rank_from_coords(src_idx.data());
    };

    for (int i = 0; i < n_recv_buffers; i++) { v_recv_buffer[i] = split_buffer_malloc(bytes, device); }

    // Retire the stream before arming any receive.
    split_sync_device(device);

    if (prepost) {
      for (int i = 0; i < n_replicates; i++) {
        int src_rank = recv_peer(i);
        int tag = src_rank * total_rank + rank;
        v_mh_recv[i] = comm_declare_recv_rank(v_recv_buffer[i], src_rank, tag, bytes);
        comm_start(v_mh_recv[i]);
      }
    }

    // Unlike split_field there is no buffer to share: every replicate carries a different slice of
    // collect_field, so each needs its own.
    std::vector<void *> v_send_buffer(n_replicates, nullptr);
    std::vector<MsgHandle *> v_mh_send(n_replicates, nullptr);

    // Two loops, not one. Every slice is packed first, then the stream is drained, and only then is
    // anything handed to MPI. Interleaving comm_start with the packing -- as this used to -- would
    // hand MPI a buffer whose copyFieldOffset kernel and device-to-device copy were still queued;
    // see split_sync_device().
    for (int i = 0; i < n_replicates; i++) {
      auto partition_idx = coordinate_from_index(i, comm_key);
      auto offset = partition_idx * field_dim;

      v_send_buffer[i] = split_buffer_malloc(bytes, device);

      if (pack_in_place) {
        param_type send_param(param);
        split_param_reference(send_param, v_send_buffer[i]);
        Field send_field(send_param);
        // copyFieldOffset writes the body only, and a reference-created field skips the zeroPad
        // its allocating counterpart gets.
        send_field.zeroPad();
        quda::copyFieldOffset(send_field, collect_field, offset, pc_type);
      } else {
        quda::copyFieldOffset(buffer_field, collect_field, offset, pc_type);
        buffer_field.copy_to_buffer(v_send_buffer[i]);
      }
    }

    split_sync_device(device);

    for (int i = 0; i < n_replicates; i++) {
      auto partition_idx = coordinate_from_index(i, comm_key);
      auto dst_idx = (comm_grid_idx % processor_dim) * partition_dim + partition_idx;

      int dst_rank = comm_rank_from_coords(dst_idx.data());
      int tag = rank * total_rank + dst_rank;

      v_mh_send[i] = comm_declare_send_rank(v_send_buffer[i], dst_rank, tag, bytes);

      comm_start(v_mh_send[i]);
    }

    // Replicate order is preserved deliberately: when n_fields < n_replicates several replicates
    // land in the same base field and the last write wins, so completion order would be observable.
    for (int i = 0; i < n_replicates; i++) {
      const int b = prepost ? i : 0;

      if (!prepost) {
        int src_rank = recv_peer(i);
        int tag = src_rank * total_rank + rank;
        v_mh_recv[i] = comm_declare_recv_rank(v_recv_buffer[b], src_rank, tag, bytes);
        comm_start(v_mh_recv[i]);
      }

      comm_wait(v_mh_recv[i]);
      v_base_field[i % n_fields].copy_from_buffer(v_recv_buffer[b]);

      // Shared receive buffer -- see the note in split_field.
      if (!prepost) { split_sync_device(device); }
    }

    // Retire the reads out of the receive buffers before they go back to the pool.
    split_sync_device(device);

    // Load-bearing -- see the note in split_field.
    comm_barrier();

    for (int i = 0; i < n_replicates; i++) {
      if (v_mh_recv[i]) { comm_free(v_mh_recv[i]); }
      if (v_mh_send[i]) { comm_free(v_mh_send[i]); }
    }
    for (auto &p : v_recv_buffer) { split_buffer_free(p, device); }
    for (auto &p : v_send_buffer) { split_buffer_free(p, device); }
  }

} // namespace quda
