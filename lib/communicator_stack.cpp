#include <communicator_quda.h>
#include <map>
#include <array.h>
#include <lattice_field.h>

namespace quda
{

  int Communicator::gpuid = -1;

  static std::map<CommKey, Communicator> communicator_stack;

  static CommKey current_key = {-1, -1, -1, -1};

  void init_communicator_stack(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data,
                               bool user_set_comm_handle, void *user_comm)
  {
    communicator_stack.emplace(
      std::piecewise_construct, std::forward_as_tuple(default_comm_key),
      std::forward_as_tuple(ndim, dims, rank_from_coords, map_data, user_set_comm_handle, user_comm));

    current_key = default_comm_key;
  }

  void finalize_communicator_stack() { communicator_stack.clear(); }

  static Communicator &get_default_communicator()
  {
    auto search = communicator_stack.find(default_comm_key);
    if (search == communicator_stack.end()) { errorQuda("Default communicator can't be found."); }
    return search->second;
  }

  Communicator &get_current_communicator()
  {
    auto search = communicator_stack.find(current_key);
    if (search == communicator_stack.end()) { errorQuda("Current communicator can't be found."); }
    return search->second;
  }

  void push_communicator(const CommKey &split_key)
  {
    if (comm_nvshmem_enabled())
      errorQuda(
        "Split-grid is currently not supported with NVSHMEM. Please set QUDA_ENABLE_NVSHMEM=0 to disable NVSHMEM.");
    auto search = communicator_stack.find(split_key);
    if (search == communicator_stack.end()) {
      communicator_stack.emplace(std::piecewise_construct, std::forward_as_tuple(split_key),
                                 std::forward_as_tuple(get_default_communicator(), split_key.data()));
    }

    LatticeField::freeGhostBuffer(); // Destroy the (IPC) Comm buffers with the old communicator.

    current_key = split_key;
  }

#if defined(QMP_COMMS) || defined(MPI_COMMS)
  MPI_Comm get_mpi_handle() { return get_current_communicator().get_mpi_handle(); }
#endif

  int comm_neighbor_rank(int dir, int dim) { return get_current_communicator().comm_neighbor_rank(dir, dim); }

  int comm_dim(int dim) { return get_current_communicator().comm_dim(dim); }

  int comm_coord(int dim) { return get_current_communicator().comm_coord(dim); }

  int comm_rank_from_coords(const int *coords) { return get_current_communicator().comm_rank_from_coords(coords); }

  void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data, bool user_set_comm_handle,
                 void *user_comm)
  {
    init_communicator_stack(ndim, dims, rank_from_coords, map_data, user_set_comm_handle, user_comm);
  }

  void comm_finalize() { finalize_communicator_stack(); }

  void comm_dim_partitioned_set(int dim) { get_current_communicator().comm_dim_partitioned_set(dim); }

  void comm_dim_partitioned_reset() { get_current_communicator().comm_dim_partitioned_reset(); }

  int comm_dim_partitioned(int dim) { return get_current_communicator().comm_dim_partitioned(dim); }

  int comm_partitioned() { return get_current_communicator().comm_partitioned(); }

  const char *comm_dim_topology_string() { return get_current_communicator().topology_string; }

  const char *comm_config_string() { return get_current_communicator().comm_config_string(); }

  const char *comm_dim_partitioned_string(const int *comm_dim_override)
  {
    return get_current_communicator().comm_dim_partitioned_string(comm_dim_override);
  }

  int comm_rank(void) { return get_current_communicator().comm_rank(); }

  int comm_rank_global(void) { return Communicator::comm_rank_global(); }

  size_t comm_size(void) { return get_current_communicator().comm_size(); }

  // XXX:
  // Note here we are always using the **default** communicator.
  // We might need to have a better approach.
  int comm_gpuid(void) { return Communicator::comm_gpuid(); }

  bool comm_deterministic_reduce() { return get_current_communicator().comm_deterministic_reduce(); }

  void comm_gather_hostname(char *hostname_recv_buf)
  {
    get_current_communicator().comm_gather_hostname(hostname_recv_buf);
  }

  void comm_gather_gpuid(int *gpuid_recv_buf) { get_current_communicator().comm_gather_gpuid(gpuid_recv_buf); }

  void comm_peer2peer_init(const char *hostname_recv_buf)
  {
    get_current_communicator().comm_peer2peer_init(hostname_recv_buf);
  }

  bool comm_peer2peer_present() { return get_current_communicator().comm_peer2peer_present(); }

  int comm_peer2peer_enabled_global() { return get_current_communicator().comm_peer2peer_enabled_global(); }

  bool comm_peer2peer_enabled(int dir, int dim) { return get_current_communicator().comm_peer2peer_enabled(dir, dim); }

  void comm_enable_peer2peer(bool enable) { get_current_communicator().comm_enable_peer2peer(enable); }

  bool comm_intranode_enabled(int dir, int dim) { return get_current_communicator().comm_intranode_enabled(dir, dim); }

  void comm_enable_intranode(bool enable) { get_current_communicator().comm_enable_intranode(enable); }

  bool comm_gdr_enabled() { return get_current_communicator().comm_gdr_enabled(); }

  bool comm_gdr_blacklist() { return get_current_communicator().comm_gdr_blacklist(); }

  bool comm_nvshmem_enabled() { return get_current_communicator().comm_nvshmem_enabled(); }

  MsgHandle *comm_declare_send_rank(void *buffer, int rank, int tag, size_t nbytes)
  {
    return get_current_communicator().comm_declare_send_rank(buffer, rank, tag, nbytes);
  }

  MsgHandle *comm_declare_recv_rank(void *buffer, int rank, int tag, size_t nbytes)
  {
    return get_current_communicator().comm_declare_recv_rank(buffer, rank, tag, nbytes);
  }

  MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
  {
    return get_current_communicator().comm_declare_send_displaced(buffer, displacement, nbytes);
  }

  MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
  {
    return get_current_communicator().comm_declare_receive_displaced(buffer, displacement, nbytes);
  }

  MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                 size_t stride)
  {
    return get_current_communicator().comm_declare_strided_send_displaced(buffer, displacement, blksize, nblocks, stride);
  }

  MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                    size_t stride)
  {
    return get_current_communicator().comm_declare_strided_receive_displaced(buffer, displacement, blksize, nblocks,
                                                                             stride);
  }

#define CHECK_MH(mh) { if (mh == nullptr) errorQuda("null message handle"); }

  void comm_free(MsgHandle *&mh) { CHECK_MH(mh); get_current_communicator().comm_free(mh); }

  void comm_start(MsgHandle *mh) { CHECK_MH(mh); get_current_communicator().comm_start(mh); }

  void comm_wait(MsgHandle *mh) { CHECK_MH(mh); get_current_communicator().comm_wait(mh); }

  int comm_query(MsgHandle *mh) { CHECK_MH(mh); return get_current_communicator().comm_query(mh); }

#undef CHECK_MH

  void comm_allreduce_sum_array(double *data, size_t size)
  {
    get_current_communicator().comm_allreduce_sum_array(data, size);
  }

  template <> void comm_allreduce_sum<std::vector<double>>(std::vector<double> &a)
  {
    comm_allreduce_sum_array(a.data(), a.size());
  }

  template <> void comm_allreduce_sum<std::vector<double2>>(std::vector<double2> &a)
  {
    comm_allreduce_sum_array(reinterpret_cast<double *>(a.data()), 2 * a.size());
  }

  template <> void comm_allreduce_sum<std::vector<std::complex<double>>>(std::vector<std::complex<double>> &a)
  {
    comm_allreduce_sum_array(reinterpret_cast<double *>(a.data()), 2 * a.size());
  }

  template <> void comm_allreduce_sum<std::vector<array<double, 2>>>(std::vector<array<double, 2>> &a)
  {
    comm_allreduce_sum_array(reinterpret_cast<double *>(a.data()), 2 * a.size());
  }

  template <> void comm_allreduce_sum<std::vector<array<double, 3>>>(std::vector<array<double, 3>> &a)
  {
    comm_allreduce_sum_array(reinterpret_cast<double *>(a.data()), 3 * a.size());
  }

  template <> void comm_allreduce_sum<std::vector<array<double, 4>>>(std::vector<array<double, 4>> &a)
  {
    comm_allreduce_sum_array(reinterpret_cast<double *>(a.data()), 4 * a.size());
  }

  template <> void comm_allreduce_sum<double>(double &a) { comm_allreduce_sum_array(&a, 1); }

  void comm_allreduce_max_array(double *data, size_t size)
  {
    get_current_communicator().comm_allreduce_max_array(data, size);
  }

  template <> void comm_allreduce_max<double>(double &a) { comm_allreduce_max_array(&a, 1); }

  template <> void comm_allreduce_max<float>(float &a)
  {
    double a_ = a;
    comm_allreduce_max_array(&a_, 1);
    a = a_;
  }

  template <> void comm_allreduce_max<std::vector<double>>(std::vector<double> &a)
  {
    comm_allreduce_max_array(a.data(), a.size());
  }

  template <> void comm_allreduce_max<std::vector<float>>(std::vector<float> &a)
  {
    std::vector<double> a_(a.size());
    for (unsigned int i = 0; i < a.size(); i++) a_[i] = a[i];
    comm_allreduce_max_array(a_.data(), a_.size());
    for (unsigned int i = 0; i < a.size(); i++) a[i] = a_[i];
  }

  template <> void comm_allreduce_max<std::vector<array<double, 2>>>(std::vector<array<double, 2>> &a)
  {
    comm_allreduce_max_array(reinterpret_cast<double *>(a.data()), 2 * a.size());
  }

  template <> void comm_allreduce_max<deviation_t<double>>(deviation_t<double> &a)
  {
    get_current_communicator().comm_allreduce_max_array(&a, 1);
  }

  template <> void comm_allreduce_max<std::vector<deviation_t<double>>>(std::vector<deviation_t<double>> &a)
  {
    get_current_communicator().comm_allreduce_max_array(a.data(), a.size());
  }

  void comm_allreduce_min_array(double *data, size_t size)
  {
    get_current_communicator().comm_allreduce_min_array(data, size);
  }

  template <> void comm_allreduce_min<std::vector<double>>(std::vector<double> &a)
  {
    comm_allreduce_min_array(a.data(), a.size());
  }

  template <> void comm_allreduce_min<std::vector<float>>(std::vector<float> &a)
  {
    std::vector<double> a_(a.size());
    for (unsigned int i = 0; i < a.size(); i++) a_[i] = a[i];
    comm_allreduce_min_array(a_.data(), a_.size());
    for (unsigned int i = 0; i < a.size(); i++) a[i] = a_[i];
  }

  template <> void comm_allreduce_max<int32_t>(int32_t &a)
  {
    std::vector<double> a_(1, static_cast<double>(a));
    comm_allreduce_max_array(a_.data(), a_.size());
    a = static_cast<int32_t>(a_[0]);
  }

  template <> void comm_allreduce_min<int32_t>(int32_t &a)
  {
    std::vector<double> a_(1, static_cast<double>(a));
    comm_allreduce_min_array(a_.data(), a_.size());
    a = static_cast<int32_t>(a_[0]);
  }

  void comm_allreduce_int(int &data) { get_current_communicator().comm_allreduce_int(data); }

  void comm_allreduce_xor(uint64_t &data) { get_current_communicator().comm_allreduce_xor(data); }

  void comm_broadcast(void *data, size_t nbytes, int root)
  {
    get_current_communicator().comm_broadcast(data, nbytes, root);
  }

  void comm_broadcast_global(void *data, size_t nbytes, int root)
  {
    get_default_communicator().comm_broadcast(data, nbytes, root);
  }

  void comm_barrier(void) { get_current_communicator().comm_barrier(); }

  void comm_abort_(int status) { Communicator::comm_abort_(status); };

  int commDim(int dim) { return get_current_communicator().commDim(dim); }

  int commCoords(int dim) { return get_current_communicator().commCoords(dim); }

  int commDimPartitioned(int dir) { return get_current_communicator().commDimPartitioned(dir); }

  void commDimPartitionedSet(int dir)
  {
    get_current_communicator().commDimPartitionedSet(dir);
  }

  void commDimPartitionedReset()
  {
    get_current_communicator().comm_dim_partitioned_reset();
  }

  bool commGlobalReduction() { return get_current_communicator().commGlobalReduction(); }

  void commGlobalReductionPush(bool global_reduce)
  {
    get_current_communicator().commGlobalReductionPush(global_reduce);
  }

  void commGlobalReductionPop() { get_current_communicator().commGlobalReductionPop(); }

  bool commAsyncReduction() { return get_current_communicator().commAsyncReduction(); }

  void commAsyncReductionSet(bool global_reduce) { get_current_communicator().commAsyncReductionSet(global_reduce); }

  int get_enable_p2p_max_access_rank() { return get_current_communicator().enable_p2p_max_access_rank; }

} // namespace quda
