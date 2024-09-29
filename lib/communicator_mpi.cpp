#include <communicator_quda.h>

#define QUDA_COMM_CHECKHANG
#define QUDA_COMM_CHECKSUM

#define MPI_CHECK(mpi_call)                                                                                            \
  do {                                                                                                                 \
    int status = mpi_call;                                                                                             \
    if (status != MPI_SUCCESS) {                                                                                       \
      char err_string[MPI_MAX_ERROR_STRING];                                                                           \
      int err_len;                                                                                                     \
      MPI_Error_string(status, err_string, &err_len);                                                                  \
      err_string[127] = '\0';                                                                                          \
      errorQuda("(MPI) %s", err_string);                                                                               \
    }                                                                                                                  \
  } while (0)

namespace quda
{

  struct MsgHandle_s {
    /**
       The persistant MPI communicator handle that is created with
       MPI_Send_init / MPI_Recv_init.
     */
    MPI_Request request;

    /**
       To create a strided communicator, a MPI_Vector datatype has to be
       created.  This is where it is stored.
     */
    MPI_Datatype datatype;

    /**
       Whether a custom datatype has been created or not.  Used to
       determine whether we need to free the datatype or not.
     */
    bool custom;

#ifdef QUDA_COMM_CHECKSUM
    bool isSend;
    void *buffer;
    size_t nbytes;
    uint64_t chksum;
    MPI_Request chkreq;
#endif
  };

#ifdef QUDA_COMM_CHECKSUM
  uint64_t chksum_cpu(void *buf, size_t n)
  {
    uint64_t sum = 0xf0f0f0f0;
    // assume buffer is aligned
    auto bufl = static_cast<uint64_t *>(buf);
    size_t nl = n/8;
    for (size_t i=0; i<nl; i++) {
      sum ^= bufl[i];
    }
    size_t nc = 8 * nl;
    char *bufc = static_cast<char *>(buf) + nc;
    size_t rem = n - nc;
    for (size_t i=0; i<rem; i++) {
      sum ^= ((uint64_t)bufc[i]) << i;
    }
    return sum;
  }
  uint64_t chksum_gpu(void *buf, size_t n)
  {
    void *bufh = safe_malloc(n);
    qudaMemcpy(bufh, buf, n, qudaMemcpyDeviceToHost);
    auto chk = chksum_cpu(bufh, n);
    host_free(bufh);
    return chk;
  }
  uint64_t chksum(void *buf, size_t n)
  {
    auto loc = get_pointer_location(buf);
    if (loc==QUDA_CPU_FIELD_LOCATION) {
      return chksum_cpu(buf, n);
    } else {
      return chksum_gpu(buf, n);
    }
  }
#endif

  Communicator::Communicator(int nDim, const int *commDims, QudaCommsMap rank_from_coords, void *map_data,
                             bool user_set_comm_handle_, void *user_comm)
  {
    user_set_comm_handle = user_set_comm_handle_;

    int initialized;
    MPI_CHECK(MPI_Initialized(&initialized));

    if (!initialized) { assert(false); }

    if (user_set_comm_handle) {
      MPI_COMM_HANDLE = *((MPI_Comm *)user_comm);
    } else {
      MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_HANDLE);
    }

    comm_init(nDim, commDims, rank_from_coords, map_data);
    globalReduce.push(true);
  }

  Communicator::Communicator(Communicator &other, const int *comm_split) : globalReduce(other.globalReduce)
  {
    user_set_comm_handle = false;

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

    int key = index(nDim, comm_dims_split.data(), comm_key_split.data());
    int color = index(nDim, comm_split, comm_color_split.data());

    MPI_CHECK(MPI_Comm_split(other.MPI_COMM_HANDLE, color, key, &MPI_COMM_HANDLE));
    int my_rank_;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_HANDLE, &my_rank_));

    QudaCommsMap func = lex_rank_from_coords_dim_t;
    comm_init(nDim, comm_dims_split.data(), func, comm_dims_split.data());
  }

  Communicator::~Communicator()
  {
    comm_finalize();
    if (!user_set_comm_handle) { MPI_Comm_free(&MPI_COMM_HANDLE); }
  }

  void Communicator::comm_gather_hostname(char *hostname_recv_buf)
  {
    // determine which GPU this rank will use
    char *hostname = comm_hostname();
    MPI_CHECK(MPI_Allgather(hostname, QUDA_MAX_HOSTNAME_STRING, MPI_CHAR, hostname_recv_buf, QUDA_MAX_HOSTNAME_STRING, MPI_CHAR, MPI_COMM_HANDLE));
  }

  void Communicator::comm_gather_gpuid(int *gpuid_recv_buf)
  {
    int gpuid = comm_gpuid();
    MPI_CHECK(MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_HANDLE));
  }

  void Communicator::comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
  {
    int initialized;
    MPI_CHECK(MPI_Initialized(&initialized));

    if (!initialized) { errorQuda("MPI has not been initialized"); }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_HANDLE, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_HANDLE, &size));

    int grid_size = 1;
    for (int i = 0; i < ndim; i++) { grid_size *= dims[i]; }
    if (grid_size != size) {
      errorQuda("Communication grid size declared via initCommsGridQuda() does not match"
                " total number of MPI ranks (%d != %d)",
                grid_size, size);
    }

    comm_init_common(ndim, dims, rank_from_coords, map_data);
  }

  int Communicator::comm_rank(void) { return rank; }

  size_t Communicator::comm_size(void) { return size; }

  /**
   * Declare a message handle for sending `nbytes` to the `rank` with `tag`.
   */
  MsgHandle *Communicator::comm_declare_send_rank(void *buffer, int rank, int tag, size_t nbytes)
  {
    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
    MPI_CHECK(MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
    mh->custom = false;
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = true;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    MPI_CHECK(MPI_Send_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif
    return mh;
  }

  /**
   * Declare a message handle for receiving `nbytes` from the `rank` with `tag`.
   */
  MsgHandle *Communicator::comm_declare_recv_rank(void *buffer, int rank, int tag, size_t nbytes)
  {
    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
    MPI_CHECK(MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
    mh->custom = false;
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = false;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    MPI_CHECK(MPI_Recv_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif
    return mh;
  }

  /**
   * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *Communicator::comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
  {
    Topology *topo = comm_default_topology();
    int ndim = comm_ndim(topo);
    check_displacement(displacement, ndim);

    int rank = comm_rank_displaced(topo, displacement);

    int tag = 0;
    for (int i = ndim - 1; i >= 0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
    tag = tag >= 0 ? tag : 2 * std::pow(4 * max_displacement, ndim) + tag;

    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
    MPI_CHECK(MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
    mh->custom = false;
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = true;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    MPI_CHECK(MPI_Send_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif

    return mh;
  }

  /**
   * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *Communicator::comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
  {
    Topology *topo = comm_default_topology();
    int ndim = comm_ndim(topo);
    check_displacement(displacement, ndim);

    int rank = comm_rank_displaced(topo, displacement);

    int tag = 0;
    for (int i = ndim - 1; i >= 0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
    tag = tag >= 0 ? tag : 2 * std::pow(4 * max_displacement, ndim) + tag;

    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
    MPI_CHECK(MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
    mh->custom = false;
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = false;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    MPI_CHECK(MPI_Recv_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif

    return mh;
  }

  /**
   * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *Communicator::comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize,
                                                               int nblocks, size_t stride)
  {
    Topology *topo = comm_default_topology();
    int ndim = comm_ndim(topo);
    check_displacement(displacement, ndim);

    int rank = comm_rank_displaced(topo, displacement);

    int tag = 0;
    for (int i = ndim - 1; i >= 0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
    tag = tag >= 0 ? tag : 2 * std::pow(4 * max_displacement, ndim) + tag;

    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

    // create a new strided MPI type
    MPI_CHECK(MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)));
    MPI_CHECK(MPI_Type_commit(&(mh->datatype)));
    mh->custom = true;

    MPI_CHECK(MPI_Send_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = true;
    mh->buffer = buffer;
    mh->nbytes = 0;  // strides not supported yet
    //MPI_CHECK(MPI_Send_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif

    return mh;
  }

  /**
   * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *Communicator::comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
                                                                  size_t blksize, int nblocks, size_t stride)
  {
    Topology *topo = comm_default_topology();
    int ndim = comm_ndim(topo);
    check_displacement(displacement, ndim);

    int rank = comm_rank_displaced(topo, displacement);

    int tag = 0;
    for (int i = ndim - 1; i >= 0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
    tag = tag >= 0 ? tag : 2 * std::pow(4 * max_displacement, ndim) + tag;

    MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

    // create a new strided MPI type
    MPI_CHECK(MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)));
    MPI_CHECK(MPI_Type_commit(&(mh->datatype)));
    mh->custom = true;

    MPI_CHECK(MPI_Recv_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
#ifdef QUDA_COMM_CHECKSUM
    mh->isSend = false;
    mh->buffer = buffer;
    mh->nbytes = 0;  // strides not supported yet
    //MPI_CHECK(MPI_Recv_init(&(mh->chksum), 1, MPI_UINT64_T, rank, tag, MPI_COMM_HANDLE, &(mh->chkreq)));
#endif

    return mh;
  }

  void Communicator::comm_free(MsgHandle *&mh)
  {
    MPI_CHECK(MPI_Request_free(&(mh->request)));
    if (mh->custom) MPI_CHECK(MPI_Type_free(&(mh->datatype)));
#ifdef QUDA_COMM_CHECKSUM
    if (mh->nbytes>0) MPI_CHECK(MPI_Request_free(&(mh->chkreq)));
#endif
    host_free(mh);
    mh = nullptr;
  }

  void Communicator::comm_start(MsgHandle *mh)
  {
#ifdef QUDA_COMM_CHECKSUM
    if (mh->isSend) {
      mh->chksum = chksum(mh->buffer, mh->nbytes);
    }
    MPI_CHECK(MPI_Start(&(mh->chkreq)));
    MPI_CHECK(MPI_Wait(&(mh->chkreq), MPI_STATUS_IGNORE));
#endif
    MPI_CHECK(MPI_Start(&(mh->request)));
  }

  void Communicator::comm_wait(MsgHandle *mh) {
    MPI_CHECK(MPI_Wait(&(mh->request), MPI_STATUS_IGNORE));
#ifdef QUDA_COMM_CHECKSUM
    if (!mh->isSend) {
      auto cs = chksum(mh->buffer, mh->nbytes);
      if (cs != mh->chksum) {
	errorQuda("comm_wait checksum failure got %lu expeted %lu\n", cs, mh->chksum);
      }
    }
#endif
  }

#ifdef QUDA_COMM_CHECKHANG
  void hang(int, siginfo_t *, void *) {
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlen;
    MPI_Get_processor_name(name, &resultlen);
    errorQuda("%s stuck in MPI_Test for 120 seconds\n", name);
  }
#endif

  int Communicator::comm_query(MsgHandle *mh)
  {
#ifdef QUDA_COMM_CHECKHANG
    static bool firstCall = true;
    if(firstCall) {
      firstCall = false;
      struct sigaction sig_action;
      memset(&sig_action, 0, sizeof(sig_action));
      sig_action.sa_sigaction = hang;
      sig_action.sa_flags = 0;
      sigemptyset(&sig_action.sa_mask);
      sigaction(SIGALRM, &sig_action, 0);
    }
    alarm(120);  // 120 seconds
#endif
    int query;
    MPI_CHECK(MPI_Test(&(mh->request), &query, MPI_STATUS_IGNORE));
#ifdef QUDA_COMM_CHECKHANG
    alarm(0);
#endif
    return query;
  }

#if 0
  void Communicator::comm_query(int n, MsgHandle *mh[], int *outcount, int array_of_indices[])
  {
    MPI_Request req[n];
    for (int i=0; i<n; i++) req[i] = mh[i]->request;
    MPI_CHECK(MPI_Testsome(n, req, outcount, array_of_indices, MPI_STATUSES_IGNORE));
  }
#endif

  void Communicator::comm_allreduce_sum_array(double *data, size_t size)
  {
    if (!comm_deterministic_reduce()) {
      std::vector<double> recvbuf(size);
      MPI_CHECK(MPI_Allreduce(data, recvbuf.data(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_HANDLE));
      memcpy(data, recvbuf.data(), size * sizeof(double));
    } else {
      size_t n = comm_size();
      std::vector<double> recv_buf(size * n);
      MPI_CHECK(MPI_Allgather(data, size, MPI_DOUBLE, recv_buf.data(), size, MPI_DOUBLE, MPI_COMM_HANDLE));

      std::vector<double> recv_trans(size * n);
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < size; j++) { recv_trans[j * n + i] = recv_buf[i * size + j]; }
      }

      for (size_t i = 0; i < size; i++) { data[i] = deterministic_reduce(recv_trans.data() + i * n, n); }
    }
  }

  void Communicator::comm_allreduce_sum(size_t &a)
  {
    if (sizeof(size_t) != sizeof(unsigned long)) {
      errorQuda("sizeof(size_t) != sizeof(unsigned long): %lu != %lu\n", sizeof(size_t), sizeof(unsigned long));
    }
    size_t recv;
    MPI_CHECK(MPI_Allreduce(&a, &recv, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_HANDLE));
    a = recv;
  }

  void Communicator::comm_allreduce_max_array(deviation_t<double> *data, size_t size)
  {
    size_t n = comm_size();
    std::vector<deviation_t<double>> recv_buf(size * n);
    MPI_CHECK(MPI_Allgather(data, 2 * size, MPI_DOUBLE, recv_buf.data(), 2 * size, MPI_DOUBLE, MPI_COMM_HANDLE));

    std::vector<deviation_t<double>> recv_trans(size * n);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < size; j++) { recv_trans[j * n + i] = recv_buf[i * size + j]; }
    }

    for (size_t i = 0; i < size; i++) {
      data[i] = recv_trans[i * n];
      for (size_t j = 1; j < n; j++) { data[i] = data[i] > recv_trans[i * n + j] ? data[i] : recv_trans[i * n + j]; }
    }
  }

  void Communicator::comm_allreduce_max_array(double *data, size_t size)
  {
    std::vector<double> recvbuf(size);
    MPI_CHECK(MPI_Allreduce(data, recvbuf.data(), size, MPI_DOUBLE, MPI_MAX, MPI_COMM_HANDLE));
    memcpy(data, recvbuf.data(), size * sizeof(double));
  }

  void Communicator::comm_allreduce_min_array(double *data, size_t size)
  {
    std::vector<double> recvbuf(size);
    MPI_CHECK(MPI_Allreduce(data, recvbuf.data(), size, MPI_DOUBLE, MPI_MIN, MPI_COMM_HANDLE));
    memcpy(data, recvbuf.data(), size * sizeof(double));
  }

  void Communicator::comm_allreduce_int(int &data)
  {
    int recvbuf;
    MPI_CHECK(MPI_Allreduce(&data, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_HANDLE));
    data = recvbuf;
  }

  void Communicator::comm_allreduce_xor(uint64_t &data)
  {
    if (sizeof(uint64_t) != sizeof(unsigned long)) errorQuda("unsigned long is not 64-bit");
    uint64_t recvbuf;
    MPI_CHECK(MPI_Allreduce(&data, &recvbuf, 1, MPI_UNSIGNED_LONG, MPI_BXOR, MPI_COMM_HANDLE));
    data = recvbuf;
  }

  /**  broadcast from rank 0 */
  void Communicator::comm_broadcast(void *data, size_t nbytes, int root)
  {
    MPI_CHECK(MPI_Bcast(data, (int)nbytes, MPI_BYTE, root, MPI_COMM_HANDLE));
  }

  void Communicator::comm_barrier(void) { MPI_CHECK(MPI_Barrier(MPI_COMM_HANDLE)); }

  void Communicator::comm_abort_(int status) { MPI_Abort(MPI_COMM_WORLD, status); }

  int Communicator::comm_rank_global()
  {
    static int global_rank = -1;
    if (global_rank < 0) { MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &global_rank)); }
    return global_rank;
  }

} // namespace quda
