#include <communicator_quda.h>
#include <mpi_comm_handle.h>

// While we can emulate an all-gather using QMP reductions, this
// scales horribly as the number of nodes increases, so for
// performance we just call MPI directly
#define USE_MPI_GATHER

#ifdef USE_MPI_GATHER
#include <mpi.h>
#endif

#define QMP_CHECK(qmp_call)                                                                                            \
  do {                                                                                                                 \
    QMP_status_t status = qmp_call;                                                                                    \
    if (status != QMP_SUCCESS) errorQuda("(QMP) %s", QMP_error_string(status));                                        \
  } while (0)

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
    QMP_msgmem_t mem;
    QMP_msghandle_t handle;
  };

  Communicator::Communicator(int nDim, const int *commDims, QudaCommsMap rank_from_coords, void *map_data,
                             bool user_set_comm_handle_, void *user_comm)
  {
    user_set_comm_handle = user_set_comm_handle_;

    int initialized;
    MPI_CHECK(MPI_Initialized(&initialized));

    if (!initialized) { assert(false); }

    if (user_set_comm_handle) {
      // The logic here being: previouly all QMP calls are based on QMP_comm_get_default(), and user can
      // feed in their own MPI_COMM_HANDLE. Now with the following this behavior should remain the same.
      QMP_COMM_HANDLE = QMP_comm_get_default();
      MPI_COMM_HANDLE = *((MPI_Comm *)user_comm);
    } else {
      QMP_COMM_HANDLE = QMP_comm_get_default();
      void *my_comm;
      QMP_get_mpi_comm(QMP_COMM_HANDLE, &my_comm);
      MPI_COMM_HANDLE = *((MPI_Comm *)my_comm);
    }

    is_qmp_handle_default = true; // the QMP handle is the default one.

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

    QMP_comm_split(other.QMP_COMM_HANDLE, color, key, &QMP_COMM_HANDLE);
    void *my_comm;
    QMP_get_mpi_comm(QMP_COMM_HANDLE, &my_comm);
    MPI_COMM_HANDLE = *((MPI_Comm *)my_comm);

    is_qmp_handle_default = false; // the QMP handle is not the default one.

    int my_rank_ = QMP_get_node_number();

    QudaCommsMap func = lex_rank_from_coords_dim_t;
    comm_init(nDim, comm_dims_split.data(), func, comm_dims_split.data());

    printf("Creating split communicator, base_rank = %5d, key = %5d, color = %5d, split_rank = %5d, gpuid = %d\n",
           other.comm_rank(), key, color, my_rank_, comm_gpuid());
  }

  Communicator::~Communicator()
  {
    comm_finalize();
    if (!(user_set_comm_handle || is_qmp_handle_default)) {
      // - if it's a user set handle, or if it's the default QMP handle, we don't free it.
      QMP_comm_free(QMP_COMM_HANDLE);
    }
  }

  // There are more efficient ways to do the following,
  // but it doesn't really matter since this function should be
  // called just once.
  void Communicator::comm_gather_hostname(char *hostname_recv_buf)
  {
    // determine which GPU this rank will use
    char *hostname = comm_hostname();

#ifdef USE_MPI_GATHER
  MPI_CHECK(MPI_Allgather(hostname, QUDA_MAX_HOSTNAME_STRING, MPI_CHAR, hostname_recv_buf, QUDA_MAX_HOSTNAME_STRING, MPI_CHAR, MPI_COMM_HANDLE));
#else
  // Abuse reductions to emulate all-gather.  We need to copy the
  // local hostname to all other nodes
  // this isn't very scalable though
  for (int i = 0; i < comm_size(); i++) {
    int data[QUDA_MAX_HOSTNAME_STRING];
    for (int j = 0; j < QUDA_MAX_HOSTNAME_STRING; j++) {
      data[j] = (i == comm_rank()) ? hostname[j] : 0;
      QMP_comm_sum_int(QMP_COMM_HANDLE, data + j);
      hostname_recv_buf[i * QUDA_MAX_HOSTNAME_STRING + j] = data[j];
    }
  }
#endif
}

// There are more efficient ways to do the following,
// but it doesn't really matter since this function should be
// called just once.
void Communicator::comm_gather_gpuid(int *gpuid_recv_buf)
{

#ifdef USE_MPI_GATHER
  int gpuid = comm_gpuid();
  MPI_CHECK(MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_HANDLE));
#else
  // Abuse reductions to emulate all-gather.  We need to copy the
  // local gpu to all other nodes
  for (int i = 0; i < comm_size(); i++) {
    int data = (i == comm_rank()) ? comm_gpuid() : 0;
    QMP_comm_sum_int(QMP_COMM_HANDLE, &data);
    gpuid_recv_buf[i] = data;
  }
#endif
}

void Communicator::comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  if (QMP_is_initialized() != QMP_TRUE) { errorQuda("QMP has not been initialized"); }

  int grid_size = 1;
  for (int i = 0; i < ndim; i++) { grid_size *= dims[i]; }
  if (grid_size != QMP_comm_get_number_of_nodes(QMP_COMM_HANDLE)) {
    errorQuda("Communication grid size declared via initCommsGridQuda() does not match"
              " total number of QMP nodes (%d != %d)",
              grid_size, QMP_comm_get_number_of_nodes(QMP_COMM_HANDLE));
  }

  comm_init_common(ndim, dims, rank_from_coords, map_data);
}

int Communicator::comm_rank(void) { return QMP_comm_get_node_number(QMP_COMM_HANDLE); }

size_t Communicator::comm_size(void) { return QMP_comm_get_number_of_nodes(QMP_COMM_HANDLE); }

/**
 * Declare a message handle for sending `nbytes` to the `rank` with `tag`.
 */
MsgHandle *Communicator::comm_declare_send_rank(void *buffer, int rank, int, size_t nbytes)
{
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_send_to(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for receiving `nbytes` from the `rank` with `tag`.
 */
MsgHandle *Communicator::comm_declare_recv_rank(void *buffer, int rank, int, size_t nbytes)
{
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_receive_from(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *Communicator::comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_send_to(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *Communicator::comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_receive_from(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for strided sending to a node displaced in
 * (x,y,z,t) according to "displacement"
 */
MsgHandle *Communicator::comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize,
                                                             int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_strided_msgmem(buffer, blksize, nblocks, stride);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_send_to(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for strided receiving from a node
 * displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *Communicator::comm_declare_strided_receive_displaced(void *buffer, const int displacement[], size_t blksize,
                                                                int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_strided_msgmem(buffer, blksize, nblocks, stride);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_comm_declare_receive_from(QMP_COMM_HANDLE, mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

void Communicator::comm_free(MsgHandle *&mh)
{
  QMP_free_msghandle(mh->handle);
  QMP_free_msgmem(mh->mem);
  host_free(mh);
  mh = nullptr;
}

void Communicator::comm_start(MsgHandle *mh) { QMP_CHECK(QMP_start(mh->handle)); }

void Communicator::comm_wait(MsgHandle *mh) { QMP_CHECK(QMP_wait(mh->handle)); }

int Communicator::comm_query(MsgHandle *mh) { return (QMP_is_complete(mh->handle) == QMP_TRUE); }

void Communicator::comm_allreduce_sum_array(double *data, size_t size)
{
  if (!comm_deterministic_reduce()) {
    QMP_CHECK(QMP_comm_sum_double_array(QMP_COMM_HANDLE, data, size));
  } else {
    // we need to break out of QMP for the deterministic floating point reductions
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
  for (size_t i = 0; i < size; i++) { QMP_CHECK(QMP_comm_max_double(QMP_COMM_HANDLE, data + i)); }
}

void Communicator::comm_allreduce_min_array(double *data, size_t size)
{
  for (size_t i = 0; i < size; i++) { QMP_CHECK(QMP_comm_min_double(QMP_COMM_HANDLE, data + i)); }
}

void Communicator::comm_allreduce_int(int &data) { QMP_CHECK(QMP_comm_sum_int(QMP_COMM_HANDLE, &data)); }

void Communicator::comm_allreduce_xor(uint64_t &data)
{
  if (sizeof(uint64_t) != sizeof(unsigned long)) errorQuda("unsigned long is not 64-bit");
  QMP_CHECK(QMP_comm_xor_ulong(QMP_COMM_HANDLE, reinterpret_cast<unsigned long *>(&data)));
}

void Communicator::comm_broadcast(void *data, size_t nbytes, int root)
{
  // break out of QMP since it can only broadcast from rank 0
  MPI_CHECK(MPI_Bcast(data, (int)nbytes, MPI_BYTE, root, MPI_COMM_HANDLE));
  // QMP_CHECK(QMP_comm_broadcast(QMP_COMM_HANDLE, data, nbytes));
}

void Communicator::comm_barrier(void) { QMP_CHECK(QMP_comm_barrier(QMP_COMM_HANDLE)); }

void Communicator::comm_abort_(int status) { QMP_abort(status); }

int Communicator::comm_rank_global() { return QMP_get_node_number(); }

} // namespace quda
