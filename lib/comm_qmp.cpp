#include <qmp.h>
#include <algorithm>
#include <numeric>
#include <quda_internal.h>
#include <comm_quda.h>
#include <mpi_comm_handle.h>

#define QMP_CHECK(qmp_call) do {                     \
  QMP_status_t status = qmp_call;                    \
  if (status != QMP_SUCCESS)                         \
    errorQuda("(QMP) %s", QMP_error_string(status)); \
} while (0)

#define MPI_CHECK(mpi_call)                                                                                            \
  do {                                                                                                                 \
    int status = mpi_call;                                                                                             \
    if (status != MPI_SUCCESS) {                                                                                       \
      char err_string[128];                                                                                            \
      int err_len;                                                                                                     \
      MPI_Error_string(status, err_string, &err_len);                                                                  \
      err_string[127] = '\0';                                                                                          \
      errorQuda("(MPI) %s", err_string);                                                                               \
    }                                                                                                                  \
  } while (0)

struct MsgHandle_s {
  QMP_msgmem_t mem;
  QMP_msghandle_t handle;
};

// While we can emulate an all-gather using QMP reductions, this
// scales horribly as the number of nodes increases, so for
// performance we just call MPI directly
#define USE_MPI_GATHER

#ifdef USE_MPI_GATHER
#include <mpi.h>
#endif

// There are more efficient ways to do the following,
// but it doesn't really matter since this function should be
// called just once.
void comm_gather_hostname(char *hostname_recv_buf) {
  // determine which GPU this rank will use
  char *hostname = comm_hostname();

#ifdef USE_MPI_GATHER
  MPI_CHECK(MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_HANDLE));
#else
  // Abuse reductions to emulate all-gather.  We need to copy the
  // local hostname to all other nodes
  // this isn't very scalable though
  for (int i=0; i<comm_size(); i++) {
    int data[128];
    for (int j=0; j<128; j++) {
      data[j] = (i == comm_rank()) ? hostname[j] : 0;
      QMP_sum_int(data+j);
      hostname_recv_buf[i*128 + j] = data[j];
    }
  }
#endif

}


// There are more efficient ways to do the following,
// but it doesn't really matter since this function should be
// called just once.
void comm_gather_gpuid(int *gpuid_recv_buf) {

#ifdef USE_MPI_GATHER
  int gpuid = comm_gpuid();
  MPI_CHECK(MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_HANDLE));
#else
  // Abuse reductions to emulate all-gather.  We need to copy the
  // local gpu to all other nodes
  for (int i=0; i<comm_size(); i++) {
    int data = (i == comm_rank()) ? comm_gpuid() : 0;
    QMP_sum_int(&data);
    gpuid_recv_buf[i] = data;
  }
#endif
}


void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  if ( QMP_is_initialized() != QMP_TRUE ) {
    errorQuda("QMP has not been initialized");
  }

  int grid_size = 1;
  for (int i = 0; i < ndim; i++) {
    grid_size *= dims[i];
  }
  if (grid_size != QMP_get_number_of_nodes()) {
    errorQuda("Communication grid size declared via initCommsGridQuda() does not match"
              " total number of QMP nodes (%d != %d)", grid_size, QMP_get_number_of_nodes());
  }

  comm_init_common(ndim, dims, rank_from_coords, map_data);
}

int comm_rank(void)
{
  return QMP_get_node_number();
}


int comm_size(void)
{
  return QMP_get_number_of_nodes();
}


/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_declare_send_to(mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_msgmem(buffer, nbytes);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_declare_receive_from(mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}


/**
 * Declare a message handle for strided sending to a node displaced in
 * (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[],
					       size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_strided_msgmem(buffer, blksize, nblocks, stride);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_declare_send_to(mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

/**
 * Declare a message handle for strided receiving from a node
 * displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  mh->mem = QMP_declare_strided_msgmem(buffer, blksize, nblocks, stride);
  if (mh->mem == NULL) errorQuda("Unable to allocate QMP message memory");

  mh->handle = QMP_declare_receive_from(mh->mem, rank, 0);
  if (mh->handle == NULL) errorQuda("Unable to allocate QMP message handle");

  return mh;
}

void comm_free(MsgHandle *&mh)
{
  QMP_free_msghandle(mh->handle);
  QMP_free_msgmem(mh->mem);
  host_free(mh);
  mh = nullptr;
}


void comm_start(MsgHandle *mh)
{
  QMP_CHECK( QMP_start(mh->handle) );
}


void comm_wait(MsgHandle *mh)
{
  QMP_CHECK( QMP_wait(mh->handle) );
}


int comm_query(MsgHandle *mh)
{
  return (QMP_is_complete(mh->handle) == QMP_TRUE);
}

template <typename T> T deterministic_reduce(T *array, int n)
{
  std::sort(array, array + n); // sort reduction into ascending order for deterministic reduction
  return std::accumulate(array, array + n, 0.0);
}

void comm_allreduce(double* data)
{
  if (!comm_deterministic_reduce()) {
    QMP_CHECK(QMP_sum_double(data));
  } else {
    // we need to break out of QMP for the deterministic floating point reductions
    const size_t n = comm_size();
    double *recv_buf = (double *)safe_malloc(n * sizeof(double));
    MPI_CHECK(MPI_Allgather(data, 1, MPI_DOUBLE, recv_buf, 1, MPI_DOUBLE, MPI_COMM_HANDLE));
    *data = deterministic_reduce(recv_buf, n);
    host_free(recv_buf);
  }
}


void comm_allreduce_max(double* data)
{
  QMP_CHECK( QMP_max_double(data) );
}

void comm_allreduce_min(double* data)
{
  QMP_CHECK( QMP_min_double(data) );
}


void comm_allreduce_array(double* data, size_t size)
{
  if (!comm_deterministic_reduce()) {
    QMP_CHECK(QMP_sum_double_array(data, size));
  } else {
    // we need to break out of QMP for the deterministic floating point reductions
    size_t n = comm_size();
    double *recv_buf = new double[size * n];
    MPI_CHECK(MPI_Allgather(data, size, MPI_DOUBLE, recv_buf, size, MPI_DOUBLE, MPI_COMM_HANDLE));

    double *recv_trans = new double[size * n];
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < size; j++) { recv_trans[j * n + i] = recv_buf[i * size + j]; }
    }

    for (size_t i = 0; i < size; i++) { data[i] = deterministic_reduce(recv_trans + i * n, n); }

    delete[] recv_buf;
    delete[] recv_trans;
  }
}

void comm_allreduce_max_array(double* data, size_t size)
{

  for (size_t i = 0; i < size; i++) { QMP_CHECK(QMP_max_double(data + i)); }
}

void comm_allreduce_int(int* data)
{
  QMP_CHECK( QMP_sum_int(data) );
}

void comm_allreduce_xor(uint64_t *data)
{
  if (sizeof(uint64_t) != sizeof(unsigned long)) errorQuda("unsigned long is not 64-bit");
  QMP_CHECK( QMP_xor_ulong( reinterpret_cast<unsigned long*>(data) ));
}

void comm_broadcast(void *data, size_t nbytes)
{
  QMP_CHECK( QMP_broadcast(data, nbytes) );
}


void comm_barrier(void)
{
  QMP_CHECK( QMP_barrier() );
}


void comm_abort_(int status)
{
  QMP_abort(status);
}
