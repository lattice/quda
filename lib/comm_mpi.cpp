#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <mpi_comm_handle.h>

#define MPI_CHECK(mpi_call) do {                    \
  int status = mpi_call;                            \
  if (status != MPI_SUCCESS) {                      \
    char err_string[128];                           \
    int err_len;                                    \
    MPI_Error_string(status, err_string, &err_len); \
    err_string[127] = '\0';                         \
    errorQuda("(MPI) %s", err_string);              \
  }                                                 \
} while (0)

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
};

static int rank = -1;
static int size = -1;

void comm_gather_hostname(char *hostname_recv_buf) {
  // determine which GPU this rank will use
  char *hostname = comm_hostname();
  MPI_CHECK(MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_HANDLE));
}

void comm_gather_gpuid(int *gpuid_recv_buf) {
  int gpuid = comm_gpuid();
  MPI_CHECK(MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_HANDLE));
}

void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  int initialized;
  MPI_CHECK( MPI_Initialized(&initialized) );

  if (!initialized) {
    errorQuda("MPI has not been initialized");
  }

  MPI_CHECK(MPI_Comm_rank(MPI_COMM_HANDLE, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_HANDLE, &size));

  int grid_size = 1;
  for (int i = 0; i < ndim; i++) {
    grid_size *= dims[i];
  }
  if (grid_size != size) {
    errorQuda("Communication grid size declared via initCommsGridQuda() does not match"
              " total number of MPI ranks (%d != %d)", grid_size, size);
  }

  comm_init_common(ndim, dims, rank_from_coords, map_data);
}

int comm_rank(void)
{
  return rank;
}


int comm_size(void)
{
  return size;
}


static const int max_displacement = 4;

static void check_displacement(const int displacement[], int ndim) {
  for (int i=0; i<ndim; i++) {
    if (abs(displacement[i]) > max_displacement){
      errorQuda("Requested displacement[%d] = %d is greater than maximum allowed", i, displacement[i]);
    }
  }
}

/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement, ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  MPI_CHECK(MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
  mh->custom = false;

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement,ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  MPI_CHECK(MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_HANDLE, &(mh->request)));
  mh->custom = false;

  return mh;
}


/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[],
					       size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement, ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;

  MPI_CHECK(MPI_Send_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_HANDLE, &(mh->request)));

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement,ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;

  MPI_CHECK(MPI_Recv_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_HANDLE, &(mh->request)));

  return mh;
}

void comm_free(MsgHandle *&mh)
{
  MPI_CHECK(MPI_Request_free(&(mh->request)));
  if (mh->custom) MPI_CHECK(MPI_Type_free(&(mh->datatype)));
  host_free(mh);
  mh = nullptr;
}


void comm_start(MsgHandle *mh)
{
  MPI_CHECK( MPI_Start(&(mh->request)) );
}


void comm_wait(MsgHandle *mh)
{
  MPI_CHECK( MPI_Wait(&(mh->request), MPI_STATUS_IGNORE) );
}


int comm_query(MsgHandle *mh)
{
  int query;
  MPI_CHECK( MPI_Test(&(mh->request), &query, MPI_STATUS_IGNORE) );

  return query;
}

template <typename T> T deterministic_reduce(T *array, int n)
{
  std::sort(array, array + n); // sort reduction into ascending order for deterministic reduction
  return std::accumulate(array, array + n, 0.0);
}

void comm_allreduce(double* data)
{
  if (!comm_deterministic_reduce()) {
    double recvbuf;
    MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_HANDLE));
    *data = recvbuf;
  } else {
    const size_t n = comm_size();
    double *recv_buf = (double *)safe_malloc(n * sizeof(double));
    MPI_CHECK(MPI_Allgather(data, 1, MPI_DOUBLE, recv_buf, 1, MPI_DOUBLE, MPI_COMM_HANDLE));
    *data = deterministic_reduce(recv_buf, n);
    host_free(recv_buf);
  }
}


void comm_allreduce_max(double* data)
{
  double recvbuf;
  MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_HANDLE));
  *data = recvbuf;
}

void comm_allreduce_min(double* data)
{
  double recvbuf;
  MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_HANDLE));
  *data = recvbuf;
}

void comm_allreduce_array(double* data, size_t size)
{
  if (!comm_deterministic_reduce()) {
    double *recvbuf = new double[size];
    MPI_CHECK(MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_HANDLE));
    memcpy(data, recvbuf, size * sizeof(double));
    delete[] recvbuf;
  } else {
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
  double *recvbuf = new double[size];
  MPI_CHECK(MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_MAX, MPI_COMM_HANDLE));
  memcpy(data, recvbuf, size*sizeof(double));
  delete []recvbuf;
}

void comm_allreduce_int(int* data)
{
  int recvbuf;
  MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_HANDLE));
  *data = recvbuf;
}

void comm_allreduce_xor(uint64_t *data)
{
  if (sizeof(uint64_t) != sizeof(unsigned long)) errorQuda("unsigned long is not 64-bit");
  uint64_t recvbuf;
  MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_UNSIGNED_LONG, MPI_BXOR, MPI_COMM_HANDLE));
  *data = recvbuf;
}


/**  broadcast from rank 0 */
void comm_broadcast(void *data, size_t nbytes)
{
  MPI_CHECK(MPI_Bcast(data, (int)nbytes, MPI_BYTE, 0, MPI_COMM_HANDLE));
}

void comm_barrier(void) { MPI_CHECK(MPI_Barrier(MPI_COMM_HANDLE)); }

void comm_abort_(int status)
{
  MPI_Abort(MPI_COMM_HANDLE, status);
}
