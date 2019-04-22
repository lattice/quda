#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <csignal>
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
static int gpuid = -1;

static char partition_string[16];
static char topology_string[128];


void comm_gather_hostname(char *hostname_recv_buf) {
  // determine which GPU this rank will use
  char *hostname = comm_hostname();
  MPI_CHECK(MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_HANDLE));
}

void comm_gather_gpuid(int *gpuid_recv_buf) {
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

  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);

  // determine which GPU this MPI rank will use
  char *hostname_recv_buf = (char *)safe_malloc(128*size);

  comm_gather_hostname(hostname_recv_buf);

  gpuid = 0;
  for (int i = 0; i < rank; i++) {
    if (!strncmp(comm_hostname(), &hostname_recv_buf[128*i], 128)) {
      gpuid++;
    }
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    errorQuda("No CUDA devices found");
  }
  if (gpuid >= device_count) {
    char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
    if (enable_mps_env && strcmp(enable_mps_env,"1") == 0) {
      gpuid = gpuid%device_count;
      printf("MPS enabled, rank=%d -> gpu=%d\n", comm_rank(), gpuid);
    } else {
      errorQuda("Too few GPUs available on %s", comm_hostname());
    }
  }

  comm_peer2peer_init(hostname_recv_buf);

  host_free(hostname_recv_buf);

  snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3));

  // if CUDA_VISIBLE_DEVICES is set, we include this information in the topology_string
  char *device_order_env = getenv("CUDA_VISIBLE_DEVICES");
  if (device_order_env) {

    // to ensure we have process consistency define using rank 0
    if (comm_rank() == 0) {
      std::stringstream device_list_raw(device_order_env); // raw input
      std::stringstream device_list;                       // formatted (no commas)

      int device;
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      while (device_list_raw >> device) {
        // check this is a valid policy choice
        if ( device < 0 ) {
          errorQuda("Invalid CUDA_VISIBLE_DEVICE ordinal %d", device);
        }

        device_list << device;
        if (device_list_raw.peek() == ',') device_list_raw.ignore();
      }
      snprintf(topology_string, 128, ",topo=%d%d%d%d,order=%s",
               comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3), device_list.str().c_str());
    }

    comm_broadcast(topology_string, 128);
  } else {
    snprintf(topology_string, 128, ",topo=%d%d%d%d", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3));
  }

}

int comm_rank(void)
{
  return rank;
}


int comm_size(void)
{
  return size;
}


int comm_gpuid(void)
{
  return gpuid;
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


void comm_allreduce(double* data)
{
  double recvbuf;
  MPI_CHECK(MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_HANDLE));
  *data = recvbuf;
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
  double *recvbuf = new double[size];
  MPI_CHECK(MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_HANDLE));
  memcpy(data, recvbuf, size*sizeof(double));
  delete []recvbuf;
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

void comm_abort(int status)
{
#ifdef HOST_DEBUG
  raise(SIGINT);
#endif
  MPI_Abort(MPI_COMM_HANDLE, status);
}

static char partition_override_string[16];

const char* comm_dim_partitioned_string(const int *comm_dim_override)
{
  if (comm_dim_override) {
    char comm[5] = {
      (!comm_dim_partitioned(0) ? '0' : comm_dim_override[0] ? '1' : '0'),
      (!comm_dim_partitioned(1) ? '0' : comm_dim_override[1] ? '1' : '0'),
      (!comm_dim_partitioned(2) ? '0' : comm_dim_override[2] ? '1' : '0'),
      (!comm_dim_partitioned(3) ? '0' : comm_dim_override[3] ? '1' : '0'),
      '\0'};
    strcpy(partition_override_string, ",comm=");
    strcat(partition_override_string, comm);
    return partition_override_string;
  } else {
    return partition_string;
  }
}

const char *comm_dim_topology_string() { return topology_string; }
