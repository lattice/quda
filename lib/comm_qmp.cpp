#include <qmp.h>
#include <csignal>
#include <quda_internal.h>
#include <comm_quda.h>

#define QMP_CHECK(qmp_call) do {                     \
  QMP_status_t status = qmp_call;                    \
  if (status != QMP_SUCCESS)                         \
    errorQuda("(QMP) %s", QMP_error_string(status)); \
} while (0)

struct MsgHandle_s {
  QMP_msgmem_t mem;
  QMP_msghandle_t handle;
};

static int gpuid = -1;

static char partition_string[16];
static char topology_string[16];

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
  MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_WORLD);
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
  MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_WORLD);
#else
  // Abuse reductions to emulate all-gather.  We need to copy the
  // local gpu to all other nodes
  for (int i=0; i<comm_size(); i++) {
    int data = (i == comm_rank()) ? gpuid : 0;
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

  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);

  // determine which GPU this rank will use
  char *hostname_recv_buf = (char *)safe_malloc(128*comm_size());
  comm_gather_hostname(hostname_recv_buf);

  gpuid = 0;
  for (int i = 0; i < comm_rank(); i++) {
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
  snprintf(topology_string, 16, ",topo=%d%d%d%d", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3));
}

int comm_rank(void)
{
  return QMP_get_node_number();
}


int comm_size(void)
{
  return QMP_get_number_of_nodes();
}


int comm_gpuid(void)
{
  return gpuid;
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


void comm_free(MsgHandle *mh)
{
  QMP_free_msghandle(mh->handle);
  QMP_free_msgmem(mh->mem);
  host_free(mh);
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


void comm_allreduce(double* data)
{
  QMP_CHECK( QMP_sum_double(data) );
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
  QMP_CHECK( QMP_sum_double_array(data, size) );
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


void comm_abort(int status)
{
#ifdef HOST_DEBUG
  raise(SIGINT);
#endif
  QMP_abort(status);
}

const char* comm_dim_partitioned_string() {
  return partition_string;
}

const char* comm_dim_topology_string() {
  return topology_string;
}
