#include <qmp.h>

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
static bool peer2peer_enabled[2][4] = { {false,false,false,false},
                                        {false,false,false,false} };
static bool peer2peer_init = false;


// this is a work around (in the absence of C++11) to do a compile
// time check that the size of float and int are the same.  Since we
// are reinterpretting a float as an int, this property is required.
template <typename A, typename B>
inline void static_assert_equal_size()
{
  typedef char sizeof_float_must_equal_sizeof_int[sizeof(A) == sizeof(B) ? 1 : -1];
  (void) sizeof(sizeof_float_must_equal_sizeof_int);
}


void get_hostnames(char *hostname_recv_buf) {
  // determine which GPU this rank will use
  char *hostname = comm_hostname();

  // Abuse reductions to emulate all-gather.  We need to copy the
  // local hostname to all other nodes
  for (int i=0; i<comm_size(); i++) {
    int data[128];
    for (int j=0; j<128; j++) {
      data[j] = (i == comm_rank()) ? hostname[j] : 0;
    }

    static_assert_equal_size<float,int>();
    QMP_sum_float_array(reinterpret_cast<float*>(&data), 128);

    for (int j=0; j<128; j++) {
      hostname_recv_buf[i*128 + j] = data[j];
    }
  }
}


void get_gpuid(int *gpuid_recv_buf) {

  // Abuse reductions to emulate all-gather.  We need to copy the
  // local hostname to all other nodes
  for (int i=0; i<comm_size(); i++) {
    int data = (i == comm_rank()) ? gpuid : 0;

    static_assert_equal_size<float,int>();
    QMP_sum_float_array(reinterpret_cast<float*>(&data), 1);

    gpuid_recv_buf[i] = data;
  }
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
  get_hostnames(hostname_recv_buf);

  gpuid = 0;
  for (int i = 0; i < comm_rank(); i++) {
    if (!strncmp(comm_hostname(), &hostname_recv_buf[128*i], 128)) {
      gpuid++;
    }
  }
  host_free(hostname_recv_buf);

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    errorQuda("No CUDA devices found");
  }

  gpuid = (comm_rank() % device_count);
}


void comm_peer2peer_init()
{
  if (!peer2peer_init) {

    // first check that the local GPU supports UVA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuid);
    if(!prop.unifiedAddressing || prop.computeMode != cudaComputeModeDefault) return;

    comm_set_neighbor_ranks();

    char *hostname = comm_hostname();
    char *hostname_recv_buf = (char *)safe_malloc(128*comm_size());
    int *gpuid_recv_buf = (int *)safe_malloc(sizeof(int)*comm_size());

    get_hostnames(hostname_recv_buf);
    get_gpuid(gpuid_recv_buf);

    for(int dir=0; dir<2; ++dir){ // forward/backward directions
      for(int dim=0; dim<4; ++dim){
	int neighbor_rank = comm_neighbor_rank(dir,dim);
	if(neighbor_rank == comm_rank()) continue;

	// if the neighbors are on the same
	if (!strncmp(hostname, &hostname_recv_buf[128*neighbor_rank], 128)) {
	  int neighbor_gpuid = gpuid_recv_buf[neighbor_rank];
	  int canAccessPeer[2];
	  cudaDeviceCanAccessPeer(&canAccessPeer[0], gpuid, neighbor_gpuid);
	  cudaDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, gpuid);
	  if(canAccessPeer[0]*canAccessPeer[1]){
	    peer2peer_enabled[dir][dim] = true;
	    printf("Peer-to-peer enabled for rank %d with neighbor %d dir=%d, dim=%d\n",
		   comm_rank(), neighbor_rank, dir, dim);
	  }
	} // on the same node
      } // different dimensions - x, y, z, t
    } // different directions - forward/backward

    host_free(hostname_recv_buf);
    host_free(gpuid_recv_buf);

    peer2peer_init = true;
  }
  return;
}


bool comm_peer2peer_enabled(int dir, int dim){
  return peer2peer_enabled[dir][dim];
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


void comm_allreduce_array(double* data, size_t size)
{
  QMP_CHECK( QMP_sum_double_array(data, size) );
}


void comm_allreduce_int(int* data)
{
  QMP_CHECK( QMP_sum_int(data) );
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
  QMP_abort(status);
}
