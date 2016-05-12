#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <quda_internal.h>
#include <comm_quda.h>


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
static bool peer2peer_enabled[2][4] = { {false,false,false,false},
                                        {false,false,false,false} };
static bool peer2peer_init = false;


void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  int initialized;
  MPI_CHECK( MPI_Initialized(&initialized) );

  if (!initialized) {
    errorQuda("MPI has not been initialized");
  }

  MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
  MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &size) );

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
  char *hostname = comm_hostname();
  char *hostname_recv_buf = (char *)safe_malloc(128*size);
  
  MPI_CHECK( MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_WORLD) );

  gpuid = 0;
  for (int i = 0; i < rank; i++) {
    if (!strncmp(hostname, &hostname_recv_buf[128*i], 128)) {
      gpuid++;
    }
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    errorQuda("No CUDA devices found");
  }
  if (gpuid >= device_count) {
    errorQuda("Too few GPUs available on %s", hostname);
  }

  comm_peer2peer_init(hostname_recv_buf);

  host_free(hostname_recv_buf);
}

void comm_peer2peer_init(const char* hostname_recv_buf)
{
  if (peer2peer_init) return;

  bool disable_peer_to_peer = false;
  char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");
  if (enable_peer_to_peer_env && strcmp(enable_peer_to_peer_env, "0") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling peer-to-peer access\n");
    disable_peer_to_peer = true;
  }

  if (!peer2peer_init && !disable_peer_to_peer) {

    // first check that the local GPU supports UVA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuid);
    if(!prop.unifiedAddressing || prop.computeMode != cudaComputeModeDefault) return;

    comm_set_neighbor_ranks();

    char *hostname = comm_hostname();

    int *gpuid_recv_buf = (int *)safe_malloc(sizeof(int)*size);

    // There are more efficient ways to do the following,
    // but it doesn't really matter since this function should be
    // called just once.
    MPI_CHECK( MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_WORLD) );

    for(int dir=0; dir<2; ++dir){ // forward/backward directions
      for(int dim=0; dim<4; ++dim){
	int neighbor_rank = comm_neighbor_rank(dir,dim);
	if(neighbor_rank == rank) continue;

	// if the neighbors are on the same
	if (!strncmp(hostname, &hostname_recv_buf[128*neighbor_rank], 128)) {
	  int neighbor_gpuid = gpuid_recv_buf[neighbor_rank];
	  int canAccessPeer[2];
	  cudaDeviceCanAccessPeer(&canAccessPeer[0], gpuid, neighbor_gpuid);
	  cudaDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, gpuid);
	  if(canAccessPeer[0]*canAccessPeer[1]){
	    peer2peer_enabled[dir][dim] = true;
	    if (getVerbosity() > QUDA_SILENT)
	      printf("Peer-to-peer enabled for rank %d gpu=%d with neighbor %d gpu=%d dir=%d, dim=%d\n",
		     comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim);
	  }
	} // on the same node
      } // different dimensions - x, y, z, t
    } // different directions - forward/backward

    host_free(gpuid_recv_buf);
  }

  peer2peer_init = true;
  return;
}


bool comm_peer2peer_enabled(int dir, int dim){
  return peer2peer_enabled[dir][dim];
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


/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = comm_rank();
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  MPI_CHECK( MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
  mh->custom = false;

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = rank;
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  MPI_CHECK( MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
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

  int rank = comm_rank_displaced(topo, displacement);
  int tag = comm_rank();
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;

  MPI_CHECK( MPI_Send_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = rank;
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;

  MPI_CHECK( MPI_Recv_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );

  return mh;
}


void comm_free(MsgHandle *mh)
{
  MPI_CHECK(MPI_Request_free(&(mh->request)));
  if (mh->custom) MPI_CHECK(MPI_Type_free(&(mh->datatype)));
  host_free(mh);
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
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  *data = recvbuf;
} 


void comm_allreduce_max(double* data)
{
  double recvbuf;
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) );
  *data = recvbuf;
} 

void comm_allreduce_array(double* data, size_t size)
{
  double *recvbuf = new double[size];
  MPI_CHECK( MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  memcpy(data, recvbuf, size*sizeof(double));
  delete []recvbuf;
}


void comm_allreduce_int(int* data)
{
  int recvbuf;
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );
  *data = recvbuf;
}


/**  broadcast from rank 0 */
void comm_broadcast(void *data, size_t nbytes)
{
  MPI_CHECK( MPI_Bcast(data, (int)nbytes, MPI_BYTE, 0, MPI_COMM_WORLD) );
}


void comm_barrier(void)
{
  MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );
}


void comm_abort(int status)
{
  MPI_Abort(MPI_COMM_WORLD, status) ;
}
