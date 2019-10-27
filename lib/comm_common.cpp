#include <unistd.h> // for gethostname()
#include <assert.h>

#include <quda_internal.h>
#include <comm_quda.h>
#include <csignal>

#ifdef QUDA_BACKWARDSCPP
#include "backward.hpp"
namespace backward {
  static backward::SignalHandling sh;
} // namespace backward
#endif 

struct Topology_s {
  int ndim;
  int dims[QUDA_MAX_DIM];
  int *ranks;
  int (*coords)[QUDA_MAX_DIM];
  int my_rank;
  int my_coords[QUDA_MAX_DIM];
  // It might be worth adding communicators to allow for efficient reductions:
  //   #if defined(MPI_COMMS)
  //     MPI_Comm comm;
  //   #elif defined(QMP_COMMS)
  //     QMP_communicator_t comm; // currently only supported by qmp-2.4.0-alpha
  //   #endif
};


/**
 * Utility function for indexing into Topology::ranks[]
 *
 * @param ndim  Number of grid dimensions in the network topology
 * @param dims  Array of grid dimensions
 * @param x     Node coordinates
 * @return      Linearized index cooresponding to the node coordinates
 */
static inline int index(int ndim, const int *dims, const int *x)
{
  int idx = x[0];
  for (int i = 1; i < ndim; i++) {
    idx = dims[i]*idx + x[i];
  }
  return idx;
}


static inline bool advance_coords(int ndim, const int *dims, int *x)
{
  bool valid = false;
  for (int i = ndim-1; i >= 0; i--) {
    if (x[i] < dims[i]-1) {
      x[i]++;
      valid = true;
      break;
    } else {
      x[i] = 0;
    }
  }
  return valid;
}


char *comm_hostname(void)
{
  static bool cached = false;
  static char hostname[128];

  if (!cached) {
    gethostname(hostname, 128);
    hostname[127] = '\0';
    cached = true;
  }

  return hostname;
}


static unsigned long int rand_seed = 137;

/**
 * We provide our own random number generator to avoid re-seeding
 * rand(), which might also be used by the calling application.  This
 * is a clone of rand48(), provided by stdlib.h on UNIX.
 *
 * @return a random double in the interval [0,1)
 */
double comm_drand(void)
{
  const double twoneg48 = 0.35527136788005009e-14;
  const unsigned long int m = 25214903917, a = 11, mask = 281474976710655;
  rand_seed = (m * rand_seed + a) & mask;
  return (twoneg48 * rand_seed);
}


// QudaCommsMap is declared in quda.h:
//   typedef int (*QudaCommsMap)(const int *coords, void *fdata);

Topology *comm_create_topology(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  if (ndim > QUDA_MAX_DIM) {
    errorQuda("ndim exceeds QUDA_MAX_DIM");
  }

  Topology *topo = (Topology *) safe_malloc(sizeof(Topology));

  topo->ndim = ndim;

  int nodes = 1;
  for (int i=0; i<ndim; i++) {
    topo->dims[i] = dims[i];
    nodes *= dims[i];
  }

  topo->ranks = (int *) safe_malloc(nodes*sizeof(int));
  topo->coords = (int (*)[QUDA_MAX_DIM]) safe_malloc(nodes*sizeof(int[QUDA_MAX_DIM]));

  int x[QUDA_MAX_DIM];
  for (int i = 0; i < QUDA_MAX_DIM; i++) x[i] = 0;

  do {
    int rank = rank_from_coords(x, map_data);
    topo->ranks[index(ndim, dims, x)] = rank;
    for (int i=0; i<ndim; i++) {
      topo->coords[rank][i] = x[i];
    }
  } while (advance_coords(ndim, dims, x));

  int my_rank = comm_rank();
  topo->my_rank = my_rank;
  for (int i = 0; i < ndim; i++) {
    topo->my_coords[i] = topo->coords[my_rank][i];
  }

  // initialize the random number generator with a rank-dependent seed
  rand_seed = 17*my_rank + 137;

  return topo;
}


void comm_destroy_topology(Topology *topo)
{
  host_free(topo->ranks);
  host_free(topo->coords);
  host_free(topo);
}

static int gpuid = -1;

int comm_gpuid(void) { return gpuid; }

static bool peer2peer_enabled[2][4] = { {false,false,false,false},
                                        {false,false,false,false} };
static bool peer2peer_init = false;

static bool intranode_enabled[2][4] = { {false,false,false,false},
					{false,false,false,false} };

/** this records whether there is any peer-2-peer capability
    (regardless whether it is enabled or not) */
static bool peer2peer_present = false;

/** by default enable both copy engines and load/store access */
static int enable_peer_to_peer = 3; 


void comm_peer2peer_init(const char* hostname_recv_buf)
{
  if (peer2peer_init) return;

  // set gdr enablement
  if (comm_gdr_enabled()) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling GPU-Direct RDMA access\n");
    comm_gdr_blacklist(); // set GDR blacklist
    // by default, if GDR is enabled we disable non-p2p policies to
    // prevent possible conflict between MPI and QUDA opening the same
    // IPC memory handles when using CUDA-aware MPI
    enable_peer_to_peer += 4;
  } else {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling GPU-Direct RDMA access\n");
  }

  char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");

  // disable peer-to-peer comms in one direction if QUDA_ENABLE_P2P=-1
  // and comm_dim(dim) == 2 (used for perf benchmarking)
  bool disable_peer_to_peer_bidir = false;

  if (enable_peer_to_peer_env) {
    enable_peer_to_peer = atoi(enable_peer_to_peer_env);

    switch ( std::abs(enable_peer_to_peer) ) {
    case 0: if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling peer-to-peer access\n"); break;
    case 1: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer copy engine access (disabling direct load/store)\n"); break;
    case 2: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer direct load/store access (disabling copy engines)\n"); break;
    case 3: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer copy engine and direct load/store access\n"); break;
    case 5: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer copy engine access (disabling direct load/store and non-p2p policies)\n"); break;
    case 6: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer direct load/store access (disabling copy engines and non-p2p policies)\n"); break;
    case 7: if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer copy engine and direct load/store access (disabling non-p2p policies)\n"); break;
    default: errorQuda("Unexpected value QUDA_ENABLE_P2P=%d\n", enable_peer_to_peer);
    }

    if (enable_peer_to_peer < 0) { // only values -1, -2, -3 can make it here
      if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling bi-directional peer-to-peer access\n");
      disable_peer_to_peer_bidir = true;
    }

    enable_peer_to_peer = abs(enable_peer_to_peer);

  } else { // !enable_peer_to_peer_env
    if (getVerbosity() > QUDA_SILENT) printfQuda("Enabling peer-to-peer copy engine and direct load/store access\n");
  }

  if (!peer2peer_init && enable_peer_to_peer) {

    // first check that the local GPU supports UVA
    const int gpuid = comm_gpuid();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuid);
    if(!prop.unifiedAddressing) return;

    comm_set_neighbor_ranks();

    char *hostname = comm_hostname();
    int *gpuid_recv_buf = (int *)safe_malloc(sizeof(int)*comm_size());

    comm_gather_gpuid(gpuid_recv_buf);

    for(int dir=0; dir<2; ++dir){ // forward/backward directions
      for(int dim=0; dim<4; ++dim){
	int neighbor_rank = comm_neighbor_rank(dir,dim);
	if(neighbor_rank == comm_rank()) continue;

	// disable peer-to-peer comms in one direction
	if ( ((comm_rank() > neighbor_rank && dir == 0) || (comm_rank() < neighbor_rank && dir == 1)) &&
	     disable_peer_to_peer_bidir && comm_dim(dim) == 2 ) continue;

	// if the neighbors are on the same
	if (!strncmp(hostname, &hostname_recv_buf[128*neighbor_rank], 128)) {
	  int neighbor_gpuid = gpuid_recv_buf[neighbor_rank];
	  int canAccessPeer[2];
	  cudaDeviceCanAccessPeer(&canAccessPeer[0], gpuid, neighbor_gpuid);
	  cudaDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, gpuid);

	  int accessRank[2] = { };
#if CUDA_VERSION >= 8000  // this was introduced with CUDA 8
	  if (canAccessPeer[0]*canAccessPeer[1] != 0) {
	    cudaDeviceGetP2PAttribute(&accessRank[0], cudaDevP2PAttrPerformanceRank, gpuid, neighbor_gpuid);
	    cudaDeviceGetP2PAttribute(&accessRank[1], cudaDevP2PAttrPerformanceRank, neighbor_gpuid, gpuid);
	  }
#endif

	  // enable P2P if we can access the peer or if peer is self
	  if (canAccessPeer[0]*canAccessPeer[1] != 0 || gpuid == neighbor_gpuid) {
	    peer2peer_enabled[dir][dim] = true;
	    if (getVerbosity() > QUDA_SILENT) {
	      printf("Peer-to-peer enabled for rank %d (gpu=%d) with neighbor %d (gpu=%d) dir=%d, dim=%d, performance rank = (%d, %d)\n",
		     comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim, accessRank[0], accessRank[1]);
	    }
	  } else {
	    intranode_enabled[dir][dim] = true;
	    if (getVerbosity() > QUDA_SILENT) {
	      printf("Intra-node (non peer-to-peer) enabled for rank %d (gpu=%d) with neighbor %d (gpu=%d) dir=%d, dim=%d\n",
		     comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim);
	    }
	  }

	} // on the same node
      } // different dimensions - x, y, z, t
    } // different directions - forward/backward

    host_free(gpuid_recv_buf);
  }

  peer2peer_init = true;

  comm_barrier();

  peer2peer_present = comm_peer2peer_enabled_global();

  checkCudaErrorNoSync();
  return;
}

bool comm_peer2peer_present() { return peer2peer_present; }

static bool enable_p2p = true;

bool comm_peer2peer_enabled(int dir, int dim){
  return enable_p2p ? peer2peer_enabled[dir][dim] : false;
}

int comm_peer2peer_enabled_global() {
  if (!enable_p2p) return false;

  static bool init = false;
  static bool p2p_global = false;

  if (!init) {
    int p2p = 0;
    for (int dim=0; dim<4; dim++)
      for (int dir=0; dir<2; dir++)
	p2p += (int)comm_peer2peer_enabled(dir,dim);

    comm_allreduce_int(&p2p);
    init = true;
    p2p_global = p2p > 0 ? true : false;
  }
  return p2p_global * enable_peer_to_peer;
}

void comm_enable_peer2peer(bool enable) {
  enable_p2p = enable;
}

static bool enable_intranode = true;

bool comm_intranode_enabled(int dir, int dim){
  return enable_intranode ? intranode_enabled[dir][dim] : false;
}

void comm_enable_intranode(bool enable) {
  enable_intranode = enable;
}

int comm_ndim(const Topology *topo)
{
  return topo->ndim;
}


const int *comm_dims(const Topology *topo)
{
  return topo->dims;
}


const int *comm_coords(const Topology *topo)
{
  return topo->my_coords;
}


const int *comm_coords_from_rank(const Topology *topo, int rank)
{
  return topo->coords[rank];
}


int comm_rank_from_coords(const Topology *topo, const int *coords)
{
  return topo->ranks[index(topo->ndim, topo->dims, coords)];
}


static inline int mod(int a, int b)
{
  return ((a % b) + b) % b;
}

int comm_rank_displaced(const Topology *topo, const int displacement[])
{
  int coords[QUDA_MAX_DIM];

  for (int i = 0; i < QUDA_MAX_DIM; i++) {
    coords[i] = (i < topo->ndim) ? 
      mod(comm_coords(topo)[i] + displacement[i], comm_dims(topo)[i]) : 0;
  }

  return comm_rank_from_coords(topo, coords);
}


// FIXME: The following routines rely on a "default" topology.
// They should probably be reworked or eliminated eventually.

Topology *default_topo = NULL;

void comm_set_default_topology(Topology *topo)
{
  default_topo = topo;
}


Topology *comm_default_topology(void)
{
  if (!default_topo) {
    errorQuda("Default topology has not been declared");
  }
  return default_topo;
}

static int neighbor_rank[2][4] = { {-1,-1,-1,-1},
                                          {-1,-1,-1,-1} };

static bool neighbors_cached = false;

void comm_set_neighbor_ranks(Topology *topo){

  if(neighbors_cached) return;

  Topology *topology = topo ? topo : default_topo; // use default topology if topo is NULL
  if(!topology){
    errorQuda("Topology not specified");
    return;
  }
     
  for(int d=0; d<4; ++d){
    int pos_displacement[4] = {0,0,0,0};
    int neg_displacement[4] = {0,0,0,0};
    pos_displacement[d] = +1;
    neg_displacement[d] = -1;
    neighbor_rank[0][d] = comm_rank_displaced(topology, neg_displacement);
    neighbor_rank[1][d] = comm_rank_displaced(topology, pos_displacement);
  }
  neighbors_cached = true;
  return;
}

int comm_neighbor_rank(int dir, int dim){
  if(!neighbors_cached){
    comm_set_neighbor_ranks();
  }
  return neighbor_rank[dir][dim];
}


int comm_dim(int dim)
{
  Topology *topo = comm_default_topology();
  return comm_dims(topo)[dim];
}


int comm_coord(int dim)
{
  Topology *topo = comm_default_topology();
  return comm_coords(topo)[dim];
}

inline bool isHost(const void *buffer)
{
  CUmemorytype memType;
  void *attrdata[] = {(void *)&memType};
  CUpointer_attribute attributes[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
  CUresult err = cuPointerGetAttributes(1, attributes, attrdata, (CUdeviceptr)buffer);
  if (err != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorName(err, &str);
    errorQuda("cuPointerGetAttributes returned error %s", str);
  }

  switch (memType) {
  case CU_MEMORYTYPE_DEVICE: return false;
  case CU_MEMORYTYPE_ARRAY: errorQuda("Using array memory for communications buffer is not supported");
  case CU_MEMORYTYPE_UNIFIED: errorQuda("Using unified memory for communications buffer is not supported");
  case CU_MEMORYTYPE_HOST:
  default: // memory not allocated by CUDA allocaters will default to being host memory
    return true;
  }
}

/**
 * Send to the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_send_relative_(const char *func, const char *file, int line,
				       void *buffer, int dim, int dir, size_t nbytes)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = safe_malloc(nbytes);
    try {
      std::copy(static_cast<char*>(buffer), static_cast<char*>(buffer)+nbytes, static_cast<char*>(tmp));
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting");
    }
    host_free(tmp);
  } else {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = device_malloc(nbytes);
    cudaError_t err = cudaMemcpy(tmp, buffer, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting with error %s", cudaGetErrorString(err));
    }
    device_free(tmp);
  }
#endif

  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_send_displaced(buffer, disp, nbytes);
}

/**
 * Receive from the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_receive_relative_(const char *func, const char *file, int line,
					  void *buffer, int dim, int dir, size_t nbytes)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by filling it
    try {
      std::fill(static_cast<char*>(buffer), static_cast<char*>(buffer)+nbytes, 0);
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting");
    }
  } else {
    // test this memory allocation is ok by doing a memset
    cudaError_t err = cudaMemset(buffer, 0, nbytes);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting with error %s", cudaGetErrorString(err));
    }
  }
#endif

  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_receive_displaced(buffer, disp, nbytes);
}

/**
 * Strided send to the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_strided_send_relative_(const char *func, const char *file, int line,
					       void *buffer, int dim, int dir, size_t blksize, int nblocks, size_t stride)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = safe_malloc(blksize*nblocks);
    try {
      for (int i=0; i<nblocks; i++)
	std::copy(static_cast<char*>(buffer)+i*stride, static_cast<char*>(buffer)+i*stride+blksize, static_cast<char*>(tmp));
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n",
		 file, line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting");
    }
    host_free(tmp);
  } else {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = device_malloc(blksize*nblocks);
    cudaError_t err = cudaMemcpy2D(tmp, blksize, buffer, stride, blksize, nblocks, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n",
		 file, line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting with error %s", cudaGetErrorString(err));
    }
    device_free(tmp);
  }
#endif

  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_strided_send_displaced(buffer, disp, blksize, nblocks, stride);
}


/**
 * Strided receive from the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_strided_receive_relative_(const char *func, const char *file, int line,
						  void *buffer, int dim, int dir, size_t blksize, int nblocks, size_t stride)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by filling it
    try {
      for (int i=0; i<nblocks; i++)
	std::fill(static_cast<char*>(buffer)+i*stride, static_cast<char*>(buffer)+i*stride+blksize, 0);
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n",
		 file, line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting");
    }
  } else {
    // test this memory allocation is ok by doing a memset
    cudaError_t err = cudaMemset2D(buffer, stride, 0, blksize, nblocks);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n",
		 file, line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting with error %s", cudaGetErrorString(err));
    }
  }
#endif

  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_strided_receive_displaced(buffer, disp, blksize, nblocks, stride);
}

void comm_finalize(void)
{
  Topology *topo = comm_default_topology();
  comm_destroy_topology(topo);
  comm_set_default_topology(NULL);
}

static char partition_string[16];          /** string that contains the job partitioning */
static char topology_string[128];          /** string that contains the job topology */
static char partition_override_string[16]; /** string that contains any overridden partitioning */

static int manual_set_partition[QUDA_MAX_DIM] = {0};

void comm_dim_partitioned_set(int dim)
{ 
#ifdef MULTI_GPU
  manual_set_partition[dim] = 1;
#endif

  snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
           comm_dim_partitioned(2), comm_dim_partitioned(3));
}

void comm_dim_partitioned_reset(){
  for (int i = 0; i < QUDA_MAX_DIM; i++) manual_set_partition[i] = 0;

  snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
           comm_dim_partitioned(2), comm_dim_partitioned(3));
}

int comm_dim_partitioned(int dim)
{
  return (manual_set_partition[dim] || (comm_dim(dim) > 1));
}

int comm_partitioned()
{
  int partitioned = 0;
  for (int i=0; i<4; i++) {
    partitioned = partitioned || comm_dim_partitioned(i);
  }
  return partitioned;
}

bool comm_gdr_enabled() {
  static bool gdr_enabled = false;
#ifdef MULTI_GPU
  static bool gdr_init = false;

  if (!gdr_init) {
    char *enable_gdr_env = getenv("QUDA_ENABLE_GDR");
    if (enable_gdr_env && strcmp(enable_gdr_env, "1") == 0) {
      gdr_enabled = true;
    }
    gdr_init = true;
  }
#endif
  return gdr_enabled;
}

bool comm_gdr_blacklist() {
  static bool blacklist = false;
  static bool blacklist_init = false;

  if (!blacklist_init) {
    char *blacklist_env = getenv("QUDA_ENABLE_GDR_BLACKLIST");

    if (blacklist_env) { // set the policies to tune for explicitly
      std::stringstream blacklist_list(blacklist_env);

      int device_count;
      cudaGetDeviceCount(&device_count);

      int excluded_device;
      while (blacklist_list >> excluded_device) {
	// check this is a valid device
	if ( excluded_device < 0 || excluded_device >= device_count ) {
	  errorQuda("Cannot blacklist invalid GPU device ordinal %d", excluded_device);
	}

	if (blacklist_list.peek() == ',') blacklist_list.ignore();
	if (excluded_device == comm_gpuid()) blacklist = true;
      }
      comm_barrier();
      if (getVerbosity() > QUDA_SILENT && blacklist) printf("Blacklisting GPU-Direct RDMA for rank %d (GPU %d)\n", comm_rank(), comm_gpuid());
    }
    blacklist_init = true;

  }

  return blacklist;
}

static bool deterministic_reduce = false;

void comm_init_common(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);

  // determine which GPU this rank will use
  char *hostname_recv_buf = (char *)safe_malloc(128 * comm_size());
  comm_gather_hostname(hostname_recv_buf);

  gpuid = 0;
  for (int i = 0; i < comm_rank(); i++) {
    if (!strncmp(comm_hostname(), &hostname_recv_buf[128 * i], 128)) { gpuid++; }
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) { errorQuda("No CUDA devices found"); }
  if (gpuid >= device_count) {
    char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
    if (enable_mps_env && strcmp(enable_mps_env, "1") == 0) {
      gpuid = gpuid % device_count;
      printf("MPS enabled, rank=%d -> gpu=%d\n", comm_rank(), gpuid);
    } else {
      errorQuda("Too few GPUs available on %s", comm_hostname());
    }
  }

  comm_peer2peer_init(hostname_recv_buf);

  host_free(hostname_recv_buf);

  char *enable_reduce_env = getenv("QUDA_DETERMINISTIC_REDUCE");
  if (enable_reduce_env && strcmp(enable_reduce_env, "1") == 0) { deterministic_reduce = true; }

  snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
           comm_dim_partitioned(2), comm_dim_partitioned(3));

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
        if (device < 0) { errorQuda("Invalid CUDA_VISIBLE_DEVICE ordinal %d", device); }

        device_list << device;
        if (device_list_raw.peek() == ',') device_list_raw.ignore();
      }
      snprintf(topology_string, 128, ",topo=%d%d%d%d,order=%s", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3),
               device_list.str().c_str());
    }

    comm_broadcast(topology_string, 128);
  } else {
    snprintf(topology_string, 128, ",topo=%d%d%d%d", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3));
  }
}

const char *comm_config_string()
{
  static char config_string[16];
  static bool config_init = false;

  if (!config_init) {
    strcpy(config_string, ",p2p=");
    strcat(config_string, std::to_string(comm_peer2peer_enabled_global()).c_str());
    strcat(config_string, ",gdr=");
    strcat(config_string, std::to_string(comm_gdr_enabled()).c_str());
    config_init = true;
  }

  return config_string;
}

const char *comm_dim_partitioned_string(const int *comm_dim_override)
{
  if (comm_dim_override) {
    char comm[5] = {(!comm_dim_partitioned(0) ? '0' : comm_dim_override[0] ? '1' : '0'),
                    (!comm_dim_partitioned(1) ? '0' : comm_dim_override[1] ? '1' : '0'),
                    (!comm_dim_partitioned(2) ? '0' : comm_dim_override[2] ? '1' : '0'),
                    (!comm_dim_partitioned(3) ? '0' : comm_dim_override[3] ? '1' : '0'), '\0'};
    strcpy(partition_override_string, ",comm=");
    strcat(partition_override_string, comm);
    return partition_override_string;
  } else {
    return partition_string;
  }
}

const char *comm_dim_topology_string() { return topology_string; }

bool comm_deterministic_reduce() { return deterministic_reduce; }

static bool globalReduce = true;
static bool asyncReduce = false;

void reduceMaxDouble(double &max) { comm_allreduce_max(&max); }

void reduceDouble(double &sum) { if (globalReduce) comm_allreduce(&sum); }

void reduceDoubleArray(double *sum, const int len)
{ if (globalReduce) comm_allreduce_array(sum, len); }

int commDim(int dir) { return comm_dim(dir); }

int commCoords(int dir) { return comm_coord(dir); }

int commDimPartitioned(int dir){ return comm_dim_partitioned(dir);}

void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir);}

void commDimPartitionedReset(){ comm_dim_partitioned_reset();}

bool commGlobalReduction() { return globalReduce; }

void commGlobalReductionSet(bool global_reduction) { globalReduce = global_reduction; }

bool commAsyncReduction() { return asyncReduce; }

void commAsyncReductionSet(bool async_reduction) { asyncReduce = async_reduction; }

void comm_abort(int status)
{
#ifdef HOST_DEBUG
  raise(SIGABRT);
#endif
#ifdef QUDA_BACKWARDSCPP
  backward::StackTrace st; 
  st.load_here(32);
  backward::Printer p; 
  p.print(st, getOutputFile());
#endif
  comm_abort_(status);
}
