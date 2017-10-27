#include <unistd.h> // for gethostname()
#include <assert.h>

#include <quda_internal.h>
#include <comm_quda.h>


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


static bool peer2peer_enabled[2][4] = { {false,false,false,false},
                                        {false,false,false,false} };
static bool peer2peer_init = false;
void comm_peer2peer_init(const char* hostname_recv_buf)
{
  if (peer2peer_init) return;

  bool disable_peer_to_peer = false;
  char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");
  if (enable_peer_to_peer_env && strcmp(enable_peer_to_peer_env, "0") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling peer-to-peer access\n");
    disable_peer_to_peer = true;
  }

  // disable peer-to-peer comms in one direction if QUDA_ENABLE_P2P=-1
  // and comm_dim(dim) == 2 (used for perf benchmarking)
  bool disable_peer_to_peer_bidir = false;
  if (enable_peer_to_peer_env && strcmp(enable_peer_to_peer_env, "-1") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling bi-directional peer-to-peer access\n");
    disable_peer_to_peer_bidir = true;
  }

  if (!peer2peer_init && !disable_peer_to_peer) {

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
	  if (canAccessPeer[0]*canAccessPeer[1]) {
	    cudaDeviceGetP2PAttribute(&accessRank[0], cudaDevP2PAttrPerformanceRank, gpuid, neighbor_gpuid);
	    cudaDeviceGetP2PAttribute(&accessRank[1], cudaDevP2PAttrPerformanceRank, neighbor_gpuid, gpuid);
	  }
#endif

	  // enable P2P if we can access the peer or if peer is self
	  if (canAccessPeer[0]*canAccessPeer[1] || gpuid == neighbor_gpuid) {
	    peer2peer_enabled[dir][dim] = true;
	    if (getVerbosity() > QUDA_SILENT) {
	      printf("Peer-to-peer enabled for rank %d (gpu=%d) with neighbor %d (gpu=%d) dir=%d, dim=%d, performance rank = (%d, %d)\n",
		     comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim, accessRank[0], accessRank[1]);
	    }
	  }

	} // on the same node
      } // different dimensions - x, y, z, t
    } // different directions - forward/backward

    host_free(gpuid_recv_buf);
  }

  peer2peer_init = true;

  // set gdr enablement
  if (comm_gdr_enabled()) {
    printfQuda("Enabling GPU-Direct RDMA access\n");
  } else {
    printfQuda("Disabling GPU-Direct RDMA access\n");
  }

  checkCudaErrorNoSync();
  return;
}

bool comm_peer2peer_enabled(int dir, int dim){
  return peer2peer_enabled[dir][dim];
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


/**
 * Send to the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_send_relative_(const char *func, const char *file, int line,
				       void *buffer, int dim, int dir, size_t nbytes)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, buffer);
  if (err != cudaSuccess || attributes.memoryType == cudaMemoryTypeHost) {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = safe_malloc(nbytes);
    try {
      std::copy(static_cast<char*>(buffer), static_cast<char*>(buffer)+nbytes, static_cast<char*>(tmp));
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting");
    }
    if (err != cudaSuccess) cudaGetLastError();
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
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, buffer);
  if (err != cudaSuccess || attributes.memoryType == cudaMemoryTypeHost) {
    // test this memory allocation is ok by filling it
    try {
      std::fill(static_cast<char*>(buffer), static_cast<char*>(buffer)+nbytes, 0);
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir, nbytes);
      errorQuda("aborting");
    }
    if (err != cudaSuccess) cudaGetLastError();
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
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, buffer);
  if (err != cudaSuccess || attributes.memoryType == cudaMemoryTypeHost) {
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
    if (err != cudaSuccess) cudaGetLastError();
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
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, buffer);
  if (err != cudaSuccess || attributes.memoryType == cudaMemoryTypeHost) {
    // test this memory allocation is ok by filling it
    try {
      for (int i=0; i<nblocks; i++)
	std::fill(static_cast<char*>(buffer)+i*stride, static_cast<char*>(buffer)+i*stride+blksize, 0);
    } catch(std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n",
		 file, line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting");
    }
    if (err != cudaSuccess) cudaGetLastError();
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


static int manual_set_partition[QUDA_MAX_DIM] = {0};

void comm_dim_partitioned_set(int dim)
{ 
#ifdef MULTI_GPU
  manual_set_partition[dim] = 1;
#endif
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

bool comm_peer2peer_enabled_global() {
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
  return p2p_global;
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

bool commGlobalReduction() { return globalReduce; }

void commGlobalReductionSet(bool global_reduction) { globalReduce = global_reduction; }

bool commAsyncReduction() { return asyncReduce; }

void commAsyncReductionSet(bool async_reduction) { asyncReduce = async_reduction; }
