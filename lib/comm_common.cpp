#include <unistd.h> // for gethostname()

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
  for (int i = 0; i < ndim; i++) {
    x[i] = 0;
  }

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

  for (int i = 0; i < topo->ndim; i++) {
    coords[i] = mod(comm_coords(topo)[i] + displacement[i], comm_dims(topo)[i]);
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
MsgHandle *comm_declare_send_relative(void *buffer, int dim, int dir, size_t nbytes)
{
  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_send_displaced(buffer, disp, nbytes);
}


/**
 * Receive from the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_receive_relative(void *buffer, int dim, int dir, size_t nbytes)
{
  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_receive_displaced(buffer, disp, nbytes);
}


/**
 * Strided send to the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_strided_send_relative(void *buffer, int dim, int dir, 
					      size_t blksize, int nblocks, size_t stride)
{
  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_strided_send_displaced(buffer, disp, blksize, nblocks, stride);
}


/**
 * Strided receive from the "dir" direction in the "dim" dimension
 */
MsgHandle *comm_declare_strided_receive_relative(void *buffer, int dim, int dir, 
						 size_t blksize, int nblocks, size_t stride)
{
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
  manual_set_partition[dim] = 1;
}


int comm_dim_partitioned(int dim)
{
  return (manual_set_partition[dim] || (comm_dim(dim) > 1));
}
