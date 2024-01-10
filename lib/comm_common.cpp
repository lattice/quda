#include <unistd.h> // for gethostname()
#include <assert.h>
#include <limits>

#include <quda_internal.h>
#include <communicator_quda.h>
#include <comm_quda.h>

namespace quda
{

  char *comm_hostname(void)
  {
    static bool cached = false;
    static char hostname[QUDA_MAX_HOSTNAME_STRING];

    if (!cached) {
      gethostname(hostname, QUDA_MAX_HOSTNAME_STRING);
      hostname[QUDA_MAX_HOSTNAME_STRING - 1] = '\0';
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

  /**
   * Send to the "dir" direction in the "dim" dimension
   */
  MsgHandle *comm_declare_send_relative_(const char *func, const char *file, int line, void *buffer, int dim, int dir,
                                         size_t nbytes)
  {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%s called (%s:%d in %s())\n", __func__, file, line, func);

    int disp[QUDA_MAX_DIM] = {0};
    disp[dim] = dir;

    return comm_declare_send_displaced(buffer, disp, nbytes);
  }

  /**
   * Receive from the "dir" direction in the "dim" dimension
   */
  MsgHandle *comm_declare_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                            int dir, size_t nbytes)
  {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%s called (%s:%d in %s())\n", __func__, file, line, func);

    int disp[QUDA_MAX_DIM] = {0};
    disp[dim] = dir;

    return comm_declare_receive_displaced(buffer, disp, nbytes);
  }

  /**
   * Strided send to the "dir" direction in the "dim" dimension
   */
  MsgHandle *comm_declare_strided_send_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                                 int dir, size_t blksize, int nblocks, size_t stride)
  {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%s called (%s:%d in %s())\n", __func__, file, line, func);

    int disp[QUDA_MAX_DIM] = {0};
    disp[dim] = dir;

    return comm_declare_strided_send_displaced(buffer, disp, blksize, nblocks, stride);
  }

  /**
   * Strided receive from the "dir" direction in the "dim" dimension
   */
  MsgHandle *comm_declare_strided_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                                    int dir, size_t blksize, int nblocks, size_t stride)
  {
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%s called (%s:%d in %s())\n", __func__, file, line, func);

    int disp[QUDA_MAX_DIM] = {0};
    disp[dim] = dir;

    return comm_declare_strided_receive_displaced(buffer, disp, blksize, nblocks, stride);
  }

  Topology *comm_create_topology(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data, int my_rank)
  {
    if (ndim > QUDA_MAX_DIM) { errorQuda("ndim exceeds QUDA_MAX_DIM"); }

    Topology *topo = new Topology;

    topo->ndim = ndim;

    int nodes = 1;
    for (int i = 0; i < ndim; i++) {
      topo->dims[i] = dims[i];
      nodes *= dims[i];
    }

    topo->ranks = new int[nodes];
    topo->coords = (int(*)[QUDA_MAX_DIM]) new int[QUDA_MAX_DIM * nodes];

    int x[QUDA_MAX_DIM];
    for (int i = 0; i < QUDA_MAX_DIM; i++) x[i] = 0;

    do {
      int rank = rank_from_coords(x, map_data);
      topo->ranks[index(ndim, dims, x)] = rank;
      for (int i = 0; i < ndim; i++) { topo->coords[rank][i] = x[i]; }
    } while (advance_coords(ndim, dims, x));

    topo->my_rank = my_rank;
    for (int i = 0; i < ndim; i++) { topo->my_coords[i] = topo->coords[my_rank][i]; }

    // initialize the random number generator with a rank-dependent seed and initialized it only once.
    if (comm_gpuid() < 0) { rand_seed = 17 * my_rank + 137; }

    return topo;
  }

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

} // namespace quda
