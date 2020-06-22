#include <unistd.h> // for gethostname()
#include <assert.h>

#include <quda_internal.h>
#include <comm_quda.h>
#include <csignal>

#ifdef QUDA_BACKWARDSCPP
#include "backward.hpp"
namespace backward
{
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
};

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

void set_rand_seed(unsigned long int seed){
  rand_seed = seed;
}

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
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = safe_malloc(nbytes);
    try {
      std::copy(static_cast<char *>(buffer), static_cast<char *>(buffer) + nbytes, static_cast<char *>(tmp));
    } catch (std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir,
                 nbytes);
      errorQuda("aborting");
    }
    host_free(tmp);
  } else {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = device_malloc(nbytes);
    cudaError_t err = cudaMemcpy(tmp, buffer, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir,
                 nbytes);
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
MsgHandle *comm_declare_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim, int dir,
                                          size_t nbytes)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by filling it
    try {
      std::fill(static_cast<char *>(buffer), static_cast<char *>(buffer) + nbytes, 0);
    } catch (std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir,
                 nbytes);
      errorQuda("aborting");
    }
  } else {
    // test this memory allocation is ok by doing a memset
    cudaError_t err = cudaMemset(buffer, 0, nbytes);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, nbytes=%zu)\n", file, line, func, dim, dir,
                 nbytes);
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
MsgHandle *comm_declare_strided_send_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                               int dir, size_t blksize, int nblocks, size_t stride)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = safe_malloc(blksize * nblocks);
    try {
      for (int i = 0; i < nblocks; i++)
        std::copy(static_cast<char *>(buffer) + i * stride, static_cast<char *>(buffer) + i * stride + blksize,
                  static_cast<char *>(tmp));
    } catch (std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n", file,
                 line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting");
    }
    host_free(tmp);
  } else {
    // test this memory allocation is ok by doing a memcpy from it
    void *tmp = device_malloc(blksize * nblocks);
    cudaError_t err = cudaMemcpy2D(tmp, blksize, buffer, stride, blksize, nblocks, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n", file,
                 line, func, dim, dir, blksize, nblocks, stride);
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
MsgHandle *comm_declare_strided_receive_relative_(const char *func, const char *file, int line, void *buffer, int dim,
                                                  int dir, size_t blksize, int nblocks, size_t stride)
{
#ifdef HOST_DEBUG
  checkCudaError(); // check and clear error state first

  if (isHost(buffer)) {
    // test this memory allocation is ok by filling it
    try {
      for (int i = 0; i < nblocks; i++)
        std::fill(static_cast<char *>(buffer) + i * stride, static_cast<char *>(buffer) + i * stride + blksize, 0);
    } catch (std::exception &e) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n", file,
                 line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting");
    }
  } else {
    // test this memory allocation is ok by doing a memset
    cudaError_t err = cudaMemset2D(buffer, stride, 0, blksize, nblocks);
    if (err != cudaSuccess) {
      printfQuda("ERROR: buffer failed (%s:%d in %s(), dim=%d, dir=%d, blksize=%zu nblocks=%d stride=%zu)\n", file,
                 line, func, dim, dir, blksize, nblocks, stride);
      errorQuda("aborting with error %s", cudaGetErrorString(err));
    }
  }
#endif

  int disp[QUDA_MAX_DIM] = {0};
  disp[dim] = dir;

  return comm_declare_strided_receive_displaced(buffer, disp, blksize, nblocks, stride);
}

static char partition_string[16];          /** string that contains the job partitioning */
static char partition_override_string[16]; /** string that contains any overridden partitioning */

void comm_dim_partitioned_reset();

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
