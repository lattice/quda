#pragma once

#include <unistd.h> // for gethostname()
#include <cassert>
#include <csignal>
#include <limits>
#include <stack>
#include <algorithm>
#include <numeric>

#include <quda_internal.h>
#include <comm_quda.h>
#include <color_spinor_field.h>
#include <field_cache.h>
#include <comm_key.h>
#include <float_vector.h>

#if defined(MPI_COMMS) || defined(QMP_COMMS)
#include <mpi.h>
#endif

#if defined(QMP_COMMS)
#include <qmp.h>
#endif

#ifdef QUDA_BACKWARDSCPP
#include "backward.hpp"
namespace backward
{
  static backward::SignalHandling sh;
} // namespace backward
#endif

namespace quda
{

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

  static const int max_displacement = 4;

  inline int lex_rank_from_coords_dim_t(const int *coords, void *fdata)
  {
    int *dims = reinterpret_cast<int *>(fdata);
    int rank = coords[0];
    for (int i = 1; i < 4; i++) { rank = dims[i] * rank + coords[i]; }
    return rank;
  }

  inline int lex_rank_from_coords_dim_x(const int *coords, void *fdata)
  {
    int *dims = reinterpret_cast<int *>(fdata);
    int rank = coords[3];
    for (int i = 2; i >= 0; i--) { rank = dims[i] * rank + coords[i]; }
    return rank;
  }

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
    for (int i = 1; i < ndim; i++) { idx = dims[i] * idx + x[i]; }
    return idx;
  }

  static inline bool advance_coords(int ndim, const int *dims, int *x)
  {
    bool valid = false;
    for (int i = ndim - 1; i >= 0; i--) {
      if (x[i] < dims[i] - 1) {
        x[i]++;
        valid = true;
        break;
      } else {
        x[i] = 0;
      }
    }
    return valid;
  }

  // QudaCommsMap is declared in quda.h:
  //   typedef int (*QudaCommsMap)(const int *coords, void *fdata);
  Topology *comm_create_topology(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data, int my_rank);

  inline void comm_destroy_topology(Topology *topo)
  {
    delete[] topo->ranks;
    delete[] topo->coords;
    delete topo;
  }

  inline int comm_ndim(const Topology *topo) { return topo->ndim; }

  inline const int *comm_dims(const Topology *topo) { return topo->dims; }

  inline const int *comm_coords(const Topology *topo) { return topo->my_coords; }

  inline const int *comm_coords_from_rank(const Topology *topo, int rank) { return topo->coords[rank]; }

  inline int comm_rank_from_coords(const Topology *topo, const int *coords)
  {
    return topo->ranks[index(topo->ndim, topo->dims, coords)];
  }

  static inline int mod(int a, int b) { return ((a % b) + b) % b; }

  inline int comm_rank_displaced(const Topology *topo, const int displacement[])
  {
    int coords[QUDA_MAX_DIM];

    for (int i = 0; i < QUDA_MAX_DIM; i++) {
      coords[i] = (i < topo->ndim) ? mod(comm_coords(topo)[i] + displacement[i], comm_dims(topo)[i]) : 0;
    }

    return comm_rank_from_coords(topo, coords);
  }

  inline void check_displacement(const int displacement[], int ndim)
  {
    for (int i = 0; i < ndim; i++) {
      if (abs(displacement[i]) > max_displacement) {
        errorQuda("Requested displacement[%d] = %d is greater than maximum allowed", i, displacement[i]);
      }
    }
  }

  struct Communicator {

    /**
      The gpuid is static, and it's set when the default communicator is initialized.
    */
    static int gpuid;
    static int comm_gpuid() { return gpuid; }

    /**
      Whether or not the MPI_COMM_HANDLE is created by user, in which case we should not free it.
    */
    bool user_set_comm_handle;

    bool peer2peer_enabled[2][4] = {{false, false, false, false}, {false, false, false, false}};
    bool peer2peer_init = false;

    bool intranode_enabled[2][4] = {{false, false, false, false}, {false, false, false, false}};

    /** this records whether there is any peer-2-peer capability
        (regardless whether it is enabled or not) */
    bool peer2peer_present = false;

    /** by default enable both copy engines and load/store access */
    int enable_peer_to_peer = 3;

    /** sets whether we cap which peers can use peer-to-peer */
    int enable_p2p_max_access_rank = std::numeric_limits<int>::max();

    void comm_peer2peer_init(const char *hostname_recv_buf)
    {
      if (peer2peer_init) return;

      // set gdr enablement
      if (comm_gdr_enabled()) {
        if (getVerbosity() > QUDA_SILENT && rank == 0) printf("Enabling GPU-Direct RDMA access\n");
        comm_gdr_blacklist(); // set GDR blacklist
        // by default, if GDR is enabled we disable non-p2p policies to
        // prevent possible conflict between MPI and QUDA opening the same
        // IPC memory handles when using CUDA-aware MPI
        enable_peer_to_peer += 4;
      } else {
        if (getVerbosity() > QUDA_SILENT && rank == 0) printf("Disabling GPU-Direct RDMA access\n");
      }

      char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");

      // disable peer-to-peer comms in one direction if QUDA_ENABLE_P2P=-1
      // and comm_dim(dim) == 2 (used for perf benchmarking)
      bool disable_peer_to_peer_bidir = false;

      if (enable_peer_to_peer_env) {
        enable_peer_to_peer = atoi(enable_peer_to_peer_env);

        switch (std::abs(enable_peer_to_peer)) {
        case 0:
          if (getVerbosity() > QUDA_SILENT && rank == 0) printf("Disabling peer-to-peer access\n");
          break;
        case 1:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer copy engine access (disabling direct load/store)\n");
          break;
        case 2:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer direct load/store access (disabling copy engines)\n");
          break;
        case 3:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer copy engine and direct load/store access\n");
          break;
        case 5:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer copy engine access (disabling direct load/store and non-p2p policies)\n");
          break;
        case 6:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer direct load/store access (disabling copy engines and non-p2p policies)\n");
          break;
        case 7:
          if (getVerbosity() > QUDA_SILENT && rank == 0)
            printf("Enabling peer-to-peer copy engine and direct load/store access (disabling non-p2p policies)\n");
          break;
        default: errorQuda("Unexpected value QUDA_ENABLE_P2P=%d\n", enable_peer_to_peer);
        }

        if (enable_peer_to_peer < 0) { // only values -1, -2, -3 can make it here
          if (getVerbosity() > QUDA_SILENT && rank == 0) printf("Disabling bi-directional peer-to-peer access\n");
          disable_peer_to_peer_bidir = true;
        }

        enable_peer_to_peer = abs(enable_peer_to_peer);

      } else { // !enable_peer_to_peer_env
        if (getVerbosity() > QUDA_SILENT && rank == 0)
          printf("Enabling peer-to-peer copy engine and direct load/store access\n");
      }

      if (!peer2peer_init && enable_peer_to_peer) {

        // set whether we are limiting p2p enablement
        char *enable_p2p_max_access_rank_env = getenv("QUDA_ENABLE_P2P_MAX_ACCESS_RANK");
        if (enable_p2p_max_access_rank_env) {
          enable_p2p_max_access_rank = atoi(enable_p2p_max_access_rank_env);
          if (enable_p2p_max_access_rank < 0)
            errorQuda("Invalid QUDA_ENABLE_P2P_MAX_ACCESS_RANK=%d\n", enable_p2p_max_access_rank);
          if (getVerbosity() > QUDA_SILENT)
            printfQuda("Limiting peer-to-peer communication to a maximum access rank of %d (lower ranks have higher "
                       "bandwidth)\n",
                       enable_p2p_max_access_rank);
        }

        const int gpuid = comm_gpuid();

        comm_set_neighbor_ranks();

        char *hostname = comm_hostname();
        int *gpuid_recv_buf = (int *)safe_malloc(sizeof(int) * comm_size());

        comm_gather_gpuid(gpuid_recv_buf);

        for (int dir = 0; dir < 2; ++dir) { // forward/backward directions
          for (int dim = 0; dim < 4; ++dim) {
            int neighbor_rank = comm_neighbor_rank(dir, dim);
            if (neighbor_rank == comm_rank()) continue;

            // disable peer-to-peer comms in one direction
            if (((comm_rank() > neighbor_rank && dir == 0) || (comm_rank() < neighbor_rank && dir == 1))
                && disable_peer_to_peer_bidir && comm_dim(dim) == 2)
              continue;

            // if the neighbors are on the same
            if (!strncmp(hostname, &hostname_recv_buf[QUDA_MAX_HOSTNAME_STRING * neighbor_rank], QUDA_MAX_HOSTNAME_STRING)) {
              int neighbor_gpuid = gpuid_recv_buf[neighbor_rank];

              bool can_access_peer = comm_peer2peer_possible(gpuid, neighbor_gpuid);
              int access_rank = comm_peer2peer_performance(gpuid, neighbor_gpuid);

              // enable P2P if we can access the peer or if peer is self
              // if (canAccessPeer[0] * canAccessPeer[1] != 0 || gpuid == neighbor_gpuid) {
              if ((can_access_peer && access_rank <= enable_p2p_max_access_rank) || gpuid == neighbor_gpuid) {
                peer2peer_enabled[dir][dim] = true;
                if (getVerbosity() > QUDA_SILENT) {
                  printf("Peer-to-peer enabled for rank %3d (gpu=%d) with neighbor %3d (gpu=%d) dir=%d, dim=%d, "
                         "access rank = (%3d)\n",
                         comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim, access_rank);
                }
              } else {
                intranode_enabled[dir][dim] = true;
                if (getVerbosity() > QUDA_SILENT) {
                  printf(
                    "Intra-node (non peer-to-peer) enabled for rank %3d (gpu=%d) with neighbor %3d (gpu=%d) dir=%d, "
                    "dim=%d\n",
                    comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim);
                }
              }

            } // on the same node
          }   // different dimensions - x, y, z, t
        }     // different directions - forward/backward

        host_free(gpuid_recv_buf);
      }

      peer2peer_init = true;

      comm_barrier();

      peer2peer_present = comm_peer2peer_enabled_global();
    }

    bool comm_peer2peer_present() { return peer2peer_present; }

    bool enable_p2p = true;

    bool comm_peer2peer_enabled(int dir, int dim) { return enable_p2p ? peer2peer_enabled[dir][dim] : false; }

    bool init = false;
    bool p2p_global = false;

    int comm_peer2peer_enabled_global()
    {
      if (!enable_p2p) return false;

      if (!init) {
        int p2p = 0;
        for (int dim = 0; dim < 4; dim++)
          for (int dir = 0; dir < 2; dir++) p2p += (int)comm_peer2peer_enabled(dir, dim);

        comm_allreduce_int(p2p);
        init = true;
        p2p_global = p2p > 0 ? true : false;
      }
      return p2p_global * enable_peer_to_peer;
    }

    void comm_enable_peer2peer(bool enable) { enable_p2p = enable; }

    bool enable_intranode = true;

    bool comm_intranode_enabled(int dir, int dim) { return enable_intranode ? intranode_enabled[dir][dim] : false; }

    void comm_enable_intranode(bool enable) { enable_intranode = enable; }

    Topology *default_topo = nullptr;

    void comm_set_default_topology(Topology *topo) { default_topo = topo; }

    Topology *comm_default_topology(void)
    {
      if (!default_topo) { errorQuda("Default topology has not been declared"); }
      return default_topo;
    }

    int neighbor_rank[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};

    bool neighbors_cached = false;

    void comm_set_neighbor_ranks(Topology *topo = nullptr)
    {

      if (neighbors_cached) return;

      Topology *topology = topo ? topo : default_topo; // use default topology if topo is NULL
      if (!topology) { errorQuda("Topology not specified"); }

      for (int d = 0; d < 4; ++d) {
        int pos_displacement[QUDA_MAX_DIM] = {};
        int neg_displacement[QUDA_MAX_DIM] = {};
        pos_displacement[d] = +1;
        neg_displacement[d] = -1;
        neighbor_rank[0][d] = comm_rank_displaced(topology, neg_displacement);
        neighbor_rank[1][d] = comm_rank_displaced(topology, pos_displacement);
      }
      neighbors_cached = true;
    }

    int comm_neighbor_rank(int dir, int dim)
    {
      if (!neighbors_cached) { comm_set_neighbor_ranks(); }
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

    void comm_finalize(void)
    {
      Topology *topo = comm_default_topology();
      comm_destroy_topology(topo);
      comm_set_default_topology(NULL);
    }

    char partition_string[16];          /** string that contains the job partitioning */
    char topology_string[256];          /** string that contains the job topology */
    char partition_override_string[16]; /** string that contains any overridden partitioning */

    int manual_set_partition[QUDA_MAX_DIM] = {0};

#ifdef MULTI_GPU
  void comm_dim_partitioned_set(int dim)
  {
    manual_set_partition[dim] = 1;

    snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
             comm_dim_partitioned(2), comm_dim_partitioned(3));

    FieldTmp<ColorSpinorField>::destroy(); // destroy field cache since message handles can be invalid
  }
#else
  void comm_dim_partitioned_set(int)
  {
    snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
             comm_dim_partitioned(2), comm_dim_partitioned(3));
  }
#endif

  void comm_dim_partitioned_reset()
  {
    for (int i = 0; i < QUDA_MAX_DIM; i++) manual_set_partition[i] = 0;

    snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
             comm_dim_partitioned(2), comm_dim_partitioned(3));

    FieldTmp<ColorSpinorField>::destroy(); // destroy field cache since message handles can be invalid
  }

#ifdef MULTI_GPU
  int comm_dim_partitioned(int dim) { return (manual_set_partition[dim] || (default_topo && comm_dim(dim) > 1)); }
#else
  int comm_dim_partitioned(int) { return 0; }
#endif

  int comm_partitioned()
  {
    int partitioned = 0;
    for (int i = 0; i < 4; i++) { partitioned = partitioned || comm_dim_partitioned(i); }
    return partitioned;
  }

  bool gdr_enabled = false;

#ifdef MULTI_GPU
  bool gdr_init = false;
#endif

  bool comm_gdr_enabled()
  {
#ifdef MULTI_GPU

    if (!gdr_init) {
      char *enable_gdr_env = getenv("QUDA_ENABLE_GDR");
      if (enable_gdr_env && strcmp(enable_gdr_env, "1") == 0) { gdr_enabled = true; }
      gdr_init = true;
    }
#endif
    return gdr_enabled;
  }

  bool blacklist = false;
  bool blacklist_init = false;

  bool comm_gdr_blacklist()
  {
    if (!blacklist_init) {
      char *blacklist_env = getenv("QUDA_ENABLE_GDR_BLACKLIST");

      if (blacklist_env) { // set the policies to tune for explicitly
        std::stringstream blacklist_list(blacklist_env);

        int excluded_device;
        while (blacklist_list >> excluded_device) {
          // check this is a valid device
          if (excluded_device < 0 || excluded_device >= device::get_device_count()) {
            errorQuda("Cannot blacklist invalid GPU device ordinal %d", excluded_device);
          }

          if (blacklist_list.peek() == ',') blacklist_list.ignore();
          if (excluded_device == comm_gpuid()) blacklist = true;
        }
        comm_barrier();
        if (getVerbosity() > QUDA_SILENT && blacklist)
          printf("Blacklisting GPU-Direct RDMA for rank %d (GPU %d)\n", comm_rank(), comm_gpuid());
      }
      blacklist_init = true;
    }

    return blacklist;
  }

  bool comm_nvshmem_enabled()
  {
#if (defined MULTI_GPU) && (defined NVSHMEM_COMMS)
    static bool nvshmem_enabled = true;
    static bool nvshmem_init = false;
    if (!nvshmem_init) {
      char *enable_nvshmem_env = getenv("QUDA_ENABLE_NVSHMEM");
      if (enable_nvshmem_env && strcmp(enable_nvshmem_env, "1") == 0) { nvshmem_enabled = true; }
      if (enable_nvshmem_env && strcmp(enable_nvshmem_env, "0") == 0) { nvshmem_enabled = false; }
      nvshmem_init = true;
    }
#else
    static bool nvshmem_enabled = false;
#endif
    return nvshmem_enabled;
  }

  bool use_deterministic_reduce = false;

  void comm_init_common(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
  {
    Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data, comm_rank());
    comm_set_default_topology(topo);

    // determine which GPU this rank will use
    char *hostname_recv_buf = (char *)safe_malloc(QUDA_MAX_HOSTNAME_STRING * comm_size());
    comm_gather_hostname(hostname_recv_buf);

    if (gpuid < 0) {
      int device_count = device::get_device_count();
      if (device_count == 0) { errorQuda("No devices found"); }

      // We initialize gpuid if it's still negative.
      gpuid = 0;
      for (int i = 0; i < comm_rank(); i++) {
        if (!strncmp(comm_hostname(), &hostname_recv_buf[QUDA_MAX_HOSTNAME_STRING * i], QUDA_MAX_HOSTNAME_STRING)) { gpuid++; }
      }

      if (gpuid >= device_count) {
        char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
        if (enable_mps_env && strcmp(enable_mps_env, "1") == 0) {
          gpuid = gpuid % device_count;
          printf("MPS enabled, rank=%3d -> gpu=%d\n", comm_rank(), gpuid);
        } else {
          errorQuda("Too few GPUs available on %s", comm_hostname());
        }
      }
    } // -ve gpuid

    comm_peer2peer_init(hostname_recv_buf);

    host_free(hostname_recv_buf);

    char *enable_reduce_env = getenv("QUDA_DETERMINISTIC_REDUCE");
    if (enable_reduce_env && strcmp(enable_reduce_env, "1") == 0) { use_deterministic_reduce = true; }

    snprintf(partition_string, 16, ",comm=%d%d%d%d", comm_dim_partitioned(0), comm_dim_partitioned(1),
             comm_dim_partitioned(2), comm_dim_partitioned(3));

    // if CUDA_VISIBLE_DEVICES is set, we include this information in the topology_string
    char device_list_string[128] = "";
    // to ensure we have process consistency define using rank 0
    if (comm_rank() == 0) {
      device::get_visible_devices_string(device_list_string);
    }
    comm_broadcast(device_list_string, 128);
    if (std::strlen(device_list_string) > 0) {
      snprintf(topology_string, 256, ",topo=%d%d%d%d,order=%s", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3),
              device_list_string);
    } else {
      snprintf(topology_string, 256, ",topo=%d%d%d%d", comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3));
    }
  }

  char config_string[64];
  bool config_init = false;

  const char *comm_config_string()
  {
    if (!config_init) {
      strcpy(config_string, ",p2p=");
      strcat(config_string, std::to_string(comm_peer2peer_enabled_global()).c_str());
      if (enable_p2p_max_access_rank != std::numeric_limits<int>::max()) {
        strcat(config_string, ",p2p_max_access_rank=");
        strcat(config_string, std::to_string(enable_p2p_max_access_rank).c_str());
      }
      strcat(config_string, ",gdr=");
      strcat(config_string, std::to_string(comm_gdr_enabled()).c_str());
      strcat(config_string, ",nvshmem=");
      strcat(config_string, std::to_string(comm_nvshmem_enabled()).c_str());
      config_init = true;
    }

    return config_string;
  }

  const char *comm_dim_partitioned_string(const int *comm_dim_override)
  {
    if (comm_dim_override) {
      char comm[5] = {(!comm_dim_partitioned(0) ? '0' :
                         comm_dim_override[0]   ? '1' :
                                                  '0'),
                      (!comm_dim_partitioned(1) ? '0' :
                         comm_dim_override[1]   ? '1' :
                                                  '0'),
                      (!comm_dim_partitioned(2) ? '0' :
                         comm_dim_override[2]   ? '1' :
                                                  '0'),
                      (!comm_dim_partitioned(3) ? '0' :
                         comm_dim_override[3]   ? '1' :
                                                  '0'),
                      '\0'};
      strcpy(partition_override_string, ",comm=");
      strcat(partition_override_string, comm);
      return partition_override_string;
    } else {
      return partition_string;
    }
  }

  const char *comm_dim_topology_string() { return topology_string; }

  bool comm_deterministic_reduce() { return use_deterministic_reduce; }

  std::stack<bool> globalReduce;
  bool asyncReduce = false;

  int commDim(int dir) { return comm_dim(dir); }

  int commCoords(int dir) { return comm_coord(dir); }

  int commDimPartitioned(int dir) { return comm_dim_partitioned(dir); }

  void commDimPartitionedSet(int dir) { comm_dim_partitioned_set(dir); }

  void commDimPartitionedReset() { comm_dim_partitioned_reset(); }

  bool commGlobalReduction() { return globalReduce.top(); }

  void commGlobalReductionPush(bool global_reduction) { globalReduce.push(global_reduction); }

  void commGlobalReductionPop() { globalReduce.pop(); }

  bool commAsyncReduction() { return asyncReduce; }

  void commAsyncReductionSet(bool async_reduction) { asyncReduce = async_reduction; }

#if defined(QMP_COMMS) || defined(MPI_COMMS)
  MPI_Comm MPI_COMM_HANDLE;
#endif

#if defined(QMP_COMMS)
  QMP_comm_t QMP_COMM_HANDLE;

  /**
   * A bool indicating if the QMP handle here is the default one, which we should not free at the end,
   * or a one that QUDA creates through `QMP_comm_split`, which we should free at the end.
   */
  bool is_qmp_handle_default;
#endif

  int rank = -1;
  int size = -1;

  Communicator() { }

  Communicator(Communicator &other, const int *comm_split);

  Communicator(int nDim, const int *commDims, QudaCommsMap rank_from_coords, void *map_data,
               bool user_set_comm_handle = false, void *user_comm = nullptr);

  ~Communicator();

#if defined(QMP_COMMS) || defined(MPI_COMMS)
  MPI_Comm get_mpi_handle() { return MPI_COMM_HANDLE; }
#endif

  void comm_gather_hostname(char *hostname_recv_buf);

  void comm_gather_gpuid(int *gpuid_recv_buf);

  void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data);

  int comm_rank(void);

  size_t comm_size(void);

  int comm_rank_from_coords(const int *coords)
  {
    Topology *topo = comm_default_topology();
    return ::quda::comm_rank_from_coords(topo, coords);
  }

  /**
   * Declare a message handle for sending `nbytes` to the `rank` with `tag`.
   */
  MsgHandle *comm_declare_send_rank(void *buffer, int rank, int tag, size_t nbytes);

  /**
   * Declare a message handle for receiving `nbytes` from the `rank` with `tag`.
   */
  MsgHandle *comm_declare_recv_rank(void *buffer, int rank, int tag, size_t nbytes);

  /**
   * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes);

  /**
   * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes);

  /**
   * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                 size_t stride);

  /**
   * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
   */
  MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[], size_t blksize, int nblocks,
                                                    size_t stride);

  void comm_free(MsgHandle *&mh);

  void comm_start(MsgHandle *mh);

  void comm_wait(MsgHandle *mh);

  int comm_query(MsgHandle *mh);

  template <typename T> T deterministic_reduce(T *array, int n)
  {
    std::sort(array, array + n); // sort reduction into ascending order for deterministic reduction
    return std::accumulate(array, array + n, 0.0);
  }

  void comm_allreduce_sum_array(double *data, size_t size);

  void comm_allreduce_max_array(double *data, size_t size);

  void comm_allreduce_max_array(deviation_t<double> *data, size_t size);

  void comm_allreduce_min_array(double *data, size_t size);

  void comm_allreduce_int(int &data);

  void comm_allreduce_xor(uint64_t &data);

  /**
     @brief Broadcast from the root rank
     @param[in,out] data The data to be read from on the root rank, and
     written to on all other ranks
     @param[in] nbytes The size in bytes of data to be broadcast
     @param[in] root The process that will be broadcasting
  */
  void comm_broadcast(void *data, size_t nbytes, int root = 0);

  void comm_barrier(void);

  static void comm_abort_(int status);

  static int comm_rank_global();
};

constexpr CommKey default_comm_key = {1, 1, 1, 1};

void push_communicator(const CommKey &split_key);

/**
   @brief Broadcast from the root rank of the default communicator
   @param[in,out] data The data to be read from on the root rank, and
   written to on all other ranks
   @param[in] nbytes The size in bytes of data to be broadcast
   @param[in] root The process that will be broadcasting
*/
void comm_broadcast_global(void *data, size_t nbytes, int root = 0);

} // namespace quda
