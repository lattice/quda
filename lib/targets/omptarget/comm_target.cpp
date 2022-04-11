#include <comm_quda.h>
#include <quda_api.h>
#include <algorithm>
#include <shmem_helper.cuh>

namespace quda
{
  bool comm_peer2peer_possible(int local_gpuid, int neighbor_gpuid)
  {
    ompwip("comm_peer2peer_possible -> false");
    return false;
  }

  int comm_peer2peer_performance(int local_gpuid, int neighbor_gpuid)
  {
    ompwip("comm_peer2peer_performance -> 0");
    return 0;
  }

  void comm_create_neighbor_memory(void *remote[QUDA_MAX_DIM][2], void *local)
  {
    ompwip("unimplemented");
  }

  void comm_destroy_neighbor_memory(void *remote[QUDA_MAX_DIM][2])
  {
    ompwip("unimplemented");
  }

  void comm_create_neighbor_event(qudaEvent_t remote[2][QUDA_MAX_DIM], qudaEvent_t local[2][QUDA_MAX_DIM])
  {
    ompwip("unimplemented");
  }

  void comm_destroy_neighbor_event(qudaEvent_t [2][QUDA_MAX_DIM], qudaEvent_t local[2][QUDA_MAX_DIM])
  {
    ompwip("unimplemented");
  }

} // namespace quda
