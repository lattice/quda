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

  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &remote, void *local)
  {
    ompwip("unimplemented");
  }

  void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &)
  {
    ompwip("unimplemented");
  }

  void comm_create_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &remote,
                                  array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
  {
    ompwip("unimplemented");
  }

  void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &,
                                   array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &local)
  {
    ompwip("unimplemented");
  }

} // namespace quda

