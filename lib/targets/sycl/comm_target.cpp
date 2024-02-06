#include <comm_quda.h>
#include <quda_api.h>
//#include <quda_cuda_api.h>
#include <algorithm>

namespace quda
{

  // bool comm_peer2peer_possible(int local_gpuid, int neighbor_gpuid)
  bool comm_peer2peer_possible(int, int)
  {
    return false;
  }

  int comm_peer2peer_performance(int local_gpuid, int neighbor_gpuid)
  {
    int accessRank[2] = {};
    if (comm_peer2peer_possible(local_gpuid, neighbor_gpuid)) {
    }
    // return the slowest direction of access (lower is faster)
    return std::max(accessRank[0], accessRank[1]);
  }

  void comm_create_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &, void *)
  {
  }

  void comm_destroy_neighbor_memory(array_2d<void *, QUDA_MAX_DIM, 2> &)
  {
  }

  void comm_create_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &,
				  array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &)
  {
  }

  void comm_destroy_neighbor_event(array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &,
				   array_2d<qudaEvent_t, QUDA_MAX_DIM, 2> &)
  {
  }

} // namespace quda
