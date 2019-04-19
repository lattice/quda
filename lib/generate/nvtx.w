#include <pthread.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#include <mpi_comm_handle.h>
// Setup event category name
{{fn name MPI_Init}}
  nvtxNameCategoryA(999, "MPI");
  {{callfn}}
  int rank;
  PMPI_Comm_rank(MPI_COMM_HANDLE, &rank);
  char name[256];
  sprintf( name, "MPI Rank %d", rank );
 
  nvtxNameOsThread(pthread_self(), name);
  nvtxNameCudaDeviceA(rank, name);
{{endfn}}
// Wrap select MPI functions with NVTX ranges
{{fn name MPI_Send MPI_Recv MPI_Allreduce MPI_Reduce MPI_Wait MPI_Waitany
MPI_Waitall MPI_Waitsome MPI_Gather MPI_Gatherv MPI_Scatter MPI_Scatterv
MPI_Allgather MPI_Allgatherv MPI_Alltoall MPI_Alltoallv MPI_Alltoallw MPI_Bcast
MPI_Sendrecv MPI_Barrier MPI_Start MPI_Test MPI_Send_init MPI_Recv_init }}
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "{{name}}";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  {{callfn}}
  nvtxRangePop();
{{endfn}}
