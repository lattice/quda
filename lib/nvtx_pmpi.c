
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */
#ifdef MPICH_HAS_C2F
_EXTERN_C_ void *MPIR_ToPointer(int);
#endif // MPICH_HAS_C2F
#ifdef PIC
/* For shared libraries, declare these weak and figure out which one was linked
   based on which init wrapper was called.  See mpi_init wrappers.  */
#pragma weak pmpi_init
#pragma weak PMPI_INIT
#pragma weak pmpi_init_
#pragma weak pmpi_init__
#endif /* PIC */
_EXTERN_C_ void pmpi_init(MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init__(MPI_Fint *ierr);
static int in_wrapper = 0;
#include <pthread.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#include <mpi_comm_handle.h>
// Setup event category name
/* ================== C Wrappers for MPI_Init ================== */
_EXTERN_C_ int PMPI_Init(int *argc, char ***argv);
_EXTERN_C_ int MPI_Init(int *argc, char ***argv) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Init(argc, argv);
    in_wrapper = 1;

  nvtxNameCategoryA(999, "MPI");
  _wrap_py_return_val = PMPI_Init(argc, argv);
  int rank;
  PMPI_Comm_rank(MPI_COMM_HANDLE, &rank);
  char name[256];
  sprintf( name, "MPI Rank %d", rank );
 
  nvtxNameOsThread(pthread_self(), name);
  nvtxNameCudaDeviceA(rank, name);
    in_wrapper = 0;
    return _wrap_py_return_val;
}


// Wrap select MPI functions with NVTX ranges
/* ================== C Wrappers for MPI_Send ================== */
_EXTERN_C_ int PMPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
_EXTERN_C_ int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Send(buf, count, datatype, dest, tag, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Send";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Send(buf, count, datatype, dest, tag, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Recv ================== */
_EXTERN_C_ int PMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
_EXTERN_C_ int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Recv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Allreduce ================== */
_EXTERN_C_ int PMPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
_EXTERN_C_ int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Allreduce";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Reduce ================== */
_EXTERN_C_ int PMPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Reduce";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Wait ================== */
_EXTERN_C_ int PMPI_Wait(MPI_Request *request, MPI_Status *status);
_EXTERN_C_ int MPI_Wait(MPI_Request *request, MPI_Status *status) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Wait(request, status);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Wait";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Wait(request, status);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Waitany ================== */
_EXTERN_C_ int PMPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status);
_EXTERN_C_ int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Waitany(count, array_of_requests, index, status);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Waitany";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Waitany(count, array_of_requests, index, status);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Waitall ================== */
_EXTERN_C_ int PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
_EXTERN_C_ int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Waitall(count, array_of_requests, array_of_statuses);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Waitall";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Waitsome ================== */
_EXTERN_C_ int PMPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
_EXTERN_C_ int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Waitsome";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Gather ================== */
_EXTERN_C_ int PMPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Gather";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Gatherv ================== */
_EXTERN_C_ int PMPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Gatherv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Scatter ================== */
_EXTERN_C_ int PMPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Scatter";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Scatterv ================== */
_EXTERN_C_ int PMPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Scatterv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Allgather ================== */
_EXTERN_C_ int PMPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
_EXTERN_C_ int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Allgather";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Allgatherv ================== */
_EXTERN_C_ int PMPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
_EXTERN_C_ int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Allgatherv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Alltoall ================== */
_EXTERN_C_ int PMPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
_EXTERN_C_ int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Alltoall";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Alltoallv ================== */
_EXTERN_C_ int PMPI_Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm);
_EXTERN_C_ int MPI_Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Alltoallv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Alltoallw ================== */
_EXTERN_C_ int PMPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);
_EXTERN_C_ int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Alltoallw";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Bcast ================== */
_EXTERN_C_ int PMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
_EXTERN_C_ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Bcast(buffer, count, datatype, root, comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Bcast";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Bcast(buffer, count, datatype, root, comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Sendrecv ================== */
_EXTERN_C_ int PMPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
_EXTERN_C_ int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Sendrecv";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Barrier ================== */
_EXTERN_C_ int PMPI_Barrier(MPI_Comm comm);
_EXTERN_C_ int MPI_Barrier(MPI_Comm comm) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Barrier(comm);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Barrier";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Barrier(comm);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Start ================== */
_EXTERN_C_ int PMPI_Start(MPI_Request *request);
_EXTERN_C_ int MPI_Start(MPI_Request *request) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Start(request);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Start";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Start(request);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Test ================== */
_EXTERN_C_ int PMPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
_EXTERN_C_ int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Test(request, flag, status);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Test";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Test(request, flag, status);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Send_init ================== */
_EXTERN_C_ int PMPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Send_init";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}

/* ================== C Wrappers for MPI_Recv_init ================== */
_EXTERN_C_ int PMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
_EXTERN_C_ int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) { 
    int _wrap_py_return_val = 0;
    if (in_wrapper) return PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
    in_wrapper = 1;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii  = "MPI_Recv_init";
  eventAttrib.category = 999;
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = 0xffffaa00; \
 
  nvtxRangePushEx(&eventAttrib);
  _wrap_py_return_val = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
  nvtxRangePop();
    in_wrapper = 0;
    return _wrap_py_return_val;
}


