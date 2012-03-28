#ifndef _UTIL_QUDA_H
#define _UTIL_QUDA_H

#include <stdio.h>
#include <stdlib.h>

#ifdef QMP_COMMS

#include <qmp.h>

#define printfQuda(...) do {	     \
 if (QMP_get_node_number() == 0) {	\
    printf(__VA_ARGS__);             \
    fflush(stdout);                  \
  }                                  \
} while (0)

#define errorQuda(...) do {		    \
  printf("QUDA error: " __VA_ARGS__);	\
  printf(" (node %d, " __FILE__ ":%d in %s())\n",   \
         QMP_get_node_number(), __LINE__, __FUNCTION__);			\
  QMP_abort(1);				    \
} while (0)

#elif defined(MPI_COMMS)

#include <comm_quda.h>

#define printfQuda(...) do {			\
    if (comm_rank() == 0) {			\
      printf(__VA_ARGS__);			\
      fflush(stdout);				\
    }						\
  } while (0)

extern char hostname[];
#define errorQuda(...) do {		    \
  printf("QUDA error: " __VA_ARGS__);	\
  printf(" (node %d, " __FILE__ ":%d in %s(), hostname=%s)\n",   \
         comm_rank(), __LINE__, __FUNCTION__, hostname);	       \
  comm_exit(1);						 \
} while (0)

#else

#define printfQuda(...) do {				\
    printf(__VA_ARGS__);				\
    fflush(stdout); } while (0)				


#define errorQuda(...) do {		     \
  printf("QUDA error: " __VA_ARGS__);	\
  printf(" (" __FILE__ ":%d in %s())\n", __LINE__, __FUNCTION__);	\
  exit(1);				     \
} while (0)

#endif // USE_QMP

#define warningQuda(...) do {                \
  printfQuda("QUDA warning: " __VA_ARGS__);  \
  printfQuda("\n");                          \
} while (0)

#ifdef HOST_DEBUG

#define checkCudaError() do {                           \
    cudaThreadSynchronize();				\
  cudaError_t error = cudaGetLastError();               \
  if (error != cudaSuccess)                             \
    errorQuda("(CUDA) %s", cudaGetErrorString(error));  \
} while (0)

#else

#define checkCudaError() do {                           \
  cudaError_t error = cudaGetLastError();               \
  if (error != cudaSuccess)                             \
    errorQuda("(CUDA) %s", cudaGetErrorString(error));  \
} while (0)

#endif

#ifdef __cplusplus
extern "C" {
#endif
  
  void stopwatchStart();
  double stopwatchReadSeconds();

#ifdef __cplusplus
}
#endif

#endif // _UTIL_QUDA_H
