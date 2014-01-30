#ifndef _UTIL_QUDA_H
#define _UTIL_QUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <enum_quda.h>
#include <comm_quda.h>

QudaTune getTuning();

/**
   @param tune Sets the whether to tune the cuda kernels or not
*/
void setTuning(QudaTune tune);

QudaVerbosity getVerbosity();
char *getOutputPrefix();
FILE *getOutputFile();

void setVerbosity(const QudaVerbosity verbosity);
void setOutputPrefix(const char *prefix);
void setOutputFile(FILE *outfile);

void pushVerbosity(QudaVerbosity verbosity);
void popVerbosity();

char *getPrintBuffer();

// Note that __func__ is part of C++11 and has long been supported by GCC.

#ifdef MULTI_GPU

#define printfQuda(...) do {                           \
  sprintf(getPrintBuffer(), __VA_ARGS__);	       \
  if (comm_rank() == 0) {	                       \
    fprintf(getOutputFile(), "%s", getOutputPrefix()); \
    fprintf(getOutputFile(), "%s", getPrintBuffer());  \
    fflush(getOutputFile());                           \
  }                                                    \
} while (0)

#define errorQuda(...) do {                                                  \
  fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());                  \
  fprintf(getOutputFile(), __VA_ARGS__);                                     \
  fprintf(getOutputFile(), " (rank %d, host %s, " __FILE__ ":%d in %s())\n", \
          comm_rank(), comm_hostname(), __LINE__, __func__);                 \
  fflush(getOutputFile());                                                   \
  comm_abort(1);                                                             \
} while (0)

#define warningQuda(...) do {                                   \
  sprintf(getPrintBuffer(), __VA_ARGS__);			\
  if (comm_rank() == 0) {                                       \
    fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix()); \
    fprintf(getOutputFile(), "%s", getPrintBuffer());		\
    fprintf(getOutputFile(), "\n");                             \
    fflush(getOutputFile());                                    \
  }                                                             \
} while (0)

#else

#define printfQuda(...) do {                         \
  fprintf(getOutputFile(), "%s", getOutputPrefix()); \
  fprintf(getOutputFile(), __VA_ARGS__);             \
  fflush(getOutputFile());                           \
} while (0)

#define errorQuda(...) do {                                 \
  fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix()); \
  fprintf(getOutputFile(), __VA_ARGS__);                    \
  fprintf(getOutputFile(), " (" __FILE__ ":%d in %s())\n",  \
	  __LINE__, __func__);                              \
  exit(1);                                                  \
} while (0)

#define warningQuda(...) do {                                 \
  fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix()); \
  fprintf(getOutputFile(), __VA_ARGS__);                      \
  fprintf(getOutputFile(), "\n");                             \
  fflush(getOutputFile());                                    \
} while (0)

#endif // MULTI_GPU


#define checkCudaErrorNoSync() do {                    \
  cudaError_t error = cudaGetLastError();              \
  if (error != cudaSuccess)                            \
    errorQuda("(CUDA) %s", cudaGetErrorString(error)); \
} while (0)


#ifdef HOST_DEBUG

#define checkCudaError() do {  \
  cudaDeviceSynchronize();     \
  checkCudaErrorNoSync();      \
} while (0)

#else

#define checkCudaError() checkCudaErrorNoSync()

#endif // HOST_DEBUG


#endif // _UTIL_QUDA_H
