#ifndef _UTIL_QUDA_H
#define _UTIL_QUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <enum_quda.h>
#include <comm_quda.h>
#include <tune_key.h>


/**
   @brief Query whether autotuning is enabled or not.  Default is enabled but can be overridden by setting QUDA_ENABLE_TUNING=0.
   @return If autotuning is enabled
 */
QudaTune getTuning();

QudaVerbosity getVerbosity();
char *getOutputPrefix();
FILE *getOutputFile();

void setVerbosity(QudaVerbosity verbosity);
void setOutputPrefix(const char *prefix);
void setOutputFile(FILE *outfile);

/**
   @brief Push a new verbosity onto the stack
*/
void pushVerbosity(QudaVerbosity verbosity);

/**
   @brief Pop the verbosity restoring the prior one on the stack
*/
void popVerbosity();

/**
   @brief Push a new output prefix onto the stack
*/
void pushOutputPrefix(const char *prefix);

/**
   @brief Pop the output prefix restoring the prior one on the stack
*/
void popOutputPrefix();

/**
   @brief This function returns true if the calling rank is enabled
   for verbosity (e.g., whether printQuda and warningQuda will print
   out from this rank).
   @return Return whether this rank will print
 */
bool getRankVerbosity();

char *getPrintBuffer();

/**
   @brief Returns a string of the form
   ",omp_threads=$OMP_NUM_THREADS", which can be used for storing the
   number of OMP threads for CPU functions recorded in the tune cache.
   @return Returns the string
*/
char* getOmpThreadStr();

namespace quda {
  // forward declaration
  void saveTuneCache(bool error);
}

#define zeroThread (threadIdx.x + blockDim.x*blockIdx.x==0 &&		\
		    threadIdx.y + blockDim.y*blockIdx.y==0 &&		\
		    threadIdx.z + blockDim.z*blockIdx.z==0)

#define printfZero(...)	do {						\
    if (zeroThread) printf(__VA_ARGS__);				\
  } while (0)

#ifdef MULTI_GPU

#define printfQuda(...) do {                           \
  sprintf(getPrintBuffer(), __VA_ARGS__);	       \
  if (getRankVerbosity()) {			       \
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
  fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	  getOutputPrefix(), getLastTuneKey().name,			     \
	  getLastTuneKey().volume, getLastTuneKey().aux);	             \
  fflush(getOutputFile());                                                   \
  quda::saveTuneCache(true);						\
  comm_abort(1);                                                             \
} while (0)

#define warningQuda(...) do {                                   \
  if (getVerbosity() > QUDA_SILENT) {				\
    sprintf(getPrintBuffer(), __VA_ARGS__);			\
    if (getRankVerbosity()) {						\
      fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
      fprintf(getOutputFile(), "%s", getPrintBuffer());			\
      fprintf(getOutputFile(), "\n");					\
      fflush(getOutputFile());						\
    }									\
  }									\
} while (0)

#else

#define printfQuda(...) do {                         \
  fprintf(getOutputFile(), "%s", getOutputPrefix()); \
  fprintf(getOutputFile(), __VA_ARGS__);             \
  fflush(getOutputFile());                           \
} while (0)

#define errorQuda(...) do {						     \
  fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());		     \
  fprintf(getOutputFile(), __VA_ARGS__);				     \
  fprintf(getOutputFile(), " (" __FILE__ ":%d in %s())\n",		     \
	  __LINE__, __func__);						     \
  fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	  getOutputPrefix(), getLastTuneKey().name,			     \
	  getLastTuneKey().volume, getLastTuneKey().aux);		     \
  quda::saveTuneCache(true);						\
  comm_abort(1);							     \
} while (0)

#define warningQuda(...) do {                                 \
  if (getVerbosity() > QUDA_SILENT) {			      \
    fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix()); \
    fprintf(getOutputFile(), __VA_ARGS__);                      \
    fprintf(getOutputFile(), "\n");                             \
    fflush(getOutputFile());                                    \
  }								\
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

#ifdef __CUDA_ARCH__
// hide from device code
#undef errorQuda
#define errorQuda(...)

#endif

#endif // _UTIL_QUDA_H
