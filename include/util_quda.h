#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <enum_quda.h>
#include <comm_quda.h>
#include <tune_key.h>
#include <malloc_quda.h>

namespace quda
{
  // strip path from __FILE__
  constexpr const char *str_end(const char *str) { return *str ? str_end(str + 1) : str; }
  constexpr bool str_slant(const char *str) { return *str == '/' ? true : (*str ? str_slant(str + 1) : false); }
  constexpr const char *r_slant(const char *str) { return *str == '/' ? (str + 1) : r_slant(str - 1); }
  constexpr const char *file_name(const char *str) { return str_slant(str) ? r_slant(str_end(str)) : str; }
} // namespace quda

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

void errorQuda_(const char *func, const char *file, int line, ...);

#define errorQuda(...)                                                                                                 \
  do {                                                                                                                 \
    fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());                                                          \
    fprintf(getOutputFile(), __VA_ARGS__);                                                                             \
    errorQuda_(__PRETTY_FUNCTION__, quda::file_name(__FILE__), __LINE__, __VA_ARGS__);                                 \
  } while (0)

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

#define warningQuda(...) do {                                 \
  if (getVerbosity() > QUDA_SILENT) {			      \
    fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix()); \
    fprintf(getOutputFile(), __VA_ARGS__);                      \
    fprintf(getOutputFile(), "\n");                             \
    fflush(getOutputFile());                                    \
  }								\
} while (0)

#endif // MULTI_GPU

#define logQuda(verbosity, ...)                                                                                        \
  if (getVerbosity() >= verbosity) { printfQuda(__VA_ARGS__); }
