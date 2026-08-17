#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <enum_quda.h>
#include <comm_quda.h>
#include <tune_key.h>
#include <malloc_quda.h>
#include <type_traits>
#include <utility>
#include <float128_t.h>

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
bool getTuning();

/**
   @brief Set the tuning state
   @param[in] tuning New tuning state
 */
void setTuning(bool tuning);

/**
   @brief Push a new tuning state, and back up the present one on the
   stack.
   @param[in] tune New tuning state
 */
void pushTuning(bool tune);

/**
   @brief Pop the present tuning state and restore what is on the stack.
 */
void popTuning();

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
const char *getOmpThreadStr();

[[noreturn]] void errorQuda_(const char *func, const char *file, int line);

namespace quda
{
  namespace printf_detail
  {
    // Convert host quads to double for fprintf/sprintf.  Use a single forwarding
    // overload so float128_t lvalues (common at logQuda call sites) are not
    // preferentially bound by a catch-all template that would skip conversion.
    template <typename T> constexpr decltype(auto) printf_arg(T &&x)
    {
#ifdef QUDA_USE_QUAD_SCALAR
      if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, float128_t>) {
        return static_cast<double>(x);
      } else
#endif
      {
        return std::forward<T>(x);
      }
    }

    template <typename... Args> inline void printfQudaImpl(const char *fmt, Args &&...args)
    {
#ifdef MULTI_GPU
      if constexpr (sizeof...(Args) == 0) {
        sprintf(getPrintBuffer(), "%s", fmt);
      } else {
        sprintf(getPrintBuffer(), fmt, printf_arg(std::forward<Args>(args))...);
      }
      if (getRankVerbosity()) {
        fprintf(getOutputFile(), "%s", getOutputPrefix());
        fprintf(getOutputFile(), "%s", getPrintBuffer());
        fflush(getOutputFile());
      }
#else
      fprintf(getOutputFile(), "%s", getOutputPrefix());
      if constexpr (sizeof...(Args) == 0) {
        fprintf(getOutputFile(), "%s", fmt);
      } else {
        fprintf(getOutputFile(), fmt, printf_arg(std::forward<Args>(args))...);
      }
      fflush(getOutputFile());
#endif
    }

    template <typename... Args> inline void warningQudaImpl(const char *fmt, Args &&...args)
    {
      if (getVerbosity() > QUDA_SILENT) {
#ifdef MULTI_GPU
        if constexpr (sizeof...(Args) == 0) {
          sprintf(getPrintBuffer(), "%s", fmt);
        } else {
          sprintf(getPrintBuffer(), fmt, printf_arg(std::forward<Args>(args))...);
        }
        if (getRankVerbosity()) {
          fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());
          fprintf(getOutputFile(), "%s", getPrintBuffer());
          fprintf(getOutputFile(), "\n");
          fflush(getOutputFile());
        }
#else
        fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());
        if constexpr (sizeof...(Args) == 0) {
          fprintf(getOutputFile(), "%s", fmt);
        } else {
          fprintf(getOutputFile(), fmt, printf_arg(std::forward<Args>(args))...);
        }
        fprintf(getOutputFile(), "\n");
        fflush(getOutputFile());
#endif
      }
    }

    template <typename... Args>
    [[noreturn]] inline void errorQudaImpl(const char *func, const char *file, int line, const char *fmt,
                                           Args &&...args)
    {
      fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());
      if constexpr (sizeof...(Args) == 0) {
        fprintf(getOutputFile(), "%s", fmt);
      } else {
        fprintf(getOutputFile(), fmt, printf_arg(std::forward<Args>(args))...);
      }
      errorQuda_(func, file, line);
    }

    template <typename... Args> inline int sprintfQuda(char *buf, const char *fmt, Args &&...args)
    {
      if constexpr (sizeof...(Args) == 0) {
        return sprintf(buf, "%s", fmt);
      } else {
        return sprintf(buf, fmt, printf_arg(std::forward<Args>(args))...);
      }
    }
  } // namespace printf_detail
} // namespace quda

#define errorQuda(...)                                                                                                \
  quda::printf_detail::errorQudaImpl(__PRETTY_FUNCTION__, quda::file_name(__FILE__), __LINE__, __VA_ARGS__)

#define zeroThread (threadIdx.x + blockDim.x*blockIdx.x==0 &&		\
		    threadIdx.y + blockDim.y*blockIdx.y==0 &&		\
		    threadIdx.z + blockDim.z*blockIdx.z==0)

#define printfZero(...)	do {						\
    if (zeroThread) printf(__VA_ARGS__);				\
  } while (0)

#ifdef MULTI_GPU

#define printfQuda(...) quda::printf_detail::printfQudaImpl(__VA_ARGS__)

#define warningQuda(...) quda::printf_detail::warningQudaImpl(__VA_ARGS__)

#else

#define printfQuda(...) quda::printf_detail::printfQudaImpl(__VA_ARGS__)

#define warningQuda(...) quda::printf_detail::warningQudaImpl(__VA_ARGS__)

#endif // MULTI_GPU

#define logQuda(verbosity, ...)                                                                                        \
  if (getVerbosity() >= verbosity) { printfQuda(__VA_ARGS__); }

#define sprintfQuda(...) quda::printf_detail::sprintfQuda(__VA_ARGS__)
