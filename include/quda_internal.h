#ifndef _QUDA_INTERNAL_H
#define _QUDA_INTERNAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <string>
#include <complex>

#if ((defined(QMP_COMMS) || defined(MPI_COMMS)) && !defined(MULTI_GPU))
#error "MULTI_GPU must be enabled to use MPI or QMP"
#endif

#if (!defined(QMP_COMMS) && !defined(MPI_COMMS) && defined(MULTI_GPU))
#error "MPI or QMP must be enabled to use MULTI_GPU"
#endif

//#ifdef USE_QDPJIT
//#include "qdp_quda.h"
//#endif

#ifdef QMP_COMMS
#include <qmp.h>
#endif

#ifdef PTHREADS
#include <pthread.h>
#endif

#define MAX_SHORT 32767.0f

#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
#include <enum_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>

#include <vector>

// Use bindless texture on Kepler
#if (__COMPUTE_CAPABILITY__ >= 300) && (CUDA_VERSION >= 5000)
#define USE_TEXTURE_OBJECTS
#endif



#ifdef INTERFACE_NVTX
#include "nvToolsExt.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif
  
  typedef void *ParityGauge;

  // replace below with ColorSpinorField
  typedef struct {
    size_t bytes;
    QudaPrecision precision;
    int length; // total length
    int volume; // geometric volume (single parity)
    int X[QUDA_MAX_DIM]; // the geometric lengths (single parity)
    int Nc; // length of color dimension
    int Ns; // length of spin dimension
    void *data; // either (double2*), (float4 *) or (short4 *), depending on precision
    float *dataNorm; // used only when precision is QUDA_HALF_PRECISION
  } ParityHw;
  
  typedef struct {
    ParityHw odd;
    ParityHw even;
  } FullHw;

  struct QUDA_DiracField{
    void *field; /**< Pointer to a ColorSpinorField */
  };

  extern cudaDeviceProp deviceProp;  
  extern cudaStream_t *streams;
 
#ifdef PTHREADS
  extern pthread_mutex_t pthread_mutex;
#endif
 
#ifdef __cplusplus
}
#endif

#define REAL(a) (*((double*)&a))
#define IMAG(a) (*((double*)&a+1))

namespace quda {

  typedef std::complex<double> Complex;

  /**
   * Use this for recording a fine-grained profile of a QUDA
   * algorithm.  This uses host-side measurement, so should be used
   * for timing fully host-device synchronous algorithms.
   */
  struct Timer {
    /**< The cumulative sum of time */
    double time;

    /**< The last recorded time interval */
    double last;

    /**< Used to store when the timer was last started */
    timeval start;

    /**< Used to store when the timer was last stopped */
    timeval stop;

    /**< Are we currently timing? */
    bool running;
    
    /**< Keep track of number of calls */
    int count;

  Timer() : time(0.0), last(0.0), running(false), count(0) { ; } 

    void Start(const char *func, const char *file, int line) {
      if (running) {
	printfQuda("ERROR: Cannot start an already running timer (%s:%d in %s())\n", file, line, func);
	errorQuda("Aborting");
      }
      gettimeofday(&start, NULL);
      running = true;
    }

    void Stop(const char *func, const char *file, int line) {
      if (!running) {
	printfQuda("ERROR: Cannot stop an unstarted timer (%s:%d in %s())\n", file, line, func);
	errorQuda("Aborting");
      }
      gettimeofday(&stop, NULL);

      long ds = stop.tv_sec - start.tv_sec;
      long dus = stop.tv_usec - start.tv_usec;
      last = ds + 0.000001*dus;
      time += last;
      count++;

      running = false;
    }

    double Last() { return last; }

  };

  /**< Enumeration type used for writing a simple but extensible profiling framework. */
  enum QudaProfileType {
    QUDA_PROFILE_H2D, /**< host -> device transfers */
    QUDA_PROFILE_D2H, /**< The time in seconds for device -> host transfers */
    QUDA_PROFILE_INIT, /**< The time in seconds taken for initiation */
    QUDA_PROFILE_PREAMBLE, /**< The time in seconds taken for any preamble */
    QUDA_PROFILE_COMPUTE, /**< The time in seconds taken for the actual computation */
    QUDA_PROFILE_EPILOGUE, /**< The time in seconds taken for any epilogue */
    QUDA_PROFILE_FREE, /**< The time in seconds for freeing resources */

    // lower level counters used in the dslash
    QUDA_PROFILE_LOWER_LEVEL, /**< dummy timer to mark beginning of lower level timers */
    QUDA_PROFILE_PACK_KERNEL, /**< face packing kernel */
    QUDA_PROFILE_DSLASH_KERNEL, /**< dslash kernel */
    QUDA_PROFILE_GATHER, /**< gather (device -> host) */
    QUDA_PROFILE_SCATTER, /**< scatter (host -> device) */
    QUDA_PROFILE_EVENT_RECORD, /**< cuda event record  */
    QUDA_PROFILE_EVENT_QUERY, /**< cuda event querying */
    QUDA_PROFILE_STREAM_WAIT_EVENT, /**< stream waiting for event completion */

    QUDA_PROFILE_COMMS, /**< synchronous communication */
    QUDA_PROFILE_COMMS_START, /**< initiating communication */
    QUDA_PROFILE_COMMS_QUERY, /**< querying communication */

    QUDA_PROFILE_CONSTANT, /**< time spent setting CUDA constant parameters */

    QUDA_PROFILE_TOTAL, /**< The total time in seconds for the algorithm. Must be the penultimate type. */
    QUDA_PROFILE_COUNT /**< The total number of timers we have.  Must be last enum type. */
  };

#ifdef INTERFACE_NVTX



#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%nvtx_num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.category = cid;\
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

  class TimeProfile {
    std::string fname;  /**< Which function are we profiling */
#ifdef INTERFACE_NVTX
    static const uint32_t nvtx_colors[];// = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
    static const int nvtx_num_colors;// = sizeof(nvtx_colors)/sizeof(uint32_t);
#endif
    Timer profile[QUDA_PROFILE_COUNT];
    static std::string pname[];

    bool switchOff;
    bool use_global;

    // global timer
    static Timer global_profile[QUDA_PROFILE_COUNT];
    static bool global_switchOff[QUDA_PROFILE_COUNT];
    static int global_total_level[QUDA_PROFILE_COUNT]; // zero initialize

    static void StopGlobal(const char *func, const char *file, int line, QudaProfileType idx) {

      global_total_level[idx]--;
      if (global_total_level[idx]==0) global_profile[idx].Stop(func,file,line);

      // switch off total timer if we need to
      if (global_switchOff[idx]) {
        global_total_level[idx]--;
        if (global_total_level[idx]==0) global_profile[idx].Stop(func,file,line);
        global_switchOff[idx] = false;
      }
    }

    static void StartGlobal(const char *func, const char *file, int line, QudaProfileType idx) {
      // if total timer isn't running, then start it running
      if (!global_profile[idx].running) {
        global_profile[idx].Start(func,file,line);
        global_total_level[idx]++;
        global_switchOff[idx] = true;
      }

      if (global_total_level[idx]==0) global_profile[idx].Start(func,file,line);
      global_total_level[idx]++;
    }

  public:
    TimeProfile(std::string fname) : fname(fname), switchOff(false), use_global(true) { ; }

    TimeProfile(std::string fname, bool use_global) : fname(fname), switchOff(false), use_global(use_global) { ; }

    /**< Print out the profile information */
    void Print();

    void Start_(const char *func, const char *file, int line, QudaProfileType idx) { 
      // if total timer isn't running, then start it running
      if (!profile[QUDA_PROFILE_TOTAL].running && idx != QUDA_PROFILE_TOTAL) {
	profile[QUDA_PROFILE_TOTAL].Start(func,file,line);
        switchOff = true;
      }

      profile[idx].Start(func, file, line); 
      PUSH_RANGE(fname.c_str(),idx)
	if (use_global) StartGlobal(func,file,line,idx);
    }


    void Stop_(const char *func, const char *file, int line, QudaProfileType idx) {
      profile[idx].Stop(func, file, line); 
      POP_RANGE

      // switch off total timer if we need to
      if (switchOff && idx != QUDA_PROFILE_TOTAL) {
        profile[QUDA_PROFILE_TOTAL].Stop(func,file,line);
        switchOff = false;
      }
      if (use_global) StopGlobal(func,file,line,idx);
    }

    double Last(QudaProfileType idx) { 
      return profile[idx].last;
    }



    static void PrintGlobal();

  };

#define TPSTART(idx) Start_(__func__, __FILE__, __LINE__, idx)
#define TPSTOP(idx) Stop_(__func__, __FILE__, __LINE__, idx)

#undef PUSH_RANGE
#undef POP_RANGE

#ifdef MULTI_GPU
#ifdef PTHREADS
  const int Nstream = 10;
#else
  const int Nstream = 9;
#endif
#else
  const int Nstream = 1;
#endif

} // namespace quda



#endif // _QUDA_INTERNAL_H
