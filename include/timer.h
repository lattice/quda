#pragma once

#ifdef __CUDACC_RTC__

namespace quda {
  // dummy Implementation that can safely be parsed by nvrtc
  enum QudaProfileType { };

  class TimeProfile {
  public:
    TimeProfile(std::string fname);
    TimeProfile(std::string fname, bool use_global);
    void Print();
    void Start_(const char *func, const char *file, int line, QudaProfileType idx);
    void Stop_(const char *func, const char *file, int line, QudaProfileType idx);
    void Reset_(const char *func, const char *file, int line);
    double Last(QudaProfileType idx);
    void PrintGlobal();
    bool isRunning(QudaProfileType idx);
  };
}

#else

#include <sys/time.h>

#ifdef INTERFACE_NVTX
#if QUDA_NVTX_VERSION == 3
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif
#endif

namespace quda {

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

    void Reset(const char *func, const char *file, int line) {
      if (running) {
	printfQuda("ERROR: Cannot reset a started timer (%s:%d in %s())\n", file, line, func);
	errorQuda("Aborting");
      }
      time = 0.0;
      last = 0.0;
      count = 0;
    }

  };

  /**< Enumeration type used for writing a simple but extensible profiling framework. */
  enum QudaProfileType {
    QUDA_PROFILE_H2D,      /**< host -> device transfers */
    QUDA_PROFILE_D2H,      /**< The time in seconds for device -> host transfers */
    QUDA_PROFILE_INIT,     /**< The time in seconds taken for initiation */
    QUDA_PROFILE_PREAMBLE, /**< The time in seconds taken for any preamble */
    QUDA_PROFILE_COMPUTE,  /**< The time in seconds taken for the actual computation */
    QUDA_PROFILE_COMMS,    /**< synchronous communication */
    QUDA_PROFILE_EPILOGUE, /**< The time in seconds taken for any epilogue */
    QUDA_PROFILE_FREE,     /**< The time in seconds for freeing resources */
    QUDA_PROFILE_IO,       /**< time spent on file i/o */
    QUDA_PROFILE_CHRONO,   /**< time spent on chronology */
    QUDA_PROFILE_EIGEN,    /**< time spent on host-side Eigen */
    QUDA_PROFILE_ARPACK,   /**< time spent on host-side ARPACK */

    // lower level counters used in the dslash and api profiling
    QUDA_PROFILE_LOWER_LEVEL, /**< dummy timer to mark beginning of lower level timers which do not count towrads global time */
    QUDA_PROFILE_PACK_KERNEL,   /**< face packing kernel */
    QUDA_PROFILE_DSLASH_KERNEL, /**< dslash kernel */
    QUDA_PROFILE_GATHER,        /**< gather (device -> host) */
    QUDA_PROFILE_SCATTER,       /**< scatter (host -> device) */

    QUDA_PROFILE_LAUNCH_KERNEL,      /**< cudaLaunchKernel */
    QUDA_PROFILE_EVENT_RECORD,       /**< cuda event record  */
    QUDA_PROFILE_EVENT_QUERY,        /**< cuda event querying */
    QUDA_PROFILE_STREAM_WAIT_EVENT,  /**< stream waiting for event completion */
    QUDA_PROFILE_FUNC_SET_ATTRIBUTE, /**< set function attribute */

    QUDA_PROFILE_EVENT_SYNCHRONIZE,  /**< event synchronization */
    QUDA_PROFILE_STREAM_SYNCHRONIZE, /**< stream synchronization */
    QUDA_PROFILE_DEVICE_SYNCHRONIZE, /**< device synchronization */

    QUDA_PROFILE_MEMCPY_D2D_ASYNC,   /**< device to device async copy */
    QUDA_PROFILE_MEMCPY_D2H_ASYNC,   /**< device to host async copy */
    QUDA_PROFILE_MEMCPY2D_D2H_ASYNC, /**< device to host 2-d memcpy async copy*/
    QUDA_PROFILE_MEMCPY_H2D_ASYNC,   /**< host to device async copy */
    QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC,   /**< default async copy */

    QUDA_PROFILE_COMMS_START, /**< initiating communication */
    QUDA_PROFILE_COMMS_QUERY, /**< querying communication */

    QUDA_PROFILE_CONSTANT, /**< time spent setting CUDA constant parameters */

    QUDA_PROFILE_TOTAL, /**< The total time in seconds for the algorithm. Must be the penultimate type. */
    QUDA_PROFILE_COUNT  /**< The total number of timers we have.  Must be last enum type. */
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

    void Reset_(const char *func, const char *file, int line) {
      for (int idx=0; idx<QUDA_PROFILE_COUNT; idx++)
	profile[idx].Reset(func, file, line);
    }

    double Last(QudaProfileType idx) { 
      return profile[idx].last;
    }

    static void PrintGlobal();

    bool isRunning(QudaProfileType idx) { return profile[idx].running; }

  };

} // namespace quda

#endif


#undef PUSH_RANGE
#undef POP_RANGE

#define TPSTART(idx) Start_(__func__, __FILE__, __LINE__, idx)
#define TPSTOP(idx) Stop_(__func__, __FILE__, __LINE__, idx)
#define TPRESET() Reset_(__func__, __FILE__, __LINE__)

