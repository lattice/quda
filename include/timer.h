#pragma once

#include <sys/time.h>
#include <stack>
#include <quda_internal.h>
#include <util_quda.h>
#include <device.h>

namespace quda {

  /**
   * Use this for recording a fine-grained profile of a QUDA
   * algorithm.  This can be used for either host-side or device side
   * measurement.  When using host-side measurement, the execution
   * should host-device synchronous.  For device-side measurement, the
   * measurement will be done on the default stream unless specified
   * otherwise in the constructor.
   */
  template <bool device = false> struct Timer {
    /**< The cumulative sum of time */
    double time;

    /**< The last recorded time interval */
    double last_interval;

    /**< Used to store when the timer was last started */
    timeval host_start;

    /**< Used to store when the timer was last stopped */
    timeval host_stop;

    /**< Used to store when the timer was last started */
    qudaEvent_t device_start;

    /**< Used to store when the timer was last stopped */
    qudaEvent_t device_stop;

    /**< Which stream are we recording on */
    qudaStream_t stream;

    /**< Are we currently timing? */
    bool running;

    /**< Keep track of number of calls */
    int count;

    Timer(qudaStream_t stream = device::get_default_stream()) :
      time(0.0), last_interval(0.0), stream(stream), running(false), count(0)
    {
      if (device) {
        device_start = qudaChronoEventCreate();
        device_stop = qudaChronoEventCreate();
      }
    }

    ~Timer()
    {
      if (device) {
        qudaEventDestroy(device_start);
        qudaEventDestroy(device_stop);
      }
    }

    int ref_count = 0;

    /**
       @brief Start the timer
     */
    void start(const char * = nullptr, const char * = nullptr, int = 0)
    {
      if (running) { // if the timer has already started, we increment the ref counter and return
        ref_count++;
        return;
      }
      if (!device) {
        gettimeofday(&host_start, NULL);
      } else {
        qudaEventRecord(device_start, stream);
      }
      running = true;
    }

    /**
       @brief Update last_interval, but doesn't stop the time or
       increment the count.
     */
    bool peek(const char *func = nullptr, const char *file = nullptr, int line = 0)
    {
      if (!running) {
        printfQuda("ERROR: Cannot peek an unstarted timer (%s:%d in %s())", file ? file : "", line, func ? func : "");
        return false;
      }
      if (!device) {
        gettimeofday(&host_stop, NULL);
        long ds = host_stop.tv_sec - host_start.tv_sec;
        long dus = host_stop.tv_usec - host_start.tv_usec;
        last_interval = ds + 0.000001 * dus;
      } else {
        qudaEventRecord(device_stop, stream);
        qudaEventSynchronize(device_stop);
        last_interval = qudaEventElapsedTime(device_start, device_stop);
      }
      return true;
    }

    /**
       @brief Updates the last_interval time, stops the timer and increments the count.
     */
    bool stop(const char *func = nullptr, const char *file = nullptr, int line = 0)
    {
      if (ref_count > 0) {
        ref_count--;
        return true;
      }
      bool rtn = peek(func, file, line);
      time += last_interval;
      count++;

      running = false;
      return rtn;
    }

    double last() { return last_interval; }

    void reset(const char *func, const char *file, int line)
    {
      if (running) {
        printfQuda("ERROR: Cannot reset a started timer (%s:%d in %s())", file ? file : "", line, func ? func : "");
        errorQuda("Aborting");
      }
      time = 0.0;
      last_interval = 0.0;
      count = 0;
    }
  };

  using device_timer_t = Timer<true>;
  using host_timer_t = Timer<false>;

  /**< Enumeration type used for writing a simple but extensible profiling framework. */
  enum QudaProfileType {
    QUDA_PROFILE_H2D,          /**< host -> device transfers */
    QUDA_PROFILE_D2H,          /**< The time in seconds for device -> host transfers */
    QUDA_PROFILE_INIT,         /**< The time in seconds taken for initiation */
    QUDA_PROFILE_PREAMBLE,     /**< The time in seconds taken for any preamble */
    QUDA_PROFILE_COMPUTE,      /**< The time in seconds taken for the actual computation */
    QUDA_PROFILE_TRAINING,     /**< The time in seconds taken for training parameters */
    QUDA_PROFILE_COMMS,        /**< synchronous communication */
    QUDA_PROFILE_EPILOGUE,     /**< The time in seconds taken for any epilogue */
    QUDA_PROFILE_FREE,         /**< The time in seconds for freeing resources */
    QUDA_PROFILE_IO,           /**< time spent on file i/o */
    QUDA_PROFILE_CHRONO,       /**< time spent on chronology */
    QUDA_PROFILE_EIGEN,        /**< time spent on host-side Eigen */
    QUDA_PROFILE_EIGENLU,      /**< time spent on host-side Eigen LU */
    QUDA_PROFILE_EIGENEV,      /**< time spent on host-side Eigen EV */
    QUDA_PROFILE_EIGENQR,      /**< time spent on host-side Eigen QR */
    QUDA_PROFILE_ARPACK,       /**< time spent on host-side ARPACK */
    QUDA_PROFILE_HOST_COMPUTE, /**< time spent on miscellaneous host-side computation */

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

    QUDA_PROFILE_MEMCPY_D2D_ASYNC,     /**< device to device async copy */
    QUDA_PROFILE_MEMCPY_D2H_ASYNC,     /**< device to host async copy */
    QUDA_PROFILE_MEMCPY2D_D2H_ASYNC,   /**< device to host 2-d memcpy async copy*/
    QUDA_PROFILE_MEMCPY_H2D_ASYNC,     /**< host to device async copy */
    QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC, /**< default async copy */

    QUDA_PROFILE_COMMS_START, /**< initiating communication */
    QUDA_PROFILE_COMMS_QUERY, /**< querying communication */

    QUDA_PROFILE_CONSTANT, /**< time spent setting CUDA constant parameters */

    QUDA_PROFILE_TOTAL, /**< The total time in seconds for the algorithm. Must be the penultimate type. */
    QUDA_PROFILE_COUNT  /**< The total number of timers we have.  Must be last enum type. */
  };

  class TimeProfile {
    std::string fname;  /**< Which function are we profiling */
#ifdef INTERFACE_NVTX
    static const uint32_t nvtx_colors[];// = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
    static const int nvtx_num_colors;// = sizeof(nvtx_colors)/sizeof(uint32_t);
#endif
    array<host_timer_t, QUDA_PROFILE_COUNT> profile;
    static std::string pname[];

    bool switchOff;
    bool use_global;

    std::stack<QudaProfileType> pt_stack; /**< A stack used for recursive profiling */

    static void StopGlobal(const char *func, const char *file, int line, QudaProfileType idx);
    static void StartGlobal(const char *func, const char *file, int line, QudaProfileType idx);

  public:
    TimeProfile() = default;
    TimeProfile(const TimeProfile &) = default;
    TimeProfile &operator=(const TimeProfile &) = default;

    TimeProfile(std::string fname) : fname(fname), switchOff(false), use_global(true) { ; }

    TimeProfile(std::string fname, bool use_global) : fname(fname), switchOff(false), use_global(use_global) { ; }

    auto Name() const { return fname; }

    /**< Print out the profile information */
    void Print();

    void StartTotal(const char *func, const char *file, int line, QudaProfileType idx);
    void StopTotal(const char *func, const char *file, int line, QudaProfileType idx);

    void Start_(const char *func, const char *file, int line, QudaProfileType idx);
    void Stop_(const char *func, const char *file, int line, QudaProfileType idx);

    void Reset_(const char *func, const char *file, int line) {
      for (int idx = 0; idx < QUDA_PROFILE_COUNT; idx++) profile[idx].reset(func, file, line);
    }

    double Last(QudaProfileType idx) { return profile[idx].last_interval; }

    static void PrintGlobal();
  };

  /**
     @brief Container that we use for pushing a profile onto the
     profile stack.  While this object is in scope it will exist on
     the profile stack, and be popped when its destructor is called.
   */
  struct pushProfile {
    static inline double secs_dummy = 0;
    static inline double gflops_dummy = 0;
    TimeProfile &profile;
    double &secs;
    double &gflops;
    uint64_t flops;
    bool active = false;
    pushProfile(TimeProfile &profile, double &secs = secs_dummy, double &gflops = gflops_dummy);
    virtual ~pushProfile();
  };

  /**
     @brief Return a reference to the present profile at the top of
     the stack
   */
  TimeProfile &getProfile();

} // namespace quda

#define TPSTART(idx) Start_(__func__, __FILE__, __LINE__, idx)
#define TPSTOP(idx) Stop_(__func__, __FILE__, __LINE__, idx)
#define TPRESET() Reset_(__func__, __FILE__, __LINE__)
