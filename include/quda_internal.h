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

#define MAX_SHORT 32767.0f

// The "Quda" prefix is added to avoid collisions with other libraries.

#define GaugeFieldOrder QudaGaugeFieldOrder
#define DiracFieldOrder QudaDiracFieldOrder
#define CloverFieldOrder QudaCloverFieldOrder
#define InverterType QudaInverterType  
#define MatPCType QudaMatPCType
#define SolutionType QudaSolutionType
#define MassNormalization QudaMassNormalization
#define PreserveSource QudaPreserveSource
#define DagType QudaDagType
#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
#include <enum_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>

// Use bindless texture on Kepler
#if (__COMPUTE_CAPABILITY__ >= 300) && (CUDA_VERSION >= 5000)
#define USE_TEXTURE_OBJECTS
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

    void Start() {
      if (running) errorQuda("Cannot start an already running timer");
      gettimeofday(&start, NULL);
      running = true;
    }

    void Stop() {
      if (!running) errorQuda("Cannot start an already running timer");
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

  struct TimeProfile {
    std::string fname;  /**< Which function are we profiling */

    Timer profile[QUDA_PROFILE_COUNT];
    static std::string pname[];

    bool switchOff;
    
    TimeProfile(std::string fname) : fname(fname), switchOff(false) { ; }

    /**< Print out the profile information */
    void Print();

    void Start(QudaProfileType idx) { 
      // if total timer isn't running, then start it running
      if (!profile[QUDA_PROFILE_TOTAL].running && idx != QUDA_PROFILE_TOTAL) {
	profile[QUDA_PROFILE_TOTAL].Start(); 
	switchOff = true;
      }

      profile[idx].Start(); 
    }

    void Stop(QudaProfileType idx) { 
      profile[idx].Stop(); 

      // switch off total timer if we need to
      if (switchOff && idx != QUDA_PROFILE_TOTAL) {
	profile[QUDA_PROFILE_TOTAL].Stop(); 
	switchOff = false;
      }
    }

    double Last(QudaProfileType idx) { 
      return profile[idx].last;
    }

  };

#ifdef MULTI_GPU
  const int Nstream = 9;
#else
  const int Nstream = 1;
#endif

} // namespace quda

#endif // _QUDA_INTERNAL_H
