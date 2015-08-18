#include <quda_internal.h>

namespace quda {

  /**< Print out the profile information */
  void TimeProfile::Print() {
    if (profile[QUDA_PROFILE_TOTAL].time > 0.0) {
      printfQuda("\n   %20s Total time = %g secs\n", fname.c_str(), 
		 profile[QUDA_PROFILE_TOTAL].time);
    }

    double accounted = 0.0;
    for (int i=0; i<QUDA_PROFILE_COUNT-1; i++) {
      if (profile[i].count > 0) {
	printfQuda("     %17s     = %f secs (%6.3g%%), with %8d calls at %e us per call\n", 
		   (const char*)&pname[i][0],  profile[i].time, 
		   100*profile[i].time/profile[QUDA_PROFILE_TOTAL].time,
		   profile[i].count, 1e6*profile[i].time/profile[i].count);
	accounted += profile[i].time;
      }
    }
    if (accounted > 0.0) {
      double missing = profile[QUDA_PROFILE_TOTAL].time - accounted;
      printfQuda("     total accounted       = %f secs (%6.3g%%)\n", 
		 accounted, 100*accounted/profile[QUDA_PROFILE_TOTAL].time);
      printfQuda("     total missing         = %f secs (%6.3g%%)\n", 
		 missing, 100*missing/profile[QUDA_PROFILE_TOTAL].time);
    }

    if (accounted > profile[QUDA_PROFILE_TOTAL].time) {
      warningQuda("Accounted time %f secs in %s is greater than total time %f secs\n", 
		  accounted, (const char*)&fname[0], profile[QUDA_PROFILE_TOTAL].time);
    }
    
  }

  std::string TimeProfile::pname[] = { "download",  "upload", "init", "preamble", "compute", 
				       "epilogue", "free", "dummy", "pack kernel", "dslash kernel", 
				       "gather", "scatter", "event record", 
				       "event query", "stream wait event", 
				       "comms", "comms start", "comms query", "constant", 
				       "total" };

#ifdef INTERFACE_NVTX
  const uint32_t TimeProfile::nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
  const int TimeProfile::nvtx_num_colors = sizeof(nvtx_colors)/sizeof(uint32_t);
#endif
  
  Timer TimeProfile::global_profile[QUDA_PROFILE_COUNT];
  bool TimeProfile::global_switchOff[QUDA_PROFILE_COUNT] = {};
  int TimeProfile::global_total_level[QUDA_PROFILE_COUNT] = {};

  void TimeProfile::PrintGlobal() {
    if (global_profile[QUDA_PROFILE_TOTAL].time > 0.0) {
      printfQuda("\n   %20s Total time = %g secs\n", "QUDA",
                 global_profile[QUDA_PROFILE_TOTAL].time);
    }

    double accounted = 0.0;
    bool print_timer = true; // whether to print that timer
    for (int i=0; i<QUDA_PROFILE_COUNT-1; i++) {
      if (i==QUDA_PROFILE_LOWER_LEVEL) print_timer=false; // we do not want to print detailed lower level timers
      if (global_profile[i].count > 0) {
        if (print_timer) printfQuda("     %17s     = %f secs (%6.3g%%), with %8d calls at %e us per call\n",
                   (const char*)&pname[i][0],  global_profile[i].time,
                   100*global_profile[i].time/global_profile[QUDA_PROFILE_TOTAL].time,
                   global_profile[i].count, 1e6*global_profile[i].time/global_profile[i].count);
        accounted += global_profile[i].time;
      }
    }
    if (accounted > 0.0) {
      double missing = global_profile[QUDA_PROFILE_TOTAL].time - accounted;
      printfQuda("     total accounted       = %f secs (%6.3g%%)\n",
                 accounted, 100*accounted/global_profile[QUDA_PROFILE_TOTAL].time);
      printfQuda("     total missing         = %f secs (%6.3g%%)\n",
                 missing, 100*missing/global_profile[QUDA_PROFILE_TOTAL].time);
    }

    if (accounted > global_profile[QUDA_PROFILE_TOTAL].time) {
      warningQuda("Accounted time %f secs in %s is greater than total time %f secs\n",
                  accounted, "QUDA", global_profile[QUDA_PROFILE_TOTAL].time);
    }

  }

}

