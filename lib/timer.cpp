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
      if (profile[i].time > 0.0) {
	printfQuda("     %15s     = %f secs (%6.3g%%)\n", (const char*)&pname[i][0],  
		   profile[i].time, 100*profile[i].time/profile[QUDA_PROFILE_TOTAL].time);
	accounted += profile[i].time;
      }
    }
    if (accounted > 0.0) {
      double missing = profile[QUDA_PROFILE_TOTAL].time - accounted;
      printfQuda("     total accounted     = %f secs (%6.3g%%)\n", 
		 accounted, 100*accounted/profile[QUDA_PROFILE_TOTAL].time);
      printfQuda("     total missing       = %f secs (%6.3g%%)\n", 
		 missing, 100*missing/profile[QUDA_PROFILE_TOTAL].time);
    }

    if (accounted > profile[QUDA_PROFILE_TOTAL].time) {
      warningQuda("Accounted time %f secs in %s is greater than total time %f secs\n", 
		  accounted, (const char*)&fname[0], profile[QUDA_PROFILE_TOTAL].time);
    }
    
  }

  std::string TimeProfile::pname[] = { "download",  "upload", "init", "preamble", "compute", 
				       "epilogue", "free", "total" };
  
}

