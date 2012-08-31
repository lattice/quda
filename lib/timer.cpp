#include <quda_internal.h>

namespace quda {

  /**< Print out the profile information */
  void TimeProfile::Print() {
    printfQuda("   %s Total time = %g secs\n", (const char*)&fname[0], profile[QUDA_PROFILE_TOTAL].time);
    double accounted = 0.0;
    for (int i=0; i<QUDA_PROFILE_COUNT-1; i++) {
      printfQuda("     %15s     = %f secs (%5.2g%%)\n", (const char*)&pname[i][0],  
		 profile[i].time, 100*profile[i].time/profile[QUDA_PROFILE_TOTAL].time);
      accounted += profile[i].time;
    }
    double missing = profile[QUDA_PROFILE_TOTAL].time - accounted;
    printfQuda("     total accounted     = %f secs (%5.2g%%)\n", 
	       accounted, 100*accounted/profile[QUDA_PROFILE_TOTAL].time);
    printfQuda("     total missing       = %f secs (%5.2g%%)\n", 
	       missing, 100*missing/profile[QUDA_PROFILE_TOTAL].time);
  }

  std::string TimeProfile::pname[] = { "download",  "upload", "init", "preamble", "compute", 
				     "epilogue", "free", "total" };

}

