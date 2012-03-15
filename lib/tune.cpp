#include <tune_quda.h>
#include <cfloat> // for FLT_MAX
#include <typeinfo>
#include <map>

static std::map<TuneKey, TuneParam> tuneCache;

TuneParam tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity)
{
  static bool tuning = false; // tuning in progress?
  static const Tunable *active_tunable; // for error checking
  static TuneParam param;

  TuneParam best_param;
  cudaError_t error;
  cudaEvent_t start, end;
  float elapsed_time, best_time;

  const TuneKey key = tunable.tuneKey();

  if (enabled == QUDA_TUNE_NO) {
    tunable.initTuneParam(param);
  } else if (tuneCache.count(key)) {
    param = tuneCache[key];
  } else if (!tuning) {

    tuning = true;
    active_tunable = &tunable;
    best_time = FLT_MAX;
    tunable.preTune();

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (verbosity >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Tuning %s with %s at vol=%s\n", key.name.c_str(), key.aux.c_str(), key.volume.c_str());
    }

    tunable.initTuneParam(param);
    while (tuning) {
      cudaThreadSynchronize();
      cudaGetLastError(); // clear error counter
      cudaEventRecord(start, 0);
      for (int i=0; i<tunable.tuningIter(); i++) {
	tunable.apply(0);  // calls tuneLaunch() again, which simply returns the currently active param
      }
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      cudaThreadSynchronize();
      error = cudaGetLastError();
      elapsed_time /= (1e3 * tunable.tuningIter());
      if ((elapsed_time < best_time) && (error == cudaSuccess)) {
	best_time = elapsed_time;
	best_param = param;
      }
      if ((verbosity >= QUDA_DEBUG_VERBOSE) && (error == cudaSuccess)) {
	printfQuda("    %s gives %s\n", tunable.paramString(param).c_str(), tunable.perfString(elapsed_time).c_str());
      }
      tuning = tunable.advanceTuneParam(param);
    }

    if (best_time == FLT_MAX) {
      errorQuda("Auto-tuning failed for %s with %s at vol=%s", key.name.c_str(), key.aux.c_str(), key.volume.c_str());
    }
    if (verbosity >= QUDA_VERBOSE) {
      printfQuda("Tuned %s giving %s", tunable.paramString(best_param).c_str(), tunable.perfString(best_time).c_str());
      printfQuda(" for %s with %s\n", key.name.c_str(), key.aux.c_str());
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    tunable.postTune();
    param = best_param;
    tuneCache[key] = best_param;

  } else if (&tunable != active_tunable) {
    errorQuda("Unexpected call to tuneLaunch() in %s::apply()", typeid(tunable).name());
  }

  return param;
}
