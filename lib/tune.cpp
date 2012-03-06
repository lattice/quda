#include <tune_quda.h>
#include <cfloat> // for FLT_MAX
#include <typeinfo>
#include <map>

static std::map<TuneKey, TuneParam> tuneCache;

TuneParam tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity)
{
  static bool tuning = false;     // tuning in progress?
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

#if 0
void TuneBase::Benchmark(TuneParam &tune)
{
  dim3 &block = tune.block;
  int &sharedBytes = tune.shared_bytes;

  int count = 1;
  int threadBlockMin = 32;
  int threadBlockMax = 512;
  double time;
  double timeMin = 1e10;
  double gflopsMax = 0.0;
  dim3 blockOpt(1,1,1);
  int sharedOpt = 0;

  // set to a max of 16KiB, since higher will switch the cache config
  int sharedMax = 16384; //deviceProp.sharedMemPerBlock;
  //sharedMax = 0;

  cudaError_t error;

  // loop over amount of shared memory to add
  // this is the minimum amount, if kernel requests more it will use more
  int shared = 0;
  while (shared <= sharedMax) {
    for (int threads=threadBlockMin; threads<=threadBlockMax; threads+=32) {
      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      
      block = dim3(threads,1,1);
      sharedBytes = shared;

      Flops(); // resets the flops counter
      cudaThreadSynchronize();
      cudaGetLastError(); // clear error counter
      
      cudaEventRecord(start, 0);
      
      for (int c=0; c<count; c++) Apply();
      
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      
      float runTime;
      cudaEventElapsedTime(&runTime, start, end);
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      
      cudaThreadSynchronize();
      error = cudaGetLastError();
      
      time = runTime / 1000;
      double flops = (double)Flops();
      double gflops = (flops*1e-9)/(time);
      
      if (time < timeMin && error == cudaSuccess && checkLaunch()) {
	timeMin = time;
	blockOpt = block;
	sharedOpt = shared;
	gflopsMax = gflops;
      }
      
      if (verbose >= QUDA_DEBUG_VERBOSE && error == cudaSuccess) 
	printfQuda("%-15s %d %d %f s, flops = %e, Gflop/s = %f\n", name, threads, shared, time, (double)flops, gflops);
    } // block loop

    if (shared == 0) shared = 128;
    else shared *= 2;
  } // shared loop
    
  block = blockOpt;
  sharedBytes = sharedOpt;
  Flops(); // reset the flop counter

  if (block.x == 1) {
    printfQuda("Auto-tuning failed for %s\n", name);
  }

  if (verbose >= QUDA_VERBOSE) 
    printfQuda("Tuned %-15s with (%d,%d,%d) threads per block, %d bytes per block, Gflop/s = %f\n", 
	       name, block.x, block.y, block.z, sharedBytes, gflopsMax);    

}
#endif // 0
