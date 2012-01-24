#include <tune_quda.h>

void TuneBase::Benchmark(TuneParam &tune)  {

  dim3 &block = tune.block;
  int &sharedBytes = tune.sharedBytes;

  int count = 10;
  int threadBlockMin = 32;
  int threadBlockMax = 256;
  double time;
  double timeMin = 1e10;
  double gflopsMax = 0.0;
  dim3 blockOpt(1,1,1);

  cudaError_t error;

  int sharedOpt = 0;
  int sharedMax = 0;

  // loop over amount of shared memory to add
  for (int shared = 0; shared<sharedMax; shared+=1024) { // 1 KiB granularity

    for (int threads=threadBlockMin; threads<=threadBlockMax; threads+=32) {
      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      
      block = dim3(threads,1,1);
      
      Flops(); // resets the flops counter
      cudaThreadSynchronize();
      cudaGetLastError(); // clear error counter
      
      cudaEventRecord(start, 0);
      cudaEventSynchronize(start);
      
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
      
      if (time < timeMin && error == cudaSuccess) {
	timeMin = time;
	blockOpt = block;
	sharedOpt = shared;
	gflopsMax = gflops;
      }
      
      if (verbose >= QUDA_DEBUG_VERBOSE && error == cudaSuccess) 
	printfQuda("%-15s %d %d %f s, flops = %e, Gflop/s = %f\n", name, threads, shared, time, (double)flops, gflops);
    } // block loop
  } // shared loop
    
  block = blockOpt;
  sharedBytes = sharedOpt;
  Flops(); // reset the flop counter

  if (block.x == 1) {
    printfQuda("Auto-tuning failed for %s\n", name);
  }

  if (verbose >= QUDA_VERBOSE) 
    printfQuda("Tuned %-15s with (%d,%d,%d) threads per block, %d KiB per block, Gflop/s = %f\n", 
	       name, block.x, block.y, block.z, sharedOpt, gflopsMax);    

}
