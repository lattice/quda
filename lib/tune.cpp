#include <tune_quda.h>

void TuneBase::Benchmark(TuneParam &tune)  {

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
