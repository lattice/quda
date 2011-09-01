#include <tune_quda.h>

void TuneBase::Benchmark(dim3 &block)  {

  int count = 10;
  int threadBlockMin = 32;
  int threadBlockMax = 256;
  double time;
  double timeMin = 1e10;
  double gflopsMax = 0.0;
  dim3 blockOpt(1,1,1);

  cudaError_t error;

  for (int threads=threadBlockMin; threads<=threadBlockMax; threads+=32) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    block = dim3(threads,1,1);

    Flops(); // resets the flops counter
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

    error = cudaGetLastError();

    time = runTime / 1000;
    double flops = (double)Flops();
    double gflops = (flops*1e-9)/(time);

    if (time < timeMin && error == cudaSuccess) {
      timeMin = time;
      blockOpt = block;
      gflopsMax = gflops;
    }

    if (verbose >= QUDA_DEBUG_VERBOSE && error == cudaSuccess) 
      printfQuda("%-15s %d %f s, flops = %e, Gflop/s = %f\n", name, threads, time, (double)flops, gflops);
  }

  block = blockOpt;
  Flops(); // reset the flop counter

  if (block.x == 1) {
    printfQuda("Auto-tuning failed for %s\n", name);
  }

  if (verbose >= QUDA_VERBOSE) 
    printfQuda("Tuned %-15s with (%d,%d,%d) threads per block, Gflop/s = %f\n", name, block.x, block.y, block.z, gflopsMax);    

}
