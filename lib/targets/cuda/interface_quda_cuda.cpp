#include <quda.h>
#include <quda_fortran.h>
#include <quda_internal.h>
#include <comm_quda.h>

/*
 * Set the device that QUDA uses.
 */
void initQudaDeviceTarget(int dev) {

  int driver_version;
  cudaDriverGetVersion(&driver_version);
  printfQuda("CUDA Driver version = %d\n", driver_version);

  int runtime_version;
  cudaRuntimeGetVersion(&runtime_version);
  printfQuda("CUDA Runtime version = %d\n", runtime_version);

#ifdef QUDA_NVML
  nvmlReturn_t result = nvmlInit();
  if (NVML_SUCCESS != result) errorQuda("NVML Init failed with error %d", result);
  const int length = 80;
  char graphics_version[length];
  result = nvmlSystemGetDriverVersion(graphics_version, length);
  if (NVML_SUCCESS != result) errorQuda("nvmlSystemGetDriverVersion failed with error %d", result);
  printfQuda("Graphic driver version = %s\n", graphics_version);
  result = nvmlShutdown();
  if (NVML_SUCCESS != result) errorQuda("NVML Shutdown failed with error %d", result);
#endif

#if defined(MULTI_GPU) && (CUDA_VERSION == 4000)
  //check if CUDA_NIC_INTEROP is set to 1 in the enviroment
  // not needed for CUDA >= 4.1
  char* cni_str = getenv("CUDA_NIC_INTEROP");
  if(cni_str == nullptr){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set");
  }
  int cni_int = atoi(cni_str);
  if (cni_int != 1){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set to 1");
  }
#endif

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    errorQuda("No CUDA devices found");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProp, i);
    checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Found device %d: %s\n", i, deviceProp.name);
    }
  }

#ifdef MULTI_GPU
  if (dev < 0) {
    if (!comms_initialized) {
      errorQuda("initDeviceQuda() called with a negative device ordinal, but comms have not been initialized");
    }
    dev = comm_gpuid();
  }
#else
  if (dev < 0 || dev >= 16) errorQuda("Invalid device number %d", dev);
#endif

  cudaGetDeviceProperties(&deviceProp, dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }


// Check GPU and QUDA build compatibiliy
// 4 cases:
// a) QUDA and GPU match: great
// b) QUDA built for higher compute capability: error
// c) QUDA built for lower major compute capability: warn if QUDA_ALLOW_JIT, else error
// d) QUDA built for same major compute capability but lower minor: warn

  const int my_major = __COMPUTE_CAPABILITY__ / 100;
  const int my_minor = (__COMPUTE_CAPABILITY__  - my_major * 100) / 10;
// b) UDA was compiled for a higher compute capability
  if (deviceProp.major * 100 + deviceProp.minor * 10 < __COMPUTE_CAPABILITY__)
    errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. ** \n --- Please set the correct QUDA_GPU_ARCH when running cmake.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);


// c) QUDA was compiled for a lower compute capability
  if (deviceProp.major < my_major) {
    char *allow_jit_env = getenv("QUDA_ALLOW_JIT");
    if (allow_jit_env && strcmp(allow_jit_env, "1") == 0) {
      if (getVerbosity() > QUDA_SILENT) warningQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- Jitting the PTX since QUDA_ALLOW_JIT=1 was set. Note that this will take some time.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);
    } else {
      errorQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n --- Please set the correct QUDA_GPU_ARCH when running cmake.\n If you want the PTX to be jitted for your current GPU arch please set the enviroment variable QUDA_ALLOW_JIT=1.", deviceProp.major, deviceProp.minor, my_major, my_minor);
    }
  }
// d) QUDA built for same major compute capability but lower minor
  if (deviceProp.major == my_major and deviceProp.minor > my_minor) {
    warningQuda("** Running on a device with compute capability %i.%i but QUDA was compiled for %i.%i. **\n -- This might result in a lower performance. Please consider adjusting QUDA_GPU_ARCH when running cmake.\n", deviceProp.major, deviceProp.minor, my_major, my_minor);
  }

  if (getVerbosity() >= QUDA_SUMMARIZE) {
    printfQuda("Using device %d: %s\n", dev, deviceProp.name);
  }
#ifndef USE_QDPJIT
  cudaSetDevice(dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
#endif


#if ((CUDA_VERSION >= 6000) && defined NUMA_NVML)
  char *enable_numa_env = getenv("QUDA_ENABLE_NUMA");
  if (enable_numa_env && strcmp(enable_numa_env, "0") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling numa_affinity\n");
  }
  else{
    setNumaAffinityNVML(dev);
  }
#endif


  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaGetDeviceProperties(&deviceProp, dev);

  { // determine if we will do CPU or GPU data reordering (default is GPU)
    char *reorder_str = getenv("QUDA_REORDER_LOCATION");

    if (!reorder_str || (strcmp(reorder_str,"CPU") && strcmp(reorder_str,"cpu")) ) {
      warningQuda("Data reordering done on GPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      reorder_location_set(QUDA_CUDA_FIELD_LOCATION);
    } else {
      warningQuda("Data reordering done on CPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      reorder_location_set(QUDA_CPU_FIELD_LOCATION);
    }
  }

  cublas::init();
}
