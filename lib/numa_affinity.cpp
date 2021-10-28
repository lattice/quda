
/* Originally from Galen Arnold, NCSA arnoldg@ncsa.illinois.edu
 * modified by Guochun Shi
 *
 */

#include <numa_affinity.h>
#include <quda_internal.h>

#ifdef NUMA_NVML
#include <nvml.h>
#endif


int setNumaAffinityNVML(int devid)
{
#ifdef NUMA_NVML
  nvmlReturn_t result;

  result = nvmlInit();
  if (NVML_SUCCESS != result)
  {
    warningQuda("Failed to determine NUMA affinity for device %d (NVML Init failed)", devid);
    return -1;
  }
  nvmlDevice_t device;
  result = nvmlDeviceGetHandleByIndex(devid, &device);
  if (NVML_SUCCESS != result)
  {
    warningQuda("Failed to determine NUMA affinity for device %d (NVML DeviceGetHandle failed)", devid);
    return -1;
  }
  result = nvmlDeviceSetCpuAffinity(device);
  if (NVML_SUCCESS != result)
  {
    warningQuda("Failed to determine NUMA affinity for device %d (NVML DeviceSetCpuAffinity failed)", devid);
    return -1;
  }
  else{
    printfQuda("Set NUMA affinity for device %d (NVML DeviceSetCpuAffinity)\n", devid);
  }
  result = nvmlShutdown();
  if (NVML_SUCCESS != result)
  {
    warningQuda("Failed to determine NUMA affinity for device %d (NVML Shutdown failed)", devid);
    return -1;
  }
  return 0;
#else
  warningQuda("Failed to determine NUMA affinity for device %d (NVML not supported in quda build)", devid);
  return -1;
#endif
}
