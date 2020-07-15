#include <quda.h>
#include <quda_internal.h>
#include <comm_quda.h>

/*
 * Set the device that QUDA uses.
 */
void
initQudaDeviceTarget(int dev)
{
  cudaGetDeviceProperties(&deviceProp, dev);

  { // determine if we will do CPU or GPU data reordering (default is GPU)
    char *reorder_str = getenv("QUDA_REORDER_LOCATION");

    if (!reorder_str || (strcmp(reorder_str,"CPU") && strcmp(reorder_str,"cpu")) ) {
      warningQuda("Data reordering done on GPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      quda::reorder_location_set(QUDA_CUDA_FIELD_LOCATION);
    } else {
      warningQuda("Data reordering done on CPU (set with QUDA_REORDER_LOCATION=GPU/CPU)");
      quda::reorder_location_set(QUDA_CPU_FIELD_LOCATION);
    }
  }

}
