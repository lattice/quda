#include <util_quda.h>
#include <quda_internal.h>

cl::sycl::queue defaultQueue;

cudaDeviceProp deviceProp;
qudaStream_t *streams;
#define qudaStreamDefault 0

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;

      qudaGetDeviceProperties(&deviceProp, dev);
    }

    void create_context()
    {
      //cl::sycl::default_selector my_selector;
      cl::sycl::host_selector my_selector;
      defaultQueue = cl::sycl::queue(my_selector);

      streams = new qudaStream_t[Nstream];

      int greatestPriority;
      int leastPriority;
      qudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
      for (int i=0; i<Nstream-1; i++) {
        qudaStreamCreateWithPriority(&streams[i], qudaStreamDefault, greatestPriority);
      }
      qudaStreamCreateWithPriority(&streams[Nstream-1], qudaStreamDefault, leastPriority);
    }

    void destroy()
    {
      if (streams) {
        for (int i=0; i<Nstream; i++) qudaStreamDestroy(streams[i]);
        delete []streams;
        streams = nullptr;
      }

      char *device_reset_env = getenv("QUDA_DEVICE_RESET");
      if (device_reset_env && strcmp(device_reset_env,"1") == 0) {
        // end this CUDA context
        qudaDeviceReset();
      }
    }

    namespace profile {

      void start()
      {
        //cudaProfilerStart();
      }

      void stop()
      {
        //cudaProfilerStop();
      }

    }

  }
}
