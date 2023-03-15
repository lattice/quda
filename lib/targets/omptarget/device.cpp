#include <util_quda.h>
#include <quda_internal.h>
#include <target_device.h>

// OMP TARGET TODO: most of what follows are guess work.
// We also don't support streams.

// until we have a way to query device properties correctly

#ifndef QUDA_OMP_MAX_TEAMS
#define QUDA_OMP_MAX_TEAMS 65536
#endif

#ifndef QUDA_MAX_THREADS_PER_PROCESSOR
#define QUDA_MAX_THREADS_PER_PROCESSOR 2048
#endif

#ifndef QUDA_PROCESSOR_COUNT
#define QUDA_PROCESSOR_COUNT 64
#endif

#ifndef QUDA_MAX_BLOCKS_PER_PROCESSOR
#define QUDA_MAX_BLOCKS_PER_PROCESSOR 32
#endif

static char ompdevname[] = "OpenMP Target Device";
static char omphostname[] = "OpenMP Host Device";
struct DeviceProp{
  char*name;
  int id;
  unsigned int num_procs;
  unsigned int max_teams;
  unsigned int max_threads;
};

static void getDeviceProperties(DeviceProp*p,int dev)
{
  constexpr int max_threads = quda::device::max_block_size();
  int m = 0;

  if(dev>=0 && dev<=omp_get_num_devices())
    p->name = ompdevname;
  else
    p->name = omphostname;

  p->id = dev;
  #pragma omp target teams device(dev) thread_limit(max_threads) map(from:m)
  if(omp_get_team_num()==0)
    m = omp_get_num_procs();
  p->num_procs = m;

/*
  #pragma omp target device(dev) map(from:m)
  m = omp_get_max_teams();
  p->max_teams = m;
*/
  p->max_teams = QUDA_OMP_MAX_TEAMS;

  #pragma omp target teams device(dev) thread_limit(max_threads) map(from:m)
  if(omp_get_team_num()==0)
    m = omp_get_max_threads();
  p->max_threads= m;
}

static DeviceProp deviceProp;

namespace quda
{

  namespace device
  {

    static bool initialized = false;

    void print_device(DeviceProp dp)
    {
      int dev = dp.id;
      printfQuda("%d - name:        %s\n", dev, dp.name);
      printfQuda("%d - num_procs:   %d\n", dev, dp.num_procs);
      printfQuda("%d - max_teams:   %d\n", dev, dp.max_teams);
      printfQuda("%d - max_threads: %d\n", dev, dp.max_threads);
    }

    void init(int dev)
    {
      if (initialized) return;
      initialized = true;

      getDeviceProperties(&deviceProp, dev);

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Using device %d: %s\n", dev, deviceProp.name);
        print_device(deviceProp);
      }
      omp_set_default_device(dev);
    }

    int get_device_count()
    {
      static int device_count = -1;
      if (device_count < 0) {
        device_count = omp_get_num_devices();
        if (device_count == 0)
          warningQuda("No non-host devices found");
      }
      return device_count;
    }

    void print_device_properties()
    {
      DeviceProp dp;
      for (int device = 0; device < get_device_count(); device++) {
        getDeviceProperties(&dp, device);
        print_device(dp);
      }
    }

    void create_context() { }

    void destroy() { }

    qudaStream_t get_stream(unsigned int i)
    {
      return qudaStream_t{static_cast<int>(i)};
    }

    qudaStream_t get_default_stream()
    {
      return qudaStream_t{0};
    }

    unsigned int get_default_stream_idx()
    {
      return 0;
    }

    bool managed_memory_supported() { return false; }

    bool shared_memory_atomic_supported() { return false; }

    size_t max_default_shared_memory() { return device::max_shared_memory_size(); }

    size_t max_dynamic_shared_memory() { return device::max_shared_memory_size(); }

    unsigned int max_threads_per_block() { return deviceProp.max_threads; }

    unsigned int max_threads_per_processor() { return QUDA_MAX_THREADS_PER_PROCESSOR; }

    unsigned int max_threads_per_block_dim(int i) { return deviceProp.max_threads; }

    unsigned int max_grid_size(int i) { return deviceProp.max_teams; }

    unsigned int processor_count() { return QUDA_PROCESSOR_COUNT; }

    unsigned int max_blocks_per_processor() { return QUDA_MAX_BLOCKS_PER_PROCESSOR; }

    namespace profile
    {

      void start() { /* cudaProfilerStart(); */ }

      void stop() { /* cudaProfilerStop(); */ }

    } // namespace profile

  } // namespace device
} // namespace quda
