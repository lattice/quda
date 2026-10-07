#include <quda_internal.h>

#ifdef QUDA_REDUCTION_ALGORITHM_REPRODUCIBLE
#include <float_vector.h>
#include <set>

namespace quda::reproducible
{

#pragma omp declare target
  RFA_bins<reduction_t> bin_device_buffer;
#pragma omp end declare target

  void init_rfa_bins()
  {
    static std::set<int> initialized;
    const int device = omp_get_default_device();
#pragma omp critical(quda_rfa_bins)
    {
      if (initialized.count(device) == 0) {
        bin_device_buffer = reducer::get_rfa_bins();
#pragma omp target update to(bin_device_buffer) device(device)
        initialized.insert(device);
      }
    }
  }

} // namespace quda::reproducible
#endif
