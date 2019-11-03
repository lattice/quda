#include <unittest/unittest.h>
#include <hip/hip_runtime_api.h>
#include <thrust/detail/util/align.h>
#include "hip/hip_runtime.h"

template<typename T>
void TestCudaMallocResultAligned(const std::size_t n)
{
  T *ptr = 0;
  hipMalloc(&ptr, n * sizeof(T));
  hipFree(ptr);

  ASSERT_EQUAL(true, thrust::detail::util::is_aligned(ptr));
}
DECLARE_VARIABLE_UNITTEST(TestCudaMallocResultAligned);

