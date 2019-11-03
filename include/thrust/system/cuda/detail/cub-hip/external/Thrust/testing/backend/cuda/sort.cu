#include "hip/hip_runtime.h"
#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Compare, typename Iterator2>
__global__
void sort_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Compare comp, Iterator2 is_supported)
{
//#if (__CUDA_ARCH__ >= 200) //Need to Recheck
if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ {
  *is_supported = true;
  thrust::sort(exec, first, last, comp);
}
//#else //commented while converting the flags
else{
  *is_supported = false;
}
//#endif //commented while converting the flags
}


template<typename T>
struct my_less
{
  __host__ __device__
  bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};


template<typename T, typename ExecutionPolicy, typename Compare>
void TestComparisonSortDevice(ExecutionPolicy exec, const size_t n, Compare comp)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::device_vector<bool> is_supported(1);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_kernel), dim3(1), dim3(1), 0, 0, exec, d_data.begin(), d_data.end(), comp, is_supported.begin());

  if(is_supported[0])
  {
    thrust::sort(h_data.begin(), h_data.end(), comp);
    
    ASSERT_EQUAL(h_data, d_data);
  }
};


template<typename T>
  struct TestComparisonSortDeviceSeq
{
  void operator()(const size_t n)
  {
    TestComparisonSortDevice<T>(thrust::seq, n, my_less<T>());
  }
};
VariableUnitTest<
  TestComparisonSortDeviceSeq,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestComparisonSortDeviceSeqInstance;


template<typename T>
  struct TestComparisonSortDeviceDevice
{
  void operator()(const size_t n)
  {
    TestComparisonSortDevice<T>(thrust::device, n, my_less<T>());
  }
};
VariableUnitTest<
  TestComparisonSortDeviceDevice,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestComparisonSortDeviceDeviceDeviceInstance;


template<typename T, typename ExecutionPolicy>
void TestSortDevice(ExecutionPolicy exec, const size_t n)
{
  TestComparisonSortDevice<T>(exec, n, thrust::less<T>());
};


template<typename T>
  struct TestSortDeviceSeq
{
  void operator()(const size_t n)
  {
    TestSortDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<
  TestSortDeviceSeq,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestSortDeviceSeqInstance;


template<typename T>
  struct TestSortDeviceDevice
{
  void operator()(const size_t n)
  {
    TestSortDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<
  TestSortDeviceDevice,
  unittest::type_list<unittest::int8_t,unittest::int32_t>
> TestSortDeviceDeviceInstance;


void TestSortCudaStreams()
{
  thrust::device_vector<int> keys(10);

  keys[0] = 9;
  keys[1] = 3;
  keys[2] = 2;
  keys[3] = 0;
  keys[4] = 4;
  keys[5] = 7;
  keys[6] = 8;
  keys[7] = 1;
  keys[8] = 5;
  keys[9] = 6;

  hipStream_t s;
  hipStreamCreate(&s);

  thrust::sort(thrust::cuda::par.on(s), keys.begin(), keys.end());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end()));
                      
  hipStreamDestroy(s);
}
DECLARE_UNITTEST(TestSortCudaStreams);


void TestComparisonSortCudaStreams()
{
  thrust::device_vector<int> keys(10);

  keys[0] = 9;
  keys[1] = 3;
  keys[2] = 2;
  keys[3] = 0;
  keys[4] = 4;
  keys[5] = 7;
  keys[6] = 8;
  keys[7] = 1;
  keys[8] = 5;
  keys[9] = 6;

  hipStream_t s;
  hipStreamCreate(&s);

  thrust::sort(thrust::cuda::par.on(s), keys.begin(), keys.end(), my_less<int>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end(), my_less<int>()));
                      
  hipStreamDestroy(s);
}
DECLARE_UNITTEST(TestComparisonSortCudaStreams);

