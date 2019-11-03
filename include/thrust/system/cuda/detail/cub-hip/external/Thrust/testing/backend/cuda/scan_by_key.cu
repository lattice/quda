#include "hip/hip_runtime.h"
#include <unittest/unittest.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void inclusive_scan_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::inclusive_scan_by_key(exec, keys_first, keys_last, values_first, result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void exclusive_scan_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::exclusive_scan_by_key(exec, keys_first, keys_last, values_first, result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename T>
__global__
void exclusive_scan_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result, T init)
{
  thrust::exclusive_scan_by_key(exec, keys_first, keys_last, values_first, result, init);
}


template<typename ExecutionPolicy>
void TestScanByKeyDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  thrust::host_vector<int> h_keys(n);
  for(size_t i = 0, k = 0; i < n; i++)
  {
    h_keys[i] = k;
    if(rand() % 10 == 0)
    {
      k++;
    }
  }
  thrust::device_vector<int> d_keys = h_keys;
  
  thrust::host_vector<int>   h_vals = unittest::random_integers<int>(n);
  for(size_t i = 0; i < n; i++)
  {
    h_vals[i] = i % 10;
  }
  thrust::device_vector<int> d_vals = h_vals;
  
  thrust::host_vector<int>   h_output(n);
  thrust::device_vector<int> d_output(n);
  
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  hipLaunchKernelGGL(HIP_KERNEL_NAME(inclusive_scan_by_key_kernel), dim3(1), dim3(1), 0, 0, exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  hipLaunchKernelGGL(HIP_KERNEL_NAME(exclusive_scan_by_key_kernel), dim3(1), dim3(1), 0, 0, exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), 11);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(exclusive_scan_by_key_kernel), dim3(1), dim3(1), 0, 0, exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin(), 11);
  ASSERT_EQUAL(d_output, h_output);
  
  // in-place scans
  h_output = h_vals;
  d_output = d_vals;
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin());
  hipLaunchKernelGGL(HIP_KERNEL_NAME(inclusive_scan_by_key_kernel), dim3(1), dim3(1), 0, 0, exec,d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
  
  h_output = h_vals;
  d_output = d_vals;
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin(), 11);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(exclusive_scan_by_key_kernel), dim3(1), dim3(1), 0, 0, exec, d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin(), 11);
  ASSERT_EQUAL(d_output, h_output);
}


void TestScanByKeyDeviceSeq()
{
  TestScanByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestScanByKeyDeviceSeq);


void TestScanByKeyDeviceDevice()
{
  TestScanByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestScanByKeyDeviceDevice);


void TestInclusiveScanByKeyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  typedef Vector::iterator   Iterator;

  Vector keys(7);
  Vector vals(7);

  Vector output(7, 0);

  keys[0] = 0; vals[0] = 1;
  keys[1] = 1; vals[1] = 2;
  keys[2] = 1; vals[2] = 3;
  keys[3] = 1; vals[3] = 4;
  keys[4] = 2; vals[4] = 5;
  keys[5] = 3; vals[5] = 6;
  keys[6] = 3; vals[6] = 7;

  hipStream_t s;
  hipStreamCreate(&s);

  Iterator iter = thrust::inclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin());
  hipStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(iter, output.end());

  ASSERT_EQUAL(output[0],  1);
  ASSERT_EQUAL(output[1],  2);
  ASSERT_EQUAL(output[2],  5);
  ASSERT_EQUAL(output[3],  9);
  ASSERT_EQUAL(output[4],  5);
  ASSERT_EQUAL(output[5],  6);
  ASSERT_EQUAL(output[6], 13);
  
  thrust::inclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>(), thrust::multiplies<T>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(output[0],  1);
  ASSERT_EQUAL(output[1],  2);
  ASSERT_EQUAL(output[2],  6);
  ASSERT_EQUAL(output[3], 24);
  ASSERT_EQUAL(output[4],  5);
  ASSERT_EQUAL(output[5],  6);
  ASSERT_EQUAL(output[6], 42);
  
  thrust::inclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(output[0],  1);
  ASSERT_EQUAL(output[1],  2);
  ASSERT_EQUAL(output[2],  5);
  ASSERT_EQUAL(output[3],  9);
  ASSERT_EQUAL(output[4],  5);
  ASSERT_EQUAL(output[5],  6);
  ASSERT_EQUAL(output[6], 13);

  hipStreamDestroy(s);
}
DECLARE_UNITTEST(TestInclusiveScanByKeyCudaStreams);


void TestExclusiveScanByKeyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  typedef Vector::iterator   Iterator;

  Vector keys(7);
  Vector vals(7);

  Vector output(7, 0);

  keys[0] = 0; vals[0] = 1;
  keys[1] = 1; vals[1] = 2;
  keys[2] = 1; vals[2] = 3;
  keys[3] = 1; vals[3] = 4;
  keys[4] = 2; vals[4] = 5;
  keys[5] = 3; vals[5] = 6;
  keys[6] = 3; vals[6] = 7;

  hipStream_t s;
  hipStreamCreate(&s);
  
  Iterator iter = thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin());
  hipStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(iter, output.end());

  ASSERT_EQUAL(output[0], 0);
  ASSERT_EQUAL(output[1], 0);
  ASSERT_EQUAL(output[2], 2);
  ASSERT_EQUAL(output[3], 5);
  ASSERT_EQUAL(output[4], 0);
  ASSERT_EQUAL(output[5], 0);
  ASSERT_EQUAL(output[6], 6);

  thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), T(10));
  hipStreamSynchronize(s);

  ASSERT_EQUAL(output[0], 10);
  ASSERT_EQUAL(output[1], 10);
  ASSERT_EQUAL(output[2], 12);
  ASSERT_EQUAL(output[3], 15);
  ASSERT_EQUAL(output[4], 10);
  ASSERT_EQUAL(output[5], 10);
  ASSERT_EQUAL(output[6], 16);
  
  thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), thrust::equal_to<T>(), thrust::multiplies<T>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(output[0], 10);
  ASSERT_EQUAL(output[1], 10);
  ASSERT_EQUAL(output[2], 20);
  ASSERT_EQUAL(output[3], 60);
  ASSERT_EQUAL(output[4], 10);
  ASSERT_EQUAL(output[5], 10);
  ASSERT_EQUAL(output[6], 60);
  
  thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), thrust::equal_to<T>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(output[0], 10);
  ASSERT_EQUAL(output[1], 10);
  ASSERT_EQUAL(output[2], 12);
  ASSERT_EQUAL(output[3], 15);
  ASSERT_EQUAL(output[4], 10);
  ASSERT_EQUAL(output[5], 10);
  ASSERT_EQUAL(output[6], 16);
}
DECLARE_UNITTEST(TestExclusiveScanByKeyCudaStreams);

