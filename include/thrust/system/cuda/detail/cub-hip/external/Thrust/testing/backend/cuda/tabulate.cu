#include "hip/hip_runtime.h"
#include <unittest/unittest.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Function>
__global__
void tabulate_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f)
{
  thrust::tabulate(exec, first, last, f);
}


template<typename ExecutionPolicy>
void TestTabulateDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  using namespace thrust::placeholders;
  typedef typename Vector::value_type T;
  
  Vector v(5);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(tabulate_kernel), dim3(1), dim3(1), 0, 0, exec, v.begin(), v.end(), thrust::identity<T>());

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(tabulate_kernel), dim3(1), dim3(1), 0, 0, exec, v.begin(), v.end(), -_1);

  ASSERT_EQUAL(v[0],  0);
  ASSERT_EQUAL(v[1], -1);
  ASSERT_EQUAL(v[2], -2);
  ASSERT_EQUAL(v[3], -3);
  ASSERT_EQUAL(v[4], -4);
  
  hipLaunchKernelGGL(HIP_KERNEL_NAME(tabulate_kernel), dim3(1), dim3(1), 0, 0, exec, v.begin(), v.end(), _1 * _1 * _1);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 27);
  ASSERT_EQUAL(v[4], 64);
}

void TestTabulateDeviceSeq()
{
  TestTabulateDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTabulateDeviceSeq);

void TestTabulateDeviceDevice()
{
  TestTabulateDevice(thrust::device);
}
DECLARE_UNITTEST(TestTabulateDeviceDevice);

void TestTabulateCudaStreams()
{
  using namespace thrust::placeholders;
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);

  hipStream_t s;
  hipStreamCreate(&s);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>());
  hipStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), -_1);
  hipStreamSynchronize(s);

  ASSERT_EQUAL(v[0],  0);
  ASSERT_EQUAL(v[1], -1);
  ASSERT_EQUAL(v[2], -2);
  ASSERT_EQUAL(v[3], -3);
  ASSERT_EQUAL(v[4], -4);
  
  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), _1 * _1 * _1);
  hipStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 27);
  ASSERT_EQUAL(v[4], 64);

  hipStreamSynchronize(s);
}
DECLARE_UNITTEST(TestTabulateCudaStreams);

