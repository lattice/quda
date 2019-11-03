#include "hip/hip_runtime.h"
#include <unittest/unittest.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Function1, typename T, typename Function2, typename Iterator2>
__global__
void transform_reduce_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Function1 f1, T init, Function2 f2, Iterator2 result)
{
  *result = thrust::transform_reduce(exec, first, last, f1, init, f2);
}


template<typename ExecutionPolicy>
void TestTransformReduceDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  Vector data(3);
  data[0] = 1; data[1] = -2; data[2] = 3;
  
  T init = 10;

  thrust::device_vector<T> result(1);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(transform_reduce_kernel), dim3(1), dim3(1), 0, 0, exec, data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>(), result.begin());
  
  ASSERT_EQUAL(8, (T)result[0]);
}


void TestTransformReduceDeviceSeq()
{
  TestTransformReduceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformReduceDeviceSeq);


void TestTransformReduceDeviceDevice()
{
  TestTransformReduceDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformReduceDeviceDevice);


void TestTransformReduceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector data(3);
  data[0] = 1; data[1] = -2; data[2] = 3;
  
  T init = 10;

  hipStream_t s;
  hipStreamCreate(&s);

  T result = thrust::transform_reduce(thrust::cuda::par.on(s), data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>());
  hipStreamSynchronize(s);
  
  ASSERT_EQUAL(8, result);

  hipStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformReduceCudaStreams);

