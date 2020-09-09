#pragma once

namespace quda {

  template <template <typename> class Functor, typename Arg>
  __global__ void Kernel1D(Arg arg)
  {
    Functor<Arg> f(arg);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= arg.threads.x) return;

    f(i);
  }

  template <template <typename> class Functor, typename Arg>
  __global__ void Kernel2D(Arg arg)
  {
    Functor<Arg> f(arg);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= arg.threads.x) return;
    if (j >= arg.threads.y) return;

    f(i, j);
  }

  template <template <typename> class Functor, typename Arg>
  __global__ void Kernel3D(Arg arg)
  {
    Functor<Arg> f(arg);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= arg.threads.x) return;
    if (j >= arg.threads.y) return;
    if (k >= arg.threads.z) return;

    f(i, j, k);
  }

}
