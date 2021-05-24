#pragma once

#include <kernel_helper.h>

namespace quda {

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel1D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < arg.threads.x) {
      f(i);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel1D(Arg arg)
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      const int tx = arg.threads.x;
      const int ty = arg.threads.y;
      const int tz = arg.threads.z;
      // printf("Kernel1D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
      Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
      // printf("dparg %p\n", dparg);
      omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
      #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        char buffer[sizeof(Arg)];
        memcpy(buffer, (void *)dparg, sizeof(Arg));
        Kernel1D_impl<Functor, Arg, grid_stride>(*(Arg *)buffer);
      }
      omp_target_free(dparg, omp_get_default_device());
    }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel1D()
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      // printf("Kernel1D: launch parameter: gd %d ld %d\n", gd, ld);
      #pragma omp target teams num_teams(gd) thread_limit(ld)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        Kernel1D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
      }
    }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel2D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= arg.threads.y) return;

    while (i < arg.threads.x) {
      f(i, j);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel2D(Arg arg)
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      const int tx = arg.threads.x;
      const int ty = arg.threads.y;
      const int tz = arg.threads.z;
      // printf("Kernel2D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
      Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
      // printf("dparg %p\n", dparg);
      omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
      #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        char buffer[sizeof(Arg)];
        memcpy(buffer, (void *)dparg, sizeof(Arg));
        Kernel2D_impl<Functor, Arg, grid_stride>(*(Arg *)buffer);
      }
      omp_target_free(dparg, omp_get_default_device());
    }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel2D()
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      // printf("Kernel2D: launch parameter: gd %d ld %d\n", gd, ld);
      #pragma omp target teams num_teams(gd) thread_limit(ld)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        Kernel2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
      }
    }


  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __forceinline__ __device__ void Kernel3D_impl(const Arg &arg)
  {
    QUDA_RT_CONSTS;
    Functor<Arg> f(arg);

    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto k = threadIdx.z + blockIdx.z * blockDim.z;
    if (j >= arg.threads.y) return;
    if (k >= arg.threads.z) return;

    while (i < arg.threads.x) {
      f(i, j, k);
      if (grid_stride) i += gridDim.x * blockDim.x; else break;
    }
  }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Kernel3D(Arg arg)
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      const int tx = arg.threads.x;
      const int ty = arg.threads.y;
      const int tz = arg.threads.z;
      // printf("Kernel3D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
      Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
      // printf("dparg %p\n", dparg);
      omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
      #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        char buffer[sizeof(Arg)];
        memcpy(buffer, (void *)dparg, sizeof(Arg));
        Kernel3D_impl<Functor, Arg, grid_stride>(*(Arg *)buffer);
      }
      omp_target_free(dparg, omp_get_default_device());
    }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Kernel3D()
    {
      const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
      // printf("Kernel3D: launch parameter: gd %d ld %d\n", gd, ld);
      #pragma omp target teams num_teams(gd) thread_limit(ld)
      #pragma omp parallel num_threads(ld)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        Kernel3D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
      }
    }

  template <template <typename> class Functor, typename Arg, bool grid_stride = false>
  __launch_bounds__(Arg::block_dim, Arg::min_blocks) __global__ void raw_kernel(Arg arg)
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    const int tx = arg.threads.x;
    const int ty = arg.threads.y;
    const int tz = arg.threads.z;
    // printf("raw_kernel: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    #pragma omp parallel num_threads(ld)
    {
      // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
      //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
      //          "omp reports: teams %d threads %d\n",
      //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
      //          omp_get_num_teams(), omp_get_num_threads());
      Functor<Arg> f(*dparg);
      f();
    }
    omp_target_free(dparg, omp_get_default_device());
  }

}
