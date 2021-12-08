#pragma once

#include <target_device.h>
#include <reduce_helper.h>

namespace quda {

  /**
     @brief This class is derived from the arg class that the functor
     creates and curries in the block size.  This allows the block
     size to be set statically at launch time in the actual argument
     class that is passed to the kernel.

     @tparam block_size_x x-dimension block-size
     @tparam block_size_y y-dimension block-size
     @tparam Arg Kernel argument struct
  */
  template <int block_size_x_, int block_size_y_, typename Arg_> struct ReduceKernelArg : Arg_ {
    using Arg = Arg_;
    static constexpr int block_size_x = block_size_x_;
    static constexpr int block_size_y = block_size_y_;
    ReduceKernelArg(const Arg &arg) : Arg(arg) { }
  };

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Reduction2D(Arg arg)
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // const int tx = arg.threads.x;
    // const int ty = arg.threads.y;
    // const int tz = arg.threads.z;
    // printf("Reduction2D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
      // OMP TARGET TODO: issues with reduction, not performed if not inlined.
      // Reduction2D_impl<Functor, Arg, grid_stride>(*dparg);
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Functor<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Functor<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Functor<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Functor<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y;

        while (idx < dparg->threads.x) {
          value = t(value, idx, j);
          if (grid_stride)
            idx += blockDim.x * gridDim.x;
          else
            break;
        }
      }

      // perform final inter-block reduction and write out result
      Functor<Arg> t(*dparg);

      // OMP TARGET TODO: fully inline everything for now
      // reduce<Arg::block_size_x, Arg::block_size_y>(*dparg, t, value);
      int idx = 0;
      bool isLastBlockDone;

      dparg->partial[idx * gridDim.x + blockIdx.x] = value;
      // __threadfence(); // flush result

      // increment global block counter
      unsigned int cvalue = 0;
      unsigned int *c = &dparg->count[idx];
      #pragma omp atomic capture
      { cvalue = *c; *c = *c + 1; }

      // determine if last block
      isLastBlockDone = (cvalue == (gridDim.x - 1));

      // __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        reduce_t sum = dparg->init();
        #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
        {
          dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
          auto i = threadIdx.y * Arg::block_size_x + threadIdx.x;
          while (i < gridDim.x) {
            sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
            i += Arg::block_size_x * Arg::block_size_y;
          }
        }

        // write out the final reduced value
        dparg->result_d[idx] = sum;
        dparg->count[idx] = 0; // set to zero for next time
      }
    }
    omp_target_free(dparg, omp_get_default_device());
  }

  /**
     @brief Reduction2D is the entry point of the generic 2-d
     reduction kernel.  This is the specialization where the kernel
     argument struct is copied to the device prior to kernel launch.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D()
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // printf("Reduction2D: launch parameter: gd %d ld %d\n", gd, ld);
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
      // Reduction2D_impl<Functor, Arg, grid_stride>(device::get_arg<Arg>());
      Arg *dparg = &device::get_arg<Arg>();
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Functor<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Functor<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Functor<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Functor<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y;

        while (idx < dparg->threads.x) {
          value = t(value, idx, j);
          if (grid_stride)
            idx += blockDim.x * gridDim.x;
          else
            break;
        }
      }

      // perform final inter-block reduction and write out result
      Functor<Arg> t(*dparg);

      // OMP TARGET TODO: fully inline everything for now
      // reduce<Arg::block_size_x, Arg::block_size_y>(*dparg, t, value);
      int idx = 0;
      bool isLastBlockDone;

      dparg->partial[idx * gridDim.x + blockIdx.x] = value;
      // __threadfence(); // flush result

      // increment global block counter
      unsigned int cvalue = 0;
      unsigned int *c = &dparg->count[idx];
      #pragma omp atomic capture
      { cvalue = *c; *c = *c + 1; }

      // determine if last block
      isLastBlockDone = (cvalue == (gridDim.x - 1));

      // __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        reduce_t sum = dparg->init();
        #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
        {
          dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
          auto i = threadIdx.y * Arg::block_size_x + threadIdx.x;
          while (i < gridDim.x) {
            sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
            i += Arg::block_size_x * Arg::block_size_y;
          }
        }

        // write out the final reduced value
        dparg->result_d[idx] = sum;
        dparg->count[idx] = 0; // set to zero for next time
      }
    }
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> MultiReduction(Arg arg)
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // const int tx = arg.threads.x;
    // const int ty = arg.threads.y;
    // const int tz = arg.threads.z;
    // printf("MultiReduction: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;

      using reduce_t = typename Functor<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Functor<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Functor<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Functor<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y + blockIdx.y * blockDim.y;
        auto k = threadIdx.z;

        if (j < dparg->threads.y) {
          while (idx < dparg->threads.x) {
            value = t(value, idx, j, k);
            if (grid_stride)
              idx += blockDim.x * gridDim.x;
            else
              break;
          }
        }
      }

      // perform final inter-block reduction and write out result
      Functor<Arg> t(*dparg);
      auto j = blockIdx.y * blockDim.y;
      // reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value, j);
      int idx = j;
      bool isLastBlockDone;

      dparg->partial[idx * gridDim.x + blockIdx.x] = value;
      // __threadfence(); // flush result

      // increment global block counter
      unsigned int cvalue = 0;
      unsigned int *c = &dparg->count[idx];
      #pragma omp atomic capture
      { cvalue = *c; *c = *c + 1; }

      // determine if last block
      isLastBlockDone = (cvalue == (gridDim.x - 1));

      // __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        reduce_t sum = dparg->init();
        #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
        {
          dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
          auto i = threadIdx.y * Arg::block_size_x + threadIdx.x;
          while (i < gridDim.x) {
            sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
            i += Arg::block_size_x * Arg::block_size_y;
          }
        }

        // write out the final reduced value
        dparg->result_d[idx] = sum;
        dparg->count[idx] = 0; // set to zero for next time
      }
    }
    omp_target_free(dparg, omp_get_default_device());
  }

  /**
     @brief MultiReduction is the entry point of the generic
     multi-reduction kernel.  This is the specialization where the
     kernel argument struct is passed by value directly to the kernel.

     @tparam Functor Kernel functor that defines the kernel
     @tparam Arg Kernel argument struct that set any required meta
     data for the kernel
     @tparam grid_stride Whether the kernel does multiple computations
     per thread (in the x dimension)
     @param[in] arg Kernel argument
   */
  template <template <typename> class Functor, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction()
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    // printf("MultiReduction: launch parameter: gd %d ld %d\n", gd, ld);
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      int cache[device::max_shared_memory_size()/sizeof(int)];
      shared_cache.addr = cache;
      Arg *dparg = device::get_arg<Arg>();
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));
      const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;

      using reduce_t = typename Functor<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Functor<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Functor<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        // if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        //   printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
        //          "omp reports: teams %d threads %d\n",
        //          launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
        //          omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Functor<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y + blockIdx.y * blockDim.y;
        auto k = threadIdx.z;

        if (j < dparg->threads.y) {
          while (idx < dparg->threads.x) {
            value = t(value, idx, j, k);
            if (grid_stride)
              idx += blockDim.x * gridDim.x;
            else
              break;
          }
        }
      }

      // perform final inter-block reduction and write out result
      Functor<Arg> t(*dparg);
      auto j = blockIdx.y * blockDim.y;
      // reduce<Arg::block_size_x, Arg::block_size_y>(arg, t, value, j);
      int idx = j;
      bool isLastBlockDone;

      dparg->partial[idx * gridDim.x + blockIdx.x] = value;
      // __threadfence(); // flush result

      // increment global block counter
      unsigned int cvalue = 0;
      unsigned int *c = &dparg->count[idx];
      #pragma omp atomic capture
      { cvalue = *c; *c = *c + 1; }

      // determine if last block
      isLastBlockDone = (cvalue == (gridDim.x - 1));

      // __syncthreads();

      // finish the reduction if last block
      if (isLastBlockDone) {
        reduce_t sum = dparg->init();
        #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
        {
          dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
          auto i = threadIdx.y * Arg::block_size_x + threadIdx.x;
          while (i < gridDim.x) {
            sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
            i += Arg::block_size_x * Arg::block_size_y;
          }
        }

        // write out the final reduced value
        dparg->result_d[idx] = sum;
        dparg->count[idx] = 0; // set to zero for next time
      }
    }
  }

} // namespace quda
