#pragma once

#include <reduce_helper.h>

namespace quda {

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void Reduction2D_impl(const Arg &arg)
  {
    const dim3
      blockDim=launch_param.block,
      gridDim=launch_param.grid,
      blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

    using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
    reduce_t value = arg.init();
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
    {
      if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
               "omp reports: teams %d threads %d\n",
               launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
               omp_get_num_teams(), omp_get_num_threads());
      dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

      Transformer<Arg> t(arg);

      auto idx = threadIdx.x + blockIdx.x * blockDim.x;
      auto j = threadIdx.y;

      while (idx < arg.threads.x) {
        value = t(value, idx, j);
        if (grid_stride) idx += blockDim.x * gridDim.x; else break;
      }
    }

    // perform final inter-block reduction and write out result
    Transformer<Arg> t(arg);
    reduce<block_size_x, block_size_y>(arg, t, value);
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> Reduction2D(Arg arg)
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    const int tx = arg.threads.x;
    const int ty = arg.threads.y;
    const int tz = arg.threads.z;
    printf("Reduction2D: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      // Reduction2D_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(*dparg);
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        if(omp_get_team_num()==0 && omp_get_thread_num()==0)
          printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
                 "omp reports: teams %d threads %d\n",
                 launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
                 omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Transformer<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y;

        while (idx < dparg->threads.x) {
          value = t(value, idx, j);
          if (grid_stride) idx += blockDim.x * gridDim.x; else break;
        }
      }
      // perform final inter-block reduction and write out result
      Transformer<Arg> t(*dparg);
      // reduce<block_size_x, block_size_y>(*dparg, t, value);
      // ../../reduce_helper.h:/reduce
      {
        const auto idx = 0;
        // In OpenMP, this runs in the main thread of each team, and `in` is already the block reduction value.
        bool isLastBlockDone;
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        { // This is the main thread per team
          if(blockIdx.x==0) printf("team %d: value: %g\n", omp_get_team_num(), *reinterpret_cast<double *>(&value));
          dparg->partial[idx * gridDim.x + blockIdx.x] = value;

          // increment global block counter
          // auto value = atomicInc(&arg.count[idx], gridDim.x);
          unsigned int cvalue = 0;
          unsigned int *c = &dparg->count[idx];
          #pragma omp atomic capture
          { cvalue = *c; *c = *c + 1; }
          // { cvalue = dparg->count[idx]; dparg->count[idx] = ((dparg->count[idx] >= gridDim.x) ? 0 : (dparg->count[idx]+1)); }

          // determine if last block
          isLastBlockDone = (cvalue == (gridDim.x - 1));
        }
        // finish the reduction if last block
        if (isLastBlockDone) {
          reduce_t sum = dparg->init();
          #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
          {
            dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
            auto i = threadIdx.y * block_size_x + threadIdx.x;
            while (i < gridDim.x) {
              sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
              i += block_size_x * block_size_y;
            }
          }

          // sum = (Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

          printf("lastBlock: team %d: idx: %d value: %g\n", omp_get_team_num(), idx, *reinterpret_cast<double *>(&sum));
          // write out the final reduced value
          // if (threadIdx.y * block_size_x + threadIdx.x == 0)
          { // This is the main thread per team
            dparg->result_d[idx] = sum;
            dparg->count[idx] = 0; // set to zero for next time
          }
        }
      }
    }
    omp_target_free(dparg, omp_get_default_device());
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> Reduction2D()
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    printf("Reduction2D: launch parameter: gd %d ld %d\n", gd, ld);
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      // Reduction2D_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(device::get_arg<Arg>());
      Arg *dparg = &device::get_arg<Arg>();
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        if(omp_get_team_num()==0 && omp_get_thread_num()==0)
          printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
                 "omp reports: teams %d threads %d\n",
                 launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
                 omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Transformer<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y;

        while (idx < dparg->threads.x) {
          value = t(value, idx, j);
          if (grid_stride) idx += blockDim.x * gridDim.x; else break;
        }
      }
      // perform final inter-block reduction and write out result
      Transformer<Arg> t(*dparg);
      // reduce<block_size_x, block_size_y>(*dparg, t, value);
      // ../../reduce_helper.h:/reduce
      {
        const auto idx = 0;
        // In OpenMP, this runs in the main thread of each team, and `in` is already the block reduction value.
        bool isLastBlockDone;
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        { // This is the main thread per team
          dparg->partial[idx * gridDim.x + blockIdx.x] = value;

          // increment global block counter
          // auto value = atomicInc(&arg.count[idx], gridDim.x);
          unsigned int cvalue = 0;
          unsigned int *c = &dparg->count[idx];
          #pragma omp atomic capture
          { cvalue = *c; *c = *c + 1; }
          // { cvalue = dparg->count[idx]; dparg->count[idx] = ((dparg->count[idx] >= gridDim.x) ? 0 : (dparg->count[idx]+1)); }

          // determine if last block
          isLastBlockDone = (cvalue == (gridDim.x - 1));
        }
        // finish the reduction if last block
        if (isLastBlockDone) {
          reduce_t sum = dparg->init();
          #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
          {
            dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
            auto i = threadIdx.y * block_size_x + threadIdx.x;
            while (i < gridDim.x) {
              sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
              i += block_size_x * block_size_y;
            }
          }

          // sum = (Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

          // write out the final reduced value
          // if (threadIdx.y * block_size_x + threadIdx.x == 0)
          { // This is the main thread per team
            dparg->result_d[idx] = sum;
            dparg->count[idx] = 0; // set to zero for next time
          }
        }
      }
    }
  }


  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __forceinline__ __device__ void MultiReduction_impl(const Arg &arg)
  {
    const dim3
      blockDim=launch_param.block,
      gridDim=launch_param.grid,
      blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

    using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
    reduce_t value = arg.init();
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
    {
      if(omp_get_team_num()==0 && omp_get_thread_num()==0)
        printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
               "omp reports: teams %d threads %d\n",
               launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
               omp_get_num_teams(), omp_get_num_threads());
      dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

      Transformer<Arg> t(arg);

      auto idx = threadIdx.x + blockIdx.x * blockDim.x;
      auto j = threadIdx.y + blockIdx.y * blockDim.y;
      auto k = threadIdx.z;

      if (j < arg.threads.y){
        while (idx < arg.threads.x) {
          value = t(value, idx, j, k);
          if (grid_stride) idx += blockDim.x * gridDim.x; else break;
        }
      }
    }
    // perform final inter-block reduction and write out result
    Transformer<Arg> t(arg);
    auto j = blockIdx.y * blockDim.y;
    reduce<block_size_x, block_size_y>(arg, t, value, j);
  }

  /** Workaround compiler limitations.  We have to inline parallel regions with target teams. **/

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
  __global__ std::enable_if_t<device::use_kernel_arg<Arg>(), void> MultiReduction(Arg arg)
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    const int tx = arg.threads.x;
    const int ty = arg.threads.y;
    const int tz = arg.threads.z;
    printf("MultiReduction: launch parameter: gd %d ld %d tx %d ty %d tz %d\n", gd, ld, tx, ty, tz);
    Arg *dparg = (Arg*)omp_target_alloc(sizeof(Arg), omp_get_default_device());
    // printf("dparg %p\n", dparg);
    omp_target_memcpy(dparg, (void *)(&arg), sizeof(Arg), 0, 0, omp_get_default_device(), omp_get_initial_device());
    #pragma omp target teams num_teams(gd) thread_limit(ld) is_device_ptr(dparg)
    {
      // MultiReduction_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(*dparg);
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        if(omp_get_team_num()==0 && omp_get_thread_num()==0)
          printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
                 "omp reports: teams %d threads %d\n",
                 launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
                 omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Transformer<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y + blockIdx.y * blockDim.y;
        auto k = threadIdx.z;

        if (j < dparg->threads.y){
          while (idx < dparg->threads.x) {
            value = t(value, idx, j, k);
            if (grid_stride) idx += blockDim.x * gridDim.x; else break;
          }
        }
      }
      // perform final inter-block reduction and write out result
      Transformer<Arg> t(*dparg);
      auto j = blockIdx.y * blockDim.y;
      // reduce<block_size_x, block_size_y>(*dparg, t, value, j);
      // ../../reduce_helper.h:/reduce
      {
        const auto idx = j;
        // In OpenMP, this runs in the main thread of each team, and `in` is already the block reduction value.
        bool isLastBlockDone;
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        { // This is the main thread per team
          if(blockIdx.x==0) printf("team %d: value: %g\n", omp_get_team_num(), *reinterpret_cast<double *>(&value));
          dparg->partial[idx * gridDim.x + blockIdx.x] = value;

          // increment global block counter
          // auto value = atomicInc(&arg.count[idx], gridDim.x);
          unsigned int cvalue = 0;
          unsigned int *c = &dparg->count[idx];
          #pragma omp atomic capture
          { cvalue = *c; *c = *c + 1; }
          // { cvalue = dparg->count[idx]; dparg->count[idx] = ((dparg->count[idx] >= gridDim.x) ? 0 : (dparg->count[idx]+1)); }

          // determine if last block
          isLastBlockDone = (cvalue == (gridDim.x - 1));
        }
        // finish the reduction if last block
        if (isLastBlockDone) {
          reduce_t sum = dparg->init();
          #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
          {
            dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
            auto i = threadIdx.y * block_size_x + threadIdx.x;
            while (i < gridDim.x) {
              sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
              i += block_size_x * block_size_y;
            }
          }

          // sum = (Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

          printf("lastBlock: team %d: idx: %d value: %g\n", omp_get_team_num(), idx, *reinterpret_cast<double *>(&sum));
          // write out the final reduced value
          // if (threadIdx.y * block_size_x + threadIdx.x == 0)
          { // This is the main thread per team
            dparg->result_d[idx] = sum;
            dparg->count[idx] = 0; // set to zero for next time
          }
        }
      }
    }
    omp_target_free(dparg, omp_get_default_device());
  }

  template <int block_size_x, int block_size_y, template <typename> class Transformer, typename Arg, bool grid_stride = true>
    __global__ std::enable_if_t<!device::use_kernel_arg<Arg>(), void> MultiReduction()
  {
    const int gd = launch_param.grid.x*launch_param.grid.y*launch_param.grid.z;
    const int ld = launch_param.block.x*launch_param.block.y*launch_param.block.z;
    printf("MultiReduction: launch parameter: gd %d ld %d\n", gd, ld);
    #pragma omp target teams num_teams(gd) thread_limit(ld)
    {
      // MultiReduction_impl<block_size_x, block_size_y, Transformer, Arg, grid_stride>(device::get_arg<Arg>());
      Arg *dparg = &device::get_arg<Arg>();
      const dim3
        blockDim=launch_param.block,
        gridDim=launch_param.grid,
        blockIdx(omp_get_team_num()%launch_param.grid.x, (omp_get_team_num()/launch_param.grid.x)%launch_param.grid.y, omp_get_team_num()/(launch_param.grid.x*launch_param.grid.y));

      using reduce_t = typename Transformer<Arg>::reduce_t;
#pragma omp declare reduction(OMPReduce_ : reduce_t : omp_out=Transformer<Arg>::reduce_omp(omp_out,omp_in)) initializer(omp_priv=Transformer<Arg>::init_omp())
      reduce_t value = dparg->init();
      #pragma omp parallel num_threads(ld) reduction(OMPReduce_:value)
      {
        if(omp_get_team_num()==0 && omp_get_thread_num()==0)
          printf("In target: launch_param.grid %d %d %d block %d %d %d\n"
                 "omp reports: teams %d threads %d\n",
                 launch_param.grid.x, launch_param.grid.y, launch_param.grid.z, launch_param.block.x, launch_param.block.y, launch_param.block.z,
                 omp_get_num_teams(), omp_get_num_threads());
        dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));

        Transformer<Arg> t(*dparg);

        auto idx = threadIdx.x + blockIdx.x * blockDim.x;
        auto j = threadIdx.y + blockIdx.y * blockDim.y;
        auto k = threadIdx.z;

        if (j < dparg->threads.y){
          while (idx < dparg->threads.x) {
            value = t(value, idx, j, k);
            if (grid_stride) idx += blockDim.x * gridDim.x; else break;
          }
        }
      }
      // perform final inter-block reduction and write out result
      Transformer<Arg> t(*dparg);
      auto j = blockIdx.y * blockDim.y;
      // reduce<block_size_x, block_size_y>(*dparg, t, value, j);
      // ../../reduce_helper.h:/reduce
      {
        const auto idx = j;
        // In OpenMP, this runs in the main thread of each team, and `in` is already the block reduction value.
        bool isLastBlockDone;
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        { // This is the main thread per team
          dparg->partial[idx * gridDim.x + blockIdx.x] = value;

          // increment global block counter
          // auto value = atomicInc(&arg.count[idx], gridDim.x);
          unsigned int cvalue = 0;
          unsigned int *c = &dparg->count[idx];
          #pragma omp atomic capture
          { cvalue = *c; *c = *c + 1; }
          // { cvalue = dparg->count[idx]; dparg->count[idx] = ((dparg->count[idx] >= gridDim.x) ? 0 : (dparg->count[idx]+1)); }

          // determine if last block
          isLastBlockDone = (cvalue == (gridDim.x - 1));
        }
        // finish the reduction if last block
        if (isLastBlockDone) {
          reduce_t sum = dparg->init();
          #pragma omp parallel num_threads(ld) reduction(OMPReduce_:sum)
          {
            dim3 threadIdx(omp_get_thread_num()%launch_param.block.x, (omp_get_thread_num()/launch_param.block.x)%launch_param.block.y, omp_get_thread_num()/(launch_param.block.x*launch_param.block.y));
            auto i = threadIdx.y * block_size_x + threadIdx.x;
            while (i < gridDim.x) {
              sum = t(sum, const_cast<reduce_t &>(static_cast<volatile reduce_t *>(dparg->partial)[idx * gridDim.x + i]));
              i += block_size_x * block_size_y;
            }
          }

          // sum = (Reducer::do_sum ? BlockReduce(cub_tmp).Sum(sum) : BlockReduce(cub_tmp).Reduce(sum, r));

          // write out the final reduced value
          // if (threadIdx.y * block_size_x + threadIdx.x == 0)
          { // This is the main thread per team
            dparg->result_d[idx] = sum;
            dparg->count[idx] = 0; // set to zero for next time
          }
        }
      }
    }
  }

}
