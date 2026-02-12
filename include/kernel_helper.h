#pragma once

#include "comm_quda.h"

namespace quda
{

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) : func(func), name(name) { }
  };

  enum class use_kernel_arg_p { FALSE, TRUE, ALWAYS };

#ifndef QUDA_WORK_STEAL
#define QUDA_WORK_STEAL true
#endif
#ifndef QUDA_WORK_STEAL_DSLASH
#define QUDA_WORK_STEAL_DSLASH false
#endif
#ifndef QUDA_WORK_STEAL_REDUCTION
#define QUDA_WORK_STEAL_REDUCTION false
#endif

  template <use_kernel_arg_p use_kernel_arg_ = use_kernel_arg_p::TRUE, bool check_bounds_ = true, bool work_steal_ = QUDA_WORK_STEAL>
  struct kernel_param {
    static constexpr use_kernel_arg_p use_kernel_arg = use_kernel_arg_;
    static constexpr bool check_bounds = check_bounds_;
    static constexpr bool work_steal = work_steal_;
    static constexpr int max_regs = 0;          // by default we don't limit register count
    static constexpr bool spill_shared = false; // whether a given kernel should use shared memory spilling
    static constexpr bool is_dslash = false;    // whether the arg is for a dslash (with its nested arg struct)
    dim3 threads;          /** number of active threads required */
    int block_size;        /** product of thread block dimensions */
    int comms_rank;        /** per process value of comm_rank() */
    int comms_rank_global; /** per process value comm_rank_global() */
    int comms_coord[4];    /** array storing {comm_coord(0), ..., comm_coord(3)} */
    int comms_dim[4];      /** array storing {comm_dim(0), ..., comm_dim(3)} */
    int comms_dim_partitioned[4]; /** array storing {comm_dim_partitioned(0), ..., comm_dim_partiitoned(3)} */

    constexpr kernel_param() = default;

    constexpr kernel_param(dim3 threads) :
      threads(threads),
      comms_rank(comm_rank()),
      comms_rank_global(comm_rank_global()),
      comms_coord {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)},
      comms_dim {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)},
      comms_dim_partitioned {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2),
                             comm_dim_partitioned(3)}
    {
    }

    /**
      @brief This helper member function may be used when templated kernel argument
      needs to check the value of use_kernel_arg without including this header file.
    */
    static constexpr bool default_use_kernel_arg()
    {
      return use_kernel_arg != use_kernel_arg_p::FALSE;
    }

    /**
      @brief This helper member function may be used when templated kernel argument
      needs to check the value of use_kernel_arg without including this header file.
    */
    static constexpr bool always_use_kernel_arg()
    {
      return use_kernel_arg == use_kernel_arg_p::ALWAYS;
    }
  };

#ifdef JITIFY
#define KERNEL(kernel) kernel_t(nullptr, #kernel)
#else
#define KERNEL(kernel) kernel_t(reinterpret_cast<const void *>(kernel<Functor, Arg, grid_stride>), #kernel)
#endif

} // namespace quda
