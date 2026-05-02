#pragma once

#include "comm_quda.h"
#include <quda_internal.h>

namespace quda
{

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) : func(func), name(name) { }
  };

  enum class use_kernel_arg_p { FALSE, TRUE, ALWAYS };

  template <use_kernel_arg_p use_kernel_arg_ = use_kernel_arg_p::TRUE, bool check_bounds_ = true> struct kernel_param {
    static constexpr use_kernel_arg_p use_kernel_arg = use_kernel_arg_;
    static constexpr bool check_bounds = check_bounds_;
    static constexpr int max_regs = 0;          // by default we don't limit register count
    static constexpr bool spill_shared = false; // whether a given kernel should use shared memory spilling
    static constexpr bool is_dslash = false;    // whether the arg is for a dslash (with its nested arg struct)
    /** Grid-stride x-dimension: process this many indices \c i, \c i+stride, ... per outer while iteration (see \c kernel.h). */
    static constexpr unsigned int work_item_unroll = 1;
    dim3 threads;          /** number of active threads required */
    int block_size;        /** product of thread block dimensions */
    int comms_rank;        /** per process value of comm_rank() */
    int comms_rank_global; /** per process value comm_rank_global() */
    int comms_coord[4];    /** array storing {comm_coord(0), ..., comm_coord(3)} */
    int comms_dim[4];      /** array storing {comm_dim(0), ..., comm_dim(3)} */
    int comms_dim_partitioned[4]; /** array storing {comm_dim_partitioned(0), ..., comm_dim_partiitoned(3)} */
    /** When \a work_item_unroll > 1: \c threads.x / work_item_unroll after exact divisibility check; else 0. */
    int item_stride = 0;
    /** Grid-stride x batch spacing (\c gridDim.x * blockDim.x); set at launch when using grid-stride unroll. */
    unsigned int x_batch_stride = 0;

    constexpr kernel_param() = default;

    /**
       @param[in] threads Active thread counts (\c threads.x is the x-domain size)
       @param[in] work_unroll When > 1, require \c threads.x % work_unroll == 0 and set \a item_stride to \c threads.x / work_unroll.
       Defaults to \c kernel_param::work_item_unroll (1). Derived args that define their own \c work_item_unroll should pass
       it explicitly in the mem-init list so lookup resolves to the derived constant (the default does not see a derived \c work_item_unroll).
     */
    kernel_param(dim3 threads, unsigned work_unroll = work_item_unroll) :
      threads(threads),
      comms_rank(comm_rank()),
      comms_rank_global(comm_rank_global()),
      comms_coord {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)},
      comms_dim {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)},
      comms_dim_partitioned {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2),
                             comm_dim_partitioned(3)}
    {
      if (work_unroll > 1u) {
        if (threads.x % work_unroll != 0u)
          errorQuda("kernel_param: threads.x (%u) must be divisible by work_unroll (%u)", threads.x, work_unroll);
        item_stride = static_cast<int>(threads.x / static_cast<int>(work_unroll));
      }
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
