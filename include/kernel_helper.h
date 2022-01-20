#pragma once

#include "comm_quda.h"

namespace quda
{

  struct kernel_t {
    const void *func;
    const std::string name;

    kernel_t(const void *func, const char *name) : func(func), name(name) { }
  };

  template <bool use_kernel_arg_ = true> struct kernel_param {
    static constexpr bool use_kernel_arg = use_kernel_arg_;
    dim3 threads;          /** number of active threads required */
    int comms_rank;        /** per process value of comm_rank() */
    int comms_rank_global; /** per process value comm_rank_global() */
    int comms_coord[4];    /** array storing {comm_coord(0), ..., comm_coord(3)} */
    int comms_dim[4];      /**  array storing {comm_dim(0), ..., comm_dim(3)} */

    constexpr kernel_param() = default;

    constexpr kernel_param(dim3 threads) :
      threads(threads),
      comms_rank(comm_rank()),
      comms_rank_global(comm_rank_global()),
      comms_coord {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)},
      comms_dim {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)}
    {
    }
  };

#ifdef JITIFY
#define KERNEL(kernel) kernel_t(nullptr, #kernel)
#else
#define KERNEL(kernel) kernel_t(reinterpret_cast<const void *>(kernel<Functor, Arg, grid_stride>), #kernel)
#endif

} // namespace quda
