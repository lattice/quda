#pragma once

#include "../generic/kernel_ops_target.h"

namespace quda
{

  // Keep unclassified operations converged.
  template <typename T> inline constexpr bool needsFullBlockImpl = true;
  template <typename... T> inline constexpr bool needsFullBlockImpl<KernelOps<T...>> = (needsFullBlockImpl<T> || ...);
  template <typename T> inline constexpr bool needsFullBlock = needsFullBlockImpl<getKernelOps<T>>;

  template <typename T> inline constexpr bool needsSharedMemImpl = true;
  template <typename... T> inline constexpr bool needsSharedMemImpl<KernelOps<T...>> = (needsSharedMemImpl<T> || ...);
  template <typename T> inline constexpr bool needsSharedMem = needsSharedMemImpl<getKernelOps<T>>;

  template <> inline constexpr bool needsSharedMemImpl<op_blockSync> = false;

  // OpenMP warp collectives use team barriers and the shared arena.
  template <typename T> inline constexpr bool needsFullBlockImpl<op_warp_combine<T>> = true;

} // namespace quda
