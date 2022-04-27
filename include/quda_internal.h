#pragma once

#include <quda_arch.h>
#include <quda_api.h>

#include <string>
#include <complex>
#include <vector>

#if ((defined(QMP_COMMS) || defined(MPI_COMMS)) && !defined(MULTI_GPU))
#error "MULTI_GPU must be enabled to use MPI or QMP"
#endif

#if (!defined(QMP_COMMS) && !defined(MPI_COMMS) && defined(MULTI_GPU))
#error "MPI or QMP must be enabled to use MULTI_GPU"
#endif

#ifdef QMP_COMMS
#include <qmp.h>
#endif

// these are helper macros used to enable spin-1, spin-2 and spin-4 building blocks as needed
#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_CLOVER_DIRAC)                           \
  || defined(GPU_TWISTED_MASS_DIRAC) || defined(GPU_TWISTED_CLOVER_DIRAC) || defined(GPU_NDEG_TWISTED_MASS_DIRAC)      \
  || defined(GPU_CLOVER_HASENBUSCH_TWIST) || defined(GPU_COVDEV) || defined(GPU_CONTRACT)
#define NSPIN4
#endif

#if defined(GPU_MULTIGRID)
#define NSPIN2
#endif

#if defined(GPU_STAGGERED_DIRAC)
#define NSPIN1
#endif

// this is a helper macro for stripping the path information from
// __FILE__.  FIXME - convert this into a consexpr routine
#define KERNEL_FILE                                                                                                    \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 :                                                               \
                            strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define ALIGN_REQ 128 // Align on 128-byte boundaries
#define ALIGNMENT_ADJUST(n) (((n + ALIGN_REQ - 1) / ALIGN_REQ) * ALIGN_REQ)

#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>
#include <device.h>
#include <array.h>

namespace quda {

  using Complex = std::complex<double>;

  /**
     Array object type used to storing lattice dimensions
   */
  using lat_dim_t = array<int, QUDA_MAX_DIM>;

  /**
   * Check that the resident gauge field is compatible with the requested inv_param
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  bool canReuseResidentGauge(QudaInvertParam *inv_param);

  class TimeProfile;

} // namespace quda
