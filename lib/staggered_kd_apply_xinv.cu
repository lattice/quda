#include <tune_quda.h>
#include <transfer.h>
#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>

#include <jitify_helper.cuh>
//#include <kernels/staggered_kd_build_xinv_kernel.cuh>

namespace quda {


  // Applies the staggered KD block inverse to a staggered ColorSpinnor
  void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    errorQuda("There's no support for applying the KD inverse op yet.");
  }

} //namespace quda
