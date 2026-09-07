#pragma once

#include <quda_define.h>

#ifdef QUDA_USE_QUAD_SCALAR

namespace quda
{

  // GNU/CUDA x86: __float128. aarch64 GCC: _Float128 (no __float128).
#if defined(__SIZEOF_FLOAT128__)
  using float128_t = __float128;
#elif defined(__FLT128_MANT_DIG__)
  using float128_t = _Float128;
#else
#error "QUDA_USE_QUAD_SCALAR requires a binary128 type (__float128 or _Float128)"
#endif

} // namespace quda

#endif
