#pragma once

#ifdef QUDA_USE_QUAD_SCALAR

#include <quda_internal.h>
#include <float128_math.h>

#include <string>

namespace quda
{

  inline real_t rel_error(real_t a, real_t b) { return fp128::fabs(a - b) / fp128::fabs(b); }

  inline real_t complex_rel_error(const complex_t &a, const complex_t &b)
  {
    const real_t num = fp128::hypot(a.real() - b.real(), a.imag() - b.imag());
    return rel_error(num, fp128::hypot(b.real(), b.imag()));
  }

  inline std::string format_real(real_t x)
  {
    char buf[256];
    const int n = fp128::snprintf(buf, sizeof(buf), x);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string("0");
  }

} // namespace quda

#endif // QUDA_USE_QUAD_SCALAR
