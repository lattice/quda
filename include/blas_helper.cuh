#include <color_spinor_field.h>

// these definitions are used to avoid calling
// std::complex<type>::real/imag which have C++11 ABI incompatibility
// issues with certain versions of GCC

#define REAL(a) (*((double*)&a))
#define IMAG(a) (*((double*)&a+1))

namespace quda {

  static void checkSpinor(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    if (a.Length() != b.Length())
      errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
    if (a.Stride() != b.Stride())
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());
  }

  static void checkLength(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    if (a.Length() != b.Length())
      errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
    if (a.Stride() != b.Stride())
      errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());
  }

} // namespace quda
