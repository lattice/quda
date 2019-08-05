/**
   @file math_helper.cuh
   @brief Helper math routines used in QUDA
 */

namespace quda
{

  /*
    @brief Fast power function that works for negative "a" argument
    @param a argument we want to raise to some power
    @param b power that we want to raise a to
    @return pow(a,b)
  */
  template <typename real> __device__ __host__ inline real __fast_pow(real a, int b)
  {
#ifdef __CUDA_ARCH__
    if (sizeof(real) == sizeof(double)) {
      return pow(a, b);
    } else {
      float sign = signbit(a) ? -1.0f : 1.0f;
      float power = __powf(fabsf(a), b);
      return b & 1 ? sign * power : power;
    }
#else
    return std::pow(a, b);
#endif
  }

} // namespace quda
