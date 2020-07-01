#include <algorithm>
#include <register_traits.h>

namespace quda
{

  namespace blas
  {

#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>

    // storage for matrix coefficients
#define MAX_MATRIX_SIZE 8192
#define MAX_ARG_SIZE 4096
    __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
    __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
    __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

    static signed char *Amatrix_h;
    static signed char *Bmatrix_h;
    static signed char *Cmatrix_h;

    /**
       @param[in] x Value we are testing
       @return True if x is a power of two
    */
    template <typename T> inline constexpr bool is_power2(T x) { return (x != 0) && ((x & (x - 1)) == 0); }

    /**
       @brief Return the maximum power of two enabled by default for
       multi-blas.  We set a lower limit for multi-reductions, since
       we can just transpose the inner product for free, and a high
       NXZ unroll for multi-reductions lead to poor performance due to
       register spilling.
       @param[in] reducer Whether we using a reducer
       @param[in] fixed Whether we are using fixed point
       @return Max power of two
    */
    inline int max_NXZ_power2(bool reducer, bool fixed = false) { return reducer ? 16 : (fixed ? 64 : 128); }

    /**
       @brief Return if the requested nxz parameter is valid or
       not.  E.g., a valid power of two, or is less than the the
       MAX_MULTI_BLAS_N parameter.
       @param[in] nxz Requested nxz parameter
       @return True if valid, false if not
     */
    inline bool is_valid_NXZ(int nxz, bool reducer, bool fixed = false)
    {
      if (nxz <= MAX_MULTI_BLAS_N || // all values below MAX_MULTI_BLAS_N are valid
          (is_power2(nxz) && nxz <= max_NXZ_power2(reducer, fixed))) {
        return true;
      } else {
        return false;
      }
    }

    template <int writeX, int writeY, int writeZ, int writeW> struct write {
      static constexpr int X = writeX;
      static constexpr int Y = writeY;
      static constexpr int Z = writeZ;
      static constexpr int W = writeW;
    };

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <int NXZ, typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename Functor>
    inline constexpr int max_YW_size()
    {
      // compute the size remaining for the Y and W accessors
      constexpr int arg_size = (MAX_ARG_SIZE - sizeof(int)                                    // NYW parameter
                                - sizeof(SpinorX[NXZ])                                        // SpinorX array
                                - (Functor::use_z ? sizeof(SpinorZ[NXZ]) : sizeof(SpinorZ *)) // SpinorZ array
                                - sizeof(int)                                                 // functor NYW member
                                - sizeof(int)                                                 // length parameter
                                - (!Functor::use_w ? sizeof(SpinorW *) : 0)   // subtract pointer if not using W
                                - (Functor::reducer ? 3 * sizeof(void *) : 0) // reduction buffers
                                - 16) // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + (Functor::use_w ? sizeof(SpinorW) : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      constexpr int coeff_size = MAX_MATRIX_SIZE / (NXZ * sizeof(typename Functor::type));

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <int NXZ, typename xType, typename yType, typename write, typename Functor>
    inline constexpr int max_YW_size()
    {
      using SpinorX = SpinorTexture<typename mapper<xType>::type, xType, 6>;
      using SpinorY = Spinor<typename mapper<yType>::type, yType, 6, write::Y>;
      using SpinorZ = SpinorX;
      using SpinorW = Spinor<typename mapper<xType>::type, xType, 6, write::W>;
      return max_YW_size<NXZ, SpinorX, SpinorY, SpinorZ, SpinorW, Functor>();
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <typename write>
    inline int max_YW_size(int NXZ, QudaPrecision x_prec, QudaPrecision y_prec, bool use_z, bool use_w, bool reduce)
    {
      bool x_fixed = x_prec < QUDA_SINGLE_PRECISION;
      bool y_fixed = y_prec < QUDA_SINGLE_PRECISION;
      size_t scalar_size = 2 * std::max(std::max(x_prec, y_prec), QUDA_SINGLE_PRECISION);
      NXZ = is_valid_NXZ(NXZ, reduce, x_fixed) ? NXZ : MAX_MULTI_BLAS_N; // ensure NXZ is a valid size
      size_t spinor_x_size
        = x_fixed ? sizeof(SpinorTexture<float4, short4, 6>) : sizeof(SpinorTexture<float4, float4, 6>);
      size_t spinor_y_size
        = y_fixed ? sizeof(Spinor<float4, short4, 6, write::Y>) : sizeof(Spinor<float4, float4, 6, write::Y>);

      size_t spinor_z_size = spinor_x_size;
      size_t spinor_w_size
        = x_fixed ? sizeof(Spinor<float4, short4, 6, write::W>) : sizeof(Spinor<float4, float4, 6, write::W>);

      // compute the size remaining for the Y and W accessors
      int arg_size = (MAX_ARG_SIZE - sizeof(int)                       // NYW parameter
                      - NXZ * spinor_x_size                            // SpinorX array
                      - (use_z ? NXZ * spinor_z_size : sizeof(void *)) // SpinorZ array (else dummy pointer)
                      - sizeof(int)                                    // functor NYW member
                      - sizeof(int)                                    // length parameter
                      - (!use_w ? sizeof(void *) : 0)                  // subtract dummy pointer if not using W
                      - (reduce ? 3 * sizeof(void *) : 0)              // reduction buffers
                      - 16) // there seems to be 16 bytes other argument space we need
        / (spinor_y_size + (use_w ? spinor_w_size : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      int coeff_size = MAX_MATRIX_SIZE / (NXZ * scalar_size);

      return std::min(arg_size, coeff_size);
    }

    template <int NXZ, typename SpinorX, typename SpinorZ, bool> struct SpinorXZ {
      SpinorX X[NXZ];
      SpinorZ *Z;
      SpinorXZ() : Z(reinterpret_cast<SpinorZ *>(X)) {}
    };

    template <int NXZ, typename SpinorX, typename SpinorZ> struct SpinorXZ<NXZ, SpinorX, SpinorZ, true> {
      SpinorX X[NXZ];
      SpinorZ Z[NXZ];
    };

    template <int NYW, typename SpinorY, typename SpinorW, bool> struct SpinorYW {
      SpinorY Y[NYW];
      SpinorW *W;
      SpinorYW() : W(reinterpret_cast<SpinorW *>(Y)) {}
    };

    template <int NYW, typename SpinorY, typename SpinorW> struct SpinorYW<NYW, SpinorY, SpinorW, true> {
      SpinorY Y[NYW];
      SpinorW W[NYW];
    };

    namespace detail
    {
      template <unsigned... digits> struct to_chars {
        static const char value[];
      };

      template <unsigned... digits> const char to_chars<digits...>::value[] = {('0' + digits)..., 0};

      template <unsigned rem, unsigned... digits> struct explode : explode<rem / 10, rem % 10, digits...> {
      };

      template <unsigned... digits> struct explode<0, digits...> : to_chars<digits...> {
      };
    } // namespace detail

    template <unsigned num> struct num_to_string : detail::explode<num / 10, num % 10> {
    };

  } // namespace blas

} // namespace quda
