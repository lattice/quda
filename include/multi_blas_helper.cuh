namespace quda
{

  namespace blas
  {

#define BLAS_SPINOR // do not include ghost functions in Spinor class to reduce parameter space overhead
#include <texture.h>

    // storage for matrix coefficients
#define MAX_MATRIX_SIZE 8192
#define MAX_ARG_SIZE 4096
    static __constant__ signed char Amatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Bmatrix_d[MAX_MATRIX_SIZE];
    static __constant__ signed char Cmatrix_d[MAX_MATRIX_SIZE];

    static signed char *Amatrix_h;
    static signed char *Bmatrix_h;
    static signed char *Cmatrix_h;

#ifdef CONSTANT_ARG
    // as a performance work around we put the argument struct into
    // __constant__ memory to prevent the compiler from spilling
    // registers on older CUDA
    static __constant__ signed char arg_buffer[MAX_ARG_SIZE];
#endif

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute how
       the maximum size of YW is and allocate this amount of space.  This
       allows for a much larger NXZ (NYW) when NYW (NXZ) is small.
    */
    template <int NXZ, typename Functor> inline constexpr int max_YW_size()
    {
      // the size of the accessor doesn't change with precision just instantiate some precision
      using SpinorX = SpinorTexture<float4,short4,6>;
      using SpinorY = Spinor<float4,short4,6,1>;
      using SpinorZ = SpinorX;
      using SpinorW = SpinorY;

      // compute the size remaining for the Y and W accessors
      constexpr int arg_size = (MAX_ARG_SIZE
                                - sizeof(int)          // NYW parameter
                                - sizeof(SpinorX[NXZ]) // SpinorX array
                                - (Functor::use_z ? sizeof(SpinorZ[NXZ]) : sizeof(SpinorZ*)) // SpinorZ array
                                - sizeof(int)          // functor NYW member
                                - sizeof(int) - 16     // length parameter
                                - (!Functor::use_w ? sizeof(SpinorW*) : 0) // subtract pointer if not using W
                                - (Functor::reducer ? 3 * sizeof(void*) : 0) // reduction buffers
                                - 16)                  // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + (Functor::use_w ? sizeof(SpinorW) : 0) );

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
    inline int max_YW_size(int NXZ, int precision, bool use_z, bool use_w, bool reduce)
    {
      // ensure NXZ is a valid size
      NXZ = std::min(NXZ, MAX_MULTI_BLAS_N);

      // the size of the accessor doesn't change with precision just instantiate some precision
      using SpinorX = SpinorTexture<float4,short4,6>;
      using SpinorY = Spinor<float4,short4,6,1>;
      using SpinorZ = SpinorX;
      using SpinorW = SpinorY;

      // compute the size remaining for the Y and W accessors
      int arg_size = (MAX_ARG_SIZE
                      - sizeof(int)         // NYW parameter
                      - NXZ*sizeof(SpinorX) // SpinorX array
                      - (use_z ? NXZ*sizeof(SpinorZ) : sizeof(SpinorZ*)) // SpinorZ array
                      - sizeof(int)         // functor NYW member
                      - sizeof(int)         // length parameter
                      - (!use_w ? sizeof(SpinorW*) : 0) // subtract pointer if not using W
                      - (reduce ? 3 * sizeof(void*) : 0) // reduction buffers
                      - 16)                  // there seems to be 16 bytes other argument space we need
        / (sizeof(SpinorY) + (use_w ? sizeof(SpinorW) : 0) );

      int coeff_size = MAX_MATRIX_SIZE / (NXZ * precision);

      return std::min(arg_size, coeff_size);
    }

    template <int NXZ, typename SpinorX, typename SpinorZ, bool> struct SpinorXZ
    {
      SpinorX X[NXZ];
      SpinorZ *Z;
      SpinorXZ() : Z(reinterpret_cast<SpinorZ*>(X)) { }
    };

    template <int NXZ, typename SpinorX, typename SpinorZ> struct SpinorXZ<NXZ,SpinorX,SpinorZ,true>
    {
      SpinorX X[NXZ];
      SpinorZ Z[NXZ];
    };

    template <int NYW, typename SpinorY, typename SpinorW, bool> struct SpinorYW
    {
      SpinorY Y[NYW];
      SpinorW *W;
      SpinorYW() : W(reinterpret_cast<SpinorW*>(Y)) { }
    };

    template <int NYW, typename SpinorY, typename SpinorW> struct SpinorYW<NYW,SpinorY,SpinorW,true>
    {
      SpinorY Y[NYW];
      SpinorW W[NYW];
    };

  } // namespace blas

} // namespace quda
