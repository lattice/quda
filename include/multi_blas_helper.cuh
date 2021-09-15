#pragma once

#include <algorithm>
#include <register_traits.h>
#include <blas_helper.cuh>
#include <kernel_helper.h>
#include <target_device.h>

namespace quda
{

  namespace blas
  {

    /**
       @brief Returns the maximum size of a coefficient array for 2-d multi-blas
    */
    constexpr size_t max_array_size() { return 8192; }

    /**
       @brief Returns the maximum size of the kernel argument
       @tparam Functor reducer and multi_1d blas kernels use kernel arg, others use constant
    */
    template <typename Functor>
    constexpr size_t max_arg_size() { return (Functor::multi_1d || Functor::reducer) ? device::max_kernel_arg_size() : device::max_constant_size(); }

    /**
       @brief set_param sets the matrix coefficient parameters for the
       multi-blas kernels.  If a precision change is required this
       occurs here.  This is the multi-1d specialization where the a,
       b, c arrays are stored in the functor directly.

       @param[in,out] arg The kernel argument struct
       @param[in] select Which array we are setting ('a', 'b', or 'c')
       @param[in] Pointer to host copy of the matrix
     */
    template <bool multi_1d = false, typename Arg, typename T> std::enable_if_t<multi_1d, void>
    set_param(Arg &arg, char select, const T &h)
    {
      using coeff_t = typename decltype(arg.f)::coeff_t;
      coeff_t *buf_arg = nullptr;
      switch (select) {
      case 'a': buf_arg = arg.f.a; break;
      case 'b': buf_arg = arg.f.b; break;
      case 'c': buf_arg = arg.f.c; break;
      default: errorQuda("Unknown buffer %c", select);
      }
      const auto N = std::max(arg.NXZ, arg.NYW);
      for (int i = 0; i < N; i++) buf_arg[i] = coeff_t(h.data[i]);
    }

    /**
       @brief set_param sets the matrix coefficient parameters for the
       multi-blas kernels.  If a precision change is required this
       occurs here.  This is the multi-2d specialization where
       the a, b, c arrays are stored in separate arrays, which are
       stored in the MultiBlasParam struct.

       @param[in,out] arg The kernel argument struct
       @param[in] select Which array we are setting ('a', 'b', or 'c')
       @param[in] Pointer to host copy of the matrix
     */
    template <bool multi_1d = false, typename Arg, typename T> std::enable_if_t<!multi_1d, void>
    set_param(Arg &arg, char select, const T &h)
    {
      using coeff_t = typename decltype(arg.f)::coeff_t;
      if (arg.NXZ * arg.NYW * sizeof(coeff_t) > max_array_size())
        printfQuda("Requested parameter size %lu larger than max %lu", arg.NXZ * arg.NYW * sizeof(coeff_t), max_array_size());

      coeff_t *host = nullptr;
      switch (select) {
      case 'a': host = reinterpret_cast<coeff_t *>(arg.f.Amatrix); break;
      case 'b': host = reinterpret_cast<coeff_t *>(arg.f.Bmatrix); break;
      case 'c': host = reinterpret_cast<coeff_t *>(arg.f.Cmatrix); break;
      default: errorQuda("Unknown buffer %c", select);
      }

      for (int i = 0; i < arg.NXZ; i++)
        for (int j = 0; j < arg.NYW; j++) host[arg.NYW * i + j] = coeff_t(h.data[arg.NYW * i + j]);
    }

    /**
       @brief Generic MultBlasParam for kernels that have inline
       (multi_1d) or no coefficient arrays (reduce)
     */
    template <typename coeff_t, bool reducer, bool multi_1d>
    struct MultiBlasParam  {
      const int NXZ;
      const int NYW;
      MultiBlasParam(int NXZ, int NYW) : NXZ(NXZ), NYW(NYW) {}
    };

    /**
       @brief Specialized MultBlasParam for multi-2d kernels that have
       large coefficient arrays
     */
    template <typename coeff_t>
    struct MultiBlasParam<coeff_t, false, false>  {
      static constexpr int param_size = max_array_size() / sizeof(coeff_t);
      coeff_t Amatrix[param_size];
      coeff_t Bmatrix[param_size];
      coeff_t Cmatrix[param_size];

      const int NXZ;
      const int NYW;

      __device__ __host__ inline coeff_t a(int i, int j) const { return Amatrix[i * NYW + j]; }
      __device__ __host__ inline coeff_t b(int i, int j) const { return Bmatrix[i * NYW + j]; }
      __device__ __host__ inline coeff_t c(int i, int j) const { return Cmatrix[i * NYW + j]; }

      MultiBlasParam(int NXZ, int NYW) : NXZ(NXZ), NYW(NYW) {}
    };

    /**
       @param[in] x Value we are testing
       @return True if x is a power of two
    */
    template <typename T> inline constexpr bool is_power2(T x) { return (x != 0) && ((x & (x - 1)) == 0); }

    /**
       @brief Return the maximum size supported by multi-blas kernels
       when we have a multi-1d kernel and the coefficients are stored
       in the functor.
    */
    constexpr int max_N_multi_1d() { return 24; }

    /**
       @brief Return the maximum power of two enabled by default for
       multi-blas.  We set a lower limit for multi-reductions, since
       we can just transpose the inner product for free, and a high
       NXZ unroll for multi-reductions lead to poor performance due to
       register spilling.
       @tparam reducer Whether we using a reducer
       @tparam fixed Whether we are using fixed point
       @return Max power of two
     */
#if QUDA_PRECISION <= 3
    // if we only have a fixed-point build then we need this WAR to avoid some invalid template instantiations
    // this is temporary - can be removed once the norm and v pointers are fused
    template <bool reducer, bool fixed> constexpr int max_NXZ_power2() { return reducer ? 16 : 64; }
#else
    template <bool reducer, bool fixed> constexpr int max_NXZ_power2() { return reducer ? 16 : (fixed ? 64 : 128); }
#endif

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

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute
       how the maximum size of YW is and allocate this amount of
       space.  This allows for a much larger NXZ (NYW) when NYW (NXZ)
       is small.  This is the constexpr variant which we use after
       template instantiation.

       @tparam[in] NXZ The dimension of the SpinorX and SpinorZ arrays
       @tparam[in] xType The underlying type of the SpinorX and SpinorZ objects
       @tparam[in] yType The underlying type of the SpinorY and SpinorW objects
       @tparam Functor The functor we are applying
       @return The maximum NYW size possible
    */
    template <int NXZ, typename xType, typename yType, typename Functor>
    inline constexpr int max_YW_size()
    {
      using SpinorX = Spinor<xType, 4>;
      using SpinorY = Spinor<yType, 4>;
      using SpinorZ = SpinorX;
      using SpinorW = SpinorX;

      // compute the size remaining for the Y and W accessors
      constexpr auto arg_size = (max_arg_size<Functor>()
                                 - sizeof(kernel_param<>)                                      // kernel_param parent
                                 - sizeof(int)                                                 // NYW parameter
                                 - sizeof(SpinorX[NXZ])                                        // SpinorX array
                                 - (Functor::use_z ? sizeof(SpinorZ[NXZ]) : sizeof(SpinorZ *)) // SpinorZ array
                                 - sizeof(Functor)                                             // functor
                                 - sizeof(dim3)                                                // threads parameter
                                 - (!Functor::use_w ? sizeof(SpinorW *) : 0)                   // subtract pointer if not using W
                                 - (Functor::reducer ? sizeof(ReduceArg<device_reduce_t>) : 0) // reduction buffers
                                 )
        / (sizeof(SpinorY) + (Functor::use_w ? sizeof(SpinorW) : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      constexpr auto coeff_size = Functor::coeff_mul ? max_array_size() / (NXZ * sizeof(typename Functor::coeff_t)) : arg_size;

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas runctions.  Since the SpinorX and SpinorZ arrays are
       statically allocated with length NXZ, we can statically compute
       how the maximum size of YW is and allocate this amount of
       space.  This allows for a much larger NXZ (NYW) when NYW (NXZ)
       is small.  This is the run-time variant which we can call prior
       to template instantiation.

       @tparam Functor The functor we are applying
       @param[in] NXZ The dimension of the SpinorX and SpinorZ arrays
       @param[in] x_prec The precision of the SpinorX and SpinorZ objects
       @param[in] y_prec The precision of the SpinorY and SpinorW objects
       @return The maximum NYW size possible
    */
    template <typename Functor>
    inline int max_YW_size(int NXZ, QudaPrecision x_prec, QudaPrecision y_prec)
    {
      bool x_fixed = x_prec < QUDA_SINGLE_PRECISION;
      bool y_fixed = y_prec < QUDA_SINGLE_PRECISION;
      NXZ = is_valid_NXZ(NXZ, Functor::reducer, x_fixed) ? NXZ : MAX_MULTI_BLAS_N; // ensure NXZ is a valid size
      size_t spinor_x_size = x_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);
      size_t spinor_y_size = y_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);
      size_t spinor_z_size = spinor_x_size;
      size_t spinor_w_size = x_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);

      // compute the size remaining for the Y and W accessors
      const auto arg_size = (max_arg_size<Functor>()
                             - sizeof(kernel_param<>)                                      // kernel_param parent
                             - sizeof(int)                                                 // NYW parameter
                             - NXZ * spinor_x_size                                         // SpinorX array
                             - (Functor::use_z ? NXZ * spinor_z_size : sizeof(void *))     // SpinorZ array (else dummy pointer)
                             - sizeof(Functor)                                             // functor
                             - sizeof(dim3)                                                // threads parameter
                             - (!Functor::use_w ? sizeof(void *) : 0)                      // subtract dummy pointer if not using W
                             - (Functor::reducer ? sizeof(ReduceArg<device_reduce_t>) : 0) // reduction buffers
                             )
        / (spinor_y_size + (Functor::use_w ? spinor_w_size : 0));

      // this is the maximum size limit imposed by the coefficient arrays
      const auto coeff_size = Functor::coeff_mul ? max_array_size() / (NXZ * sizeof(typename Functor::coeff_t)) : arg_size;

      return std::min(arg_size, coeff_size);
    }

    /**
       @brief Helper function that we use ensure that the instantiated
       sizes are valid, prior to launching the kernel.
     */
    template <int NXZ, typename store_t, typename y_store_t, typename Functor>
    void staticCheck(const Functor &f, const std::vector<ColorSpinorField*> &x, const std::vector<ColorSpinorField*> &y)
    {
      constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Functor>();
      const int NYW_max_check = max_YW_size<Functor>(x.size(), x[0]->Precision(), y[0]->Precision());

      if (!is_valid_NXZ(NXZ, f.reducer, x[0]->Precision() < QUDA_SINGLE_PRECISION))
        errorQuda("NXZ=%d is not a valid size ( MAX_MULTI_BLAS_N %d)", NXZ, MAX_MULTI_BLAS_N);
      if (NYW_max != NYW_max_check) errorQuda("Compile-time %d and run-time %d limits disagree", NYW_max, NYW_max_check);
      if (f.NYW > NYW_max) errorQuda("NYW exceeds max size (%d > %d)", f.NYW, NYW_max);
      if ( !(f.reducer || f.multi_1d) && NXZ * f.NYW * sizeof(typename Functor::coeff_t) > max_array_size())
        errorQuda("Coefficient matrix exceeds max size (%lu > %lu)", NXZ * f.NYW * sizeof(typename Functor::coeff_t), max_array_size());
      if (f.reducer && NXZ * f.NYW > max_n_reduce())
        errorQuda("NXZ * NYW = %d exceeds maximum number of reductions %d * %d > %d",
                  NXZ * f.NYW, NXZ, f.NYW, max_n_reduce());
      if (Functor::multi_1d && std::min(NXZ, f.NYW) != 1)
        errorQuda("Expected 1-d multi-blas but appears 2-d (NXZ = %d, NYW = %d)", NXZ, f.NYW);
      if (Functor::multi_1d && std::max(NXZ, f.NYW) > max_N_multi_1d())
        errorQuda("1-d size %d exceeds maximum %d", std::max(NXZ,f.NYW), max_N_multi_1d());
    }

    template <int NXZ, typename store_t, int N, bool> struct SpinorXZ {
      Spinor<store_t, N> X[NXZ];
      Spinor<store_t, N> *Z;
      SpinorXZ() : Z(X) {}
    };

    template <int NXZ, typename store_t, int N> struct SpinorXZ<NXZ, store_t, N, true> {
      Spinor<store_t, N> X[NXZ];
      Spinor<store_t, N> Z[NXZ];
    };

    template <int NYW, typename x_store_t, int Nx, typename y_store_t, int Ny, bool> struct SpinorYW {
      Spinor<y_store_t, Ny> Y[NYW];
      Spinor<y_store_t, Ny> *W;
      SpinorYW() : W(Y) {}
    };

    template <int NYW, typename x_store_t, int Nx, typename y_store_t, int Ny>
    struct SpinorYW<NYW, x_store_t, Nx, y_store_t, Ny, true> {
      Spinor<y_store_t, Ny> Y[NYW];
      Spinor<x_store_t, Nx> W[NYW];
    };

    template <typename T> struct coeff_array {
      using type = T;
      const T *data;
      coeff_array() : data(nullptr) {}
      coeff_array(const T *data) : data(data) {}
    };

  } // namespace blas

} // namespace quda
