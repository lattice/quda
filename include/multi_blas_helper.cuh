#pragma once

#include <algorithm>
#include <register_traits.h>
#include <blas_helper.cuh>
#include <reduce_helper.h>
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
       @brief Returns the minimim size of YW array, will use constant buffer if arg size isn't large enough
    */
    constexpr size_t min_YW_size() { return 8; }

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
      const auto N = std::max(arg.NXZ, arg.f.NYW);
      if (h.size() != (size_t)N) errorQuda("coefficient size %lu does not match expected %lu", h.size(), (size_t)N);
      using coeff_t = typename decltype(arg.f)::coeff_t;
      coeff_t *buf_arg = nullptr;
      switch (select) {
      case 'a': buf_arg = arg.f.a; break;
      case 'b': buf_arg = arg.f.b; break;
      case 'c': buf_arg = arg.f.c; break;
      default: errorQuda("Unknown buffer %c", select);
      }
      for (int i = 0; i < N; i++) buf_arg[i] = coeff_t(h[i]);
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
      if (arg.NXZ * arg.f.NYW * sizeof(coeff_t) > max_array_size())
        errorQuda("Requested parameter size %lu larger than max %lu", arg.NXZ * arg.f.NYW * sizeof(coeff_t), max_array_size());
      if (h.size() != (size_t)(arg.NXZ * arg.f.NYW))
        errorQuda("coefficient size %lu does not match expected %lu * %lu", h.size(), (size_t)arg.NXZ, (size_t)arg.f.NYW);

      coeff_t *host = nullptr;
      switch (select) {
      case 'a': host = reinterpret_cast<coeff_t *>(arg.f.Amatrix); break;
      case 'b': host = reinterpret_cast<coeff_t *>(arg.f.Bmatrix); break;
      case 'c': host = reinterpret_cast<coeff_t *>(arg.f.Cmatrix); break;
      default: errorQuda("Unknown buffer %c", select);
      }

      for (int i = 0; i < arg.NXZ; i++)
        for (int j = 0; j < arg.f.NYW; j++) host[arg.f.NYW * i + j] = coeff_t(h[arg.f.NYW * i + j]);
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
       @brief Return the maximum size supported by multi-blas kernels
       (max_N_multi_1d()) rounded down to the largest power of two.
       This is used for the NXZ power-of-two instantiation.
    */
    constexpr int max_N_multi_1d_pow2()
    {
      unsigned int v = max_N_multi_1d();
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;
      return v >> 1;
    }

    /**
       @brief Return the maximum power of two enabled by default for
       multi-blas.  We set a lower limit for multi-reductions, since
       we can just transpose the inner product for free, and a high
       NXZ unroll for multi-reductions lead to poor performance due to
       register spilling.  We also set a lower limit if using double
       precision to avoid register spilling.
       @param[in] reducer Whether we using a reducer
       @param[in] precision What precision we are using
       @return Max power of two
    */
    constexpr int max_NXZ_power2(bool reducer, QudaPrecision precision = QUDA_DOUBLE_PRECISION)
    {
      return reducer ? 16 : precision == QUDA_DOUBLE_PRECISION ? 32 : 64;
    }

    /**
       @brief Return if the requested nxz parameter is valid or
       not.  E.g., a valid power of two, or is less than the the
       MAX_MULTI_BLAS_N parameter.
       @param[in] nxz Requested nxz parameter
       @return True if valid, false if not
     */
    inline bool is_valid_NXZ(int nxz, bool reducer, QudaPrecision precision = QUDA_DOUBLE_PRECISION)
    {
      if (nxz <= MAX_MULTI_BLAS_N || // all values below MAX_MULTI_BLAS_N are valid
          (is_power2(nxz) && nxz <= max_NXZ_power2(reducer, precision))) {
        return true;
      } else {
        return false;
      }
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas functions.  Since the SpinorX and SpinorZ arrays are
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

      constexpr auto arg_known_size_naive = (sizeof(kernel_param<>)                                  // kernel_param parent
				       + sizeof(SpinorX[NXZ])                                        // SpinorX array
				       + (Functor::use_z ? sizeof(SpinorZ[NXZ]) : sizeof(SpinorZ *)) // SpinorZ array
				       + sizeof(Functor)                                             // functor
				       + (!Functor::use_w ? sizeof(SpinorW *) : 0)                   // subtract pointer if not using W
				       + (Functor::reducer ? sizeof(ReduceArg<device_reduce_t>) : 0) // reduction buffers
				       );
      constexpr auto align_factor = 16;
      constexpr auto arg_known_size = ((arg_known_size_naive + align_factor - 1) / align_factor) * align_factor;

      constexpr auto yw_size = (sizeof(SpinorY) + (Functor::use_w ? sizeof(SpinorW) : 0));
      constexpr auto min_size = arg_known_size + min_YW_size() * yw_size;
      constexpr auto max_arg = (max_arg_size<Functor>() >= min_size) ? max_arg_size<Functor>() : device::max_constant_size();

      // size remaining for the Y and W accessors
      constexpr auto arg_remainder_size = max_arg - arg_known_size;
      static_assert(static_cast<int64_t>(max_arg) - static_cast<int64_t>(arg_known_size) > 0, "Remainder size not positive");

      // maximum NYW size based on max arg size
      constexpr auto arg_nyw = arg_remainder_size / yw_size;
      static_assert(arg_nyw != 0, "arg_nyw size is zero");

      // maximum NYW imposed by the coefficients
      constexpr auto coeff_nyw = Functor::coeff_mul ? max_array_size() / (NXZ * sizeof(typename Functor::coeff_t)) : arg_nyw;
      static_assert(coeff_nyw != 0, "coeff_nyw is zero");

      return std::min(arg_nyw, coeff_nyw);
    }

    /**
       @brief Helper function to compute the maximum YW size for the
       multi-blas functions.  Since the SpinorX and SpinorZ arrays are
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
      NXZ = is_valid_NXZ(NXZ, Functor::reducer, y_prec) ? NXZ : MAX_MULTI_BLAS_N; // ensure NXZ is a valid size
      size_t spinor_x_size = x_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);
      size_t spinor_y_size = y_fixed ? sizeof(Spinor<short, 4>) : sizeof(Spinor<float, 4>);
      size_t spinor_z_size = spinor_x_size;
      size_t spinor_w_size = spinor_x_size;

      const auto arg_known_size_naive = (sizeof(kernel_param<>)                                  // kernel_param parent
				   + (NXZ * spinor_x_size)                                       // SpinorX array
				   + (Functor::use_z ? NXZ * spinor_z_size : sizeof(void *))     // SpinorZ array (else dummy pointer)
				   + sizeof(Functor)                                             // functor
				   + (!Functor::use_w ? sizeof(void *) : 0)                      // subtract dummy pointer if not using W
				   + (Functor::reducer ? sizeof(ReduceArg<device_reduce_t>) : 0) // reduction buffers
                                   );
      const auto align_factor = 16;
      const auto arg_known_size = ((arg_known_size_naive + align_factor - 1) / align_factor) * align_factor;

      const auto yw_size = (spinor_y_size + (Functor::use_w ? spinor_w_size : 0));
      const auto min_size = arg_known_size + min_YW_size() * yw_size;
      const auto max_arg = (max_arg_size<Functor>() >= min_size) ? max_arg_size<Functor>() : device::max_constant_size();

      // size remaining for the Y and W accessors
      const auto arg_remainder_size = max_arg - arg_known_size;

      // maximum NYW size based on max arg size
      const auto arg_nyw = arg_remainder_size / yw_size;

      // maximum NYW imposed by the coefficients
      const auto coeff_nyw = Functor::coeff_mul ? max_array_size() / (NXZ * sizeof(typename Functor::coeff_t)) : arg_nyw;

      return std::min(arg_nyw, coeff_nyw);
    }

    /**
       @brief Helper function we use to ensure that the instantiated
       sizes are valid, prior to launching the kernel.
     */
    template <int NXZ, typename store_t, typename y_store_t, typename Functor, typename V>
    void staticCheck(const Functor &f, const std::vector<V> &x, const std::vector<V> &y)
    {
      constexpr int NYW_max = max_YW_size<NXZ, store_t, y_store_t, Functor>();
      const int NYW_max_check = max_YW_size<Functor>(x.size(), static_cast<ColorSpinorField&>(x[0]).Precision(),
                                                     static_cast<ColorSpinorField&>(y[0]).Precision());

      if (!is_valid_NXZ(NXZ, f.reducer, static_cast<QudaPrecision>(sizeof(y_store_t)))) errorQuda("NXZ=%d is not a valid size (MAX_MULTI_BLAS_N %d)", NXZ, MAX_MULTI_BLAS_N);
      if (NXZ != (int)x.size()) errorQuda("Compile-time %d and run-time %lu NXZ do not match", NXZ, x.size());
      if (NYW_max != NYW_max_check) errorQuda("Compile-time %d and run-time %d limits disagree", NYW_max, NYW_max_check);
      if (f.NYW > NYW_max) errorQuda("NYW exceeds max size (%d > %d)", f.NYW, NYW_max);
      if ( !(f.reducer || f.multi_1d) && NXZ * f.NYW * sizeof(typename Functor::coeff_t) > max_array_size())
        errorQuda("Coefficient matrix exceeds max size (%lu > %lu)", NXZ * f.NYW * sizeof(typename Functor::coeff_t), max_array_size());
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

    // the below are helpers used in the recursion

    /**
       @brief Split the set into two.  The optional parameter split is
       used to specify the split point (needed for 2-d splitting).
       @param[in] x The input set to split
       @param[in] split The split point
       @return Resulting split set
    */
    template <typename T> inline auto bisect(T &x, size_t split = 0)
    {
      if (!split) split = x.size() / 2 ;
      return std::make_pair( T{x.begin(), x.begin() + split}, T{x.begin() + split, x.end()} );
    }

    /**
       @brief Split the set into two, cutting across the columns
       (assuming a row-major 2-d data set).  The optional parameter
       split is used to specify the split point (needed for 2-d
       splitting).
       @param[in] x The input set to split
       @param[in] width The width of the 2-d data set
       @param[in] height0 The height of the resulting first set
       @param[in] height1 The height of the resulting second set
       @return Resulting split set
    */
    template <typename T> inline auto bisect_col(T &x, size_t width, size_t height0, size_t height1)
    {
      auto x_ = std::make_pair( T(width * height0), T(width * height1) );

      unsigned int count = 0;
      unsigned count0 = 0;
      unsigned count1 = 0;
      for (unsigned int i = 0; i < width; i++)
      {
        for (unsigned int j = 0; j < height0; j++)
          x_.first[count0++] = x[count++];
        for (unsigned int j = 0; j < height1; j++)
          x_.second[count1++] = x[count++];
      }

      return x_;
    }

    /**
       @brief Join a pair of sets into a single set
       @param[in] pair The pair of sets to join
       @return The newly joined set
     */
    template <typename T> inline auto join(std::pair<T,T> &pair)
    {
      T x;
      x.reserve(pair.first.size() + pair.second.size());
      x.insert(x.end(), pair.first.begin(), pair.first.end());
      x.insert(x.end(), pair.second.begin(), pair.second.end());
      return x;
    }

    /**
       @brief Join a pair of 2-d sets into a single set, joining
       across rows.
       @param[in] width0 The width of the first set
       @param[in] width1 The width of the second set
       @param[in] height The height of the sets
       @return The newly joined set
     */
    template <typename T> inline auto join_row(std::pair<T,T> &pair, size_t width0, size_t width1, size_t height)
    {
      T x((width0 + width1) * height);

      unsigned int count = 0;
      unsigned count0 = 0;
      unsigned count1 = 0;
      for (unsigned int j = 0; j < height; j++)
      {
        for (unsigned int i = 0; i < width0; i++)
          x[count++] = pair.first[count0++];
        for (unsigned int i = 0; i < width1; i++)
          x[count++] = pair.second[count1++];
      }

      return x;
    }

    /**
       @brief Return the 2-d transpose of a set
       @param[in] x Row-major set of size (M x N)
       @param[in] width Width of
       @param[in] height Height of
       @return Transposed set of size (N x M)
    */
    template <typename T> inline auto transpose(const T &x, size_t M, size_t N)
    {
      T x_t(x.size());
      for (unsigned int j = 0; j < N; j++)
        for (unsigned int i = 0; i < M; i++)
          x_t[j * M + i] = x[i * N + j];
      return x_t;
    }

  } // namespace blas

} // namespace quda
