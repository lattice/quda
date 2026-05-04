#pragma once

#include <type_traits>
#include <utility>

#include <blas_helper.cuh>
#include <reducer.h>
#include <array.h>
#include <kernel.h>

namespace quda
{

  namespace blas
  {

    constexpr bool grid_stride = false;

    constexpr unsigned int blas_unroll = QUDA_BLAS_UNROLL_STREAMING;

    /**
       Parameter struct for generic blas kernel
       @tparam real_ The precision of the calculation
       @tparam n_ The number of real elements per thread
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam Ny Y-field vector i/o length
       @tparam Functor_ Functor used to operate on data
    */
    template <typename real_, int n_, typename store_t, int N, typename y_store_t, int Ny, typename Functor_>
    struct BlasArg : kernel_param<> {
      using real = real_;
      using Functor = Functor_;
      static constexpr unsigned int work_item_unroll = blas_unroll;
      static constexpr int n = n_;
      Spinor<store_t, N> X[MAX_MULTI_RHS];
      Spinor<y_store_t, Ny> Y[MAX_MULTI_RHS];
      Spinor<store_t, N> Z[MAX_MULTI_RHS];
      Spinor<store_t, N> W[MAX_MULTI_RHS];
      Spinor<y_store_t, Ny> V[MAX_MULTI_RHS];
      Functor f;

      const int nParity;
      BlasArg(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
              cvector_ref<ColorSpinorField> &w, cvector_ref<ColorSpinorField> &v, Functor f, int length, int nParity) :
        kernel_param(dim3(length, x.size(), nParity), grid_stride ? 1u : work_item_unroll), f(f), nParity(nParity)
      {
        for (auto i = 0u; i < x.size(); i++) {
          X[i] = x[i];
          Y[i] = y[i];
          Z[i] = z[i];
          W[i] = w[i];
          V[i] = v[i];
        }
      }
    };

    /**
       Generic blas functor  with four loads and up to four stores.
    */
    template <typename Arg> struct Blas_ {
      Arg &arg;
      constexpr Blas_(const Arg &arg) : arg(const_cast<Arg&>(arg))
      {
      }
      static constexpr const char *filename() { return KERNEL_FILE; }

      /**
         @brief Device/host entry point for generic BLAS with optional work-item unroll over the x index.

         \a UnrollCount must be \c std::integral_constant<int, N> (not a bare \c int template parameter) so it cannot
         collide with other \c operator() template overloads. For \c Kernel3D launches, \a stride is
         \c gridDim.x * blockDim.x when grid-stride unroll is active, or \c arg.item_stride for non-grid-stride
         work-item unroll.

         @tparam UnrollCount Compile-time unroll width as \c std::integral_constant<int, N> (\c N >= 1).

         @param[in] i Base x-domain index for this thread.
         @param[in] src_idx Right-hand-side / vector index within the batch (second kernel dimension).
         @param[in] parity Parity or site subset index (third kernel dimension).
         @param[in] stride Spacing between unrolled x indices: \c i + j*stride for \c j in \c [0, N).

         @return None.
       */
      template <typename UnrollCount = std::integral_constant<int, 1>>
      __device__ __host__ inline void operator()(int i, int src_idx, int parity, int stride = 0) const
      {
        static_assert(std::is_same_v<UnrollCount, std::integral_constant<int, UnrollCount::value>>,
                      "work-item unroll uses std::integral_constant<int, N> as the template argument");
        constexpr int n = UnrollCount::value;
        static_assert(n >= 1, "unroll count must be positive");

        using vec = array<complex<typename Arg::real>, Arg::n/2>;

        arg.f.init(src_idx);

        vec x[n], y[n], z[n], w[n], v[n];

#pragma unroll
        for (int j = 0; j < n; j++) {
          if constexpr (arg.f.read.X) arg.X[src_idx].load(x[j], i + j * stride, parity);
          if constexpr (arg.f.read.Y) arg.Y[src_idx].load(y[j], i + j * stride, parity);
          if constexpr (arg.f.read.Z) arg.Z[src_idx].load(z[j], i + j * stride, parity);
          if constexpr (arg.f.read.W) arg.W[src_idx].load(w[j], i + j * stride, parity);
          if constexpr (arg.f.read.V) arg.V[src_idx].load(v[j], i + j * stride, parity);
        }

#pragma unroll
        for (int j = 0; j < n; j++) arg.f(x[j], y[j], z[j], w[j], v[j], src_idx);

#pragma unroll
        for (int j = 0; j < n; j++) {
          if constexpr (arg.f.write.X) arg.X[src_idx].save(x[j], i + j * stride, parity);
          if constexpr (arg.f.write.Y) arg.Y[src_idx].save(y[j], i + j * stride, parity);
          if constexpr (arg.f.write.Z) arg.Z[src_idx].save(z[j], i + j * stride, parity);
          if constexpr (arg.f.write.W) arg.W[src_idx].save(w[j], i + j * stride, parity);
          if constexpr (arg.f.write.V) arg.V[src_idx].save(v[j], i + j * stride, parity);
        }
      }

      __device__ __host__ inline void prefetch(int i, int src_idx, int parity) const
      {
        if constexpr (blas_prefetch_enabled_v) {
          if constexpr (arg.f.read.X)
            arg.X[src_idx].template prefetch<typename Arg::real, Arg::n / 2>(i, parity);
          if constexpr (arg.f.read.Y)
            arg.Y[src_idx].template prefetch<typename Arg::real, Arg::n / 2>(i, parity);
          if constexpr (arg.f.read.Z)
            arg.Z[src_idx].template prefetch<typename Arg::real, Arg::n / 2>(i, parity);
          if constexpr (arg.f.read.W)
            arg.W[src_idx].template prefetch<typename Arg::real, Arg::n / 2>(i, parity);
          if constexpr (arg.f.read.V)
            arg.V[src_idx].template prefetch<typename Arg::real, Arg::n / 2>(i, parity);
        }
      }
    };

    /**
       Base class from which all blas functors should derive
    */
    struct BlasFunctor {
      //! pre-computation routine before the main loop
      __device__ __host__ void init(int) const { }
    };

    /**
       Functor to perform the operation z = a*x + b*y
    */
    template <typename real> struct axpbyz_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 0, 0, 0> read{ };
      static constexpr memory_access<0, 0, 0, 0, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};

      axpbyz_(cvector<double> &a, cvector<double> &b, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }

      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &, T &, T &v, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) v[i] = a[j] * x[i] + b[j] * y[i];
      }                                  // use v not z to ensure same precision as y
      constexpr int flops() const { return 3; }   //! flops per element
    };

    /**
       Functor to perform the operation y = a * x
    */
    template <typename real> struct axy_ : public BlasFunctor {
      static constexpr memory_access<1, 0> read{ };
      static constexpr memory_access<0, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      axy_(cvector<Complex> &a, cvector<Complex> &, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }

      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) y[i] = a[j] * x[i];
      }
      constexpr int flops() const { return 3; } //! flops per element
    };

    /**
       Functor to perform the operator y += a*x (complex-valued)
    */
    template <typename real> struct caxpy_ : public BlasFunctor {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      caxpy_(cvector<Complex> &a, cvector<Complex> &, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) y[i] = cmac(a[j], x[i], y[i]);
      }
      constexpr int flops() const { return 4; }   //! flops per element
    };

    /**
       Functor to perform the operation y = a*x + b*y  (complex-valued)
    */
    template <typename T>
    __device__ __host__ void _caxpby(const complex<T> &a, const typename VectorType<T, 2>::type &x,
                                     const complex<T> &b, typename VectorType<T, 2>::type &y)
    {
      typename VectorType<T, 2>::type yy;
      yy.x = a.x * x.x;
      yy.x -= a.y * x.y;
      yy.x += b.x * y.x;
      yy.x -= b.y * y.y;
      yy.y = a.y * x.x;
      yy.y += a.x * x.y;
      yy.y += b.y * y.x;
      yy.y += b.x * y.y;
      y = yy;
    }

    template <typename real> struct caxpby_ : public BlasFunctor {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpby_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < a.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) _caxpby(a[j], x[i], b[j], y[i]);
      }
      constexpr int flops() const { return 7; }   //! flops per element
    };

    /**
       Functor performing the operation: w[i] = a*x[i] + b*y[i] + c*z[i]
    */
    template <typename real> struct axpbypczw_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 0, 0, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      real c[MAX_MULTI_RHS] = {};
      axpbypczw_(cvector<double> &a, cvector<double> &b, cvector<double> &c)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
        for (auto i = 0u; i < c.size(); i++) this->c[i] = c[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) { w[i] = a[j] * x[i] + b[j] * y[i] + c[j] * z[i]; }
      }
      constexpr int flops() const { return 5; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <typename real> struct axpyBzpcx_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      real c[MAX_MULTI_RHS] = {};
      axpyBzpcx_(cvector<double> &a, cvector<double> &b, cvector<double> &c)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
        for (auto i = 0u; i < c.size(); i++) this->c[i] = c[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] += a[j] * x[i];
          x[i] = b[j] * z[i] + c[j] * x[i];
        }
      }
      constexpr int flops() const { return 5; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]
    */
    template <typename real> struct axpyZpbx_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      axpyZpbx_(cvector<double> &a, cvector<double> &b, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] += a[j] * x[i];
          x[i] = z[i] + b[j] * x[i];
        }
      }
      constexpr int flops() const { return 4; }   //! flops per element
    };

    /**
       Functor performing the operation z[i] = x[i] + a * y[i] + b * z[i]
    */
    template <typename real> struct cxpaypbz_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<0, 0, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      cxpaypbz_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          _caxpby(a[j], y[i], b[j], z[i]);
          z[i] += x[i];
        }
      }
      constexpr int flops() const { return 9; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and x[i] = b*z[i] + x[i]
    */
    template <typename real> struct caxpyBzpx_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpyBzpx_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a[j], x[i], y[i]);
          x[i] = cmac(b[j], z[i], x[i]);
        }
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <typename real> struct caxpyBxpz_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpyBxpz_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a[j], x[i], y[i]);
          z[i] = cmac(b[j], x[i], z[i]);
        }
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
    */
    template <typename real> struct caxpbypzYmbw_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpbypzYmbw_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          z[i] = cmac(a[j], x[i], z[i]);
          z[i] = cmac(b[j], y[i], z[i]);
          y[i] = cmac(-b[j], w[i], y[i]);
        }
      }
      constexpr int flops() const { return 12; }  //! flops per element
    };

    /**
       Functor performing the operation y[i] += a*b*x[i], x[i] *= a
    */
    template <typename real> struct cabxpyAx_ : public BlasFunctor {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      cabxpyAx_(cvector<Complex> &a, cvector<Complex> &b, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i].real();
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          x[i] *= a[j];
          y[i] = cmac(b[j], x[i], y[i]);
        }
      }
      constexpr int flops() const { return 5; }   //! flops per element
    };

    /**
       double caxpyXmaz(c a, V x, V y, V z){}
       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename real> struct caxpyxmaz_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      caxpyxmaz_(cvector<Complex> &a, cvector<Complex> &, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a[j], x[i], y[i]);
          x[i] = cmac(-a[j], z[i], x[i]);
        }
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       double caxpyXmazMR(c a, V x, V y, V z){}

       This is a special variant of caxpyxmaz where we source the scalar multiplier from device memory.

       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename real> struct caxpyxmazMR_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      double4 *Ar4;
      caxpyxmazMR_(cvector<double> &a, cvector<double> &, cvector<double> &) :
        Ar4(static_cast<double4 *>(reducer::get_device_buffer()))
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }

      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &, T &, int j) const
      {
        auto ar4 = Ar4[j];
        auto aj = a[j].real() * complex<real>((real)ar4.x, (real)ar4.y) * ((real)1.0 / (real)ar4.z);

#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(aj, x[i], y[i]);
          x[i] = cmac(-aj, z[i], x[i]);
        }
      }

      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       double tripleCGUpdate(d a, d b, V x, V y, V z, V w){}
       First performs the operation y[i] = y[i] + a*w[i]
       Second performs the operation z[i] = z[i] - a*x[i]
       Third performs the operation w[i] = z[i] + b*w[i]
    */
    template <typename real> struct tripleCGUpdate_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      tripleCGUpdate_(cvector<double> &a, cvector<double> &b, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < a.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] += a[j] * w[i];
          z[i] -= a[j] * x[i];
          w[i] = z[i] + b[j] * w[i];
        }
      }
      constexpr int flops() const { return 6; }   //! flops per element
    };

  } // namespace blas
} // namespace quda
