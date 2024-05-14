#pragma once

#include <blas_helper.cuh>
#include <reducer.h>
#include <array.h>
#include <kernel.h>

namespace quda
{

  namespace blas
  {

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
    struct BlasArg : kernel_param<Functor_::use_kernel_arg> {
      using real = real_;
      using Functor = Functor_;
      static constexpr int n = n_;
      static constexpr unsigned int max_n_rhs = MAX_MULTI_RHS;
      Spinor<store_t, N> X[max_n_rhs];
      Spinor<y_store_t, Ny> Y[max_n_rhs];
      Spinor<store_t, N> Z[max_n_rhs];
      Spinor<store_t, N> W[max_n_rhs];
      Spinor<y_store_t, Ny> V[max_n_rhs];
      Functor f;

      const int nParity;
      BlasArg(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
              cvector_ref<ColorSpinorField> &w, cvector_ref<ColorSpinorField> &v, Functor f, int length, int nParity) :
        kernel_param<Functor::use_kernel_arg>(dim3(length, x.size(), nParity)), f(f), nParity(nParity)
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
        // The safety of making the arg non-const (required for caxpyxmazMR) is guaranteed
        // by settting `use_kernel_arg = use_kernel_arg_p::ALWAYS` inside the functor.
      }
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline void operator()(int i, int src_idx, int parity) const
      {
        using vec = array<complex<typename Arg::real>, Arg::n/2>;

        arg.f.init(src_idx);

        vec x, y, z, w, v;
        if (arg.f.read.X) arg.X[src_idx].load(x, i, parity);
        if (arg.f.read.Y) arg.Y[src_idx].load(y, i, parity);
        if (arg.f.read.Z) arg.Z[src_idx].load(z, i, parity);
        if (arg.f.read.W) arg.W[src_idx].load(w, i, parity);
        if (arg.f.read.V) arg.V[src_idx].load(v, i, parity);

        arg.f(x, y, z, w, v, src_idx);

        if (arg.f.write.X) arg.X[src_idx].save(x, i, parity);
        if (arg.f.write.Y) arg.Y[src_idx].save(y, i, parity);
        if (arg.f.write.Z) arg.Z[src_idx].save(z, i, parity);
        if (arg.f.write.W) arg.W[src_idx].save(w, i, parity);
        if (arg.f.write.V) arg.V[src_idx].save(v, i, parity);
      }
    };

    /**
       Base class from which all blas functors should derive
    */
    struct BlasFunctor {
      static constexpr use_kernel_arg_p use_kernel_arg = use_kernel_arg_p::TRUE;
      //! pre-computation routine before the main loop
      __device__ __host__ void init(int) const { ; }
    };

    /**
       Functor to perform the operation z = a*x + b*y
    */
    template <typename real> struct axpbyz_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 0, 0, 0> read{ };
      static constexpr memory_access<0, 0, 0, 0, 1> write{ };
      real a[MAX_MULTI_RHS];
      real b[MAX_MULTI_RHS];

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
      complex<real> a[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      real a[MAX_MULTI_RHS];
      real b[MAX_MULTI_RHS];
      real c[MAX_MULTI_RHS];
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
      real a[MAX_MULTI_RHS];
      real b[MAX_MULTI_RHS];
      real c[MAX_MULTI_RHS];
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
      real a[MAX_MULTI_RHS];
      real b[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      real a[MAX_MULTI_RHS];
      complex<real> b[MAX_MULTI_RHS];
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
      complex<real> a[MAX_MULTI_RHS];
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
    template <typename real> struct caxpyxmazMR_ {
      static constexpr use_kernel_arg_p use_kernel_arg = use_kernel_arg_p::ALWAYS;
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS];
      double3 *Ar3;
      bool init_;
      caxpyxmazMR_(cvector<double> &a, cvector<double> &, cvector<double> &) :
        Ar3(static_cast<double3 *>(reducer::get_device_buffer())), init_(false)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }

      __device__ __host__ void init(int j)
      {
        if (!init_) {
          double3 result = Ar3[j];
          a[j] = a[j].real() * complex<real>((real)result.x, (real)result.y) * ((real)1.0 / (real)result.z);
          init_ = true;
        }
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
       double tripleCGUpdate(d a, d b, V x, V y, V z, V w){}
       First performs the operation y[i] = y[i] + a*w[i]
       Second performs the operation z[i] = z[i] - a*x[i]
       Third performs the operation w[i] = z[i] + b*w[i]
    */
    template <typename real> struct tripleCGUpdate_ : public BlasFunctor {
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1, 1> write{ };
      real a[MAX_MULTI_RHS];
      real b[MAX_MULTI_RHS];
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
