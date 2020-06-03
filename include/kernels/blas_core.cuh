#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>

namespace quda
{

  namespace blas
  {

#include <texture.h>

    /**
       Parameter struct for generic blas kernel
    */
    template <typename SpinorX, typename SpinorY, typename SpinorZ, typename SpinorW, typename SpinorV, typename Functor>
    struct BlasArg {
      SpinorX X;
      SpinorY Y;
      SpinorZ Z;
      SpinorW W;
      SpinorV V;
      Functor f;
      const int length;
      BlasArg(SpinorX X, SpinorY Y, SpinorZ Z, SpinorW W, SpinorV V, Functor f, int length) :
          X(X),
          Y(Y),
          Z(Z),
          W(W),
          V(V),
          f(f),
          length(length)
      { ; }
    };

    /**
       Generic blas kernel with four loads and up to four stores.
    */
    template <typename FloatN, int M, typename Arg> __global__ void blasKernel(Arg arg)
    {
      unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
      unsigned int parity = blockIdx.y;
      unsigned int gridSize = gridDim.x * blockDim.x;

      arg.f.init();

      while (i < arg.length) {
        FloatN x[M], y[M], z[M], w[M], v[M];
        arg.X.load(x, i, parity);
        arg.Y.load(y, i, parity);
        arg.Z.load(z, i, parity);
        arg.W.load(w, i, parity);
        arg.V.load(v, i, parity);

#pragma unroll
        for (int j = 0; j < M; j++) arg.f(x[j], y[j], z[j], w[j], v[j]);

        if (arg.f.write.X) arg.X.save(x, i, parity);
        if (arg.f.write.Y) arg.Y.save(y, i, parity);
        if (arg.f.write.Z) arg.Z.save(z, i, parity);
        if (arg.f.write.W) arg.W.save(w, i, parity);
        if (arg.f.write.V) arg.V.save(v, i, parity);
        i += gridSize;
      }
    }

    /**
       Base class from which all blas functors should derive
     */
    struct BlasFunctor {
      //! pre-computation routine before the main loop
      virtual __device__ __host__ void init() { ; }
    };

    /**
       Functor to perform the operation z = a*x + b*y
    */
    template <typename real> struct axpbyz_ : public BlasFunctor {
      static constexpr write<0, 0, 0, 0, 1> write{ };
      const real a;
      const real b;
      axpbyz_(const real &a, const real &b, const real &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        v = a * x + b * y;
      }                                  // use v not z to ensure same precision as y
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 3; }   //! flops per element
    };

    /**
       Functor to perform the operation x *= a
    */
    template <typename real> struct ax_ : public BlasFunctor {
      static constexpr write<1> write{ };
      const real a;
      ax_(const real &a, const real &b, const real &c) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        x *= a;
      }
      static int streams() { return 2; } //! total number of input and output streams
      static int flops() { return 1; }   //! flops per element
    };

    /**
       Functor to perform the operation y += a * x  (complex-valued)
    */

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 4>::type &x, typename VectorType<T, 4>::type &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
      y.z += a.x * x.z;
      y.z -= a.y * x.w;
      y.w += a.y * x.z;
      y.w += a.x * x.w;
    }

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 2>::type &x, typename VectorType<T, 2>::type &y)
    {
      y.x += a.x * x.x;
      y.x -= a.y * x.y;
      y.y += a.y * x.x;
      y.y += a.x * x.y;
    }

    template <typename T>
    __device__ __host__ void _caxpy(const complex<T> &a, const typename VectorType<T, 8>::type &x, typename VectorType<T, 8>::type &y)
    {
      _caxpy(a, x.x, y.x);
      _caxpy(a, x.y, y.y);
    }

    template <typename real> struct caxpy_ : public BlasFunctor {
      static constexpr write<0, 1> write{ };
      const complex<real> a;
      caxpy_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v) { _caxpy(a, x, y); }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    /**
       Functor to perform the operation y = a*x + b*y  (complex-valued)
    */

    template <typename T>
    __device__ __host__ void _caxpby(const complex<T> &a, const typename VectorType<T, 4>::type &x,
                                     const complex<T> &b, typename VectorType<T, 4>::type &y)
    {
      typename VectorType<T, 4>::type yy;
      yy.x = a.x * x.x;
      yy.x -= a.y * x.y;
      yy.x += b.x * y.x;
      yy.x -= b.y * y.y;
      yy.y = a.y * x.x;
      yy.y += a.x * x.y;
      yy.y += b.y * y.x;
      yy.y += b.x * y.y;
      yy.z = a.x * x.z;
      yy.z -= a.y * x.w;
      yy.z += b.x * y.z;
      yy.z -= b.y * y.w;
      yy.w = a.y * x.z;
      yy.w += a.x * x.w;
      yy.w += b.y * y.z;
      yy.w += b.x * y.w;
      y = yy;
    }

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

    template <typename T>
    __device__ __host__ void _caxpby(const complex<T> &a, const typename VectorType<T, 8>::type &x,
                                     const complex<T> &b, typename VectorType<T, 8>::type &y)
    {
      _caxpby(a, x.x, b, y.x);
      _caxpby(a, x.y, b, y.y);
    }

    template <typename real> struct caxpby_ : public BlasFunctor {
      static constexpr write<0, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      caxpby_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpby(a, x, b, y);
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 7; }   //! flops per element
    };

    template <typename real> struct caxpbypczw_ : public BlasFunctor {
      static constexpr write<0, 0, 0, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      const complex<real> c;
      caxpbypczw_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a), b(b), c(c) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        w = y;
        _caxpby(a, x, b, w);
        _caxpy(c, z, w);
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = b*z[i] + c*x[i]
    */
    template <typename real> struct axpyBzpcx_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const real a;
      const real b;
      const real c;
      axpyBzpcx_(const real &a, const real &b, const real &c) : a(a), b(b), c(c) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        y += a * x;
        x = b * z + c * x;
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 5; }   //! flops per element
    };

    /**
       Functor performing the operations: y[i] = a*x[i] + y[i]; x[i] = z[i] + b*x[i]
    */
    template <typename real> struct axpyZpbx_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const real a;
      const real b;
      axpyZpbx_(const real &a, const real &b, const real &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        y += a * x;
        x = z + b * x;
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 4; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and x[i] = b*z[i] + x[i]
    */
    template <typename real> struct caxpyBzpx_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      caxpyBzpx_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpy(a, x, y);
        _caxpy(b, z, x);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations y[i] = a*x[i] + y[i] and z[i] = b*x[i] + z[i]
    */
    template <typename real> struct caxpyBxpz_ : public BlasFunctor {
      static constexpr write<0, 1, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      caxpyBxpz_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpy(a, x, y);
        _caxpy(b, x, z);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       Functor performing the operations z[i] = a*x[i] + b*y[i] + z[i] and y[i] -= b*w[i]
    */
    template <typename real> struct caxpbypzYmbw_ : public BlasFunctor {
      static constexpr write<0, 1, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      caxpbypzYmbw_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpy(a, x, z);
        _caxpy(b, y, z);
        _caxpy(-b, w, y);
      }

      static int streams() { return 6; } //! total number of input and output streams
      static int flops() { return 12; }  //! flops per element
    };

    /**
       Functor performing the operation y[i] += a*b*x[i], x[i] *= a
    */
    template <typename real> struct cabxpyAx_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const real a;
      const complex<real> b;
      cabxpyAx_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a.real()), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        x *= a;
        _caxpy(b, x, y);
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 5; }   //! flops per element
    };

    /**
       double caxpyXmaz(c a, V x, V y, V z){}
       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename real> struct caxpyxmaz_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const complex<real> a;
      caxpyxmaz_(const complex<real> &a, const complex<real> &b, const complex<real> &c) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpy(a, x, y);
        _caxpy(-a, z, x);
      }
      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       double caxpyXmazMR(c a, V x, V y, V z){}

       This is a special variant of caxpyxmaz where we source the scalar multiplier from device memory. 

       First performs the operation y[i] += a*x[i]
       Second performs the operator x[i] -= a*z[i]
    */
    template <typename real> struct caxpyxmazMR_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      complex<real> a;
      double3 *Ar3;
      caxpyxmazMR_(const real &a, const real &b, const real &c) :
        a(a),
        Ar3(static_cast<double3 *>(blas::getDeviceReduceBuffer()))
      { ; }

      __device__ __host__ void init()
      {
        double3 result = *Ar3;
        a = a.real() * complex<real>((real)result.x, (real)result.y) * ((real)1.0 / (real)result.z);
      }

      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        _caxpy(a, x, y);
        _caxpy(-a, z, x);
      }

      static int streams() { return 5; } //! total number of input and output streams
      static int flops() { return 8; }   //! flops per element
    };

    /**
       double tripleCGUpdate(d a, d b, V x, V y, V z, V w){}
       First performs the operation y[i] = y[i] + a*w[i]
       Second performs the operation z[i] = z[i] - a*x[i]
       Third performs the operation w[i] = z[i] + b*w[i]
    */
    template <typename real> struct tripleCGUpdate_ : public BlasFunctor {
      static constexpr write<0, 1, 1, 1> write{ };
      const real a;
      const real b;
      tripleCGUpdate_(const real &a, const real &b, const real &c) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        y += a * w;
        z -= a * x;
        w = z + b * w;
      }
      static int streams() { return 7; } //! total number of input and output streams
      static int flops() { return 6; }   //! flops per element
    };

    /**
       void doubleCG3Init(d a, V x, V y, V z){}
        y = x;
        x += a.x*z;
    */
    template <typename real> struct doubleCG3Init_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const real a;
      doubleCG3Init_(const real &a, const real &b, const real &c) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        y = x;
        x += a * z;
      }
      static int streams() { return 3; } //! total number of input and output streams
      static int flops() { return 3; }   //! flops per element
    };

    /**
       void doubleCG3Update(d a, d b, V x, V y, V z){}
        tmp = x;
        x = b.x*(x+a.x*z) + b.y*y;
        y = tmp;
    */
    template <typename real> struct doubleCG3Update_ : public BlasFunctor {
      static constexpr write<1, 1> write{ };
      const real a;
      const real b;
      const real c;
      doubleCG3Update_(const real &a, const real &b, const real &c) : a(a), b(b), c(c) { ; }
      template <typename T> __device__ __host__ void operator()(T &x, T &y, T &z, T &w, T &v)
      {
        auto tmp = x;
        x = b * (x + a * z) + c * y;
        y = tmp;
      }
      static int streams() { return 4; } //! total number of input and output streams
      static int flops() { return 7; }   //! flops per element
    };

  } // namespace blas
} // namespace quda
