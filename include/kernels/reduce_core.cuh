#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>
#include <reduce_helper.h>
#include <reduction_kernel.h>

namespace quda
{

  namespace blas
  {

    /**
       @brief Parameter struct for generic reduction blas kernel.
       @tparam real_ The precision of the calculation
       @tparam n_ The number of real elements per thread
       @tparam store_t Default store type for the fields
       @tparam N Default field vector i/o length
       @tparam y_store_t Store type for the y fields
       @tparam Ny Y-field vector i/o length
       @tparam Reducer_ Functor used to operate on data
    */
    template <typename real_, int n_, typename store_t, int N, typename y_store_t, int Ny, typename Reducer_>
    struct ReductionArg : public ReduceArg<typename Reducer_::reduce_t> {
      using real = real_;
      static constexpr int n = n_;
      using Reducer = Reducer_;
      using reduce_t = typename Reducer_::reduce_t;
      Spinor<store_t, N> X;
      Spinor<y_store_t, Ny> Y;
      Spinor<store_t, N> Z;
      Spinor<store_t, N> W;
      Spinor<store_t, N> V;
      Reducer r;

      const int length_cb;
      const int nParity;

      ReductionArg(ColorSpinorField &x, ColorSpinorField &y, ColorSpinorField &z, ColorSpinorField &w,
                   ColorSpinorField &v, Reducer r, int length, int nParity) :
        ReduceArg<reduce_t>(dim3(length, 1, 1)),
        X(x),
        Y(y),
        Z(z),
        W(w),
        V(v),
        r(r),
        length_cb(length / nParity),
        nParity(nParity) { }

      __device__ __host__ auto init() const { return ::quda::zero<reduce_t>(); }
    };

    /**
       Generic reduction kernel with up to five loads and saves.
    */
    template <typename Arg> struct Reduce_ : plus<typename Arg::Reducer::reduce_t> {
      using reduce_t = typename Arg::Reducer::reduce_t;
      using plus<reduce_t>::operator();
      Arg &arg;
      constexpr Reduce_(const Arg &arg) : arg(const_cast<Arg&>(arg))
      {
        // this assertion ensures it's safe to make the arg non-const (required for HQ residual)
        static_assert(Arg::use_kernel_arg, "This functor must be passed as a kernel argument");
      }
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int tid, int) const
      {
        using vec = vector_type<complex<typename Arg::real>, Arg::n/2>;

        unsigned int parity = tid >= arg.length_cb ? 1 : 0;
        unsigned int i = tid - parity * arg.length_cb;

        vec x, y, z, w, v;
        if (arg.r.read.X) arg.X.load(x, i, parity);
        if (arg.r.read.Y) arg.Y.load(y, i, parity);
        if (arg.r.read.Z) arg.Z.load(z, i, parity);
        if (arg.r.read.W) arg.W.load(w, i, parity);
        if (arg.r.read.V) arg.V.load(v, i, parity);

        arg.r.pre();
        arg.r(sum, x, y, z, w, v);
        arg.r.post(sum);

        if (arg.r.write.X) arg.X.save(x, i, parity);
        if (arg.r.write.Y) arg.Y.save(y, i, parity);
        if (arg.r.write.Z) arg.Z.save(z, i, parity);
        if (arg.r.write.W) arg.W.save(w, i, parity);
        if (arg.r.write.V) arg.V.save(v, i, parity);

        return sum;
      }
    };

    /**
       Base class from which all reduction functors should derive.

       @tparam reduce_t The fundamental reduction type
       @tparam site_unroll Whether each thread must update the entire site
    */
    template <typename reduce_t_, bool site_unroll_ = false>
    struct ReduceFunctor {
      using reduce_t = reduce_t_;
      static constexpr bool site_unroll = site_unroll_;

      //! pre-computation routine called before the "M-loop"
      __device__ __host__ void pre() const { ; }

      //! post-computation routine called after the "M-loop"
      __device__ __host__ void post(reduce_t &) const { ; }
    };

    /**
       Return the L1 norm of x
    */
    template <typename reduce_t, typename T> __device__ __host__ reduce_t norm1_(const typename VectorType<T, 2>::type &a)
    {
      return static_cast<reduce_t>(sqrt(a.x * a.x + a.y * a.y));
    }

    template <typename reduce_t, typename real>
    struct Norm1 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1> read{ };
      static constexpr memory_access<> write{ };
      Norm1(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &, T &, T &, T &) const
      {
#pragma unroll
        for (int i=0; i < x.size(); i++) sum += norm1_<reduce_t, real>(x[i]);
      }
      constexpr int flops() const { return 2; }   //! flops per element
    };

    /**
       Return the L2 norm of x
    */
    template <typename reduce_t, typename T> __device__ __host__ void norm2_(reduce_t &sum, const typename VectorType<T, 2>::type &a)
    {
      sum += static_cast<reduce_t>(a.x) * static_cast<reduce_t>(a.x);
      sum += static_cast<reduce_t>(a.y) * static_cast<reduce_t>(a.y);
    }

    template <typename reduce_t, typename real>
    struct Norm2 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1> read{ };
      static constexpr memory_access<> write{ };
      Norm2(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) norm2_<reduce_t, real>(sum, x[i]);
      }
      constexpr int flops() const { return 2; }   //! flops per element
    };

    /**
       Return the real dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void dot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      sum += static_cast<reduce_t>(a.x) * static_cast<reduce_t>(b.x);
      sum += static_cast<reduce_t>(a.y) * static_cast<reduce_t>(b.y);
    }

    template <typename reduce_t, typename real>
    struct Dot : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1,1> read{ };
      static constexpr memory_access<> write{ };
      Dot(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) dot_<reduce_t, real>(sum, x[i], y[i]);
      }
      constexpr int flops() const { return 2; }   //! flops per element
    };

    /**
       First performs the operation z[i] = a*x[i] + b*y[i]
       Return the norm of y
    */
    template <typename reduce_t, typename real>
    struct axpbyzNorm2 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 0> read{ };
      static constexpr memory_access<0, 0, 1> write{ };
      const real a;
      const real b;
      axpbyzNorm2(const real &a, const real &b) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          z[i] = a * x[i] + b * y[i];
          norm2_<reduce_t, real>(sum, z[i]);
        }
      }
      constexpr int flops() const { return 4; }   //! flops per element
    };

    /**
       First performs the operation y[i] += a*x[i]
       Return real dot product (x,y)
    */
    template <typename reduce_t, typename real>
    struct AxpyReDot : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      const real a;
      AxpyReDot(const real &a, const real &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] += a * x[i];
          dot_<reduce_t, real>(sum, x[i], y[i]);
        }
      }
      constexpr int flops() const { return 4; }   //! flops per element
    };

    /**
       First performs the operation y[i] = a*x[i] + y[i] (complex-valued)
       Second returns the norm of y
    */
    template <typename reduce_t, typename real>
    struct caxpyNorm2 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      const complex<real> a;
      caxpyNorm2(const complex<real> &a, const complex<real> &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a, x[i], y[i]);
          norm2_<reduce_t, real>(sum, y[i]);
        }
      }
      constexpr int flops() const { return 6; }   //! flops per element
    };

    /**
       double caxpyXmayNormCuda(float a, float *x, float *y, n){}
       First performs the operation y[i] = a*x[i] + y[i]
       Second performs the operator x[i] -= a*z[i]
       Third returns the norm of x
    */
    template <typename reduce_t, typename real>
    struct caxpyxmaznormx : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<1, 1> write{ };
      const complex<real> a;
      caxpyxmaznormx(const complex<real> &a, const complex<real> &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a, x[i], y[i]);
          x[i] = cmac(-a, z[i], x[i]);
          norm2_<reduce_t, real>(sum, x[i]);
        }
      }
      constexpr int flops() const { return 10; }  //! flops per element
    };

    /**
       double cabxpyzAxNorm(float a, complex b, float *x, float *y, float *z){}
       First performs the operation z[i] = y[i] + a*b*x[i]
       Second performs x[i] *= a
       Third returns the norm of x
    */
    template <typename reduce_t, typename real>
    struct cabxpyzaxnorm : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 0> read{ };
      static constexpr memory_access<1, 0, 1> write{ };
      const real a;
      const complex<real> b;
      cabxpyzaxnorm(const complex<real> &a, const complex<real> &b) : a(a.real()), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          x[i] *= a;
          z[i] = cmac(b, x[i], y[i]);
          norm2_<reduce_t, real>(sum, z[i]);
        }
      }
      constexpr int flops() const { return 10; }  //! flops per element
    };

    /**
       Returns complex-valued dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const typename VectorType<T, 2>::type &a, const typename VectorType<T, 2>::type &b)
    {
      using scalar = typename scalar<reduce_t>::type;
      sum.x += static_cast<scalar>(a.x) * static_cast<scalar>(b.x);
      sum.x += static_cast<scalar>(a.y) * static_cast<scalar>(b.y);
      sum.y += static_cast<scalar>(a.x) * static_cast<scalar>(b.y);
      sum.y -= static_cast<scalar>(a.y) * static_cast<scalar>(b.x);
    }

    template <typename real_reduce_t, typename real>
    struct Cdot : public ReduceFunctor<typename VectorType<real_reduce_t, 2>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 2>::type;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      Cdot(const complex<real> &, const complex<real> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) cdot_<reduce_t, real>(sum, x[i], y[i]);
      }
      constexpr int flops() const { return 4; }   //! flops per element
    };

    /**
       double caxpyDotzyCuda(float a, float *x, float *y, float *z, n){}
       First performs the operation y[i] = a*x[i] + y[i]
       Second returns the dot product (z,y)
    */
    template <typename real_reduce_t, typename real>
    struct caxpydotzy : public ReduceFunctor<typename VectorType<real_reduce_t, 2>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 2>::type;
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      const complex<real> a;
      caxpydotzy(const complex<real> &a, const complex<real> &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a, x[i], y[i]);
          cdot_<reduce_t, real>(sum, z[i], y[i]);
        }
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       First returns the dot product (x,y)
       Returns the norm of x
    */
    template <typename reduce_t, typename InputType>
    __device__ __host__ void cdotNormA_(reduce_t &sum, const InputType &a, const InputType &b)
    {
      using real = typename scalar<InputType>::type;
      using scalar = typename scalar<reduce_t>::type;
      cdot_<reduce_t, real>(sum, a, b);
      norm2_<scalar, real>(sum.z, a);
    }

    /**
       First returns the dot product (x,y)
       Returns the norm of y
    */
    template <typename reduce_t, typename InputType>
    __device__ __host__ void cdotNormB_(reduce_t &sum, const InputType &a, const InputType &b)
    {
      using real = typename scalar<InputType>::type;
      using scalar = typename scalar<reduce_t>::type;
      cdot_<reduce_t, real>(sum, a, b);
      norm2_<scalar, real>(sum.z, b);
    }

    template <typename real_reduce_t, typename real>
    struct CdotNormA : public ReduceFunctor<typename VectorType<real_reduce_t, 3>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 3>::type;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      CdotNormA(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) cdotNormA_<reduce_t>(sum, x[i], y[i]);
      }
      constexpr int flops() const { return 6; }   //! flops per element
    };

    /**
       This convoluted kernel does the following:
       y += a*x + b*z, z -= b*w, norm = (z,z), dot = (u, z)
    */
    template <typename real_reduce_t, typename real>
    struct caxpbypzYmbwcDotProductUYNormY_ : public ReduceFunctor<typename VectorType<real_reduce_t, 3>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 3>::type;
      static constexpr memory_access<1, 1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1> write{ };
      const complex<real> a;
      const complex<real> b;
      caxpbypzYmbwcDotProductUYNormY_(const complex<real> &a, const complex<real> &b) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a, x[i], y[i]);
          y[i] = cmac(b, z[i], y[i]);
          z[i] = cmac(-b, w[i], z[i]);
          cdotNormB_<reduce_t>(sum, v[i], z[i]);
        }
      }
      constexpr int flops() const { return 18; }  //! flops per element
    };

    /**
       Specialized kernel for the modified CG norm computation for
       computing beta.  Computes y = y + a*x and returns norm(y) and
       dot(y, delta(y)) where delta(y) is the difference between the
       input and out y vector.
    */
    template <typename real_reduce_t, typename real>
    struct axpyCGNorm2 : public ReduceFunctor<typename VectorType<real_reduce_t, 2>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 2>::type;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      const real a;
      axpyCGNorm2(const real &a, const real &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          auto y_new = y[i] + a * x[i];
          norm2_<real_reduce_t, real>(sum.x, y_new);
          dot_<real_reduce_t, real>(sum.y, y_new, y_new - y[i]);
          y[i] = y_new;
        }
      }
      constexpr int flops() const { return 6; }   //! flops per real element
    };

    /**
       This kernel returns (x, x) and (r,r) and also returns the
       so-called heavy quark norm as used by MILC: 1 / N * \sum_i (r,
       r)_i / (x, x)_i, where i is site index and N is the number of
       sites.  We must enforce that each thread updates an entire
       lattice hence the site_unroll template parameter must be set
       true.
    */
    template <typename real_reduce_t, typename real>
    struct HeavyQuarkResidualNorm_ {
      using reduce_t = typename VectorType<real_reduce_t, 3>::type;
      static constexpr bool site_unroll = true;

      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      reduce_t aux;
      HeavyQuarkResidualNorm_(const real &, const real &) : aux {} { ; }

      __device__ __host__ void pre()
      {
        aux.x = 0;
        aux.y = 0;
      }

      template <typename T> __device__ __host__ void operator()(reduce_t &, T &x, T &y, T &, T &, T &)
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(aux.x, x[i]);
          norm2_<real_reduce_t, real>(aux.y, y[i]);
        }
      }

      //! sum the solution and residual norms, and compute the heavy-quark norm
      __device__ __host__ void post(reduce_t &sum)
      {
        sum.x += aux.x;
        sum.y += aux.y;
        sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : static_cast<real>(1.0);
      }

      constexpr int flops() const { return 4; }   //! undercounts since it excludes the per-site division
    };

    /**
      Variant of the HeavyQuarkResidualNorm kernel: this takes three
      arguments, the first two are summed together to form the
      solution, with the third being the residual vector.  This
      removes the need an additional xpy call in the solvers,
      improving performance.  We must enforce that each thread updates
      an entire lattice hence the site_unroll template parameter must
      be set true.
    */
    template <typename real_reduce_t, typename real>
    struct xpyHeavyQuarkResidualNorm_ {
      using reduce_t = typename VectorType<real_reduce_t, 3>::type;
      static constexpr bool site_unroll = true;

      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      reduce_t aux;
      xpyHeavyQuarkResidualNorm_(const real &, const real &) : aux {} { ; }

      __device__ __host__ void pre()
      {
        aux.x = 0;
        aux.y = 0;
      }

      template <typename T> __device__ __host__ void operator()(reduce_t &, T &x, T &y, T &z, T &, T &)
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(aux.x, x[i] + y[i]);
          norm2_<real_reduce_t, real>(aux.y, z[i]);
        }
      }

      //! sum the solution and residual norms, and compute the heavy-quark norm
      __device__ __host__ void post(reduce_t &sum)
      {
        sum.x += aux.x;
        sum.y += aux.y;
        sum.z += (aux.x > 0.0) ? (aux.y / aux.x) : static_cast<real>(1.0);
      }

      constexpr int flops() const { return 5; }
    };

    /**
       double3 tripleCGReduction(V x, V y, V z){}
       First performs the operation norm2(x)
       Second performs the operatio norm2(y)
       Third performs the operation dotPropduct(y,z)
    */
    template <typename real_reduce_t, typename real>
    struct tripleCGReduction_ : public ReduceFunctor<typename VectorType<real_reduce_t, 3>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 3>::type;
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      using scalar = typename scalar<reduce_t>::type;
      tripleCGReduction_(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(sum.x, x[i]);
          norm2_<real_reduce_t, real>(sum.y, y[i]);
          dot_<real_reduce_t, real>(sum.z, y[i], z[i]);
        }
      }
      constexpr int flops() const { return 6; }   //! flops per element
    };

    /**
       double4 quadrupleCGReduction(V x, V y, V z){}
       First performs the operation norm2(x)
       Second performs the operatio norm2(y)
       Third performs the operation dotPropduct(y,z)
       Fourth performs the operation norm(z)
    */
    template <typename real_reduce_t, typename real>
    struct quadrupleCGReduction_ : public ReduceFunctor<typename VectorType<real_reduce_t, 4>::type> {
      using reduce_t = typename VectorType<real_reduce_t, 4>::type;
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      quadrupleCGReduction_(const real &, const real &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(sum.x, x[i]);
          norm2_<real_reduce_t, real>(sum.y, y[i]);
          dot_<real_reduce_t, real>(sum.z, y[i], z[i]);
          norm2_<real_reduce_t, real>(sum.w, w[i]);
        }
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       double quadrupleCG3InitNorm(d a, d b, V x, V y, V z, V w, V v){}
        z = x;
        w = y;
        x += a*y;
        y -= a*v;
        norm2(y);
    */
    template <typename reduce_t, typename real>
    struct quadrupleCG3InitNorm_ : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 0, 0, 1> read{ };
      static constexpr memory_access<1, 1, 1, 1> write{ };
      const real a;
      quadrupleCG3InitNorm_(const real &a, const real &) : a(a) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          z[i] = x[i];
          w[i] = y[i];
          x[i] += a * y[i];
          y[i] -= a * v[i];
          norm2_<reduce_t, real>(sum, y[i]);
        }
      }
      constexpr int flops() const { return 6; }   //! flops per element check if it's right
    };

    /**
       double quadrupleCG3UpdateNorm(d gamma, d rho, V x, V y, V z, V w, V v){}
        tmpx = x;
        tmpy = y;
        x = b*(x + a*y) + (1-b)*z;
        y = b*(y + a*v) + (1-b)*w;
        z = tmpx;
        w = tmpy;
        norm2(y);
    */
    template <typename reduce_t, typename real>
    struct quadrupleCG3UpdateNorm_ : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 1, 1, 1> read{ };
      static constexpr memory_access<1, 1, 1, 1> write{ };
      const real a;
      const real b;
      quadrupleCG3UpdateNorm_(const real &a, const real &b) : a(a), b(b) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          auto tmpx = x[i];
          auto tmpy = y[i];
          x[i] = b * (x[i] + a * y[i]) + ((real)1.0 - b) * z[i];
          y[i] = b * (y[i] - a * v[i]) + ((real)1.0 - b) * w[i];
          z[i] = tmpx;
          w[i] = tmpy;
          norm2_<reduce_t, real>(sum, y[i]);
        }
      }
      constexpr int flops() const { return 16; }  //! flops per element check if it's right
    };

  } // namespace blas

} // namespace quda
