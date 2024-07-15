#pragma once

#include <color_spinor_field_order.h>
#include <blas_helper.cuh>
#include <reduce_helper.h>
#include <array.h>
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
      Spinor<store_t, N> X[MAX_MULTI_RHS] = {};
      Spinor<y_store_t, Ny> Y[MAX_MULTI_RHS] = {};
      Spinor<store_t, N> Z[MAX_MULTI_RHS] = {};
      Spinor<store_t, N> W[MAX_MULTI_RHS] = {};
      Spinor<store_t, N> V[MAX_MULTI_RHS] = {};
      Reducer r;

      const int length_cb;
      const int nParity;

      ReductionArg(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &y, cvector_ref<ColorSpinorField> &z,
                   cvector_ref<ColorSpinorField> &w, cvector_ref<ColorSpinorField> &v, Reducer r, int length,
                   int nParity) :
        ReduceArg<reduce_t>(dim3(length, 1, x.size()), x.size()), r(r), length_cb(length / nParity), nParity(nParity)
      {
        for (auto i = 0u; i < x.size(); i++) X[i] = x[i];
        for (auto i = 0u; i < y.size(); i++) Y[i] = y[i];
        for (auto i = 0u; i < z.size(); i++) Z[i] = z[i];
        for (auto i = 0u; i < w.size(); i++) W[i] = w[i];
        for (auto i = 0u; i < v.size(); i++) V[i] = v[i];
      }
    };

    /**
       Generic reduction kernel with up to five loads and saves.
    */
    template <typename Arg> struct Reduce_ : Arg::Reducer::reducer {
      using reduce_t = typename Arg::reduce_t;
      using Arg::Reducer::reducer::operator();
      static constexpr int reduce_block_dim = 1; // x_cb and parity are mapped to x dim
      Arg &arg;
      constexpr Reduce_(const Arg &arg) : arg(const_cast<Arg&>(arg))
      {
      }
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ inline reduce_t operator()(reduce_t &sum, int tid, int, int src_idx) const
      {
        using vec = array<complex<typename Arg::real>, Arg::n/2>;

        unsigned int parity = tid >= arg.length_cb ? 1 : 0;
        unsigned int i = tid - parity * arg.length_cb;

        vec x, y, z, w, v;
        if (arg.r.read.X) arg.X[src_idx].load(x, i, parity);
        if (arg.r.read.Y) arg.Y[src_idx].load(y, i, parity);
        if (arg.r.read.Z) arg.Z[src_idx].load(z, i, parity);
        if (arg.r.read.W) arg.W[src_idx].load(w, i, parity);
        if (arg.r.read.V) arg.V[src_idx].load(v, i, parity);

        arg.r(sum, x, y, z, w, v, src_idx);

        if (arg.r.write.X) arg.X[src_idx].save(x, i, parity);
        if (arg.r.write.Y) arg.Y[src_idx].save(y, i, parity);
        if (arg.r.write.Z) arg.Z[src_idx].save(z, i, parity);
        if (arg.r.write.W) arg.W[src_idx].save(w, i, parity);
        if (arg.r.write.V) arg.V[src_idx].save(v, i, parity);

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
      using reducer = plus<reduce_t>;
      static constexpr bool site_unroll = site_unroll_;
    };

    template <typename reduce_t, typename real>
    struct Max : public ReduceFunctor<reduce_t> {
      using reducer = maximum<reduce_t>;
      static constexpr memory_access<1> read{ };
      static constexpr memory_access<> write{ };
      Max(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &max, T &x, T &, T &, T &, T &, int) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          max = max > abs(x[i].real()) ? max : abs(x[i].real());
          max = max > abs(x[i].imag()) ? max : abs(x[i].imag());
        }
      }
      constexpr int flops() const { return 0; }   //! flops per element
    };

    template <typename real_reduce_t, typename real>
    struct MaxDeviation : public ReduceFunctor<deviation_t<real_reduce_t>> {
      using reduce_t = deviation_t<real_reduce_t>;
      using reducer = maximum<reduce_t>;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      MaxDeviation(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &max, T &x, T &y, T &, T &, T &, int) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          complex<real_reduce_t> diff = {abs(x[i].real() - y[i].real()), abs(x[i].imag() - y[i].imag())};
          if (diff.real() > max.diff ) {
            max.diff = diff.real();
            max.ref = abs(y[i].real());
          }
          if (diff.imag() > max.diff) {
            max.diff = diff.imag();
            max.ref = abs(y[i].imag());
          }
        }
      }
      constexpr int flops() const { return 0; }   //! flops per element
    };

    /**
       Return the L1 norm of x
    */
    template <typename reduce_t, typename T> __device__ __host__ reduce_t norm1_(const complex<T> &a)
    {
      return static_cast<reduce_t>(sqrt(a.real() * a.real() + a.imag() * a.imag()));
    }

    template <typename reduce_t, typename real>
    struct Norm1 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1> read{ };
      static constexpr memory_access<> write{ };
      Norm1(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &, T &, T &, T &, int) const
      {
#pragma unroll
        for (int i=0; i < x.size(); i++) sum += norm1_<reduce_t, real>(x[i]);
      }
      constexpr int flops() const { return 2; }   //! flops per element
    };

    /**
       Return the L2 norm of x
    */
    template <typename reduce_t, typename T> __device__ __host__ void norm2_(reduce_t &sum, const complex<T> &a)
    {
      sum += static_cast<reduce_t>(a.real()) * static_cast<reduce_t>(a.real());
      sum += static_cast<reduce_t>(a.imag()) * static_cast<reduce_t>(a.imag());
    }

    template <typename reduce_t, typename real>
    struct Norm2 : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1> read{ };
      static constexpr memory_access<> write{ };
      Norm2(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &, T &, T &, T &, int) const
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
    __device__ __host__ void dot_(reduce_t &sum, const complex<T> &a, const complex<T> &b)
    {
      sum += static_cast<reduce_t>(a.real()) * static_cast<reduce_t>(b.real());
      sum += static_cast<reduce_t>(a.imag()) * static_cast<reduce_t>(b.imag());
    }

    template <typename reduce_t, typename real>
    struct Dot : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1,1> read{ };
      static constexpr memory_access<> write{ };
      Dot(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int) const
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
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      axpbyzNorm2(cvector<double> &a, cvector<double> &b)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          z[i] = a[j] * x[i] + b[j] * y[i];
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
      real a[MAX_MULTI_RHS] = {};
      AxpyReDot(cvector<double> &a, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] += a[j] * x[i];
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
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpyNorm2(cvector<Complex> &a, cvector<Complex> &b)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] *= b[j];
          y[i] = cmac(a[j], x[i], y[i]);
          norm2_<reduce_t, real>(sum, y[i]);
        }
      }
      constexpr int flops() const { return 8; } //! flops per element
    };

    /**
       double cabxpyzAxNorm(float a, complex b, float *x, float *y, float *z){}
       First performs the operation z[i] = y[i] + a*b*x[i]
       Second performs x[i] *= a
       Third returns the norm of z
    */
    template <typename reduce_t, typename real>
    struct cabxpyzaxnorm : public ReduceFunctor<reduce_t> {
      static constexpr memory_access<1, 1, 0> read{ };
      static constexpr memory_access<1, 0, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      cabxpyzaxnorm(cvector<Complex> &a, cvector<Complex> &b)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i].real();
        for (auto i = 0u; i < a.size(); i++) this->b[i] = b[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          x[i] *= a[j];
          z[i] = cmac(b[j], x[i], y[i]);
          norm2_<reduce_t, real>(sum, z[i]);
        }
      }
      constexpr int flops() const { return 10; }  //! flops per element
    };

    /**
       Returns complex-valued dot product of x and y
    */
    template <typename reduce_t, typename T>
    __device__ __host__ void cdot_(reduce_t &sum, const complex<T> &a, const complex<T> &b)
    {
      using scalar_t = typename reduce_t::value_type;
      sum[0] += static_cast<scalar_t>(a.real()) * static_cast<scalar_t>(b.real());
      sum[0] += static_cast<scalar_t>(a.imag()) * static_cast<scalar_t>(b.imag());
      sum[1] += static_cast<scalar_t>(a.real()) * static_cast<scalar_t>(b.imag());
      sum[1] -= static_cast<scalar_t>(a.imag()) * static_cast<scalar_t>(b.real());
    }

    template <typename real_reduce_t, typename real>
    struct Cdot : public ReduceFunctor<array<real_reduce_t, 2>> {
      using reduce_t = array<real_reduce_t, 2>;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      Cdot(cvector<Complex> &, cvector<Complex> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int) const
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
    struct caxpydotzy : public ReduceFunctor<array<real_reduce_t, 2>> {
      using reduce_t = array<real_reduce_t, 2>;
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      caxpydotzy(cvector<Complex> &a, cvector<Complex> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a[j], x[i], y[i]);
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
    __device__ __host__ void cdotNormAB_(reduce_t &sum, const InputType &a, const InputType &b)
    {
      using real = typename InputType::value_type;
      using scalar = typename reduce_t::value_type;
      cdot_<reduce_t, real>(sum, a, b);
      norm2_<scalar, real>(sum[2], a);
      norm2_<scalar, real>(sum[3], b);
    }

    template <typename real_reduce_t, typename real>
    struct CdotNormAB : public ReduceFunctor<array<real_reduce_t, 4>> {
      using reduce_t = array<real_reduce_t, 4>;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      CdotNormAB(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) cdotNormAB_<reduce_t>(sum, x[i], y[i]);
      }
      constexpr int flops() const { return 8; }   //! flops per element
    };

    /**
       First returns the dot product (x,y)
       Returns the norm of y
    */
    template <typename reduce_t, typename InputType>
    __device__ __host__ void cdotNormB_(reduce_t &sum, const InputType &a, const InputType &b)
    {
      using real = typename InputType::value_type;
      using scalar = typename reduce_t::value_type;
      cdot_<reduce_t, real>(sum, a, b);
      norm2_<scalar, real>(sum[2], b);
    }

    /**
       This convoluted kernel does the following:
       y += a*x + b*z, z -= b*w, norm = (z,z), dot = (u, z)
    */
    template <typename real_reduce_t, typename real>
    struct caxpbypzYmbwcDotProductUYNormY_ : public ReduceFunctor<array<real_reduce_t, 3>> {
      using reduce_t = array<real_reduce_t, 3>;
      static constexpr memory_access<1, 1, 1, 1, 1> read{ };
      static constexpr memory_access<0, 1, 1> write{ };
      complex<real> a[MAX_MULTI_RHS] = {};
      complex<real> b[MAX_MULTI_RHS] = {};
      caxpbypzYmbwcDotProductUYNormY_(cvector<Complex> &a, cvector<Complex> &b)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < b.size(); i++) this->b[i] = b[i];
      }
      template <typename T>
      __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          y[i] = cmac(a[j], x[i], y[i]);
          y[i] = cmac(b[j], z[i], y[i]);
          z[i] = cmac(-b[j], w[i], z[i]);
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
    struct axpyCGNorm2 : public ReduceFunctor<array<real_reduce_t, 2>> {
      using reduce_t = array<real_reduce_t, 2>;
      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<0, 1> write{ };
      real a[MAX_MULTI_RHS] = {};
      axpyCGNorm2(cvector<double> &a, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          auto y_new = y[i] + a[j] * x[i];
          norm2_<real_reduce_t, real>(sum[0], y_new);
          dot_<real_reduce_t, real>(sum[1], y_new, y_new - y[i]);
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
       lattice site hence the site_unroll template parameter must be
       set true.
    */
    template <typename real_reduce_t, typename real>
    struct HeavyQuarkResidualNorm_ {
      using reduce_t = array<real_reduce_t, 3>;
      using reducer = plus<reduce_t>;
      static constexpr bool site_unroll = true;

      static constexpr memory_access<1, 1> read{ };
      static constexpr memory_access<> write{ };
      HeavyQuarkResidualNorm_(cvector<double> &, cvector<double> &) { }

      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &, T &, T &, int)
      {
        reduce_t aux = {};

#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(aux[0], x[i]);
          norm2_<real_reduce_t, real>(aux[1], y[i]);
        }

        sum[0] += aux[0];
        sum[1] += aux[1];
        sum[2] += (aux[0] > 0.0) ? (aux[1] / aux[0]) : static_cast<real>(1.0);
      }

      constexpr int flops() const { return 4; }   //! undercounts since it excludes the per-site division
    };

    /**
      Variant of the HeavyQuarkResidualNorm kernel: this takes three
      arguments, the first two are summed together to form the
      solution, with the third being the residual vector.  This
      removes the need an additional xpy call in the solvers,
      improving performance.  We must enforce that each thread updates
      an entire lattice site hence the site_unroll template parameter
      must be set true.
    */
    template <typename real_reduce_t, typename real>
    struct xpyHeavyQuarkResidualNorm_ {
      using reduce_t = array<real_reduce_t, 3>;
      using reducer = plus<reduce_t>;
      static constexpr bool site_unroll = true;

      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      xpyHeavyQuarkResidualNorm_(cvector<double> &, cvector<double> &) { }

      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &, int)
      {
        reduce_t aux = {};

#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(aux[0], x[i] + y[i]);
          norm2_<real_reduce_t, real>(aux[1], z[i]);
        }

        sum[0] += aux[0];
        sum[1] += aux[1];
        sum[2] += (aux[0] > 0.0) ? (aux[1] / aux[0]) : static_cast<real>(1.0);
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
    struct tripleCGReduction_ : public ReduceFunctor<array<real_reduce_t, 3>> {
      using reduce_t = array<real_reduce_t, 3>;
      static constexpr memory_access<1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      tripleCGReduction_(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &, T &, int) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(sum[0], x[i]);
          norm2_<real_reduce_t, real>(sum[1], y[i]);
          dot_<real_reduce_t, real>(sum[2], y[i], z[i]);
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
    struct quadrupleCGReduction_ : public ReduceFunctor<array<real_reduce_t, 4>> {
      using reduce_t = array<real_reduce_t, 4>;
      static constexpr memory_access<1, 1, 1, 1> read{ };
      static constexpr memory_access<> write{ };
      quadrupleCGReduction_(cvector<double> &, cvector<double> &) { ; }
      template <typename T> __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &, int) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          norm2_<real_reduce_t, real>(sum[0], x[i]);
          norm2_<real_reduce_t, real>(sum[1], y[i]);
          dot_<real_reduce_t, real>(sum[2], y[i], z[i]);
          norm2_<real_reduce_t, real>(sum[3], w[i]);
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
      real a[MAX_MULTI_RHS] = {};
      quadrupleCG3InitNorm_(cvector<double> &a, cvector<double> &)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
      }
      template <typename T>
      __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          z[i] = x[i];
          w[i] = y[i];
          x[i] += a[j] * y[i];
          y[i] -= a[j] * v[i];
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
      real a[MAX_MULTI_RHS] = {};
      real b[MAX_MULTI_RHS] = {};
      quadrupleCG3UpdateNorm_(cvector<double> &a, cvector<double> &b)
      {
        for (auto i = 0u; i < a.size(); i++) this->a[i] = a[i];
        for (auto i = 0u; i < a.size(); i++) this->b[i] = b[i];
      }
      template <typename T>
      __device__ __host__ void operator()(reduce_t &sum, T &x, T &y, T &z, T &w, T &v, int j) const
      {
#pragma unroll
        for (int i = 0; i < x.size(); i++) {
          auto tmpx = x[i];
          auto tmpy = y[i];
          x[i] = b[j] * (x[i] + a[j] * y[i]) + ((real)1.0 - b[j]) * z[i];
          y[i] = b[j] * (y[i] - a[j] * v[i]) + ((real)1.0 - b[j]) * w[i];
          z[i] = tmpx;
          w[i] = tmpy;
          norm2_<reduce_t, real>(sum, y[i]);
        }
      }
      constexpr int flops() const { return 16; }  //! flops per element check if it's right
    };

  } // namespace blas

} // namespace quda
