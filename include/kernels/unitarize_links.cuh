#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <suN_project.cuh>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <svd_quda.h>
#include <kernel.h>

namespace quda {

  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct UnitarizeArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    Gauge out;
    const Gauge in;

    int X[4]; // grid dimensions
    int *fails;
    const int max_iter;
    const double unitarize_eps;
    const double max_error;
    const int reunit_allow_svd;
    const int reunit_svd_only;
    const double svd_rel_error;
    const double svd_abs_error;
    const static bool check_unitarization = true;

    UnitarizeArg(GaugeField &out, const GaugeField &in, int* fails, int max_iter,
                 double unitarize_eps, double max_error, int reunit_allow_svd,
                 int reunit_svd_only, double svd_rel_error, double svd_abs_error) :
      kernel_param(dim3(in.VolumeCB(), 2, 4)),
      out(out),
      in(in),
      fails(fails),
      max_iter(max_iter),
      unitarize_eps(unitarize_eps),
      max_error(max_error),
      reunit_allow_svd(reunit_allow_svd),
      reunit_svd_only(reunit_svd_only),
      svd_rel_error(svd_rel_error),
      svd_abs_error(svd_abs_error)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = in.X()[dir];
    }
  };

  template <typename mat>
  __device__ __host__ bool isUnitarizedLinkConsistent(const mat &initial_matrix,
                                                      const mat &unitary_matrix, double max_error)
  {
    auto n = initial_matrix.rows();
    mat temporary = conj(initial_matrix)*unitary_matrix;
    temporary = temporary*temporary - conj(initial_matrix)*initial_matrix;

    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
	if (fabs(temporary(i,j).x) > max_error ||
	    fabs(temporary(i,j).y) > max_error ||
	    temporary(i,j).x != temporary(i,j).x ||
	    temporary(i,j).y != temporary(i,j).y ) {
	  printf("is not consistent at %d,%d, %e, %e\n", i, j, temporary(i,j).x, temporary(i,j).y);
	  return false;
	}
      }
    }
    return true;
  }

  template <class T> constexpr T getAbsMin(const T* const array, int size)
  {
    T min = fabs(array[0]);
    for(int i=1; i<size; ++i){
      T abs_val = fabs(array[i]);
      if((abs_val) < min){ min = abs_val; }
    }
    return min;
  }

  template <class Real> constexpr bool checkAbsoluteError(Real a, Real b, Real epsilon) { return fabs(a-b) < epsilon; }

  template <class Real> constexpr bool checkRelativeError(Real a, Real b, Real epsilon) { return fabs((a-b)/b) < epsilon; }

  // Compute the reciprocal square root of the matrix q
  // Also modify q if the eigenvalues are dangerously small.
  template <typename real, typename mat, typename Arg>
  __device__  __host__ bool reciprocalRoot(mat &res, const mat& q, const Arg &arg)
  {
    real c[3];
    real g[3];

    mat qsq = q*q;
    mat tempq = qsq*q;

    c[0] = getTrace(q).real();
    c[1] = getTrace(qsq).real() * static_cast<real>(0.5);
    c[2] = getTrace(tempq).real() * static_cast<real>(1.0 / 3.0);

    g[0] = g[1] = g[2] = c[0] * static_cast<real>(1.0 / 3.0);
    real r,theta;
    real s = c[1] * static_cast<real>(1.0 / 3.0) - c[0] * c[0] * static_cast<real>(1.0 / 18.0);

    real cosTheta;
    if (fabs(s) >= arg.unitarize_eps) { // faster when this conditional is removed?
      const real rsqrt_s = quda::rsqrt(s);
      r = c[2]*0.5 - (c[0] * static_cast<real>(1.0 / 3.0)) * (c[1] - c[0] * c[0] * static_cast<real>(1.0 / 9.0));
      cosTheta = r*rsqrt_s*rsqrt_s*rsqrt_s;

      if (fabs(cosTheta) >= static_cast<real>(1.0)) {
	theta = (r > 0) ? 0.0 : M_PI;
      } else {
	theta = acos(cosTheta); // this is the primary performance limiter
      }
      
      const real sqrt_s = s*rsqrt_s;
      
#if 0 // experimental version
      real as, ac;
      quda::sincos( theta*static_cast<real>(1.0 / 3.0), &as, &ac );
      g[0] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*ac;
      //g[1] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*(ac*cos(2 * M_PI / 3.0) - as*sin(2 * M_PI / 3));
      g[1] = c[0]*static_cast<real>(1.0 / 3.0) - 2*sqrt_s*(0.5*ac + as*0.8660254037844386467637);
      //g[2] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*(ac*cos(4 * M_PI / 3.0) - as*sin(4 * M_PI / 3));
      g[2] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*(-0.5*ac + as*0.8660254037844386467637);
#else
      g[0] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*cos( theta*static_cast<real>(1.0 / 3.0) );
      g[1] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*cos( theta*static_cast<real>(1.0 / 3.0) + static_cast<real>(2.0 * M_PI / 3.0));
      g[2] = c[0]*static_cast<real>(1.0 / 3.0) + 2*sqrt_s*cos( theta*static_cast<real>(1.0 / 3.0) + static_cast<real>(4.0 * M_PI / 3.0));
#endif
    }
    
    // Check the eigenvalues, if the determinant does not match the product of the eigenvalues
    // return false. Then call SVD instead.
    real det = getDeterminant(q).real();
    if (fabs(det) < arg.svd_abs_error) return false;
    if (!checkRelativeError<double>(g[0]*g[1]*g[2], det, arg.svd_rel_error)) return false;
      
    // At this point we have finished with the c's
    // use these to store sqrt(g)
    for(int i=0; i<3; ++i) c[i] = sqrt(g[i]);
    
    // done with the g's, use these to store u, v, w
    g[0] = c[0]+c[1]+c[2];
    g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
    g[2] = c[0]*c[1]*c[2];
    
    const real denominator = 1.0 / ( g[2]*(g[0]*g[1]-g[2]) );
    c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1])) * denominator;
    c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1]) * denominator;
    c[2] = g[0] * denominator;
    
    tempq = c[1]*q + c[2]*qsq;
    // Add a real scalar
    tempq(0,0).x += c[0];
    tempq(1,1).x += c[0];
    tempq(2,2).x += c[0];
    
    res = tempq;
    
    return true;
  }
  
  template <typename real, typename mat, typename Arg>
  __host__ __device__ bool unitarizeLinkMILC(mat &out, const mat &in, const Arg &arg)
  {
    // Canonnical Nc = 3 strategy
#if (N_COLORS == 3)
    mat u;
    if (!arg.reunit_svd_only) {
      if (reciprocalRoot<real>(u, conj(in)*in, arg) ) {
	out = in * u;
	return true;
      }
    }

    // If we've got this far, then the Caley-Hamilton unitarization
    // has failed. If SVD is not allowed, the unitarization has failed.
    if (!arg.reunit_allow_svd) return false;

    mat v;
    real singular_values[3];
    computeSVD<real>(in, u, v, singular_values);    
    out = u * conj(v);
    return true;
#else
    // Brute force Nc != 3 strategy
    return unitarizeLinkNewton(out, in, 10000);
#endif
    
  } // unitarizeMILC
  
  template <typename mat>
  __host__ __device__ bool unitarizeLinkNewton(mat &out, const mat& in, int max_iter)
  {
    mat u = in;
    bool is_unitarized = false;
    int i = 0;
    while (!is_unitarized && i < max_iter) {
      mat uinv = inverse(u);
      u = 0.5*(u + conj(uinv));
      if((i+1)%10 == 0) is_unitarized = isUnitarizedLinkConsistent(in, u, 1e-7);	
      i++;
    }
    
    if (!is_unitarized) {
      //printf("ERROR: Unitarized link is not consistent with incoming link"
      //" tol %e and %d iterations\n", 1e-7, max_iter);
      return false;
    }
    //printf("is %sunitary at %d\n", is_unitarized ? "" : "not ", i);
    out = u;
    return true;
  }
  
  template <typename Arg> struct Unitarize
  {
    const Arg &arg;
    constexpr Unitarize(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int mu)
    {
      // result is always in double precision
      Matrix<complex<double>,Arg::nColor> v, result;
      Matrix<complex<typename Arg::real>,Arg::nColor> tmp = arg.in(mu, x_cb, parity);

      v = tmp;
      unitarizeLinkMILC<double>(result, v, arg);
      if (arg.check_unitarization) {
        if (result.isUnitary(arg.max_error) == false) atomic_fetch_add(arg.fails, 1);
      }
      tmp = result;
      arg.out(mu, x_cb, parity) = tmp;
    }
  };

  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct ProjectSUNArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    Gauge u;

    real tol;
    int *fails;
    ProjectSUNArg(GaugeField &u, real tol, int *fails) :
      kernel_param(dim3(u.VolumeCB(), 2, 4)),
      u(u),
      tol(tol),
      fails(fails) { }
  };

  template <typename Arg> struct Projector
  {
    const Arg &arg;
    constexpr Projector(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int mu)
    {
      Matrix<complex<typename Arg::real>, Arg::nColor> u = arg.u(mu, x_cb, parity);

      polarSUN<typename Arg::real>(u, arg.tol);

      // count number of failures
      if (u.isUnitary(arg.tol) == false) atomic_fetch_add(arg.fails, 1);
      
      arg.u(mu, x_cb, parity) = u;
    }
  };

}
