#pragma once

#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <svd_quda.h>
#include <kernel.h>

namespace quda {

  static double unitarize_eps;
  static double force_filter;
  static double max_det_error;
  static bool   allow_svd;
  static bool   svd_only;
  static double svd_rel_error;
  static double svd_abs_error;

  namespace fermion_force {

    template <typename store_t, int nColor_, QudaReconstructType recon_, QudaGaugeFieldOrder order = QUDA_NATIVE_GAUGE_ORDER>
    struct UnitarizeForceArg : kernel_param<> {
      using real = double; // we always use double precision for this kernel
      static constexpr int nColor = nColor_;
      static constexpr QudaReconstructType recon = recon_;
      // use long form here to allow specification of order
      typedef typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO,2*nColor*nColor,QUDA_STAGGERED_PHASE_NO,gauge::default_huge_alloc, QUDA_GHOST_EXCHANGE_INVALID,false,order>::type F;
      typedef typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO,2*nColor*nColor,QUDA_STAGGERED_PHASE_NO,gauge::default_huge_alloc, QUDA_GHOST_EXCHANGE_INVALID,false,order>::type G;
      F force;
      const F force_old;
      const G u;
      int *fails;
      const real unitarize_eps;
      const real force_filter;
      const real max_det_error;
      const int allow_svd;
      const int svd_only;
      const real svd_rel_error;
      const real svd_abs_error;

      UnitarizeForceArg(GaugeField &force, const GaugeField &force_old, const GaugeField &u, int *fails,
			double unitarize_eps, double force_filter, double max_det_error, int allow_svd,
			int svd_only, double svd_rel_error, double svd_abs_error) :
        kernel_param(dim3(u.VolumeCB(), 2, 1)),
        force(force),
        force_old(force_old),
        u(u),
        fails(fails),
        unitarize_eps(unitarize_eps),
        force_filter(force_filter),
        max_det_error(max_det_error),
        allow_svd(allow_svd),
        svd_only(svd_only),
        svd_rel_error(svd_rel_error),
        svd_abs_error(svd_abs_error) { }
    };

    template <class Real> class DerivativeCoefficients {
      Real b[6];
      constexpr Real computeC00(const Real &u, const Real &v, const Real &w)
      {
        return -fpow(w,3) * fpow(u,6) + 3*v*fpow(w,3)*fpow(u,4) + 3*fpow(v,4)*w*fpow(u,4)
          - fpow(v,6)*fpow(u,3) - 4*fpow(w,4)*fpow(u,3) - 12*fpow(v,3)*fpow(w,2)*fpow(u,3)
          + 16*fpow(v,2)*fpow(w,3)*fpow(u,2) + 3*fpow(v,5)*w*fpow(u,2) - 8*v*fpow(w,4)*u
          - 3*fpow(v,4)*fpow(w,2)*u + fpow(w,5) + fpow(v,3)*fpow(w,3);
      }

      constexpr Real computeC01(const Real & u, const Real & v, const Real & w)
      {
        return -fpow(w,2)*fpow(u,7) - fpow(v,2)*w*fpow(u,6) + fpow(v,4)*fpow(u,5) + 6*v*fpow(w,2)*fpow(u,5)
          - 5*fpow(w,3)*fpow(u,4) - fpow(v,3)*w*fpow(u,4)- 2*fpow(v,5)*fpow(u,3) - 6*fpow(v,2)*fpow(w,2)*fpow(u,3)
          + 10*v*fpow(w,3)*fpow(u,2) + 6*fpow(v,4)*w*fpow(u,2) - 3*fpow(w,4)*u - 6*fpow(v,3)*fpow(w,2)*u + 2*fpow(v,2)*fpow(w,3);
      }

      constexpr Real computeC02(const Real & u, const Real & v, const Real & w)
      {
        return fpow(w,2)*fpow(u,5) + fpow(v,2)*w*fpow(u,4)- fpow(v,4)*fpow(u,3)- 4*v*fpow(w,2)*fpow(u,3)
          + 4*fpow(w,3)*fpow(u,2) + 3*fpow(v,3)*w*fpow(u,2) - 3*fpow(v,2)*fpow(w,2)*u + v*fpow(w,3);
      }

      constexpr Real computeC11(const Real & u, const Real & v, const Real & w)
      {
        return -w*fpow(u,8) - fpow(v,2)*fpow(u,7) + 7*v*w*fpow(u,6) + 4*fpow(v,3)*fpow(u,5)
          - 5*fpow(w,2)*fpow(u,5) - 16*fpow(v,2)*w*fpow(u,4) - 4*fpow(v,4)*fpow(u,3) + 16*v*fpow(w,2)*fpow(u,3)
          - 3*fpow(w,3)*fpow(u,2) + 12*fpow(v,3)*w*fpow(u,2) - 12*fpow(v,2)*fpow(w,2)*u + 3*v*fpow(w,3);
      }

      constexpr Real computeC12(const Real &u, const Real &v, const Real &w)
      {
        return w*fpow(u,6) + fpow(v,2)*fpow(u,5) - 5*v*w*fpow(u,4) - 2*fpow(v,3)*fpow(u,3)
          + 4*fpow(w,2)*fpow(u,3) + 6*fpow(v,2)*w*fpow(u,2) - 6*v*fpow(w,2)*u + fpow(w,3);
      }

      constexpr Real computeC22(const Real &u, const Real &v, const Real &w)
      {
        return -w*fpow(u,4) - fpow(v,2)*fpow(u,3) + 3*v*w*fpow(u,2) - 3*fpow(w,2)*u;
      }

    public:
      constexpr void set(const Real &u, const Real &v, const Real &w)
      {
        const Real denominator = 1.0 / (2.0*fpow(w*(u*v-w),3));
        b[0] = computeC00(u,v,w) * denominator;
        b[1] = computeC01(u,v,w) * denominator;
        b[2] = computeC02(u,v,w) * denominator;
        b[3] = computeC11(u,v,w) * denominator;
        b[4] = computeC12(u,v,w) * denominator;
        b[5] = computeC22(u,v,w) * denominator;
      }

      constexpr Real getB00() const { return b[0]; }
      constexpr Real getB01() const { return b[1]; }
      constexpr Real getB02() const { return b[2]; }
      constexpr Real getB11() const { return b[3]; }
      constexpr Real getB12() const { return b[4]; }
      constexpr Real getB22() const { return b[5]; }
    };

    template <typename mat>
    __device__ __host__ void accumBothDerivatives(mat &result, const mat &left, const mat &right, const mat &outer_prod)
    {
      auto temp = (2.0*getTrace(left*outer_prod)).real();
      for (int k=0; k<3; ++k) {
	for (int l=0; l<3; ++l) {
          result(k,l) += temp*right(k,l);
	}
      }
    }

    template <class mat>
    __device__ __host__ void accumDerivatives(mat &result, const mat &left, const mat &right, const mat &outer_prod)
    {
      auto temp = getTrace(left*outer_prod);
      for(int k=0; k<3; ++k){
	for(int l=0; l<3; ++l){
	  result(k,l) = temp*right(k,l);
	}
      }
    }

    template<class T> constexpr T getAbsMin(const T* const array, int size)
    {
      T min = fabs(array[0]);
      for (int i=1; i<size; ++i) {
        T abs_val = fabs(array[i]);
        if ((abs_val) < min){ min = abs_val; }
      }
      return min;
    }

    template<class Real> constexpr bool checkAbsoluteError(Real a, Real b, Real epsilon) { return fabs(a-b) < epsilon; }
    template<class Real> constexpr bool checkRelativeError(Real a, Real b, Real epsilon) { return fabs((a-b)/b) < epsilon; }

    // Compute the reciprocal square root of the matrix q
    // Also modify q if the eigenvalues are dangerously small.
    template <typename Link, typename coeff_t, typename Arg>
    __device__  __host__  void reciprocalRoot(Link &res, coeff_t &deriv_coeffs, typename Arg::real f[3], Link &q, const Arg &arg)
    {
      using real = typename Arg::real;
      Link qsq, tempq;

      real c[3] = { };
      real g[3] = { };

      if (!arg.svd_only) {
	qsq = q*q;
	tempq = qsq*q;

	c[0] = getTrace(q).real();
	c[1] = getTrace(qsq).real() / static_cast<real>(2.0);
	c[2] = getTrace(tempq).real() / static_cast<real>(3.0);

	g[0] = g[1] = g[2] = c[0] / static_cast<real>(3.0);
	real theta;
	real s = c[1]/3. - c[0] * c[0] / static_cast<real>(18.0);
	real r = c[2]/2. - (c[0] / static_cast<real>(3.0)) * (c[1] - c[0] * c[0] / static_cast<real>(9.0));

	real cosTheta = r * quda::rsqrt(s * s * s);
	if (fabs(s) < arg.unitarize_eps) {
	  cosTheta = static_cast<real>(1.0);
	  s = static_cast<real>(0.0);
	}
	if (fabs(cosTheta) > static_cast<real>(1.0)) {
          if (r > static_cast<real>(0.0)) theta = static_cast<real>(0.0);
          else theta = static_cast<real>(M_PI)/static_cast<real>(3.0);
        } else {
          theta = acos(cosTheta) / static_cast<real>(3.0);
        }

	s = 2.0 * sqrt(s);
	for (int i=0; i<3; ++i) {
	  g[i] += s * cos(theta + (i-1) * static_cast<real>(M_PI * 2.0 / 3.0));
	}

      } // !REUNIT_SVD_ONLY?

	//
	// Compare the product of the eigenvalues computed thus far to the
	// absolute value of the determinant.
	// If the determinant is very small or the relative error is greater than some predefined value
	// then recompute the eigenvalues using a singular-value decomposition.
	// Note that this particular calculation contains multiple branches,
	// so it doesn't appear to be particularly well-suited to the GPU
	// programming model. However, the analytic calculation of the
	// unitarization is extremely fast, and if the SVD routine is not called
	// too often, we expect pretty good performance.
	//

      if (arg.allow_svd) {
	bool perform_svd = true;
	if (!arg.svd_only) {
	  const real det = getDeterminant(q).real();
	  if( fabs(det) >= arg.svd_abs_error) {
	    if( checkRelativeError(g[0]*g[1]*g[2], det, arg.svd_rel_error) ) perform_svd = false;
	  }
	}

	if (perform_svd) {
	  Link tmp2;
	  // compute the eigenvalues using the singular value decomposition
	  computeSVD<real>(q, tempq, tmp2, g);
	  // The array g contains the eigenvalues of the matrix q
	  // The determinant is the product of the eigenvalues, and I can use this
	  // to check the SVD
	  const real determinant = getDeterminant(q).real();
	  const real gprod = g[0] * g[1] * g[2];
	  // Check the svd result for errors
	  if (fabs(gprod - determinant) > arg.max_det_error) {
	    //printf("Warning: Error in determinant computed by SVD : %g > %g\n", fabs(gprod-determinant), arg.max_det_error);
	    //printLink(q);

            atomic_fetch_add(arg.fails, 1);
	  }
	} // perform_svd?

      } // REUNIT_ALLOW_SVD?

      real delta = getAbsMin(g,3);
      if (delta < arg.force_filter) {
	for (int i=0; i<3; ++i) {
	  g[i]     += arg.force_filter;
	  q(i,i).x += arg.force_filter;
	}
	qsq = q*q; // recalculate Q^2
      }


      // At this point we have finished with the c's
      // use these to store sqrt(g)
      for (int i=0; i<3; ++i) c[i] = sqrt(g[i]);

      // done with the g's, use these to store u, v, w
      g[0] = c[0]+c[1]+c[2];
      g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
      g[2] = c[0]*c[1]*c[2];

      // set the derivative coefficients!
      deriv_coeffs.set(g[0], g[1], g[2]);

      const real denominator  = g[2] * (g[0] * g[1] - g[2]);
      c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1])) / denominator;
      c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1]) / denominator;
      c[2] = g[0] / denominator;

      tempq = c[1] * q + c[2] * qsq;
      // Add a real scalar
      tempq(0,0).x += c[0];
      tempq(1,1).x += c[0];
      tempq(2,2).x += c[0];

      f[0] = c[0];
      f[1] = c[1];
      f[2] = c[2];

      res = tempq;
    }

    // "v" denotes a "fattened" link variable
    template <typename Link, typename Arg>
    __device__ __host__ void getUnitarizeForceSite(Link &result, const Link &v, const Link &outer_prod, const Arg &arg)
    {
      using real = typename Arg::real;
      real f[3];
      real b[6];

      Link v_dagger = conj(v);  // okay!
      Link q   = v_dagger*v;    // okay!
      Link rsqrt_q;

      DerivativeCoefficients<real> deriv_coeffs;

      reciprocalRoot(rsqrt_q, deriv_coeffs, f, q, arg); // approx 529 flops (assumes no SVD)

      // Pure hack here
      b[0] = deriv_coeffs.getB00();
      b[1] = deriv_coeffs.getB01();
      b[2] = deriv_coeffs.getB02();
      b[3] = deriv_coeffs.getB11();
      b[4] = deriv_coeffs.getB12();
      b[5] = deriv_coeffs.getB22();

      result = rsqrt_q*outer_prod;

      // We are now finished with rsqrt_q
      Link qv_dagger  = q*v_dagger;
      Link vv_dagger  = v*v_dagger;
      Link vqv_dagger = v*qv_dagger;
      Link temp = f[1]*vv_dagger + f[2]*vqv_dagger;

      temp = f[1]*v_dagger + f[2]*qv_dagger;
      Link conj_outer_prod = conj(outer_prod);

      temp = f[1]*v + f[2]*v*q;
      result = result + outer_prod*temp*v_dagger + f[2]*q*outer_prod*vv_dagger;
      result = result + v_dagger*conj_outer_prod*conj(temp) + f[2]*qv_dagger*conj_outer_prod*v_dagger;

      Link qsqv_dagger = q*qv_dagger;
      Link pv_dagger   = b[0]*v_dagger + b[1]*qv_dagger + b[2]*qsqv_dagger;
      accumBothDerivatives(result, v, pv_dagger, outer_prod); // 41 flops

      Link rv_dagger = b[1]*v_dagger + b[3]*qv_dagger + b[4]*qsqv_dagger;
      Link vq = v*q;
      accumBothDerivatives(result, vq, rv_dagger, outer_prod); // 41 flops

      Link sv_dagger = b[2]*v_dagger + b[4]*qv_dagger + b[5]*qsqv_dagger;
      Link vqsq = vq*q;
      accumBothDerivatives(result, vqsq, sv_dagger, outer_prod); // 41 flops

      // 4528 flops - 17 matrix multiplies (198 flops each) + reciprocal root (approx 529 flops) + accumBothDerivatives (41 each) + miscellaneous
    } // get unit force term

    template <typename Arg> struct UnitarizeForce
    {
      const Arg &arg;
      constexpr UnitarizeForce(const Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __host__ void operator()(int x_cb, int parity)
      {
        using real = typename Arg::real;

        // This part of the calculation is always done in double precision
        Matrix<complex<real>, 3> v, result, oprod;

        for (int dir=0; dir<4; ++dir) {
          oprod = arg.force_old(dir, x_cb, parity);
          v = arg.u(dir, x_cb, parity);

          getUnitarizeForceSite(result, v, oprod, arg);

          arg.force(dir, x_cb, parity) = result;
        } // 4*4528 flops per site
      } // getUnitarizeForceField
    };

  }
}
