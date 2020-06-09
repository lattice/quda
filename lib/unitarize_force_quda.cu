#include <cstdlib>
#include <cstdio>

#include <gauge_field.h>
#include <tune_quda.h>
#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <instantiate.h>
#include <color_spinor.h>

namespace quda {

  namespace { // anonymous
#include <svd_quda.h>
  }

#define HISQ_UNITARIZE_PI 3.14159265358979323846
#define HISQ_UNITARIZE_PI23 HISQ_UNITARIZE_PI*2.0/3.0

  static double unitarize_eps;
  static double force_filter;
  static double max_det_error;
  static bool   allow_svd;
  static bool   svd_only;
  static double svd_rel_error;
  static double svd_abs_error;

  namespace fermion_force {

    template <typename Float_, int nColor_, QudaReconstructType recon_, QudaGaugeFieldOrder order = QUDA_NATIVE_GAUGE_ORDER>
    struct UnitarizeForceArg {
      using Float = Float_;
      static constexpr int nColor = nColor_;
      static constexpr QudaReconstructType recon = recon_;
      // use long form here to allow specification of order
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO,2*nColor*nColor,QUDA_STAGGERED_PHASE_NO,gauge::default_huge_alloc, QUDA_GHOST_EXCHANGE_INVALID,false,order>::type F;
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO,2*nColor*nColor,QUDA_STAGGERED_PHASE_NO,gauge::default_huge_alloc, QUDA_GHOST_EXCHANGE_INVALID,false,order>::type G;
      F force;
      const F force_old;
      const G u;
      int *fails;
      int threads;
      const double unitarize_eps;
      const double force_filter;
      const double max_det_error;
      const int allow_svd;
      const int svd_only;
      const double svd_rel_error;
      const double svd_abs_error;

      UnitarizeForceArg(GaugeField &force, const GaugeField &force_old, const GaugeField &u, int *fails,
			double unitarize_eps, double force_filter, double max_det_error, int allow_svd,
			int svd_only, double svd_rel_error, double svd_abs_error) :
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
        svd_abs_error(svd_abs_error),
        threads(u.VolumeCB()) { }
    };

    void setUnitarizeForceConstants(double unitarize_eps_, double force_filter_,
				    double max_det_error_, bool allow_svd_, bool svd_only_,
				    double svd_rel_error_, double svd_abs_error_)
    {
      unitarize_eps = unitarize_eps_;
      force_filter = force_filter_;
      max_det_error = max_det_error_;
      allow_svd = allow_svd_;
      svd_only = svd_only_;
      svd_rel_error = svd_rel_error_;
      svd_abs_error = svd_abs_error_;
    }

    template <class Real> class DerivativeCoefficients {
      Real b[6];
      constexpr Real computeC00(const Real &u, const Real &v, const Real &w)
      {
        return -pow(w,3) * pow(u,6) + 3*v*pow(w,3)*pow(u,4) + 3*pow(v,4)*w*pow(u,4)
          - pow(v,6)*pow(u,3) - 4*pow(w,4)*pow(u,3) - 12*pow(v,3)*pow(w,2)*pow(u,3)
          + 16*pow(v,2)*pow(w,3)*pow(u,2) + 3*pow(v,5)*w*pow(u,2) - 8*v*pow(w,4)*u
          - 3*pow(v,4)*pow(w,2)*u + pow(w,5) + pow(v,3)*pow(w,3);
      }

      constexpr Real computeC01(const Real & u, const Real & v, const Real & w)
      {
        return -pow(w,2)*pow(u,7) - pow(v,2)*w*pow(u,6) + pow(v,4)*pow(u,5) + 6*v*pow(w,2)*pow(u,5)
          - 5*pow(w,3)*pow(u,4) - pow(v,3)*w*pow(u,4)- 2*pow(v,5)*pow(u,3) - 6*pow(v,2)*pow(w,2)*pow(u,3)
          + 10*v*pow(w,3)*pow(u,2) + 6*pow(v,4)*w*pow(u,2) - 3*pow(w,4)*u - 6*pow(v,3)*pow(w,2)*u + 2*pow(v,2)*pow(w,3);
      }

      constexpr Real computeC02(const Real & u, const Real & v, const Real & w)
      {
        return pow(w,2)*pow(u,5) + pow(v,2)*w*pow(u,4)- pow(v,4)*pow(u,3)- 4*v*pow(w,2)*pow(u,3)
          + 4*pow(w,3)*pow(u,2) + 3*pow(v,3)*w*pow(u,2) - 3*pow(v,2)*pow(w,2)*u + v*pow(w,3);
      }

      constexpr Real computeC11(const Real & u, const Real & v, const Real & w)
      {
        return -w*pow(u,8) - pow(v,2)*pow(u,7) + 7*v*w*pow(u,6) + 4*pow(v,3)*pow(u,5)
          - 5*pow(w,2)*pow(u,5) - 16*pow(v,2)*w*pow(u,4) - 4*pow(v,4)*pow(u,3) + 16*v*pow(w,2)*pow(u,3)
          - 3*pow(w,3)*pow(u,2) + 12*pow(v,3)*w*pow(u,2) - 12*pow(v,2)*pow(w,2)*u + 3*v*pow(w,3);
      }

      constexpr Real computeC12(const Real &u, const Real &v, const Real &w)
      {
        return w*pow(u,6) + pow(v,2)*pow(u,5) - 5*v*w*pow(u,4) - 2*pow(v,3)*pow(u,3)
          + 4*pow(w,2)*pow(u,3) + 6*pow(v,2)*w*pow(u,2) - 6*v*pow(w,2)*u + pow(w,3);
      }

      constexpr Real computeC22(const Real &u, const Real &v, const Real &w)
      {
        return -w*pow(u,4) - pow(v,2)*pow(u,3) + 3*v*w*pow(u,2) - 3*pow(w,2)*u;
      }

    public:
      constexpr void set(const Real &u, const Real &v, const Real &w)
      {
        const Real denominator = 1.0 / (2.0*pow(w*(u*v-w),3));
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
    template<class Float, typename Arg>
    __device__  __host__  void reciprocalRoot(Matrix<complex<Float>,3>* res, DerivativeCoefficients<Float>* deriv_coeffs,
                                              Float f[3], Matrix<complex<Float>,3> & q, Arg &arg)
    {
      Matrix<complex<Float>,3> qsq, tempq;

      Float c[3];
      Float g[3];

      if(!arg.svd_only){
	qsq = q*q;
	tempq = qsq*q;

	c[0] = getTrace(q).x;
	c[1] = getTrace(qsq).x/2.0;
	c[2] = getTrace(tempq).x/3.0;

	g[0] = g[1] = g[2] = c[0]/3.;
	Float r,s,theta;
	s = c[1]/3. - c[0]*c[0]/18;
	r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);

	Float cosTheta = r*rsqrt(s*s*s);
	if (fabs(s) < arg.unitarize_eps) {
	  cosTheta = 1.;
	  s = 0.0;
	}
	if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=HISQ_UNITARIZE_PI/3.0; }
	else{ theta = acos(cosTheta)/3.0; }

	s = 2.0*sqrt(s);
	for (int i=0; i<3; ++i) {
	  g[i] += s*cos(theta + (i-1)*HISQ_UNITARIZE_PI23);
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
	  const Float det = getDeterminant(q).x;
	  if( fabs(det) >= arg.svd_abs_error) {
	    if( checkRelativeError(g[0]*g[1]*g[2],det,arg.svd_rel_error) ) perform_svd = false;
	  }
	}

	if(perform_svd){
	  Matrix<complex<Float>,3> tmp2;
	  // compute the eigenvalues using the singular value decomposition
	  computeSVD<Float>(q,tempq,tmp2,g);
	  // The array g contains the eigenvalues of the matrix q
	  // The determinant is the product of the eigenvalues, and I can use this
	  // to check the SVD
	  const Float determinant = getDeterminant(q).x;
	  const Float gprod = g[0]*g[1]*g[2];
	  // Check the svd result for errors
	  if (fabs(gprod - determinant) > arg.max_det_error) {
	    printf("Warning: Error in determinant computed by SVD : %g > %g\n", fabs(gprod-determinant), arg.max_det_error);
	    printLink(q);

#ifdef __CUDA_ARCH__
	    atomicAdd(arg.fails, 1);
#else
	    (*arg.fails)++;
#endif
	  }
	} // perform_svd?

      } // REUNIT_ALLOW_SVD?

      Float delta = getAbsMin(g,3);
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
      deriv_coeffs->set(g[0], g[1], g[2]);

      const Float& denominator  = g[2]*(g[0]*g[1]-g[2]);
      c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
      c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
      c[2] = g[0]/denominator;

      tempq = c[1]*q + c[2]*qsq;
      // Add a real scalar
      tempq(0,0).x += c[0];
      tempq(1,1).x += c[0];
      tempq(2,2).x += c[0];

      f[0] = c[0];
      f[1] = c[1];
      f[2] = c[2];

      *res = tempq;
    }

    // "v" denotes a "fattened" link variable
    template <class Float, typename Arg>
    __device__ __host__ void getUnitarizeForceSite(Matrix<complex<Float>,3>& result, const Matrix<complex<Float>,3> & v,
                                                   const Matrix<complex<Float>,3> & outer_prod, Arg &arg)
    {
      typedef Matrix<complex<Float>,3> Link;
      Float f[3];
      Float b[6];

      Link v_dagger = conj(v);  // okay!
      Link q   = v_dagger*v;    // okay!
      Link rsqrt_q;

      DerivativeCoefficients<Float> deriv_coeffs;

      reciprocalRoot<Float>(&rsqrt_q, &deriv_coeffs, f, q, arg); // approx 529 flops (assumes no SVD)

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

    template <typename Arg> __global__ void getUnitarizeForceField(Arg arg)
    {
      using real = typename Arg::Float;
      int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
      if (x_cb >= arg.threads) return;
      int parity = blockIdx.y*blockDim.y + threadIdx.y;

      // This part of the calculation is always done in double precision
      Matrix<complex<double>,3> v, result, oprod;
      Matrix<complex<real>,3> v_tmp, result_tmp, oprod_tmp;

      for (int dir=0; dir<4; ++dir) {
	oprod_tmp = arg.force_old(dir, x_cb, parity);
	v_tmp = arg.u(dir, x_cb, parity);
	v = v_tmp;
	oprod = oprod_tmp;

	getUnitarizeForceSite<double>(result, v, oprod, arg);
	result_tmp = result;

	arg.force(dir, x_cb, parity) = result_tmp;
      } // 4*4528 flops per site
    } // getUnitarizeForceField

    template <typename Float, int nColor, QudaReconstructType recon> class UnitarizeForce : public TunableVectorY {
      UnitarizeForceArg<Float, nColor, recon> arg;
      const GaugeField &meta;

      // don't tune the grid dimension
      bool tuneGridDim() const { return false; }
      unsigned int minThreads() const { return arg.threads; }

    public:
      UnitarizeForce(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u, int* fails) :
        TunableVectorY(2),
        arg(newForce, oldForce, u, fails, unitarize_eps, force_filter,
            max_det_error, allow_svd, svd_only, svd_rel_error, svd_abs_error),
        meta(u)
      {
        apply(0);
        qudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
        checkCudaError();
      }

      void apply(const qudaStream_t &stream) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	getUnitarizeForceField<<<tp.grid,tp.block>>>(arg);
      }

      void preTune() { ; }
      void postTune() { cudaMemset(arg.fails, 0, sizeof(int)); } // reset fails counter

      long long flops() const { return 4ll*4528*meta.Volume(); }
      long long bytes() const { return 4ll * meta.Volume() * (arg.force.Bytes() + arg.force_old.Bytes() + arg.u.Bytes()); }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    }; // UnitarizeForce

    void unitarizeForce(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u,
			int* fails)
    {
#ifdef GPU_HISQ_FORCE
      checkReconstruct(u, oldForce, newForce);
      checkPrecision(u, oldForce, newForce);

      if (!u.isNative() || !oldForce.isNative() || !newForce.isNative())
        errorQuda("Only native order supported");

      instantiate<UnitarizeForce,ReconstructNone>(newForce, oldForce, u, fails);
#else
      errorQuda("HISQ force has not been built");
#endif
    }

    template <typename Float, typename Arg> void unitarizeForceCPU(Arg &arg)
    {
#ifdef GPU_HISQ_FORCE
      Matrix<complex<double>, 3> v, result, oprod;
      Matrix<complex<Float>, 3> v_tmp, result_tmp, oprod_tmp;

      for (int parity = 0; parity < 2; parity++) {
        for (int i = 0; i < arg.threads; i++) {
          for (int dir = 0; dir < 4; dir++) {
            oprod_tmp = arg.force_old(dir, i, parity);
            v_tmp = arg.u(dir, i, parity);
            v = v_tmp;
            oprod = oprod_tmp;

            getUnitarizeForceSite<double>(result, v, oprod, arg);

            result_tmp = result;
            arg.force(dir, i, parity) = result_tmp;
          }
        }
      }
#else
      errorQuda("HISQ force has not been built");
#endif
    }

    void unitarizeForceCPU(GaugeField &newForce, const GaugeField &oldForce, const GaugeField &u)
    {
      if (checkLocation(newForce, oldForce, u) != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
      int num_failures = 0;
      constexpr int nColor = 3;
      Matrix<complex<double>, nColor> old_force, new_force, v;
      if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
        if (u.Precision() == QUDA_DOUBLE_PRECISION) {
          UnitarizeForceArg<double, nColor, QUDA_RECONSTRUCT_NO, QUDA_MILC_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<double>(arg);
        } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
          UnitarizeForceArg<float, nColor, QUDA_RECONSTRUCT_NO, QUDA_MILC_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<float>(arg);
        } else {
          errorQuda("Precision = %d not supported", u.Precision());
        }
      } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
        if (u.Precision() == QUDA_DOUBLE_PRECISION) {
          UnitarizeForceArg<double, nColor, QUDA_RECONSTRUCT_NO, QUDA_QDP_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<double>(arg);
        } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
          UnitarizeForceArg<float, nColor, QUDA_RECONSTRUCT_NO, QUDA_QDP_GAUGE_ORDER> arg(
            newForce, oldForce, u, &num_failures, unitarize_eps, force_filter, max_det_error, allow_svd, svd_only,
            svd_rel_error, svd_abs_error);
          unitarizeForceCPU<float>(arg);
        } else {
          errorQuda("Precision = %d not supported", u.Precision());
        }
      } else {
        errorQuda("Only MILC and QDP gauge orders supported\n");
      }
      if (num_failures) errorQuda("Unitarization failed, failures = %d", num_failures);
    } // unitarize_force_cpu

  } // namespace fermion_force

} // namespace quda
