#include <cstdlib>
#include <cstdio>

#include <gauge_field.h>
#include <gauge_field_order.h>
#include <tune_quda.h>
#include <quda_matrix.h>
#include <unitarization_links.h>
#include <su3_project.cuh>
#include <index_helper.cuh>
#include <instantiate.h>
#include <color_spinor.h>

namespace quda {

namespace {
#include <svd_quda.h>
}

#ifndef FL_UNITARIZE_PI
#define FL_UNITARIZE_PI 3.14159265358979323846
#endif
#ifndef FL_UNITARIZE_PI23
#define FL_UNITARIZE_PI23 FL_UNITARIZE_PI*0.66666666666666666666
#endif

    
  // supress compiler warnings about unused variables when GPU_UNITARIZE is not set
  // when we switch to C++17 consider [[maybe_unused]]
  __attribute__((unused)) static const int max_iter_newton = 20;
  __attribute__((unused))static const int max_iter = 20;

  __attribute__((unused)) static double unitarize_eps = 1e-14;
  __attribute__((unused)) static double max_error = 1e-10;
  __attribute__((unused)) static int reunit_allow_svd = 1;
  __attribute__((unused)) static int reunit_svd_only  = 0;
  __attribute__((unused)) static double svd_rel_error = 1e-6;
  __attribute__((unused)) static double svd_abs_error = 1e-6;

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct UnitarizeLinksArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    Gauge out;
    const Gauge in;

    int threads; // number of active threads required
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

    UnitarizeLinksArg(GaugeField &out, const GaugeField &in, int* fails, int max_iter,
                      double unitarize_eps, double max_error, int reunit_allow_svd,
                      int reunit_svd_only, double svd_rel_error, double svd_abs_error) :
      out(out),
      in(in),
      threads(in.VolumeCB()),
      fails(fails),
      unitarize_eps(unitarize_eps),
      max_iter(max_iter),
      max_error(max_error),
      reunit_allow_svd(reunit_allow_svd),
      reunit_svd_only(reunit_svd_only),
      svd_rel_error(svd_rel_error),
      svd_abs_error(svd_abs_error)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = in.X()[dir];
    }
  };

  void setUnitarizeLinksConstants(double unitarize_eps_, double max_error_,
				  bool reunit_allow_svd_, bool reunit_svd_only_,
				  double svd_rel_error_, double svd_abs_error_) {
    unitarize_eps = unitarize_eps_;
    max_error = max_error_;
    reunit_allow_svd = reunit_allow_svd_;
    reunit_svd_only = reunit_svd_only_;
    svd_rel_error = svd_rel_error_;
    svd_abs_error = svd_abs_error_;
  }

  template <typename mat>
  __device__ __host__ bool isUnitarizedLinkConsistent(const mat &initial_matrix,
                                                      const mat &unitary_matrix, double max_error)
  {
    auto n = initial_matrix.size();
    mat temporary = conj(initial_matrix)*unitary_matrix;
    temporary = temporary*temporary - conj(initial_matrix)*initial_matrix;

    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
	if (fabs(temporary(i,j).x) > max_error || fabs(temporary(i,j).y) > max_error) {
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
  __device__  __host__ bool reciprocalRoot(mat &res, const mat& q, Arg &arg)
  {
    mat qsq, tempq;

    real c[3];
    real g[3];

    const real one_third = 0.333333333333333333333;
    const real one_ninth = 0.111111111111111111111;
    const real one_eighteenth = 0.055555555555555555555;

    qsq = q*q;
    tempq = qsq*q;

    c[0] = getTrace(q).x;
    c[1] = getTrace(qsq).x * 0.5;
    c[2] = getTrace(tempq).x * one_third;;

    g[0] = g[1] = g[2] = c[0] * one_third;
    real r,s,theta;
    s = c[1]*one_third - c[0]*c[0]*one_eighteenth;

    real cosTheta;
    if (fabs(s) >= arg.unitarize_eps) { // faster when this conditional is removed?
      const real rsqrt_s = rsqrt(s);
      r = c[2]*0.5 - (c[0]*one_third)*(c[1] - c[0]*c[0]*one_ninth);
      cosTheta = r*rsqrt_s*rsqrt_s*rsqrt_s;

      if(fabs(cosTheta) >= 1.0){
	theta = (r > 0) ? 0.0 : FL_UNITARIZE_PI;
      }else{
	theta = acos(cosTheta); // this is the primary performance limiter
      }

      const real sqrt_s = s*rsqrt_s;

#if 0 // experimental version
      real as, ac;
      sincos( theta*one_third, &as, &ac );
      g[0] = c[0]*one_third + 2*sqrt_s*ac;
      //g[1] = c[0]*one_third + 2*sqrt_s*(ac*cos(1*FL_UNITARIZE_PI23) - as*sin(1*FL_UNITARIZE_PI23));
      g[1] = c[0]*one_third - 2*sqrt_s*(0.5*ac + as*0.8660254037844386467637);
      //g[2] = c[0]*one_third + 2*sqrt_s*(ac*cos(2*FL_UNITARIZE_PI23) - as*sin(2*FL_UNITARIZE_PI23));
      g[2] = c[0]*one_third + 2*sqrt_s*(-0.5*ac + as*0.8660254037844386467637);
#else
      g[0] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third );
      g[1] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third + FL_UNITARIZE_PI23 );
      g[2] = c[0]*one_third + 2*sqrt_s*cos( theta*one_third + 2*FL_UNITARIZE_PI23 );
#endif
    }

    // Check the eigenvalues, if the determinant does not match the product of the eigenvalues
    // return false. Then call SVD instead.
    real det = getDeterminant(q).x;
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
  __host__ __device__ bool unitarizeLinkMILC(mat &out, const mat &in, Arg &arg)
  {
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
  } // unitarizeMILC

  template <typename mat>
  __host__ __device__ bool unitarizeLinkNewton(mat &out, const mat& in, int max_iter)
  {
    mat u = in;

    for (int i=0; i<max_iter; ++i) {
      mat uinv = inverse(u);
      u = 0.5*(u + conj(uinv));
    }

    if (isUnitarizedLinkConsistent(in,u,0.0000001)==false) {
      printf("ERROR: Unitarized link is not consistent with incoming link\n");
      return false;
    }
    out = u;

    return true;
  }

  void unitarizeLinksCPU(GaugeField &outfield, const GaugeField& infield)
  {
#ifdef GPU_UNITARIZE
    if (checkLocation(outfield, infield) != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
    checkPrecision(outfield, infield);

    int num_failures = 0;
    Matrix<complex<double>,3> inlink, outlink;

    for (unsigned int i = 0; i < infield.Volume(); ++i) {
      for (int dir=0; dir<4; ++dir){
	if (infield.Precision() == QUDA_SINGLE_PRECISION) {
	  copyArrayToLink(&inlink, ((float*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if (unitarizeLinkNewton(outlink, inlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((float*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink);
	} else if (infield.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyArrayToLink(&inlink, ((double*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if (unitarizeLinkNewton(outlink, inlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((double*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink);
	} // precision?
      } // dir
    }   // loop over volume
#else
    errorQuda("Unitarization has not been built");
#endif
  }

  // CPU function which checks that the gauge field is unitary
  bool isUnitary(const GaugeField& field, double max_error)
  {
#ifdef GPU_UNITARIZE
    if (field.Location() != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
    Matrix<complex<double>,3> link, identity;

    for (unsigned int i = 0; i < field.Volume(); ++i) {
      for (int dir=0; dir<4; ++dir) {
	if (field.Precision() == QUDA_SINGLE_PRECISION) {
	  copyArrayToLink(&link, ((float*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	} else if (field.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyArrayToLink(&link, ((double*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	} else {
	  errorQuda("Unsupported precision\n");
	}
	if (link.isUnitary(max_error) == false) {
	  printf("Unitarity failure\n");
	  printf("site index = %u,\t direction = %d\n", i, dir);
	  printLink(link);
	  identity = conj(link)*link;
	  printLink(identity);
	  return false;
	}
      } // dir
    }   // i
    return true;
#else
    errorQuda("Unitarization has not been built");
    return false;
#endif
  } // is unitary


  template <typename Arg> __global__ void DoUnitarizedLink(Arg arg)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int mu = threadIdx.z + blockIdx.z*blockDim.z;
    if (idx >= arg.threads) return;
    if (mu >= 4) return;

    // result is always in double precision
    Matrix<complex<double>,Arg::nColor> v, result;
    Matrix<complex<typename Arg::Float>,Arg::nColor> tmp = arg.in(mu, idx, parity);

    v = tmp;
    unitarizeLinkMILC<double>(result, v, arg);
    if (arg.check_unitarization) {
      if (result.isUnitary(arg.max_error) == false) atomicAdd(arg.fails, 1);
    }
    tmp = result;

    arg.out(mu, idx, parity) = tmp;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class UnitarizeLinks : TunableVectorYZ {
    UnitarizeLinksArg<Float, nColor, recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    UnitarizeLinks(GaugeField &out, const GaugeField &in, int* fails) :
      TunableVectorYZ(2,4),
      arg(out, in, fails, max_iter, unitarize_eps, max_error, reunit_allow_svd,
          reunit_svd_only, svd_rel_error, svd_abs_error),
      meta(in)
    {
      apply(0);
      qudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
      checkCudaError();
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DoUnitarizedLink<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    void preTune() { if (arg.in.gauge == arg.out.gauge) arg.out.save(); }
    void postTune() {
      if (arg.in.gauge == arg.out.gauge) arg.out.load();
      cudaMemset(arg.fails, 0, sizeof(int)); // reset fails counter
    }

    long long flops() const {
      // Accounted only the minimum flops for the case reunitarize_svd_only=0
      return 4ll * 2 * arg.threads * 1147;
    }
    long long bytes() const { return 4ll * 2 * arg.threads * (arg.in.Bytes() + arg.out.Bytes()); }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
  };

  void unitarizeLinks(GaugeField& out, const GaugeField &in, int* fails)
  {
#ifdef GPU_UNITARIZE
    checkPrecision(out, in);
    instantiate<UnitarizeLinks, ReconstructWilson>(out, in, fails);
#else
    errorQuda("Unitarization has not been built");
#endif
  }

  void unitarizeLinks(GaugeField &links, int* fails) { unitarizeLinks(links, links, fails); }

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct ProjectSU3Arg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    Gauge u;

    int threads; // number of active threads required
    Float tol;
    int *fails;
    ProjectSU3Arg(GaugeField &u, Float tol, int *fails) :
      threads(u.VolumeCB()),
      u(u),
      tol(tol),
      fails(fails) { }
  };

  template<typename Arg>
  __global__ void ProjectSU3kernel(Arg arg){
    using real = typename Arg::Float;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int mu = threadIdx.z + blockIdx.z*blockDim.z;
    if (idx >= arg.threads) return;
    if (mu >= 4) return;

    Matrix<complex<real>, Arg::nColor> u = arg.u(mu, idx, parity);

    polarSu3<real>(u, arg.tol);

    // count number of failures
    if (u.isUnitary(arg.tol) == false) {
      atomicAdd(arg.fails, 1);
    }

    arg.u(mu, idx, parity) = u;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class ProjectSU3 : TunableVectorYZ {
    ProjectSU3Arg<Float, nColor, recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    ProjectSU3(GaugeField &u, double tol, int *fails) :
      arg(u, static_cast<Float>(tol), fails),
      TunableVectorYZ(2, 4),
      meta(u)
    {
      apply(0);
      qudaDeviceSynchronize();
      checkCudaError();
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ProjectSU3kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    void preTune() { arg.u.save(); }
    void postTune() {
      arg.u.load();
      cudaMemset(arg.fails, 0, sizeof(int)); // reset fails counter
    }

    long long flops() const { return 0; } // depends on number of iterations
    long long bytes() const { return 4ll * 2 * arg.threads * 2 * arg.u.Bytes(); }
    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
  };

  void projectSU3(GaugeField &u, double tol, int *fails) {
#ifdef GPU_GAUGE_TOOLS
    // check the the field doesn't have staggered phases applied
    if (u.StaggeredPhaseApplied())
      errorQuda("Cannot project gauge field with staggered phases applied");

    instantiate<ProjectSU3, ReconstructWilson>(u, tol, fails);
#else
    errorQuda("Gauge tools have not been built");
#endif
  }

} // namespace quda
