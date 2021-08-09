#include <gauge_field.h>
#include <unitarization_links.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/unitarize_links.cuh>

namespace quda {

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

  void setUnitarizeLinksConstants(double unitarize_eps_, double max_error_,
				  bool reunit_allow_svd_, bool reunit_svd_only_,
				  double svd_rel_error_, double svd_abs_error_)
  {
    unitarize_eps = unitarize_eps_;
    max_error = max_error_;
    reunit_allow_svd = reunit_allow_svd_;
    reunit_svd_only = reunit_svd_only_;
    svd_rel_error = svd_rel_error_;
    svd_abs_error = svd_abs_error_;
  }

#ifdef GPU_UNITARIZE
  template <typename T, int n, class Real>
  void copyArrayToLink(Matrix<T,n> &link, Real* array)
  {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        link(i, j).real(array[(i * n + j) * 2 + 0]);
        link(i, j).imag(array[(i * n + j) * 2 + 1]);
      }
    }
  }

  template <typename T, int n, class Real>
  void copyLinkToArray(Real* array, const Matrix<T, n> &link)
  {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        array[(i * n + j) * 2 + 0] = link(i, j).real();
        array[(i * n + j) * 2 + 1] = link(i, j).imag();
      }
    }
  }

  void unitarizeLinksCPU(GaugeField &outfield, const GaugeField& infield)
  {
    if (checkLocation(outfield, infield) != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
    checkPrecision(outfield, infield);

    int num_failures = 0;
    Matrix<complex<double>,3> inlink, outlink;

    for (unsigned int i = 0; i < infield.Volume(); ++i) {
      for (int dir=0; dir<4; ++dir){
	if (infield.Precision() == QUDA_SINGLE_PRECISION) {
	  copyArrayToLink(inlink, ((float*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if (unitarizeLinkNewton(outlink, inlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((float*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink);
	} else if (infield.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyArrayToLink(inlink, ((double*)(infield.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	  if (unitarizeLinkNewton(outlink, inlink, max_iter_newton) == false ) num_failures++;
	  copyLinkToArray(((double*)(outfield.Gauge_p()) + (i*4 + dir)*18), outlink);
	} // precision?
      } // dir
    }   // loop over volume
  }
#else
  void unitarizeLinksCPU(GaugeField &, const GaugeField &)
  {
    errorQuda("Unitarization has not been built");
  }
#endif

  // CPU function which checks that the gauge field is unitary
#ifdef GPU_UNITARIZE
  bool isUnitary(const GaugeField& field, double max_error)
  {
    if (field.Location() != QUDA_CPU_FIELD_LOCATION) errorQuda("Location must be CPU");
    Matrix<complex<double>,3> link, identity;

    for (unsigned int i = 0; i < field.Volume(); ++i) {
      for (int dir=0; dir<4; ++dir) {
	if (field.Precision() == QUDA_SINGLE_PRECISION) {
	  copyArrayToLink(link, ((float*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
	} else if (field.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyArrayToLink(link, ((double*)(field.Gauge_p()) + (i*4 + dir)*18)); // order of arguments?
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
  }
#else
  bool isUnitary(const GaugeField &, double)
  {
    errorQuda("Unitarization has not been built");
    return false;
  }
#endif

  template <typename Float, int nColor, QudaReconstructType recon>
  class UnitarizeLinks : TunableKernel3D {
    GaugeField &out;
    const GaugeField &in;
    int *fails;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    UnitarizeLinks(GaugeField &out, const GaugeField &in, int* fails) :
      TunableKernel3D(in, 2, 4),
      out(out),
      in(in),
      fails(fails)
    {
      apply(device::get_default_stream());
      qudaDeviceSynchronize(); // need to synchronize to ensure failure write has completed
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Unitarize>(tp, stream,
                        UnitarizeArg<Float, nColor, recon>(out, in, fails, max_iter, unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error, svd_abs_error));
    }

    void preTune() { if (in.Gauge_p() == out.Gauge_p()) out.backup(); }
    void postTune() {
      if (in.Gauge_p() == out.Gauge_p()) out.restore();
      qudaMemset(fails, 0, sizeof(int)); // reset fails counter
    }

    // Accounted only the minimum flops for the case reunitarize_svd_only=0
    long long flops() const { return 4ll * in.Volume() * 1147; }
    long long bytes() const { return in.Bytes() + out.Bytes(); }
  };

#ifdef GPU_UNITARIZE
  void unitarizeLinks(GaugeField& out, const GaugeField &in, int* fails)
  {
    checkPrecision(out, in);
    instantiate<UnitarizeLinks, ReconstructWilson>(out, in, fails);
  }
#else
  void unitarizeLinks(GaugeField &, const GaugeField &, int*)
  {
    errorQuda("Unitarization has not been built");
  }
#endif

  void unitarizeLinks(GaugeField &links, int* fails) { unitarizeLinks(links, links, fails); }

  template <typename Float, int nColor, QudaReconstructType recon> class ProjectSU3 : TunableKernel3D {
    using real = typename mapper<Float>::type;
    GaugeField &u;
    real tol;
    int *fails;
    unsigned int minThreads() const { return u.VolumeCB(); }

  public:
    ProjectSU3(GaugeField &u, double tol, int *fails) :
      TunableKernel3D(u, 2, 4),
      u(u),
      tol(tol),
      fails(fails)
    {
      apply(device::get_default_stream());
      qudaDeviceSynchronize();
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<Projector>(tp, stream, ProjectSU3Arg<Float, nColor, recon>(u, tol, fails));
    }

    void preTune() { u.backup(); }
    void postTune() {
      u.restore();
      qudaMemset(fails, 0, sizeof(int)); // reset fails counter
    }

    long long flops() const { return 0; } // depends on number of iterations
    long long bytes() const { return 2 * u.Bytes(); }
  };

#ifdef GPU_GAUGE_TOOLS
  void projectSU3(GaugeField &u, double tol, int *fails)
  {
    // check the the field doesn't have staggered phases applied
    if (u.StaggeredPhaseApplied())
      errorQuda("Cannot project gauge field with staggered phases applied");

    instantiate<ProjectSU3, ReconstructWilson>(u, tol, fails);
  }
#else
  void projectSU3(GaugeField &, double, int *)
  {
    errorQuda("Gauge tools have not been built");
  }
#endif

} // namespace quda
