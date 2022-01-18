#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <linalg.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>

namespace quda
{
  template <typename Float, int nColor_, QudaReconstructType recon_>
  struct GaugeFundamentalArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;
    typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GaugeFun;
    
    const Gauge in;
    GaugeFun out;
    
    int_fastdiv X[4];    // grid dimensions
    int border[4];
    int_fastdiv E[4];
    const double qr_tol;
    const int qr_max_iter;
    const int taylor_N;
    
    GaugeFundamentalArg(const GaugeField &in, GaugeField &out, const double qr_tol, const int qr_max_iter, const int taylor_N) :
      kernel_param(dim3(in.VolumeCB(), 2, 4)),
      in(in),
      out(out),
      qr_tol(qr_tol),
      qr_max_iter(qr_max_iter),
      taylor_N(taylor_N)      
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        E[dir] = in.X()[dir];
      }
    }
  };

  template <typename Arg> struct Fundamental
  {
    const Arg &arg;
    constexpr Fundamental(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      //Get stacetime and local coords
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr];

      // compute the  eigendecomposition
      Link U = arg.in(dir, linkIndex(x, arg.E), parity);
      linalg::Eigensolve<Matrix, complex<real>, Arg::nColor> eigensolve(U, arg.qr_tol, arg.qr_max_iter);
      
      // Place V S V^{\dag} in the out field
      Link H = eigensolve.exponent();
      
      if(x[0] == 0 && x[1] == 0 && x[2] == 0 && dir == 3) {
	printf("FUNDAMENTALS %d %d %d %d %d: %+.8e %+.8e %+.8e %+.8e %+.8e %+.8e %+.8e %+.8e\n",
	       x[0], x[1], x[2], x[3], dir,
	       H(0,1).real(), H(0,1).imag(),
	       0.5*(H(0,0).real() - H(1,1).real()),
	       H(0,2).real(), H(0,2).imag(),
	       H(1,2).real(), H(1,2).imag(),
	       sqrt(3)*0.5*(H(0,0).real() + H(1,1).real()));
      }

      Link Hout = H;
      expsuNTaylor(Hout, arg.taylor_N);
      arg.out(dir, linkIndex(x, arg.E), parity) = Hout;
      
      if(x_cb < 0) {

	// Eigenvalues of U and the their deviation from the unit circle
	printf("LAMBDA: %d %d %d: (%+.8e,%+.8e) (%+.8e,%+.8e) (%+.8e,%+.8e) %+.8e %+.8e %+.8e\n",
	       x_cb, parity, dir,
	       eigensolve.eval(0).real(), eigensolve.eval(0).imag(),
	       eigensolve.eval(1).real(), eigensolve.eval(1).imag(),
	       eigensolve.eval(2).real(), eigensolve.eval(2).imag(),
	       1.0 - abs(eigensolve.eval(0)), 1.0 - abs(eigensolve.eval(1)), 1.0 - abs(eigensolve.eval(2)));

	// Eigenvalues of H and their sum
	printf("SIGMA %d %d %d: %+.8e %+.8e %+.8e %+.8e\n",
	       x_cb, parity, dir,
	       atan2(eigensolve.eval(0).imag(), eigensolve.eval(0).real()),
	       atan2(eigensolve.eval(1).imag(), eigensolve.eval(1).real()),
	       atan2(eigensolve.eval(2).imag(), eigensolve.eval(2).real()),
	       atan2(eigensolve.eval(0).imag(), eigensolve.eval(0).real()) +
	       atan2(eigensolve.eval(1).imag(), eigensolve.eval(1).real()) +
	       atan2(eigensolve.eval(2).imag(), eigensolve.eval(2).real()));

	// Error of || U * v - \lambda * v ||
	printf("ERROR %d %d %d: %.8e %.8e %.8e\n",
	       x_cb, parity, dir,
	       eigensolve.checkEigendecomposition(0),
	       eigensolve.checkEigendecomposition(1),
	       eigensolve.checkEigendecomposition(2));
	
	// Debug tools
	// Test for hermiticity:
	Link H_diff = conj(H) - H; //This should be the zero matrix. Test by ReTr(Q_diff^2);
	H_diff *= H_diff;
	double error = getTrace(H_diff).real();
	printf("Herm test %d %d %d %.15e %+.15e\n", x_cb, parity, dir, error, getTrace(H).real());

	Link test = H;
	expsuNTaylor(test, arg.taylor_N);
	// Test for unitarity
	printf("SUN %d %d %d %.15e\n", x_cb, parity, dir, ErrorSUN(test));

	// Test for reproducibility
	test *= conj(U); //This should be the identity matrix. Test by ReTr((test - I)^2);	
	Link ID;
	setIdentity(&ID);
	test -= ID;
	test *= test;
	printf("expiH test %d %d %d %.6e: QR iter = %d, eig solver converged = %s\n",
	       x_cb, parity, dir, getTrace(test).real(), eigensolve.queryQR(), eigensolve.queryTriSolver() ? "true" : "false");
	
      }
    }
  };

} // namespace quda
