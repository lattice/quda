#include <numeric>
#include <eigen_helper.h>
#include <overlap_kernel.h>

namespace quda
{
  // Chebyshev polynomial the first kind
  // T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
  double Tn(double x, int n)
  {
    if (abs(x) <= 1.0) { return cos(n * std::acos(x)); }
    double T0 = 1, T1 = x, Tk = 2 * x * x - 1;
    switch (n) {
    case 0: return T0;
    case 1: return T1;
    case 2: return Tk;
    default:
      for (int k = 3; k <= n; ++k) {
        T0 = T1;
        T1 = Tk;
        Tk = 2 * x * T1 - T0;
      }
      return Tk;
    }
  }

  // \sum_{i=0}^n c_i T_i
  // T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
  // Use Clenshaw algorithm
  double ciTi(double x, std::vector<double> c, int n)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    for (int k = n; k >= 1; --k) {
      bk = c[k] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[0] + x * b1 - b2;
  }

  // (\sum_{i=0}^n c_i T_i)' = \sum_{i=1}^n i c_i U_{i-1}
  // U_{k+1}(x) = 2 x U_k(x) - U_{k-1}
  // Use Clenshaw algorithm
  double iciUim1(double x, std::vector<double> c, int n)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    for (int k = n - 1; k >= 1; --k) {
      bk = (k + 1) * c[k + 1] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[1] + 2 * x * b1 - b2;
  }

  double residual(double x, std::vector<double> c, int n, double epsilon, bool derivative)
  {
    const double z = (x * 2 - (1 + epsilon)) / (1 - epsilon);
    if (derivative) {
      return -1 / (2 * sqrt(x)) * ciTi(z, c, n) - sqrt(x) * iciUim1(z, c, n) * (2 / (1 - epsilon));
    } else {
      return 1 - sqrt(x) * ciTi(z, c, n);
    }
  }

  double findRoot(double x_l, double x_r, std::vector<double> c, int n, double epsilon, bool derivative)
  {
    double x_m, res_r, res_l, res_m;

    res_l = residual(x_l, c, n, epsilon, derivative);
    res_r = residual(x_r, c, n, epsilon, derivative);
    if (abs(res_l) < 1e-15) return x_l;
    if (abs(res_r) < 1e-15) return x_r;
    if (res_r * res_l > 0)
      errorQuda("ERROR: findRoot with derivative=%d called with wrong ends: (%e %e)->(%e %e)\n", derivative, x_l, x_r,
                res_l, res_r);
    for (int i = 0; i < 10; i++) {
      x_m = (res_l * x_r - res_r * x_l) / (res_l - res_r);
      res_m = residual(x_m, c, n, epsilon, derivative);
      if (res_m * res_l > 0) {
        x_l = x_m;
        res_l = res_m;
      } else {
        x_r = x_m;
        res_r = res_m;
      }
    }
    return (res_l * x_r - res_r * x_l) / (res_l - res_r);
  }

  std::vector<double> minimaxApproximationRemez(double delta, double epsilon)
  {
    const int n_ref = ceil(-log(delta / 0.41) / (2.083 * sqrt(epsilon))) + 1;
    bool converged = false;
    constexpr int max_iter = 5;
    std::vector<double> y, z, c, b;
    for (int n = n_ref; n < n_ref * 1.1; n++) {
      y.resize(n + 1);
      z.resize(n + 1);
      c.resize(n + 1);
      b.resize(n + 1);
      Eigen::Map<Eigen::VectorXd> b_eigen(b.data(), b.size()), c_eigen(c.data(), c.size());
      Eigen::MatrixXd M_eigen(n + 1, n + 1);

      for (int i = 0; i < n + 1; ++i) {
        z[i] = cos(M_PI * i / n);
        y[i] = (z[i] * (1 - epsilon) + (1 + epsilon)) / 2;
      }

      int iter = 0;
      while (iter < max_iter) {
        // Construct matrix M_ij=\sqrt{y_i}T_j(z_i)
        for (int i = 0; i < n + 1; ++i) {
          for (int j = 0; j < n; ++j) { M_eigen(i, j) = sqrt(y[i]) * Tn(z[i], j); }
          M_eigen(i, n) = i % 2 == 0 ? 1 : -1; // T_n is not a real Chebyshev polynomial
          b_eigen(i) = 1.0;
        }
        c_eigen = M_eigen.lu().solve(b_eigen);

        // Drop T_n
        for (int i = 0; i < n; ++i) { b[i] = findRoot(y[i], y[i + 1], c, n - 1, epsilon, false); }
        for (int i = n - 1; i > 0; --i) { y[i] = findRoot(b[i], b[i - 1], c, n - 1, epsilon, true); }
        for (int i = 1; i < n; ++i) { z[i] = (2 * y[i] - (1 + epsilon)) / (1 - epsilon); }
        for (int i = 0; i < n + 1; ++i) { b[i] = abs(1 - sqrt(y[i]) * ciTi(z[i], c, n - 1)); }
        if (*std::max_element(b.begin(), b.end()) <= delta) { break; }
        iter += 1;
      }
      if (iter < max_iter) {
        converged = true;
        break;
      }
    }
    if (!converged) errorQuda("Remez algorithm did not converge\n");
    return {c.begin(), c.end() - 1};
  }

  OverlapKernel::OverlapKernel(std::vector<ColorSpinorField> &evecs, const std::vector<Complex> &evals, double kappa,
                               const std::vector<double> remez_tol) :
    evals(evals.size()),
    kappa(kappa),
    epsilon(pow(evals.back().real() / (1.0 + 8.0 * kappa), 2)),
    remez_tol(remez_tol),
    remez_coeff(remez_tol.size()),
    remez_order(remez_tol.size())
  {
    this->evecs = std::move(evecs);
    for (size_t i = 0; i < evals.size(); i++) { this->evals[i] = evals[i].real(); }
    for (size_t i = 0; i < remez_tol.size(); i++) {
      remez_coeff[i] = minimaxApproximationRemez(remez_tol[i], epsilon);
      remez_order[i] = remez_coeff[i].size() - 1;
    }
  }

  OverlapKernel::OverlapKernel(const OverlapKernel *overlap_kernel, QudaPrecision precision) :
    evals(overlap_kernel->evals),
    kappa(overlap_kernel->kappa),
    epsilon(overlap_kernel->epsilon),
    remez_tol(overlap_kernel->remez_tol),
    remez_coeff(overlap_kernel->remez_tol.size()),
    remez_order(overlap_kernel->remez_tol.size())
  {
    ColorSpinorParam param(overlap_kernel->evecs[0]);
    param.setPrecision(precision, precision, true);
    evecs.resize(overlap_kernel->evecs.size(), ColorSpinorField(param));
    for (size_t i = 0; i < overlap_kernel->evecs.size(); i++) { evecs[i].copy(overlap_kernel->evecs[i]); }
    double prec_tol;
    switch (precision) {
    case QUDA_DOUBLE_PRECISION: prec_tol = std::numeric_limits<double>::epsilon() / 2.; break;
    case QUDA_SINGLE_PRECISION: prec_tol = std::numeric_limits<float>::epsilon() / 2.; break;
    case QUDA_HALF_PRECISION: prec_tol = pow(2., -16); break;
    case QUDA_QUARTER_PRECISION: prec_tol = pow(2., -8); break;
    default: errorQuda("Invalid precision %d", precision); break;
    }
    for (size_t i = 0; i < remez_tol.size(); i++) {
      double tol = std::max(remez_tol[i], prec_tol);
      remez_tol[i] = tol;
      remez_coeff[i] = minimaxApproximationRemez(tol, epsilon);
      remez_order[i] = remez_coeff[i].size() - 1;
    }
  }
} // namespace quda
