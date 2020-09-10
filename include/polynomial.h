// From https://github.com/opencv/opencv/blob/master/modules/calib3d/src/polynom_solver.cpp

#include <cmath>

constexpr double CV_PI = 3.14159265359;

int solve_deg2(double a, double b, double c, double &x1, double &x2)
{
  double delta = b * b - 4 * a * c;

  if (delta < 0) return 0;

  double inv_2a = 0.5 / a;

  if (delta == 0) {
    x1 = -b * inv_2a;
    x2 = x1;
    return 1;
  }

  double sqrt_delta = sqrt(delta);
  x1 = (-b + sqrt_delta) * inv_2a;
  x2 = (-b - sqrt_delta) * inv_2a;
  return 2;
}

/// Reference : Eric W. Weisstein. "Cubic Equation." From MathWorld--A Wolfram Web Resource.
/// http://mathworld.wolfram.com/CubicEquation.html
/// \return Number of real roots found.
int solve_deg3(double a, double b, double c, double d, double &x0, double &x1, double &x2)
{
  if (a == 0) {
    // Solve second order system
    if (b == 0) {
      // Solve first order system
      if (c == 0) return 0;

      x0 = -d / c;
      return 1;
    }

    x2 = 0;
    return solve_deg2(b, c, d, x0, x1);
  }

  // Calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0
  double inv_a = 1. / a;
  double b_a = inv_a * b, b_a2 = b_a * b_a;
  double c_a = inv_a * c;
  double d_a = inv_a * d;

  // Solve the cubic equation
  double Q = (3 * c_a - b_a2) / 9;
  double R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54;
  double Q3 = Q * Q * Q;
  double D = Q3 + R * R;
  double b_a_3 = (1. / 3.) * b_a;

  if (Q == 0) {
    if (R == 0) {
      x0 = x1 = x2 = -b_a_3;
      return 3;
    } else {
      x0 = pow(2 * R, 1 / 3.0) - b_a_3;
      return 1;
    }
  }

  if (D <= 0) {
    // Three real roots
    double theta = acos(R / sqrt(-Q3));
    double sqrt_Q = sqrt(-Q);
    x0 = 2 * sqrt_Q * cos(theta / 3.0) - b_a_3;
    x1 = 2 * sqrt_Q * cos((theta + 2 * CV_PI) / 3.0) - b_a_3;
    x2 = 2 * sqrt_Q * cos((theta + 4 * CV_PI) / 3.0) - b_a_3;

    return 3;
  }

  // D > 0, only one real root
  double AD = pow(fabs(R) + sqrt(D), 1.0 / 3.0) * (R > 0 ? 1 : (R < 0 ? -1 : 0));
  double BD = (AD == 0) ? 0 : -Q / AD;

  // Calculate the only real root
  x0 = AD + BD - b_a_3;

  return 1;
}

inline double eval_deg4(double a, double b, double c, double d, double e, double x)
{

  double x2 = x * x;
  double x3 = x * x2;
  double x4 = x2 * x2;

  return a * x4 + b * x3 + c * x2 + d * x + e;
}
