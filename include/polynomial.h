#pragma once

#include <cmath>
#include <array>
#include <vector>

// Implemented according to https://en.wikipedia.org/wiki/Cubic_equation

namespace quda
{

  inline std::vector<double> quadratic_formula(std::array<double, 3> coeff)
  {

    std::vector<double> z;
    z.reserve(2);

    double &a = coeff[0];
    double &b = coeff[1];
    double &c = coeff[2];

    // a x^2 + b x + c = 0
    if (a == 0) {
      // actually a linear equation
      if (b != 0) { z.push_back(-c / b); }
    } else {
      double delta = b * b - 4.0 * a * c;
      if (delta >= 0) {
        z.push_back((-b + std::sqrt(delta)) / (2.0 * a));
        z.push_back((-b - std::sqrt(delta)) / (2.0 * a));
      }
    }

    return z;
  }

  inline std::vector<double> cubic_formula(std::array<double, 4> coeff)
  {

    std::vector<double> t;
    t.reserve(3);

    // a x^3 + b x^2 + c x + d = 0
    double &a = coeff[0];
    double &b = coeff[1];
    double &c = coeff[2];
    double &d = coeff[3];

    if (a == 0) {
      // actually a quadratic equation.
      std::array<double, 3> quadratic_coeff = {coeff[1], coeff[2], coeff[3]};
      auto quad = quadratic_formula(quadratic_coeff);
      for (size_t i = 0; i < quad.size(); i++) { t.push_back(quad[i]); }
      return t;
    }

    double a2 = a * a;
    double a3 = a * a * a;

    double b2 = b * b;
    double b3 = b * b * b;

    double p = (3.0 * a * c - b2) / (3.0 * a2);
    double q = (2.0 * b3 - 9.0 * a * b * c + 27.0 * a2 * d) / (27.0 * a3);

    // Now solving t^3 + p t + q = 0
    if (p == 0) {

      t.push_back(std::cbrt(-q));

    } else {

      double delta = -4.0 * p * p * p - 27.0 * q * q;

      if (delta == 0) {

        t.push_back(+3.0 * q / p);
        t.push_back(-1.5 * q / p);
        t.push_back(-1.5 * q / p);

      } else if (delta > 0) {

        double theta = std::acos(1.5 * (q / p) * std::sqrt(-3.0 / p));
        double tmp = 2.0 * std::sqrt(-p / 3.0);
        for (int k = 0; k < 3; k++) { t.push_back(tmp * std::cos((theta - 2.0 * M_PI * k) / 3.0)); }

      } else if (delta < 0) {

        if (p < 0) {
          double theta = std::acosh(-1.5 * std::abs(q) / p * std::sqrt(-3.0 / p));
          t.push_back(-2.0 * std::abs(q) / q * std::sqrt(-p / 3.0) * cosh(theta / 3.0));
        } else if (p > 0) {
          double theta = std::asinh(+1.5 * q / p * std::sqrt(3.0 / p));
          t.push_back(-2.0 * std::sqrt(p / 3.0) * sinh(theta / 3.0));
        }
      }
    }

    for (auto &p : t) { p += -b / (3.0 * a); }

    return t;
  }

  inline double poly4(std::array<double, 5> coeffs, double x)
  {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = x2 * x2;
    return x4 * coeffs[4] + x3 * coeffs[3] + x2 * coeffs[2] + x * coeffs[1] + coeffs[0];
  }

} // namespace quda
