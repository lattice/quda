#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <quda_internal.h>
#include <math_helper.h>

// Implemented according to https://en.wikipedia.org/wiki/Cubic_equation

namespace quda
{

  inline std::vector<real_t> quadratic_formula(std::array<real_t, 3> coeff)
  {

    std::vector<real_t> z;
    z.reserve(2);

    real_t &a = coeff[0];
    real_t &b = coeff[1];
    real_t &c = coeff[2];

    // a x^2 + b x + c = 0
    if (a == 0) {
      // actually a linear equation
      if (b != 0) { z.push_back(-c / b); }
    } else {
      real_t delta = b * b - real_t(4.0) * a * c;
      if (delta >= 0) {
        z.push_back((-b + sqrt(delta)) / (real_t(2.0) * a));
        z.push_back((-b - sqrt(delta)) / (real_t(2.0) * a));
      }
    }

    return z;
  }

  inline std::vector<real_t> cubic_formula(std::array<real_t, 4> coeff)
  {

    std::vector<real_t> t;
    t.reserve(3);

    // a x^3 + b x^2 + c x + d = 0
    real_t &a = coeff[0];
    real_t &b = coeff[1];
    real_t &c = coeff[2];
    real_t &d = coeff[3];

    if (a == 0) {
      // actually a quadratic equation.
      std::array<real_t, 3> quadratic_coeff = {coeff[1], coeff[2], coeff[3]};
      auto quad = quadratic_formula(quadratic_coeff);
      for (size_t i = 0; i < quad.size(); i++) { t.push_back(quad[i]); }
      return t;
    }

    real_t a2 = a * a;
    real_t a3 = a * a * a;

    real_t b2 = b * b;
    real_t b3 = b * b * b;

    real_t p = (real_t(3.0) * a * c - b2) / (real_t(3.0) * a2);
    real_t q = (real_t(2.0) * b3 - real_t(9.0) * a * b * c + real_t(27.0) * a2 * d) / (real_t(27.0) * a3);

    // Now solving t^3 + p t + q = 0
    if (p == 0) {

      t.push_back(cbrt(-q));

    } else {

      real_t delta = real_t(-4.0) * p * p * p - real_t(27.0) * q * q;

      if (delta == 0) {

        t.push_back(+3.0 * q / p);
        t.push_back(-1.5 * q / p);
        t.push_back(-1.5 * q / p);

      } else if (delta > 0) {

        real_t theta = acos(real_t(1.5) * (q / p) * sqrt(real_t(-3.0) / p));
        real_t tmp = real_t(2.0) * sqrt(-p / real_t(3.0));
        for (int k = 0; k < 3; k++) { t.push_back(tmp * cos((theta - real_t(2.0) * M_PI * k) / real_t(3.0))); }

      } else if (delta < 0) {

        if (p < 0) {
          real_t aq = fabs(q);
          real_t theta = acosh(real_t(-1.5) * aq / p * sqrt(real_t(-3.0) / p));
          t.push_back(real_t(-2.0) * aq / q * sqrt(-p / real_t(3.0)) * cosh(theta / real_t(3.0)));
        } else if (p > 0) {
          real_t theta = asinh(real_t(+1.5) * q / p * sqrt(real_t(3.0) / p));
          t.push_back(real_t(-2.0) * sqrt(p / real_t(3.0)) * sinh(theta / real_t(3.0)));
        }
      }
    }

    for (auto &p : t) { p += -b / (real_t(3.0) * a); }

    return t;
  }

  inline real_t poly4(std::array<real_t, 5> coeffs, real_t x)
  {
    real_t x2 = x * x;
    real_t x3 = x * x2;
    real_t x4 = x2 * x2;
    return x4 * coeffs[4] + x3 * coeffs[3] + x2 * coeffs[2] + x * coeffs[1] + coeffs[0];
  }

} // namespace quda
