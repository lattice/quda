#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <util_quda.h>
#include <io/reunitarize.h>

namespace quda
{

  namespace io
  {

    namespace
    {
      constexpr int link_reals = 18;

      // element (i, j) of a 3x3 complex matrix stored row by row as [row][column][re, im]
      template <typename P> inline P &re(P *m, int i, int j) { return m[6 * i + 2 * j]; }
      template <typename P> inline P &im(P *m, int i, int j) { return m[6 * i + 2 * j + 1]; }

      // real and imaginary parts of the dot product of row a (conjugated) with row b
      template <typename T> inline T dot_re(const T *m, int a, int b)
      {
        return re(m, a, 0) * re(m, b, 0) + im(m, a, 0) * im(m, b, 0) + re(m, a, 1) * re(m, b, 1)
          + im(m, a, 1) * im(m, b, 1) + re(m, a, 2) * re(m, b, 2) + im(m, a, 2) * im(m, b, 2);
      }

      template <typename T> inline T dot_im(const T *m, int a, int b)
      {
        return re(m, a, 0) * im(m, b, 0) - im(m, a, 0) * re(m, b, 0) + re(m, a, 1) * im(m, b, 1)
          - im(m, a, 1) * re(m, b, 1) + re(m, a, 2) * im(m, b, 2) - im(m, a, 2) * re(m, b, 2);
      }

      // MILC's check_su3 (generic/check_unitarity.c, with STRONG defined)
      template <typename T> T check_su3(const T *m)
      {
        T max = 0;
        for (int i = 0; i < 3; i++) {
          T ar = dot_re(m, i, i);
          ar = std::fabs(std::sqrt(static_cast<double>(ar)) - 1.);
          if (max < ar) max = ar;
        }
        for (auto [a, b] : {std::pair {0, 1}, std::pair {0, 2}, std::pair {1, 2}}) {
          T ar = dot_re(m, a, b);
          T ai = dot_im(m, a, b);
          T ari = std::sqrt(static_cast<double>(ar * ar + ai * ai));
          if (max < ari) max = ari;
        }
        return max;
      }

      template <typename T> inline void normalize_row(T *m, int i)
      {
        T ar = dot_re(m, i, i);
        ar = 1.0 / std::sqrt(static_cast<double>(ar));
        for (int j = 0; j < 3; j++) {
          re(m, i, j) *= ar;
          im(m, i, j) *= ar;
        }
      }

      // MILC's reunit_su3 and fixsu3 (generic/reunitarize2.c), without its own deviation checks
      template <typename T> void reunit_su3(T *m)
      {
        normalize_row(m, 0);

        // row 1 -= (row 0 . row 1) row 0
        T ar = dot_re(m, 0, 1);
        T ai = dot_im(m, 0, 1);
        for (int j = 0; j < 3; j++) {
          re(m, 1, j) -= ar * re(m, 0, j) - ai * im(m, 0, j);
          im(m, 1, j) -= ar * im(m, 0, j) + ai * re(m, 0, j);
        }

        normalize_row(m, 1);

        // row 2 = conj(row 0 x row 1), as fixsu3 computes it
        const T bj0r = re(m, 0, 0), bj0i = im(m, 0, 0);
        const T bj1r = re(m, 0, 1), bj1i = im(m, 0, 1);
        const T bj2r = re(m, 0, 2), bj2i = im(m, 0, 2);
        T tr, ti;

        ar = re(m, 1, 2);
        ai = im(m, 1, 2);
        tr = bj1r * ar - bj1i * ai;
        ti = bj1r * ai + bj1i * ar;
        ar = re(m, 1, 1);
        ai = im(m, 1, 1);
        tr = tr - bj2r * ar + bj2i * ai;
        ti = ti - bj2r * ai - bj2i * ar;
        re(m, 2, 0) = tr;
        im(m, 2, 0) = -ti;

        ar = re(m, 1, 0);
        ai = im(m, 1, 0);
        tr = bj2r * ar - bj2i * ai;
        ti = bj2r * ai + bj2i * ar;
        ar = re(m, 1, 2);
        ai = im(m, 1, 2);
        tr = tr - bj0r * ar + bj0i * ai;
        ti = ti - bj0r * ai - bj0i * ar;
        re(m, 2, 1) = tr;
        im(m, 2, 1) = -ti;

        ar = re(m, 1, 1);
        ai = im(m, 1, 1);
        tr = bj0r * ar - bj0i * ai;
        ti = bj0r * ai + bj0i * ar;
        ar = re(m, 1, 0);
        ai = im(m, 1, 0);
        tr = tr - bj1r * ar + bj1i * ai;
        ti = ti - bj1r * ai - bj1i * ar;
        re(m, 2, 2) = tr;
        im(m, 2, 2) = -ti;
      }

      template <typename T> double max_deviation(const void *const gauge[], size_t volume)
      {
        double largest = 0.0;
        for (int dir = 0; dir < 4; dir++) {
          auto u = static_cast<const T *>(gauge[dir]);
#pragma omp parallel for reduction(max : largest)
          for (size_t s = 0; s < volume; s++) {
            const T *link = u + s * link_reals;
            // check_su3 ignores NaNs, since its comparisons with them are false
            bool finite = std::all_of(link, link + link_reals, [](T x) { return std::isfinite(x); });
            double deviation = finite ? check_su3(link) : std::numeric_limits<double>::infinity();
            largest = std::max(largest, deviation);
          }
        }
        return largest;
      }

      template <typename T> void reunitarize(void *const gauge[], size_t volume)
      {
        for (int dir = 0; dir < 4; dir++) {
          auto u = static_cast<T *>(gauge[dir]);
#pragma omp parallel for
          for (size_t s = 0; s < volume; s++) reunit_su3(u + s * link_reals);
        }
      }

      void check_precision(QudaPrecision prec)
      {
        if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
          errorQuda("Unsupported gauge field precision %d", prec);
      }
    } // namespace

    double unitarity_deviation(const void *const gauge[], QudaPrecision prec, size_t volume)
    {
      check_precision(prec);
      return prec == QUDA_DOUBLE_PRECISION ? max_deviation<double>(gauge, volume) : max_deviation<float>(gauge, volume);
    }

    void reunitarize_gram_schmidt(void *const gauge[], QudaPrecision prec, size_t volume)
    {
      check_precision(prec);
      if (prec == QUDA_DOUBLE_PRECISION)
        reunitarize<double>(gauge, volume);
      else
        reunitarize<float>(gauge, volume);
    }

  } // namespace io

} // namespace quda
