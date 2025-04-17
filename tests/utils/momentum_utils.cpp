#include <math.h>

#include <quda.h>

#include "command_line_params.h"
#include "host_utils.h"
#include "instantiate_host.hpp"
#include "rng_utils.hpp"
#include "momentum_utils.h"

/**
 * @brief Create a random traceless anti-Hermitian matrix with the correct
 * Gaussian distribution, times an (optional) scaling factor
 *
 * @tparam real_t Floating point type of the field
 * @param[out] mat Output random anti-Hermitian matrix
 * @param[in] max_val Optional scaling factor
 */
template <typename real_t>
void create_random_traceless_antiherm(real_t mat[10], int i, int parity, real_t max_val = 1.0)
{
  // The ordering of components is given in the RECONSTRUCT_10 unpack routine
  // in include/gauge_field_order.h
  // out[0] = complex(0.0, in[6]);
  // out[1] = complex(in[0], in[1]);
  // out[2] = complex(in[2], in[3]);
  // out[3] = complex(-out[1].real(), out[1].imag());
  // out[4] = complex(0.0, in[7]);
  // out[5] = complex(in[4], in[5]);
  // out[6] = complex(-out[2].real(), out[2].imag());
  // out[7] = complex(-out[5].real(), out[5].imag());
  // out[8] = complex(0.0, in[8]);

  // Normalization for generators on the diagonal
  real_t inv_sqrt3 = sqrt(1. / 3.);
  real_t r3 = max_val * random_gaussian_host(i, parity);
  real_t r8 = max_val * random_gaussian_host(i, parity);

  // contributes to the (0, 1), (1, 0) components
  mat[0] = max_val * random_gaussian_host(i, parity);
  mat[1] = max_val * random_gaussian_host(i, parity);

  // contributes to the (0, 2), (2, 0) components
  mat[2] = max_val * random_gaussian_host(i, parity);
  mat[3] = max_val * random_gaussian_host(i, parity);

  // contributes to the (1, 2), (2, 1) components
  mat[4] = max_val * random_gaussian_host(i, parity);
  mat[5] = max_val * random_gaussian_host(i, parity);

  // (0, 0) imaginary bit
  mat[6] = r3 + inv_sqrt3 * r8;

  // (1, 1) imaginary bit
  mat[7] = -r3 + inv_sqrt3 * r8;

  // (2, 2) imaginary bit
  mat[8] = -2. * inv_sqrt3 * r8;

  // null component
  mat[9] = 0.;
}

void createMomCPU(void *mom, QudaPrecision precision, double max_val)
{
  if (max_val == 0) {
    memset(mom, 0, 4ul * V * mom_site_size * (precision == QUDA_DOUBLE_PRECISION ? 8 : 4));
    return;
  }

  for (int i = 0; i < Vh; i++) {
    for (int parity = 0; parity < 2; parity++) {
      if (precision == QUDA_DOUBLE_PRECISION) {
        for (int dir = 0; dir < 4; dir++) {
          create_random_traceless_antiherm((double *)mom + (4 * (parity * Vh + i) + dir) * mom_site_size, i, parity,
                                           max_val);
        }
      } else {
        float max_val_f = static_cast<float>(max_val);
        for (int dir = 0; dir < 4; dir++) {
          for (auto k = 0lu; k < mom_site_size; k++) {
            create_random_traceless_antiherm((float *)mom + (4 * (parity * Vh + i) + dir) * mom_site_size, i, parity,
                                             max_val_f);
          }
        }
      }
    }
  }
}

/**
 * @brief Compute and print a robust comparison of agreement between two
 *        momentum fields
 *
 * @tparam real_t Floating point type of the field
 * @param[in] momA First momentum field
 * @param[in] momB Second momentum field
 * @param[in] len Length of the momentum field
 */
template <typename real_t> struct CompareMomentum {
  int operator()(const void *momA_, const void *momB_, int len)
  {
    auto momA = reinterpret_cast<const real_t *>(momA_);
    auto momB = reinterpret_cast<const real_t *>(momB_);

    const int fail_check = 16;
    int fail[fail_check];
    for (int f = 0; f < fail_check; f++) fail[f] = 0;

    int iter[mom_site_size];
    for (auto i = 0lu; i < mom_site_size; i++) iter[i] = 0;

#pragma omp parallel for
    for (int i = 0; i < len; i++) {
      for (auto j = 0lu; j < mom_site_size - 1; j++) {
        int is = i * mom_site_size + j;
        double diff = fabs(momA[is] - momB[is]);
        for (int f = 0; f < fail_check; f++)
          if (diff > pow(10.0, -(f + 1)) || std::isnan(diff)) {
#pragma omp atomic
            fail[f]++;
          }
        // if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
        if (diff > 1e-3 || std::isnan(diff)) {
#pragma omp atomic
          iter[j]++;
        }
      }
    }

    int accuracy_level = 0;
    for (int f = 0; f < fail_check; f++) {
      if (fail[f] == 0) { accuracy_level = f + 1; }
    }

    for (auto i = 0u; i < mom_site_size; i++) printfQuda("%u fails = %d\n", i, iter[i]);

    for (int f = 0; f < fail_check; f++) {
      printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0, -(f + 1)), fail[f], len * 9, fail[f] / (double)(len * 9));
    }

    return accuracy_level;
  }
};

/**
 * @brief Print the components of a momentum field at a given site
 *
 * @param[in] mom Momentum field
 * @param[in] X Site that is printed
 * @param[in] precision Floating-point precision of field
 */
void printMomElement(const void *mom, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    const double *thismom = ((const double *)mom) + X * mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  } else {
    const float *thismom = ((const float *)mom) + X * mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }
}

int strong_check_mom(const void *momA, const void *momB, int len, QudaPrecision precision)
{
  if (verbosity >= QUDA_VERBOSE) {
    printfQuda("mom:\n");
    printMomElement(momA, 0, prec);
    printfQuda("\n");
    printMomElement(momA, 1, prec);
    printfQuda("\n");
    printMomElement(momA, 2, prec);
    printfQuda("\n");
    printMomElement(momA, 3, prec);
    printfQuda("...\n");

    printfQuda("\nreference mom:\n");
    printMomElement(momB, 0, prec);
    printfQuda("\n");
    printMomElement(momB, 1, prec);
    printfQuda("\n");
    printMomElement(momB, 2, prec);
    printfQuda("\n");
    printMomElement(momB, 3, prec);
    printfQuda("\n");
  }

  return instantiate_host_reduce<CompareMomentum, int>(precision, momA, momB, len);
}

/**
 * @brief Host reference implementation of the momentum action
 * contribution, including the MILC convention of subtracting 4
 * from each site norm to improve stability.
 *
 * @tparam real_t Floating point type of the field
 * @param[in] mom Momentum field
 * @param[in] len Length of the momentum field
 */
template <typename real_t> struct MomentumAction {
  double operator()(const void *mom_, int len)
  {
    double action = 0.0;
    for (int i = 0; i < len; i++) {
      auto mom = reinterpret_cast<const real_t *>(mom_) + i * mom_site_size;
      double local = 0.0;
      for (int j = 0; j < 6; j++) local += mom[j] * mom[j];
      for (int j = 6; j < 9; j++) local += 0.5 * mom[j] * mom[j];
      local -= 4.0;
      action += local;
    }

    return action;
  }
};

double momentumActionCPU(const void *mom, int len, QudaPrecision prec)
{
  double action = instantiate_host_reduce<MomentumAction, double>(prec, mom, len);
  quda::comm_allreduce_sum(action);
  return action;
}