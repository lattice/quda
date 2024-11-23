#pragma once

#include <host_utils.h>
#include <quda_internal.h>
#include "color_spinor_field.h"
#include <limits>
#include <algorithm>
#include <array>
#include <vector>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;
template <typename T> using complex = std::complex<T>;

// Color contract two ColorSpinors at a site returning a [nSpin x nSpin] matrix.
template <typename Float>
inline void contractColors(const Float *const spinorX, const Float *const spinorY, const int nSpin, Float M[])
{
  for (int s1 = 0; s1 < nSpin; s1++) {
    for (int s2 = 0; s2 < nSpin; s2++) {
      Float re = 0.0;
      Float im = 0.0;
      for (int c = 0; c < 3; c++) {
        re += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 0]
               + spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 1]);

        im += (spinorX[6 * s1 + 2 * c + 0] * spinorY[6 * s2 + 2 * c + 1]
               - spinorX[6 * s1 + 2 * c + 1] * spinorY[6 * s2 + 2 * c + 0]);
      }

      M[2 * (nSpin * s1 + s2) + 0] = re;
      M[2 * (nSpin * s1 + s2) + 1] = im;
    }
  }
};

// accumulate Fourier phase
template <typename Float> inline void FourierPhase(Float z[2], const Float theta, const QudaFFTSymmType fft_type)
{
  Float w[2] {z[0], z[1]};
  if (fft_type == QUDA_FFT_SYMM_EVEN) {
    Float costh = cos(theta);
    z[0] = w[0] * costh;
    z[1] = w[1] * costh;
  } else if (fft_type == QUDA_FFT_SYMM_ODD) {
    Float sinth = sin(theta);
    z[0] = -w[1] * sinth;
    z[1] = w[0] * sinth;
  } else if (fft_type == QUDA_FFT_SYMM_EO) {
    Float costh = cos(theta);
    Float sinth = sin(theta);
    z[0] = w[0] * costh - w[1] * sinth;
    z[1] = w[1] * costh + w[0] * sinth;
  }
};

template <typename Float>
void contractFTHost(void **h_prop_array_flavor_1, void **h_prop_array_flavor_2, double *h_result,
                    const QudaContractType cType, const int src_colors, const int *X, const int *const source_position,
                    const int n_mom, const int *const mom_modes, const QudaFFTSymmType *const fft_type)
{
  int nSpin = 4;
  if (cType == QUDA_CONTRACT_TYPE_STAGGERED_FT_T) nSpin = 1;

  // The number of contraction results expected in the output
  size_t num_out_results = nSpin * nSpin;

  int reduct_dim = 3; // t-dir is default
  if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z) reduct_dim = 2;

  // The number of slices in the decay dimension on this MPI rank.
  size_t local_reduct_slices = X[reduct_dim];

  // The number of slices in the decay dimension globally.
  size_t global_reduct_slices = local_reduct_slices * comm_dim(reduct_dim);

  // Array for all momenta, reduction slices, and channels. It is zeroed prior to kernel launch.
  std::vector<std::complex<double>> result_global(n_mom * global_reduct_slices * num_out_results);
  std::fill(result_global.begin(), result_global.end(), Complex {0.0, 0.0});

  // Strides for computing local coordinates
  int strides[4] {1, X[0], X[1] * X[0], X[2] * X[1] * X[0]};

  // Global lattice dimensions
  int L[4];
  for (int dir = 0; dir < 4; ++dir) L[dir] = X[dir] * comm_dim(dir);

  std::vector<double> phase(n_mom * 2);
  std::vector<Float> M(num_out_results * 2);
  // size_t x ;
  int sink[4];
  int red_coord = -1;
  for (int sindx = 0; sindx < V; ++sindx) {
    // compute local coordinates; lexicographical with x fastest
    int parity = 0;
    int rem = sindx;
    for (int dir = 3; dir >= 0; --dir) {
      sink[dir] = rem / strides[dir];
      rem -= sink[dir] * strides[dir];
      parity += sink[dir];
    }
    parity &= 1;
    int cb_idx = sindx / 2;
    // global coords
    for (int dir = 0; dir < 4; ++dir) {
      sink[dir] += comm_coord(dir) * X[dir];
      if (reduct_dim == dir) red_coord = sink[dir]; // project to this coord
    }

    // compute Fourier phases
    for (int mom_idx = 0; mom_idx < n_mom; ++mom_idx) {
      phase[2 * mom_idx + 0] = 1.;
      phase[2 * mom_idx + 1] = 0.;
      for (int dir = 0; dir < 4; ++dir) {
        double theta = 2. * M_PI / L[dir];
        theta *= (sink[dir] - source_position[dir]) * mom_modes[4 * mom_idx + dir];
        FourierPhase<double>(phase.data() + 2 * mom_idx, theta, fft_type[4 * mom_idx + dir]);
      }
    }

    for (int s1 = 0; s1 < nSpin; s1++) {
      for (int s2 = 0; s2 < nSpin; s2++) {
        for (int c1 = 0; c1 < src_colors; c1++) {
          // color contraction
          size_t off = nSpin * 3 * 2 * (Vh * parity + cb_idx);
          contractColors<Float>(static_cast<Float *>(h_prop_array_flavor_1[s1 * src_colors + c1]) + off,
                                static_cast<Float *>(h_prop_array_flavor_2[s2 * src_colors + c1]) + off, nSpin, M.data());

          // apply gamma matrices here

          // mutiply by Fourier phases and accumulate
          for (int mom_idx = 0; mom_idx < n_mom; ++mom_idx) {
            for (size_t m_idx = 0; m_idx < num_out_results; ++m_idx) {
              Float prod[2];
              prod[0] = phase[2 * mom_idx + 0] * M[2 * m_idx + 0] - phase[2 * mom_idx + 1] * M[2 * m_idx + 1];
              prod[1] = phase[2 * mom_idx + 1] * M[2 * m_idx + 0] + phase[2 * mom_idx + 0] * M[2 * m_idx + 1];
              // result[mom_idx][red_coord][m_idx]
              size_t g_idx = global_reduct_slices * num_out_results * mom_idx + num_out_results * red_coord + m_idx;
              result_global[g_idx] += std::complex<double> {prod[0], prod[1]};
            }
          }
        }
      }
    }
  } // sites

  // global reduction
  quda::comm_allreduce_sum(result_global);

  // copy to output array
  for (size_t idx = 0; idx < n_mom * global_reduct_slices * num_out_results; ++idx) {
    h_result[2 * idx + 0] = result_global[idx].real();
    h_result[2 * idx + 1] = result_global[idx].imag();
  }
};

template <typename Float>
int contractionFT_reference(void **spinorX, void **spinorY, const double *const d_result, const QudaContractType cType,
                            const int src_colors, const int *X, const int *const source_position, const int n_mom,
                            const int *const mom_modes, const QudaFFTSymmType *const fft_type)
{
  int nSpin = 4;
  if (cType == QUDA_CONTRACT_TYPE_STAGGERED_FT_T) nSpin = 1;

  size_t reduct_dim = 3; // t-dir is default
  if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z) reduct_dim = 2;

  // The number of slices in the reduction dimension.
  size_t reduction_slices = X[reduct_dim] * comm_dim(reduct_dim);

  // space for the host result
  const size_t n_floats = n_mom * reduction_slices * nSpin * nSpin * 2;
  double *h_result = static_cast<double *>(safe_malloc(n_floats * sizeof(double)));

  // compute contractions on the host
  contractFTHost<Float>(spinorX, spinorY, h_result, cType, src_colors, X, source_position, n_mom, mom_modes, fft_type);

  const int ntol = 7;
  auto epsilon = std::numeric_limits<Float>::epsilon();
  auto fact = epsilon;
  fact *= sqrt((double)nSpin * 6 * V * comm_size() * 2 / reduction_slices); // account for repeated roundoff in float ops
  fact *= 10; // account for variation in phase computation
  std::vector<double> tolerance(ntol);
  std::generate(tolerance.begin(), tolerance.end(), [step = 1e-6 * fact]() mutable { return step *= 10; });

  int check_tol = 5;
  std::vector<int> fails(ntol, 0.0);

  for (size_t idx = 0; idx < n_floats; ++idx) {
    double rel = abs(d_result[idx] - h_result[idx]);
    // printfQuda("%5ld: %10.3e %10.3e: %10.3e\n", idx, d_result[idx], h_result[idx], rel);
    for (int d = 0; d < ntol; ++d)
      if (rel > tolerance[d]) ++fails[d];
  }

  printfQuda("tolerance  n_diffs\n");
  printfQuda("---------- --------\n");
  for (int j = 0; j < ntol; ++j) { printfQuda("%9.1e: %8d\n", tolerance[j], fails[j]); }
  printfQuda("---------- --------\n");
  printfQuda("check tolerance is %9.1e\n", tolerance[check_tol]);

  host_free(h_result);

  return fails[check_tol];
};
