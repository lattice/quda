#pragma once

#include <host_utils.h>
#include <quda_internal.h>
#include <limits>
#include <algorithm>
#include <array>
#include <vector>
#include <complex>

#include "contract_ft_reference.h" // for FourierPhase

extern int Vh;
extern int V;

/**
   Host reference for the nucleon two-point contraction.  This is a
   deliberately brute-force evaluation of the Wick contraction of

     chi(x) = eps_{abc} [u^{aT}(x) (C g5) d^b(x)] u^c_alpha(x),

     C_{alpha alpha'}(x) = eps_{abc} eps_{a'b'c'} (Cg5)_{beta gamma} (Cg5)_{beta' gamma'} S_d^{bb'}_{gamma gamma'}
                           * [ S_u^{aa'}_{beta beta'} S_u^{cc'}_{alpha alpha'}
                               - S_u^{ac'}_{beta alpha'} S_u^{ca'}_{alpha beta'} ],

   with C gamma_5 built numerically from the DeGrand-Rossi gamma
   matrices at run time.  No algebraic rearrangement is shared with the
   optimized device kernel, so agreement checks both the arithmetic and
   the collapsed two-term form used there.
 */
namespace baryon_ref
{

using cplx = std::complex<double>;

// DeGrand-Rossi gamma matrices, QUDA conventions (mu = 1,2,3,4,5)
inline void dr_gamma(int mu, cplx g[4][4])
{
  const cplx I(0.0, 1.0);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) g[i][j] = 0.0;

  switch (mu) {
  case 1:
    g[0][3] = I;
    g[1][2] = I;
    g[2][1] = -I;
    g[3][0] = -I;
    break;
  case 2:
    g[0][3] = -1.0;
    g[1][2] = 1.0;
    g[2][1] = 1.0;
    g[3][0] = -1.0;
    break;
  case 3:
    g[0][2] = I;
    g[1][3] = -I;
    g[2][0] = -I;
    g[3][1] = I;
    break;
  case 4:
    g[0][2] = 1.0;
    g[1][3] = 1.0;
    g[2][0] = 1.0;
    g[3][1] = 1.0;
    break;
  case 5:
    g[0][0] = 1.0;
    g[1][1] = 1.0;
    g[2][2] = -1.0;
    g[3][3] = -1.0;
    break;
  default: errorQuda("Unexpected gamma index %d", mu);
  }
}

inline void mat_mul(const cplx a[4][4], const cplx b[4][4], cplx c[4][4])
{
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      cplx acc = 0.0;
      for (int k = 0; k < 4; k++) acc += a[i][k] * b[k][j];
      c[i][j] = acc;
    }
}

// C gamma_5 with C = gamma_4 gamma_2
inline void c_gamma5(cplx cg5[4][4])
{
  cplx g2[4][4], g4[4][4], g5[4][4], c[4][4];
  dr_gamma(2, g2);
  dr_gamma(4, g4);
  dr_gamma(5, g5);
  mat_mul(g4, g2, c);
  mat_mul(c, g5, cg5);
}

/**
   @brief Numerical check that C = gamma_4 gamma_2 satisfies the
   defining property C gamma_mu C^{-1} = -gamma_mu^T in the
   DeGrand-Rossi basis.  Returns the maximum absolute deviation.
 */
inline double check_charge_conjugation()
{
  cplx g2[4][4], g4[4][4], c[4][4];
  dr_gamma(2, g2);
  dr_gamma(4, g4);
  mat_mul(g4, g2, c);

  // C^{-1} = C^dagger for unitary C
  cplx cinv[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) cinv[i][j] = std::conj(c[j][i]);

  double max_dev = 0.0;
  for (int mu = 1; mu <= 4; mu++) {
    cplx g[4][4], t1[4][4], t2[4][4];
    dr_gamma(mu, g);
    mat_mul(c, g, t1);
    mat_mul(t1, cinv, t2);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) max_dev = std::max(max_dev, std::abs(t2[i][j] + g[j][i]));
  }
  return max_dev;
}

template <typename Float>
void baryonContractFTHost(void **prop_u, void **prop_d, double *h_result, const int *X,
                          const int *const source_position, const int n_mom, const int *const mom_modes,
                          const QudaFFTSymmType *const fft_type)
{
  constexpr int nSpin = 4;
  constexpr size_t num_out = nSpin * nSpin;

  // The number of slices in the decay dimension, locally and globally.
  size_t local_slices = X[3];
  size_t global_slices = local_slices * quda::comm_dim(3);

  std::vector<cplx> result_global(n_mom * global_slices * num_out);
  std::fill(result_global.begin(), result_global.end(), cplx {0.0, 0.0});

  cplx cg5[4][4];
  c_gamma5(cg5);

  // non-zero pattern of C g5: one entry per row
  int cg5_col[4];
  cplx cg5_val[4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (std::abs(cg5[i][j]) > 1e-14) {
        cg5_col[i] = j;
        cg5_val[i] = cg5[i][j];
      }
    }
  }

  constexpr int eps[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
  constexpr int eps_sign[6] = {1, 1, 1, -1, -1, -1};

  // Strides for computing local coordinates
  int strides[4] {1, X[0], X[1] * X[0], X[2] * X[1] * X[0]};

  // Global lattice dimensions
  int L[4];
  for (int dir = 0; dir < 4; ++dir) L[dir] = X[dir] * quda::comm_dim(dir);

  std::vector<double> phase(n_mom * 2);
  int sink[4];

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
    for (int dir = 0; dir < 4; ++dir) sink[dir] += quda::comm_coord(dir) * X[dir];
    int red_coord = sink[3];

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

    // Propagators at this site: S[spin_sink][color_sink][spin_source][color_source]
    size_t off = nSpin * 3 * 2 * (Vh * parity + cb_idx);
    cplx Su[4][3][4][3], Sd[4][3][4][3];
    for (int sj = 0; sj < 4; sj++) {
      for (int cj = 0; cj < 3; cj++) {
        const Float *pu = static_cast<Float *>(prop_u[sj * 3 + cj]) + off;
        const Float *pd = static_cast<Float *>(prop_d[sj * 3 + cj]) + off;
        for (int si = 0; si < 4; si++) {
          for (int ci = 0; ci < 3; ci++) {
            Su[si][ci][sj][cj] = cplx(pu[6 * si + 2 * ci + 0], pu[6 * si + 2 * ci + 1]);
            Sd[si][ci][sj][cj] = cplx(pd[6 * si + 2 * ci + 0], pd[6 * si + 2 * ci + 1]);
          }
        }
      }
    }

    // open-spin correlator at this site by direct Wick contraction
    cplx C[4][4] = {};
    for (int ip = 0; ip < 6; ip++) {
      const int a = eps[ip][0], b = eps[ip][1], c = eps[ip][2];
      for (int jp = 0; jp < 6; jp++) {
        const int ap = eps[jp][0], bp = eps[jp][1], cp = eps[jp][2];
        const double sgn = eps_sign[ip] * eps_sign[jp];

        // (C g5) has a single non-zero column per row: gamma = cg5_col[beta]
        for (int beta = 0; beta < 4; beta++) {
          const int gamma = cg5_col[beta];
          for (int betap = 0; betap < 4; betap++) {
            const int gammap = cg5_col[betap];
            const cplx w = sgn * cg5_val[beta] * cg5_val[betap] * Sd[gamma][b][gammap][bp];
            for (int alpha = 0; alpha < 4; alpha++) {
              for (int alphap = 0; alphap < 4; alphap++) {
                C[alpha][alphap] += w
                  * (Su[beta][a][betap][ap] * Su[alpha][c][alphap][cp]
                     - Su[beta][a][alphap][cp] * Su[alpha][c][betap][ap]);
              }
            }
          }
        }
      }
    }

    // multiply by Fourier phases and accumulate
    for (int mom_idx = 0; mom_idx < n_mom; ++mom_idx) {
      const cplx ph(phase[2 * mom_idx + 0], phase[2 * mom_idx + 1]);
      for (int alpha = 0; alpha < 4; alpha++) {
        for (int alphap = 0; alphap < 4; alphap++) {
          size_t m_idx = 4 * alpha + alphap;
          size_t g_idx = global_slices * num_out * mom_idx + num_out * red_coord + m_idx;
          result_global[g_idx] += ph * C[alpha][alphap];
        }
      }
    }
  } // sites

  // global reduction
  quda::comm_allreduce_sum(result_global);

  // copy to output array
  for (size_t idx = 0; idx < n_mom * global_slices * num_out; ++idx) {
    h_result[2 * idx + 0] = result_global[idx].real();
    h_result[2 * idx + 1] = result_global[idx].imag();
  }
}

template <typename Float>
int baryonContractFT_reference(void **prop_u, void **prop_d, const double *const d_result, const int *X,
                               const int *const source_position, const int n_mom, const int *const mom_modes,
                               const QudaFFTSymmType *const fft_type)
{
  constexpr int nSpin = 4;

  // The number of slices in the reduction dimension.
  size_t reduction_slices = X[3] * quda::comm_dim(3);

  // space for the host result
  const size_t n_floats = n_mom * reduction_slices * nSpin * nSpin * 2;
  std::vector<double> h_result(n_floats, 0.0);

  // compute contractions on the host
  baryonContractFTHost<Float>(prop_u, prop_d, h_result.data(), X, source_position, n_mom, mom_modes, fft_type);

  const int ntol = 7;
  auto epsilon = std::numeric_limits<Float>::epsilon();
  double fact = epsilon;
  // account for repeated roundoff: 36 color permutation pairs each with
  // O(nSpin^2 * 16) triple products, accumulated over the volume
  fact *= sqrt((double)nSpin * nSpin * 36 * 16 * V * quda::comm_size() * 2 / reduction_slices);
  fact *= 100; // account for variation in phase computation and magnitude spread
  std::vector<double> tolerance(ntol);
  std::generate(tolerance.begin(), tolerance.end(), [step = 1e-6 * fact]() mutable { return step *= 10; });

  int check_tol = 5;
  std::vector<int> fails(ntol, 0);

  // normalize the comparison to the typical magnitude of the correlator
  double norm = 0.0;
  for (size_t idx = 0; idx < n_floats; ++idx) norm += h_result[idx] * h_result[idx];
  norm = sqrt(norm / n_floats);
  if (norm == 0.0) norm = 1.0;

  for (size_t idx = 0; idx < n_floats; ++idx) {
    double rel = fabs(d_result[idx] - h_result[idx]) / norm;
    for (int d = 0; d < ntol; ++d)
      if (rel > tolerance[d]) ++fails[d];
  }

  printfQuda("tolerance  n_diffs\n");
  printfQuda("---------- --------\n");
  for (int j = 0; j < ntol; ++j) { printfQuda("%9.1e: %8d\n", tolerance[j], fails[j]); }
  printfQuda("---------- --------\n");
  printfQuda("check tolerance is %9.1e (relative to rms %9.3e)\n", tolerance[check_tol], norm);

  return fails[check_tol];
}

} // namespace baryon_ref
