#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <blas_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

namespace quda {

  SD::SD(const DiracMatrix &mat, SolverParam &param) : Solver(mat, mat, mat, mat, param) { }

  void SD::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      resize(Ar, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      init = true;
    }
  }

  cvector_ref<const ColorSpinorField> SD::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    return r;
  }

  void SD::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    commGlobalReductionPush(param.global_reduction);

    create(x, b);

    vector<double> b2 = blas::norm2(b);
    vector<double> r2;

    // Check to see that we're not trying to invert on a zero-field source
    if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      bool zero_src = true;
      for (auto i = 0u; i < b.size(); i++) {
        if (b2[i] == 0) {
          warningQuda("inverting on zero-field source");
          x[i] = b[i];
          param.true_res[i] = 0.0;
          param.true_res_hq[i] = 0.0;
        } else {
          zero_src = false;
        }
      }
      if (zero_src) return;
    }

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute the true residual
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      for (auto i = 0u; i < b.size(); i++) if (b2[i] == 0) b2[i] = r2[i];
    } else {
      blas::zero(x);
      blas::copy(r, b);
      r2 = b2;
    }

    auto stop = stopping(param.tol, b2, param.residual_type);

    int res_increase = 0;
    int k = 0;
    while (k < param.maxiter) {
      mat(Ar, r);
      auto rAr = blas::cDotProductNormA(r, Ar);
      vector<double> alpha(b.size());
      for (auto i = 0u; i < b.size(); i++) {
        alpha[i] = rAr[i].z / rAr[i].x;
        r2[i] = rAr[i].z; // this is r2 from the prior iteration
      }

      PrintStats("SD", k, r2, b2);

      if (r2 < stop) {
        mat(r, x);
        r2 = blas::xmyNorm(b, r);
        if (r2 < stop) break;
        if (++res_increase > param.max_res_increase) {
          warningQuda("SD: solver exiting due to too many residual increases");
          break;
        }
      } else {
        blas::axpy(alpha, r, x);
        blas::axpy(-alpha, Ar, r);
      }
      k++;
    }

    param.iter += k;
    if (param.compute_true_res) {
      // Compute the true residual
      mat(r, x);
      auto true_r2 = blas::xmyNorm(b, r);
      PrintSummary("SD", k, true_r2, b2, stop);
      for (auto i = 0u; i < b2.size(); i++) param.true_res[i] = sqrt(true_r2[i] / b2[i]);
    } else {
      PrintSummary("SD", k, r2, b2, stop);
    }

    commGlobalReductionPop();
  }

} // namespace quda
