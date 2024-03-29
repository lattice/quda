#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <blas_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

namespace quda {

  SD::SD(const DiracMatrix &mat, SolverParam &param) : Solver(mat, mat, mat, mat, param) { }

  void SD::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      r = ColorSpinorField(csParam);
      Ar = ColorSpinorField(csParam);
      init = true;
    }
  }

  ColorSpinorField &SD::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    return r;
  }

  void SD::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    commGlobalReductionPush(param.global_reduction);

    create(x, b);

    double b2 = blas::norm2(b);
    double r2;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute the true residual
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
    } else {
      blas::zero(x);
      blas::copy(r, b);
      r2 = b2;
    }

    // if invalid residual then convergence is set by iteration count only
    double stop = param.residual_type == QUDA_INVALID_RESIDUAL ? 0.0 : b2 * param.tol * param.tol;

    int res_increase = 0;
    int k = 0;
    while (k < param.maxiter) {
      mat(Ar, r);
      double3 rAr = blas::cDotProductNormA(r, Ar);
      auto alpha = rAr.z / rAr.x;
      r2 = rAr.z; // this is r2 from the prior iteration

      PrintStats("SD", k, r2, b2, 0.0);

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
      double true_r2 = blas::xmyNorm(b, r);
      PrintSummary("SD", k, true_r2, b2, 0.0, 0.0);
      param.true_res = sqrt(true_r2 / b2);
    } else {
      PrintSummary("SD", k, r2, b2, 0.0, 0.0);
    }

    commGlobalReductionPop();
  }

} // namespace quda
