// QUDA header (for HeavyQuarkResidualNorm)
#include <blas_quda.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <dslash_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <command_line_params.h>

// Overload for workflows without multishift
std::array<double, 2> verifyInversion(void *spinorOut, void *spinorIn, void *spinorCheck, QudaGaugeParam &gauge_param,
                                      QudaInvertParam &inv_param, void **gauge, void *clover, void *clover_inv,
                                      int src_idx)
{
  void **spinorOutMulti = nullptr;
  return verifyInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge, clover,
                         clover_inv, src_idx);
}

std::array<double, 2> verifyInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                      QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge,
                                      void *clover, void *clover_inv, int src_idx)
{
  std::array<double, 2> res = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    res = verifyDomainWallTypeInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge,
                                        clover, clover_inv, src_idx);
  } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH
             || dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    res = verifyWilsonTypeInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge,
                                    clover, clover_inv, src_idx);
  } else {
    errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
  }
  return res;
}

std::array<double, 2> verifyDomainWallTypeInversion(void *spinorOut, void **, void *spinorIn, void *spinorCheck,
                                                    QudaGaugeParam &gauge_param, QudaInvertParam &inv_param,
                                                    void **gauge, void *, void *, int src_idx)
{
  if (multishift > 1) errorQuda("Multishift not supported");

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_mat(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.dagger, inv_param.cpu_prec, gauge_param,
              inv_param.mass, inv_param.b_5, inv_param.c_5);
      host_free(kappa_b);
      host_free(kappa_c);
    } else if (dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      mdw_eofa_mat(spinorCheck, gauge, spinorOut, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass,
                   inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                   inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    } else {
      errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      ax(0.5 / kappa5, spinorCheck, V * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    }

  } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {

    // DOMAIN_WALL START
    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
               inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                  inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_matpc(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      host_free(kappa_b);
      host_free(kappa_c);
      // DOMAIN_WALL END
    } else if (dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      mdw_eofa_matpc(spinorCheck, gauge, spinorOut, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    } else {
      errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      ax(0.25 / (kappa5 * kappa5), spinorCheck, V * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    }

  } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

    void *spinorTmp = safe_malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
    ax(0, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);

    // DOMAIN_WALL START
    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
               inv_param.mass);
      dw_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param,
               inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                  inv_param.mass);
      dw_4d_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param,
                  inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_matpc(spinorTmp, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                inv_param.mass, inv_param.b_5, inv_param.c_5);
      mdw_matpc(spinorCheck, gauge, spinorTmp, kappa_b, kappa_c, inv_param.matpc_type, 1, inv_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      host_free(kappa_b);
      host_free(kappa_c);
      // DOMAIN_WALL END
    } else if (dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      mdw_eofa_matpc(spinorTmp, gauge, spinorOut, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
      mdw_eofa_matpc(spinorCheck, gauge, spinorTmp, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);

    } else {
      errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      errorQuda("Mass normalization %s not implemented", get_mass_normalization_str(inv_param.mass_normalization));
    }

    host_free(spinorTmp);
  } else {
    errorQuda("Solution type %s not implemented", get_solution_str(inv_param.solution_type));
  }

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %9.6e, QUDA = %9.6e, host = %9.6e; (heavy-quark) tol %9.6e, QUDA = %9.6e\n",
             inv_param.tol, inv_param.true_res[src_idx], l2r, inv_param.tol_hq, inv_param.true_res_hq[src_idx]);

  return {l2r, inv_param.tol_hq};
  ;
}

std::array<double, 2> verifyWilsonTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                                QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge,
                                                void *clover, void *clover_inv, int src_idx)
{
  int vol
    = (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION ? V : Vh);

  double l2r_max = 0.0;
  if (multishift > 1) {

    // ONLY WILSON/CLOVER/TWISTED TYPES
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      errorQuda("Mass normalization %s not implemented", get_mass_normalization_str(inv_param.mass_normalization));
    }

    void *spinorTmp = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
    printfQuda("Host residuum checks: \n");
    for (int i = 0; i < inv_param.num_offset; i++) {
      ax(0, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tm_ndeg_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tmc_ndeg_matpc(spinorTmp, gauge, clover, clover_inv, spinorOutMulti[i], inv_param.kappa, inv_param.mu,
                         inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tmc_ndeg_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.mu,
                         inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tmc_matpc(spinorTmp, gauge, clover, clover_inv, spinorOutMulti[i], inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tmc_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec,
                  gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1, inv_param.cpu_prec,
                  gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
                     inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                     inv_param.cpu_prec, gauge_param);
      } else {
        printfQuda("Domain wall not supported for multi-shift\n");
        exit(-1);
      }

      axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, vol * spinor_site_size * inv_param.Ls,
           inv_param.cpu_prec);
      mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      double l2r = sqrt(nrm2 / src2);
      l2r_max = std::max(l2r, l2r_max);

      printfQuda("Shift %2d residuals: (L2 relative) tol %9.6e, QUDA = %9.6e, host = %9.6e; (heavy-quark) tol %9.6e, "
                 "QUDA = %9.6e\n",
                 i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, inv_param.tol_hq_offset[i],
                 inv_param.true_res_hq_offset[i]);
    }
    host_free(spinorTmp);

  } else {
    // Non-multishift workflow
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                 inv_param.cpu_prec, gauge_param);
        } else {
          tm_ndeg_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                      inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                  inv_param.cpu_prec, gauge_param);
        } else {
          tmc_ndeg_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                       inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else {
        errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        ax(0.5 / inv_param.kappa, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      }

    } else if (inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
      void *spinorTmp = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
      ax(0, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                 inv_param.cpu_prec, gauge_param);
          tm_mat(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 1,
                 inv_param.cpu_prec, gauge_param);
        } else {
          tm_ndeg_mat(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                      inv_param.cpu_prec, gauge_param);
          tm_ndeg_mat(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon, 1,
                      inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_mat(spinorTmp, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                  inv_param.cpu_prec, gauge_param);
          tmc_mat(spinorCheck, gauge, clover, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 1,
                  inv_param.cpu_prec, gauge_param);
        } else {
          tmc_ndeg_mat(spinorTmp, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                       inv_param.cpu_prec, gauge_param);
          tmc_ndeg_mat(spinorCheck, gauge, clover, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon, 1,
                       inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_mat(spinorTmp, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
        wil_mat(spinorCheck, gauge, spinorTmp, inv_param.kappa, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_mat(spinorTmp, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
        clover_mat(spinorCheck, gauge, clover, spinorTmp, inv_param.kappa, 1, inv_param.cpu_prec, gauge_param);
      } else {
        errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        ax(0.25 / (inv_param.kappa * inv_param.kappa), spinorCheck, vol * spinor_site_size * inv_param.Ls,
           inv_param.cpu_prec);
      }
      host_free(spinorTmp);
    } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tm_ndeg_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tmc_ndeg_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.mu,
                         inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        } else {
          tmc_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec,
                  gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                     inv_param.cpu_prec, gauge_param);
      } else {
        errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        ax(0.25 / (inv_param.kappa * inv_param.kappa), spinorCheck, vol * spinor_site_size * inv_param.Ls,
           inv_param.cpu_prec);
      }

    } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

      void *spinorTmp = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
      ax(0, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tm_ndeg_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          tmc_ndeg_matpc(spinorTmp, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.mu,
                         inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tmc_ndeg_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.mu,
                         inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tmc_matpc(spinorTmp, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tmc_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1, inv_param.cpu_prec,
                  gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                     inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                     inv_param.cpu_prec, gauge_param);
      } else {
        errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        errorQuda("Mass normalization %s not implemented", get_mass_normalization_str(inv_param.mass_normalization));
      }

      host_free(spinorTmp);
    } else {
      errorQuda("Solution type %s not implemented", get_solution_str(inv_param.solution_type));
    }

    mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);
    l2r_max = l2r;

    printfQuda(
      "Residuals: (L2 relative) tol %9.6e, QUDA = %9.6e, host = %9.6e; (heavy-quark) tol %9.6e, QUDA = %9.6e\n",
      inv_param.tol, inv_param.true_res[src_idx], l2r, inv_param.tol_hq, inv_param.true_res_hq[src_idx]);
  }

  return {l2r_max, inv_param.tol_hq};
}

double verifyWilsonTypeEigenvector(void *spinor, double _Complex lambda, int i, QudaGaugeParam &gauge_param,
                                   QudaEigParam &eig_param, void **gauge, void *clover, void *clover_inv)
{
  bool use_pc = (eig_param.use_pc == QUDA_BOOLEAN_TRUE ? true : false);
  bool normop = (eig_param.use_norm_op == QUDA_BOOLEAN_TRUE ? true : false);
  int vol = (use_pc ? Vh : V);

  QudaInvertParam inv_param = *(eig_param.invert_param);
  int Ls = inv_param.Ls;
  double mass = inv_param.mass;
  double kappa = inv_param.kappa;
  double epsilon = inv_param.epsilon;
  double mu = inv_param.mu;
  QudaPrecision cpu_prec = inv_param.cpu_prec;
  QudaTwistFlavorType twist_flavor = inv_param.twist_flavor;
  QudaMatPCType matpc_type = inv_param.matpc_type;
  int dagger = inv_param.dagger == QUDA_DAG_YES ? 1 : 0;
  int dagger_opposite = dagger == 1 ? 0 : 1;

  int spinor_length = vol * spinor_site_size * Ls;

  // ONLY WILSON/CLOVER/TWISTED/DWF TYPES
  if (eig_param.invert_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
    errorQuda("Mass normalization %s not implemented",
              get_mass_normalization_str(eig_param.invert_param->mass_normalization));
  }

  void *spinorTmp = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * Ls);
  void *spinorTmp2 = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * Ls);

  switch (dslash_type) {
  case QUDA_DOMAIN_WALL_DSLASH: {
    if (use_pc) {
      dw_matpc(spinorTmp, gauge, spinor, kappa5, matpc_type, dagger, cpu_prec, gauge_param, mass);
      if (normop)
        dw_matpc(spinorTmp2, gauge, spinorTmp, kappa5, matpc_type, dagger_opposite, cpu_prec, gauge_param, mass);
    } else {
      dw_mat(spinorTmp, gauge, spinor, kappa5, dagger, cpu_prec, gauge_param, mass);
      if (normop) dw_mat(spinorTmp2, gauge, spinorTmp, kappa5, dagger_opposite, cpu_prec, gauge_param, mass);
    }
    break;
  }
  case QUDA_DOMAIN_WALL_4D_DSLASH: {
    if (use_pc) {
      dw_4d_matpc(spinorTmp, gauge, spinor, kappa5, matpc_type, dagger, cpu_prec, gauge_param, mass);
      if (normop)
        dw_4d_matpc(spinorTmp, gauge, spinor, kappa5, matpc_type, dagger_opposite, cpu_prec, gauge_param, mass);
    } else {
      dw_4d_mat(spinorTmp, gauge, spinor, kappa5, dagger, cpu_prec, gauge_param, mass);
      if (normop) dw_4d_mat(spinorTmp, gauge, spinor, kappa5, dagger_opposite, cpu_prec, gauge_param, mass);
    }
    break;
  }
  case QUDA_MOBIUS_DWF_DSLASH: {
    double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
    double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
    for (int xs = 0; xs < Lsdim; xs++) {
      kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
      kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
    }
    if (use_pc) {
      mdw_matpc(spinorTmp, gauge, spinor, kappa_b, kappa_c, matpc_type, dagger, cpu_prec, gauge_param, mass,
                inv_param.b_5, inv_param.c_5);
      if (normop)
        mdw_matpc(spinorTmp2, gauge, spinorTmp, kappa_b, kappa_c, matpc_type, dagger_opposite, cpu_prec, gauge_param,
                  mass, inv_param.b_5, inv_param.c_5);
    } else {
      mdw_mat(spinorTmp, gauge, spinor, kappa_b, kappa_c, dagger, cpu_prec, gauge_param, mass, inv_param.b_5,
              inv_param.c_5);
      if (normop)
        mdw_mat(spinorTmp2, gauge, spinorTmp, kappa_b, kappa_c, dagger_opposite, cpu_prec, gauge_param, mass,
                inv_param.b_5, inv_param.c_5);
    }
    host_free(kappa_b);
    host_free(kappa_c);
    break;
  }
  case QUDA_MOBIUS_DWF_EOFA_DSLASH: {
    if (use_pc) {
      mdw_eofa_matpc(spinorTmp, gauge, spinor, matpc_type, dagger, cpu_prec, gauge_param, mass, inv_param.m5,
                     (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                     inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
      if (normop)
        mdw_eofa_matpc(spinorTmp2, gauge, spinorTmp, matpc_type, dagger_opposite, cpu_prec, gauge_param, mass,
                       inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1,
                       inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    } else {
      mdw_eofa_mat(spinorTmp, gauge, spinor, dagger, cpu_prec, gauge_param, mass, inv_param.m5,
                   (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                   inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
      if (normop)
        mdw_eofa_mat(spinorTmp2, gauge, spinorTmp, dagger_opposite, cpu_prec, gauge_param, mass, inv_param.m5,
                     (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                     inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    }
    break;
  }
  case QUDA_TWISTED_MASS_DSLASH: {
    if (twist_flavor != QUDA_TWIST_SINGLET) {
      if (use_pc) {
        tm_ndeg_matpc(spinorTmp, gauge, spinor, kappa, mu, epsilon, matpc_type, dagger, cpu_prec, gauge_param);
        if (normop)
          tm_ndeg_matpc(spinorTmp2, gauge, spinorTmp, kappa, mu, epsilon, matpc_type, dagger_opposite, cpu_prec,
                        gauge_param);
      } else {
        tm_ndeg_mat(spinorTmp, gauge, spinor, kappa, mu, epsilon, dagger, cpu_prec, gauge_param);
        if (normop)
          tm_ndeg_mat(spinorTmp2, gauge, spinorTmp, kappa, mu, epsilon, dagger_opposite, cpu_prec, gauge_param);
      }
    } else {
      if (use_pc) {
        tm_matpc(spinorTmp, gauge, spinor, kappa, mu, twist_flavor, matpc_type, dagger, cpu_prec, gauge_param);
        if (normop)
          tm_matpc(spinorTmp2, gauge, spinorTmp, kappa, mu, twist_flavor, matpc_type, dagger_opposite, cpu_prec,
                   gauge_param);
      } else {
        tm_mat(spinorTmp, gauge, spinor, kappa, mu, twist_flavor, dagger, cpu_prec, gauge_param);
        if (normop)
          tm_mat(spinorTmp2, gauge, spinorTmp, kappa, mu, twist_flavor, dagger_opposite, cpu_prec, gauge_param);
      }
    }
    break;
  }
  case QUDA_TWISTED_CLOVER_DSLASH: {
    if (twist_flavor != QUDA_TWIST_SINGLET) {
      if (use_pc) {
        tmc_ndeg_matpc(spinorTmp, gauge, clover, clover_inv, spinor, kappa, mu, epsilon, matpc_type, dagger, cpu_prec,
                       gauge_param);
        if (normop)
          tmc_ndeg_matpc(spinorTmp2, gauge, clover, clover_inv, spinorTmp, kappa, mu, epsilon, matpc_type,
                         dagger_opposite, cpu_prec, gauge_param);
      } else {
        tmc_ndeg_mat(spinorTmp, gauge, clover, spinor, kappa, mu, epsilon, dagger, cpu_prec, gauge_param);
        if (normop)
          tmc_ndeg_mat(spinorTmp2, gauge, clover, spinorTmp, kappa, mu, epsilon, dagger_opposite, cpu_prec, gauge_param);
      }
    } else {
      if (use_pc) {
        tmc_matpc(spinorTmp, gauge, clover, clover_inv, spinor, kappa, mu, twist_flavor, matpc_type, dagger, cpu_prec,
                  gauge_param);
        if (normop)
          tmc_matpc(spinorTmp2, gauge, clover, clover_inv, spinorTmp, kappa, mu, twist_flavor, matpc_type,
                    dagger_opposite, cpu_prec, gauge_param);
      } else {
        tmc_mat(spinorTmp, gauge, clover, spinor, kappa, mu, twist_flavor, dagger, cpu_prec, gauge_param);
        if (normop)
          tmc_mat(spinorTmp2, gauge, clover, spinorTmp, kappa, mu, twist_flavor, dagger_opposite, cpu_prec, gauge_param);
      }
    }
    break;
  }
  case QUDA_WILSON_DSLASH: {
    if (use_pc) {
      wil_matpc(spinorTmp, gauge, spinor, kappa, matpc_type, dagger, cpu_prec, gauge_param);
      if (normop) wil_matpc(spinorTmp2, gauge, spinorTmp, kappa, matpc_type, dagger_opposite, cpu_prec, gauge_param);
    } else {
      wil_mat(spinorTmp, gauge, spinor, kappa, dagger, cpu_prec, gauge_param);
      if (normop) wil_mat(spinorTmp2, gauge, spinorTmp, kappa, dagger_opposite, cpu_prec, gauge_param);
    }
    break;
  }
  case QUDA_CLOVER_WILSON_DSLASH: {
    if (use_pc) {
      clover_matpc(spinorTmp, gauge, clover, clover_inv, spinor, kappa, matpc_type, dagger, cpu_prec, gauge_param);
      if (normop)
        clover_matpc(spinorTmp2, gauge, clover, clover_inv, spinorTmp, kappa, matpc_type, dagger_opposite, cpu_prec,
                     gauge_param);
    } else {
      clover_mat(spinorTmp, gauge, clover, spinor, kappa, dagger, cpu_prec, gauge_param);
      if (normop) clover_mat(spinorTmp2, gauge, clover, spinorTmp, kappa, dagger_opposite, cpu_prec, gauge_param);
    }
    break;
  }
  default: errorQuda("Action not supported");
  }

  // Compute M * x - \lambda * x
  double nrm2, src2, l2r;
  if (normop) {
    caxpy(-lambda, spinor, spinorTmp2, spinor_length, cpu_prec);
    nrm2 = norm_2(spinorTmp2, spinor_length, cpu_prec);
  } else {
    caxpy(-lambda, spinor, spinorTmp, spinor_length, cpu_prec);
    nrm2 = norm_2(spinorTmp, spinor_length, cpu_prec);
  }

  src2 = norm_2(spinor, spinor_length, cpu_prec);
  l2r = sqrt(nrm2 / src2);

  printfQuda("Eigenvector %4d: tol %.2e, host residual = %.15e\n", i, eig_param.tol, l2r);

  host_free(spinorTmp);
  host_free(spinorTmp2);
  return l2r;
}

double verifyWilsonTypeSingularVector(void *spinor_left, void *spinor_right, double _Complex sigma, int i,
                                      QudaGaugeParam &gauge_param, QudaEigParam &eig_param, void **gauge, void *clover,
                                      void *clover_inv)
{
  bool use_pc = (eig_param.use_pc == QUDA_BOOLEAN_TRUE ? true : false);
  int vol = (use_pc ? Vh : V);

  QudaInvertParam inv_param = *(eig_param.invert_param);
  int Ls = inv_param.Ls;
  double mass = inv_param.mass;
  double kappa = inv_param.kappa;
  double epsilon = inv_param.epsilon;
  double mu = inv_param.mu;
  QudaPrecision cpu_prec = inv_param.cpu_prec;
  QudaTwistFlavorType twist_flavor = inv_param.twist_flavor;
  QudaMatPCType matpc_type = inv_param.matpc_type;
  int dagger = inv_param.dagger == QUDA_DAG_YES ? 1 : 0;

  int spinor_length = vol * spinor_site_size * Ls;

  // ONLY WILSON/CLOVER/TWISTED/DWF TYPES
  if (eig_param.invert_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
    errorQuda("Mass normalization %s not implemented",
              get_mass_normalization_str(eig_param.invert_param->mass_normalization));
  }

  void *spinorTmp = safe_malloc(vol * spinor_site_size * host_spinor_data_type_size * Ls);

  switch (dslash_type) {
  case QUDA_DOMAIN_WALL_DSLASH: {
    if (use_pc)
      dw_matpc(spinorTmp, gauge, spinor_left, kappa5, matpc_type, dagger, cpu_prec, gauge_param, mass);
    else
      dw_mat(spinorTmp, gauge, spinor_left, kappa5, dagger, cpu_prec, gauge_param, mass);
    break;
  }
  case QUDA_DOMAIN_WALL_4D_DSLASH: {
    if (use_pc)
      dw_4d_matpc(spinorTmp, gauge, spinor_left, kappa5, matpc_type, dagger, cpu_prec, gauge_param, mass);
    else
      dw_4d_mat(spinorTmp, gauge, spinor_left, kappa5, dagger, cpu_prec, gauge_param, mass);
    break;
  }
  case QUDA_MOBIUS_DWF_DSLASH: {
    double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
    double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
    for (int xs = 0; xs < Lsdim; xs++) {
      kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
      kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
    }
    if (use_pc)
      mdw_matpc(spinorTmp, gauge, spinor_left, kappa_b, kappa_c, matpc_type, dagger, cpu_prec, gauge_param, mass,
                inv_param.b_5, inv_param.c_5);
    else
      mdw_mat(spinorTmp, gauge, spinor_left, kappa_b, kappa_c, dagger, cpu_prec, gauge_param, mass, inv_param.b_5,
              inv_param.c_5);
    host_free(kappa_b);
    host_free(kappa_c);
    break;
  }
  case QUDA_MOBIUS_DWF_EOFA_DSLASH: {
    if (use_pc)
      mdw_eofa_matpc(spinorTmp, gauge, spinor_left, matpc_type, dagger, cpu_prec, gauge_param, mass, inv_param.m5,
                     (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                     inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    else
      mdw_eofa_mat(spinorTmp, gauge, spinor_left, dagger, cpu_prec, gauge_param, mass, inv_param.m5,
                   (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                   inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
    break;
  }
  case QUDA_TWISTED_MASS_DSLASH: {
    if (twist_flavor != QUDA_TWIST_SINGLET) {
      if (use_pc)
        tm_ndeg_matpc(spinorTmp, gauge, spinor_left, kappa, mu, epsilon, matpc_type, dagger, cpu_prec, gauge_param);
      else
        tm_ndeg_mat(spinorTmp, gauge, spinor_left, kappa, mu, epsilon, dagger, cpu_prec, gauge_param);
    } else {
      if (use_pc)
        tm_matpc(spinorTmp, gauge, spinor_left, kappa, mu, twist_flavor, matpc_type, dagger, cpu_prec, gauge_param);
      else
        tm_mat(spinorTmp, gauge, spinor_left, kappa, mu, twist_flavor, dagger, cpu_prec, gauge_param);
    }
    break;
  }
  case QUDA_TWISTED_CLOVER_DSLASH: {
    if (twist_flavor != QUDA_TWIST_SINGLET) {
      if (use_pc)
        tmc_ndeg_matpc(spinorTmp, gauge, clover, clover_inv, spinor_left, kappa, mu, epsilon, matpc_type, dagger,
                       cpu_prec, gauge_param);
      else
        tmc_ndeg_mat(spinorTmp, gauge, spinor_left, clover, kappa, mu, epsilon, dagger, cpu_prec, gauge_param);
    } else {
      if (use_pc)
        tmc_matpc(spinorTmp, gauge, clover, clover_inv, spinor_left, kappa, mu, twist_flavor, matpc_type, dagger,
                  cpu_prec, gauge_param);
      else
        tmc_mat(spinorTmp, gauge, spinor_left, clover, kappa, mu, twist_flavor, dagger, cpu_prec, gauge_param);
    }
    break;
  }
  case QUDA_WILSON_DSLASH: {
    if (use_pc)
      wil_matpc(spinorTmp, gauge, spinor_left, kappa, matpc_type, dagger, cpu_prec, gauge_param);
    else
      wil_mat(spinorTmp, gauge, spinor_left, kappa, dagger, cpu_prec, gauge_param);
    break;
  }
  case QUDA_CLOVER_WILSON_DSLASH: {
    if (use_pc)
      clover_matpc(spinorTmp, gauge, clover, clover_inv, spinor_left, kappa, matpc_type, dagger, cpu_prec, gauge_param);
    else
      clover_mat(spinorTmp, gauge, clover, spinor_left, kappa, dagger, cpu_prec, gauge_param);
    break;
  }
  default: errorQuda("Action not supported");
  }

  // Compute M * x_left - \sigma * x_right
  double nrm2, src2, l2r;
  caxpy(-sigma, spinor_right, spinorTmp, spinor_length, cpu_prec);
  nrm2 = norm_2(spinorTmp, spinor_length, cpu_prec);

  src2 = norm_2(spinor_left, spinor_length, cpu_prec);
  l2r = sqrt(nrm2 / src2);

  printfQuda("Singular vector pair %4d: tol %.2e, host residual = %.15e\n", i, eig_param.tol, l2r);

  host_free(spinorTmp);
  return l2r;
}

std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in, quda::ColorSpinorField &out,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int laplace3D, int src_idx)
{
  std::vector<quda::ColorSpinorField> out_vector(1);
  out_vector[0] = out;
  return verifyStaggeredInversion(in, out_vector, fat_link, long_link, inv_param, laplace3D, src_idx);
}

std::array<double, 2> verifyStaggeredInversion(quda::ColorSpinorField &in,
                                               std::vector<quda::ColorSpinorField> &out_vector,
                                               quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                               QudaInvertParam &inv_param, int laplace3D, int src_idx)
{
  int dagger = inv_param.dagger == QUDA_DAG_YES ? 1 : 0;
  double l2r_max = 0.0;
  double hqr_max = 0.0;

  // Create temporary spinors
  quda::ColorSpinorParam csParam(in);
  quda::ColorSpinorField ref(csParam);

  if (multishift > 1) {
    if (dslash_type == QUDA_LAPLACE_DSLASH) errorQuda("Multishift solves do not support the laplace operator (yet)");

    if (inv_param.solution_type != QUDA_MATPC_SOLUTION)
      errorQuda("Invalid staggered multishift solution type %d, expected QUDA_MATPC_SOLUTION", inv_param.solution_type);

    // Check the mat_pc type and make sure it's sane
    QudaParity parity = QUDA_INVALID_PARITY;
    switch (inv_param.matpc_type) {
    case QUDA_MATPC_EVEN_EVEN: parity = QUDA_EVEN_PARITY; break;
    case QUDA_MATPC_ODD_ODD: parity = QUDA_ODD_PARITY; break;
    default: errorQuda("Unexpected matpc_type %s", get_matpc_str(inv_param.matpc_type)); break;
    }

    for (int i = 0; i < multishift; i++) {
      auto &out = out_vector[i];
      double mass = 0.5 * sqrt(inv_param.offset[i]);
      stag_matpc(ref, fat_link, long_link, out, mass, 0, parity, dslash_type, laplace3D);

      mxpy(in.data(), ref.data(), in.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
      double nrm2 = norm_2(ref.data(), ref.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
      double src2 = norm_2(in.data(), in.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
      double hqr = sqrt(quda::blas::HeavyQuarkResidualNorm(out, ref).z);
      double l2r = sqrt(nrm2 / src2);

      printfQuda("%dth solution: mass=%f, ", i, mass);
      printfQuda("Shift %2d residuals: (L2 relative) tol %9.6e, QUDA = %9.6e, host = %9.6e; (heavy-quark) tol %9.6e, "
                 "QUDA = %9.6e, host = %9.6e\n",
                 i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, inv_param.tol_hq_offset[i],
                 inv_param.true_res_hq_offset[i], hqr);
      // Empirical: if the cpu residue is more than 1 order the target accuracy, then it fails to converge
      if (sqrt(nrm2 / src2) > 10 * inv_param.tol_offset[i]) {
        printfQuda("Shift %2d has empirically failed to converge\n", i);
      }

      l2r_max = std::max(l2r_max, l2r);
      hqr_max = std::max(hqr_max, hqr);
    }

  } else {
    auto &out = out_vector[0];
    double mass = inv_param.mass;
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
      stag_mat(ref, fat_link, long_link, out, mass, dagger, dslash_type, laplace3D);
    } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {
      QudaParity parity = QUDA_INVALID_PARITY;
      switch (inv_param.matpc_type) {
      case QUDA_MATPC_EVEN_EVEN: parity = QUDA_EVEN_PARITY; break;
      case QUDA_MATPC_ODD_ODD: parity = QUDA_ODD_PARITY; break;
      default: errorQuda("Unexpected matpc_type %s", get_matpc_str(inv_param.matpc_type)); break;
      }
      stag_matpc(ref, fat_link, long_link, out, mass, 0, parity, dslash_type, laplace3D);
    } else if (inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
      stag_matdag_mat(ref, fat_link, long_link, out, mass, dagger, dslash_type, laplace3D);
    } else {
      errorQuda("Invalid staggered solution type %d", inv_param.solution_type);
    }

    mxpy(in.data(), ref.data(), in.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double nrm2 = norm_2(ref.data(), ref.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double src2 = norm_2(in.data(), in.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double hqr = sqrt(quda::blas::HeavyQuarkResidualNorm(out, ref).z);
    double l2r = sqrt(nrm2 / src2);

    printfQuda("Residuals: (L2 relative) tol %9.6e, QUDA = %9.6e, host = %9.6e; (heavy-quark) tol %9.6e, QUDA = %9.6e, "
               "host = %9.6e\n",
               inv_param.tol, inv_param.true_res[src_idx], l2r, inv_param.tol_hq, inv_param.true_res_hq[src_idx], hqr);

    l2r_max = l2r;
    hqr_max = hqr;
  }

  return {l2r_max, hqr_max};
}

double verifyStaggeredTypeEigenvector(quda::ColorSpinorField &spinor, const std::vector<double _Complex> &lambda, int i,
                                      QudaEigParam &eig_param, quda::GaugeField &fat_link, quda::GaugeField &long_link,
                                      int laplace3D)
{
  QudaInvertParam &inv_param = *(eig_param.invert_param);
  int dagger = inv_param.dagger == QUDA_DAG_YES ? 1 : 0;
  bool use_pc = (eig_param.use_pc == QUDA_BOOLEAN_TRUE ? true : false);
  bool normop = (eig_param.use_norm_op == QUDA_BOOLEAN_TRUE ? true : false);
  double mass = inv_param.mass;

  // Reverse engineer a "solution_type" to help determine which host dslash needs to be applied
  QudaSolutionType sol_type = QUDA_INVALID_SOLUTION;
  if (normop) {
    if (use_pc)
      errorQuda("The normal preconditioned staggered op is not supported");
    else
      sol_type = QUDA_MATDAG_MAT_SOLUTION;
  } else {
    sol_type = use_pc ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;
  }

  // Create temporary spinors
  quda::ColorSpinorParam csParam(spinor);
  quda::ColorSpinorField ref(csParam);

  if (sol_type == QUDA_MAT_SOLUTION) {
    stag_mat(ref, fat_link, long_link, spinor, mass, dagger, dslash_type, laplace3D);
  } else if (sol_type == QUDA_MATPC_SOLUTION) {
    QudaParity parity = QUDA_INVALID_PARITY;
    switch (inv_param.matpc_type) {
    case QUDA_MATPC_EVEN_EVEN: parity = QUDA_EVEN_PARITY; break;
    case QUDA_MATPC_ODD_ODD: parity = QUDA_ODD_PARITY; break;
    default: errorQuda("Unexpected matpc_type %s", get_matpc_str(inv_param.matpc_type)); break;
    }
    stag_matpc(ref, fat_link, long_link, spinor, mass, 0, parity, dslash_type, laplace3D);
  } else if (sol_type == QUDA_MATDAG_MAT_SOLUTION) {
    stag_matdag_mat(ref, fat_link, long_link, spinor, mass, dagger, dslash_type, laplace3D);
  }

  if (laplace3D == 3) {
    auto batch_size = (spinor.VolumeCB() / spinor.X()[3]) * stag_spinor_site_size;
    auto t_offset = spinor.X()[3] * comm_coord(3);
    std::vector<double> nrm2(spinor.X()[3] * comm_dim(3), 0.0);
    std::vector<double> src2(spinor.X()[3] * comm_dim(3), 0.0);
    // Compute M * x - \lambda * x on each slice
    for (auto t = 0; t < spinor.X()[3]; t++) {
      auto t_global = t_offset + t;
      auto offset = t * batch_size * inv_param.cpu_prec;

      for (int parity = 0; parity < spinor.SiteSubset(); parity++) {
        caxpy(-lambda[t_global], static_cast<char *>(spinor.data()) + offset, static_cast<char *>(ref.data()) + offset,
              batch_size, inv_param.cpu_prec);

        nrm2[t_global] += norm_2(static_cast<char *>(ref.data()) + offset, batch_size, inv_param.cpu_prec, false);
        src2[t_global] += norm_2(static_cast<char *>(spinor.data()) + offset, batch_size, inv_param.cpu_prec, false);

        offset += spinor.VolumeCB() * stag_spinor_site_size * inv_param.cpu_prec;
      }
    }

    comm_allreduce_sum(nrm2);
    comm_allreduce_sum(src2);

    for (auto t = 0; t < spinor.X()[3] * comm_dim(3); t++) {
      auto l = ((double *)&(lambda[t]))[0];
      printfQuda("Eigenvector %4d, t = %d lambda = %15.14e: tol %.2e, host residual = %.15e\n", i, t, l, eig_param.tol,
                 sqrt(nrm2[t] / src2[t]));
    }
    auto total_nrm2 = std::accumulate(nrm2.begin(), nrm2.end(), 0);
    auto total_src2 = std::accumulate(src2.begin(), src2.end(), 0);

    return sqrt(total_nrm2 / total_src2);
  } else {
    // Compute M * x - \lambda * x
    caxpy(-lambda[0], spinor.data(), ref.data(), spinor.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double nrm2 = norm_2(ref.data(), ref.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double src2 = norm_2(spinor.data(), spinor.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);
    printfQuda("Eigenvector %4d: tol %.2e, host residual = %.15e\n", i, eig_param.tol, l2r);
    return l2r;
  }
}

double verifyStaggeredTypeSingularVector(quda::ColorSpinorField &spinor_left, quda::ColorSpinorField &spinor_right,
                                         const std::vector<double _Complex> &sigma, int i, QudaEigParam &eig_param,
                                         quda::GaugeField &fat_link, quda::GaugeField &long_link, int laplace3D)
{
  if (laplace3D == 3) errorQuda("3-d Laplace operator not supported");
  QudaInvertParam &inv_param = *(eig_param.invert_param);
  int dagger = inv_param.dagger == QUDA_DAG_YES ? 1 : 0;
  bool use_pc = (eig_param.use_pc == QUDA_BOOLEAN_TRUE ? true : false);
  double mass = inv_param.mass;

  if (use_pc) errorQuda("The SVD of the preconditioned staggered op is not supported");

  // Create temporary spinors
  quda::ColorSpinorParam csParam(spinor_left);
  quda::ColorSpinorField ref(csParam);

  // Only `mat` is used here
  stag_mat(ref, fat_link, long_link, spinor_left, mass, dagger, dslash_type, laplace3D);

  // Compute M * x_left - \sigma * x_right
  caxpy(-sigma[0], spinor_right.data(), ref.data(), spinor_right.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
  double nrm2 = norm_2(ref.data(), ref.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
  double src2 = norm_2(spinor_left.data(), spinor_left.Volume() * stag_spinor_site_size, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Singular vector pair %4d: tol %.2e, host residual = %.15e\n", i, eig_param.tol, l2r);

  return l2r;
}

double verifySpinorDistanceReweight(quda::ColorSpinorField &spinor, double alpha0, int t0)
{
  if (spinor.Precision() != QUDA_DOUBLE_PRECISION) {
    errorQuda("Spinor distance reweighting should only apply to double precision field");
  }

  alpha0 = abs(alpha0);

  quda::ColorSpinorParam csParam(spinor);
  csParam.create = QUDA_COPY_FIELD_CREATE;
  quda::ColorSpinorField spinorTmp1(csParam);
  csParam.create = QUDA_NULL_FIELD_CREATE;
  quda::ColorSpinorField spinorTmp2(csParam);

  auto *in = spinor.data<double *>();
  auto *out = spinorTmp2.data<double *>();

  for (int parity = 0; parity < spinor.SiteSubset(); parity++) {
    for (int sid = 0; sid < Vh; sid++) {
      auto offset = (parity * Vh + sid) * spinor_site_size;
      int t = Z[3] * comm_coord(3) + (int)(sid / Vsh_t);
      int nt = Z[3] * comm_dim(3);
      double weight = cosh(alpha0 * (double)((t - t0 + nt) % nt - nt / 2));
      for (auto j = 0ul; j < spinor_site_size; j++) { out[offset + j] = in[offset + j] / weight; }
    }
  }

  spinorDistanceReweight(spinorTmp1, -alpha0, t0);

  mxpy(spinorTmp1.data(), spinorTmp2.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  double nrm2 = norm_2(spinorTmp2.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  double src2 = norm_2(spinorTmp1.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  double l2r = sqrt(nrm2 / src2);
  printfQuda("Apply distance reweighting: alpha0 = %.2e, t0 = %d, host residual = %.15e\n", -alpha0, t0, l2r);

  spinorDistanceReweight(spinorTmp1, alpha0, t0);

  mxpy(spinor.data(), spinorTmp1.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  nrm2 = norm_2(spinorTmp1.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  src2 = norm_2(spinor.data(), spinor.Volume() * spinor_site_size, spinor.Precision());
  l2r = sqrt(nrm2 / src2);
  printfQuda("Remove distance reweighting: alpha0 = %.2e, t0 = %d, host residual = %.15e\n", alpha0, t0, l2r);

  return l2r;
}
