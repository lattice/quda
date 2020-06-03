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
void verifyInversion(void *spinorOut, void *spinorIn, void *spinorCheck, QudaGaugeParam &gauge_param,
                     QudaInvertParam &inv_param, void **gauge, void *clover, void *clover_inv)
{
  void **spinorOutMulti = nullptr;
  verifyInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge, clover, clover_inv);
}

void verifyInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                     QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                     void *clover_inv)
{

  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    verifyDomainWallTypeInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge,
                                  clover, clover_inv);
  } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH
             || dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    verifyWilsonTypeInversion(spinorOut, spinorOutMulti, spinorIn, spinorCheck, gauge_param, inv_param, gauge, clover,
                              clover_inv);
  } else {
    errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
  }
}

void verifyDomainWallTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                   QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                                   void *clover_inv)
{
  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_mat(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.dagger, inv_param.cpu_prec, gauge_param,
              inv_param.mass, inv_param.b_5, inv_param.c_5);
      free(kappa_b);
      free(kappa_c);
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
      double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_matpc(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      free(kappa_b);
      free(kappa_c);
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

    void *spinorTmp = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
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
      double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_matpc(spinorTmp, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                inv_param.mass, inv_param.b_5, inv_param.c_5);
      mdw_matpc(spinorCheck, gauge, spinorTmp, kappa_b, kappa_c, inv_param.matpc_type, 1, inv_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      free(kappa_b);
      free(kappa_c);
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

    free(spinorTmp);
  } else {
    errorQuda("Solution type %s not implemented", get_solution_str(inv_param.solution_type));
  }

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n", inv_param.tol,
             inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);
}

void verifyWilsonTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                               QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                               void *clover_inv)
{
  if (multishift > 1) {
    // ONLY WILSON/CLOVER/TWISTED TYPES
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      errorQuda("Mass normalization %s not implemented", get_mass_normalization_str(inv_param.mass_normalization));
    }

    void *spinorTmp = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
    printfQuda("Host residuum checks: \n");
    for (int i = 0; i < inv_param.num_offset; i++) {
      ax(0, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh * spinor_site_size;
          void *out0 = spinorCheck;
          void *out1 = (char *)out0 + tm_offset * cpu_prec;

          void *tmp0 = spinorTmp;
          void *tmp1 = (char *)tmp0 + tm_offset * cpu_prec;

          void *in0 = spinorOutMulti[i];
          void *in1 = (char *)in0 + tm_offset * cpu_prec;

          tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
          errorQuda("Twisted mass solution type %s not supported", get_flavor_str(inv_param.twist_flavor));
        tmc_matpc(spinorTmp, gauge, spinorOutMulti[i], clover, clover_inv, inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
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

      axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, Vh * spinor_site_size, inv_param.cpu_prec);
      mxpy(spinorIn, spinorCheck, Vh * spinor_site_size, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, Vh * spinor_site_size, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, Vh * spinor_site_size, inv_param.cpu_prec);
      double l2r = sqrt(nrm2 / src2);

      printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n", i,
                 inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, inv_param.tol_hq_offset[i],
                 inv_param.true_res_hq_offset[i]);
    }
    free(spinorTmp);

  } else {
    // Non-multishift workflow
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                 inv_param.cpu_prec, gauge_param);
        } else {
          int tm_offset = V * spinor_site_size;
          void *evenOut = spinorCheck;
          void *oddOut = (char *)evenOut + tm_offset * cpu_prec;

          void *evenIn = spinorOut;
          void *oddIn = (char *)evenIn + tm_offset * cpu_prec;

          tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                      inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        tmc_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
                inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else {
        errorQuda("Unsupported dslash_type=%s", get_dslash_str(dslash_type));
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
          ax(0.5 / inv_param.kappa, spinorCheck, 2 * V * spinor_site_size, inv_param.cpu_prec);
          // CAREFULL
        } else {
          ax(0.5 / inv_param.kappa, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);
        }
      }

    } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh * spinor_site_size;
          void *out0 = spinorCheck;
          void *out1 = (char *)out0 + tm_offset * cpu_prec;

          void *in0 = spinorOut;
          void *in1 = (char *)in0 + tm_offset * cpu_prec;

          tm_ndeg_matpc(out0, out1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
          errorQuda("Twisted mass solution type %s not supported", get_flavor_str(inv_param.twist_flavor));
        tmc_matpc(spinorCheck, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
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
        ax(0.25 / (inv_param.kappa * inv_param.kappa), spinorCheck, Vh * spinor_site_size, inv_param.cpu_prec);
      }

    } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

      void *spinorTmp = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
      ax(0, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh * spinor_site_size;
          void *out0 = spinorCheck;
          void *out1 = (char *)out0 + tm_offset * cpu_prec;

          void *tmp0 = spinorTmp;
          void *tmp1 = (char *)tmp0 + tm_offset * cpu_prec;

          void *in0 = spinorOut;
          void *in1 = (char *)in0 + tm_offset * cpu_prec;

          tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        } else {
          tm_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
        }
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
          errorQuda("Twisted mass solution type %s not supported", get_flavor_str(inv_param.twist_flavor));
        tmc_matpc(spinorTmp, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
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

      free(spinorTmp);
    } else {
      errorQuda("Solution type %s not implemented", get_solution_str(inv_param.solution_type));
    }

    int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
    mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
               inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);
  }
}

void verifyStaggeredInversion(quda::ColorSpinorField *tmp, quda::ColorSpinorField *ref, quda::ColorSpinorField *in,
                              quda::ColorSpinorField *out, double mass, void *qdp_fatlink[], void *qdp_longlink[],
                              void **ghost_fatlink, void **ghost_longlink, QudaGaugeParam &gauge_param,
                              QudaInvertParam &inv_param, int shift)
{

  switch (test_type) {
  case 0: // full parity solution, full parity system
  case 1: // full parity solution, solving EVEN EVEN prec system
  case 2: // full parity solution, solving ODD ODD prec system

    // In QUDA, the full staggered operator has the sign convention
    // {{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
    // have the minus sign. Passing in QUDA_DAG_YES solves this
    // discrepancy.
    staggeredDslash(reinterpret_cast<quda::cpuColorSpinorField *>(&ref->Even()), qdp_fatlink, qdp_longlink,
                    ghost_fatlink, ghost_longlink, reinterpret_cast<quda::cpuColorSpinorField *>(&out->Odd()),
                    QUDA_EVEN_PARITY, QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    staggeredDslash(reinterpret_cast<quda::cpuColorSpinorField *>(&ref->Odd()), qdp_fatlink, qdp_longlink,
                    ghost_fatlink, ghost_longlink, reinterpret_cast<quda::cpuColorSpinorField *>(&out->Even()),
                    QUDA_ODD_PARITY, QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);

    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      xpay(out->V(), kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
      ax(0.5 / kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    } else {
      axpy(2 * mass, out->V(), ref->V(), ref->Length(), gauge_param.cpu_prec);
    }
    break;

  case 3: // even parity solution, solving EVEN system
  case 4: // odd parity solution, solving ODD system
  case 5: // multi mass CG, even parity solution, solving EVEN system
  case 6: // multi mass CG, odd parity solution, solving ODD system

    staggeredMatDagMat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, out, mass, 0, inv_param.cpu_prec,
                       gauge_param.cpu_prec, tmp,
                       (test_type == 3 || test_type == 5) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY, dslash_type);
    break;
  }

  int len = 0;
  if (solution_type == QUDA_MAT_SOLUTION || solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    len = V;
  } else {
    len = Vh;
  }

  mxpy(in->V(), ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  double nrm2 = norm_2(ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  double src2 = norm_2(in->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  double hqr = sqrt(quda::blas::HeavyQuarkResidualNorm(*out, *ref).z);
  double l2r = sqrt(nrm2 / src2);

  if (multishift == 1) {
    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
               inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);
  } else {
    printfQuda(
      "Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
      shift, inv_param.tol_offset[shift], inv_param.true_res_offset[shift], l2r, inv_param.tol_hq_offset[shift],
      inv_param.true_res_hq_offset[shift], hqr);
    // Empirical: if the cpu residue is more than 1 order the target accuracy, then it fails to converge
    if (sqrt(nrm2 / src2) > 10 * inv_param.tol_offset[shift]) {
      printfQuda("Shift %d has empirically failed to converge\n", shift);
    }
  }
}
