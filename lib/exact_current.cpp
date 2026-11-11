#include <vector>

#include <blas_quda.h>
#include <contract_quda.h>
#include <dirac_quda.h>
#include <exact_current.h>
#include <malloc_quda.h>
#include <spin_taste.h>
#include <timer.h>

namespace quda
{

  void exactCurrent(std::vector<ColorSpinorField> &src_even, std::vector<ColorSpinorField> &src_odd,
                    const std::vector<real_t> &eval, const std::vector<real_t> &mass, QudaInvertParam &invert_param,
                    std::vector<ColorSpinorField> &acc_even, std::vector<ColorSpinorField> &acc_odd,
                    std::vector<ColorSpinorField> &acc2_even, std::vector<ColorSpinorField> &acc2_odd)
  {
    const int n_ev = src_even.size();
    const int nmasses = mass.size();
    if (src_odd.size() != static_cast<size_t>(n_ev) || eval.size() != static_cast<size_t>(n_ev))
      errorQuda("Eigenvector and eigenvalue counts do not match");

    DiracParam diracParam;
    setDiracParam(diracParam, &invert_param, true);
    Dirac *dirac = Dirac::create(diracParam);

    invert_param.dslash_type = QUDA_COVDEV_DSLASH;
    DiracParam cDParam;
    setDiracParam(cDParam, &invert_param, false);
    GaugeCovDev myCovDev(cDParam);

    ColorSpinorParam gpuParam(src_even[0]);
    gpuParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gpuParam.x[0] *= 2;
    gpuParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField gr0(gpuParam), gr_mu(gpuParam), tmp(gpuParam), evec(gpuParam);

    // One complex per full-volume site. Even sites occupy [0, V/2), odd sites [V/2, V).
    size_t data_bytes = 2 * gr0.Volume() * gr0.Precision();
    void *d_result = pool_device_malloc(data_bytes);

    ColorSpinorParam resParam(src_even[0]);
    resParam.nColor = 1;
    resParam.nSpin = 1;
    resParam.pad = 0;
    resParam.create = QUDA_REFERENCE_FIELD_CREATE;
    resParam.v = d_result;
    ColorSpinorField res_even(resParam);
    resParam.v = static_cast<char *>(d_result) + data_bytes / 2;
    ColorSpinorField res_odd(resParam);

    ColorSpinorParam accParam(src_even[0]);
    accParam.nColor = 1;
    accParam.nSpin = 1;
    accParam.pad = 0;
    accParam.create = QUDA_ZERO_FIELD_CREATE;
    resize(acc_even, 4, accParam);
    resize(acc_odd, 4, accParam);
    if (nmasses == 3) {
      resize(acc2_even, 4, accParam);
      resize(acc2_odd, 4, accParam);
    }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    for (int i = 0; i < n_ev; i++) {
      dirac->Dslash(gr0.Odd(), src_even[i], QUDA_ODD_PARITY);

      blas::copy(evec.Even(), src_even[i]);
      blas::copy(evec.Odd(), src_odd[i]);

      real_t zscale = 0.0, zscale2 = 0.0;
      switch (nmasses) {
      case 1: zscale = 1.0 / (eval[i] + 4.0 * mass[0] * mass[0]); break;
      case 2: {
        const real_t dl = eval[i] + 4.0 * mass[0] * mass[0];
        const real_t ds = eval[i] + 4.0 * mass[1] * mass[1];
        zscale = 4.0 * (mass[1] * mass[1] - mass[0] * mass[0]) / (dl * ds);
        break;
      }
      case 3: {
        const real_t du = eval[i] + 4.0 * mass[0] * mass[0];
        const real_t dd = eval[i] + 4.0 * mass[1] * mass[1];
        const real_t ds = eval[i] + 4.0 * mass[2] * mass[2];
        zscale = 4.0 * (mass[1] * mass[1] - mass[0] * mass[0]) / (du * dd);
        zscale2 = 4.0 * (mass[2] * mass[2] - mass[0] * mass[0]) / (du * ds);
        break;
      }
      default: break;
      }

      for (int mu = 0; mu < 4; mu++) {
        myCovDev.MCD(gr_mu, gr0, mu);
        blas::axy(-1.0, gr0.Odd(), gr_mu.Odd());

        applySpinTaste(tmp, gr_mu, QUDA_SPIN_TASTE_G1);
        applySpinTaste(gr_mu, tmp, QUDA_SPIN_TASTE_G5);

        contractField(evec, gr_mu, d_result, QUDA_CONTRACT_TYPE_STAGGERED);
        blas::axpy(-zscale, res_even, acc_even[mu]);
        if (nmasses == 3) blas::axpy(-zscale2, res_even, acc2_even[mu]);

        myCovDev.MCD(gr_mu, evec, mu);
        blas::ax(-1.0, gr_mu.Odd());

        applySpinTaste(tmp, gr_mu, QUDA_SPIN_TASTE_G1);
        applySpinTaste(gr_mu, tmp, QUDA_SPIN_TASTE_G5);

        contractField(gr0, gr_mu, d_result, QUDA_CONTRACT_TYPE_STAGGERED);
        blas::axpy(+zscale, res_odd, acc_odd[mu]);
        if (nmasses == 3) blas::axpy(+zscale2, res_odd, acc2_odd[mu]);
      }
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    pool_device_free(d_result);
    delete dirac;
  }

} // namespace quda
