#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

#include "CloverForce_reference.h"
#include "TMCloverForce_reference.h"
#include "gauge_field.h"
#include "host_utils.h"
#include "misc.h"
#include "quda.h"
#include <dirac_quda.h>
#include <domain_wall_dslash_reference.h>
#include <dslash_reference.h>
#include <timer.h>
#include <wilson_dslash_reference.h>

void Gamma5_host(double *out, double *in, const int V)
{

  for (int i = 0; i < V; i++) {
    for (int c = 0; c < 3; c++) {
      for (int reim = 0; reim < 2; reim++) {
        out[i * 24 + 0 * 6 + c * 2 + reim] = in[i * 24 + 0 * 6 + c * 2 + reim];
        out[i * 24 + 1 * 6 + c * 2 + reim] = in[i * 24 + 1 * 6 + c * 2 + reim];
        out[i * 24 + 2 * 6 + c * 2 + reim] = -in[i * 24 + 2 * 6 + c * 2 + reim];
        out[i * 24 + 3 * 6 + c * 2 + reim] = -in[i * 24 + 3 * 6 + c * 2 + reim];
      }
    }
  }
}
void Gamma5_host_UKQCD(double *out, double *in, const int V)
{

  for (int i = 0; i < V; i++) {
    for (int c = 0; c < 3; c++) {
      for (int reim = 0; reim < 2; reim++) {
        out[i * 24 + 0 * 6 + c * 2 + reim] = in[i * 24 + 2 * 6 + c * 2 + reim];
        out[i * 24 + 1 * 6 + c * 2 + reim] = in[i * 24 + 3 * 6 + c * 2 + reim];
        out[i * 24 + 2 * 6 + c * 2 + reim] = in[i * 24 + 0 * 6 + c * 2 + reim];
        out[i * 24 + 3 * 6 + c * 2 + reim] = in[i * 24 + 1 * 6 + c * 2 + reim];
      }
    }
  }
}
template <typename Float> void add_mom(Float *a, Float *b, int len, double coeff)
{
  for (int i = 0; i < len; i++) { a[i] += coeff * b[i]; }
}

template <typename Float> void set_to_zero(void *oprod_)
{
  Float *oprod = (Float *)oprod_;
  for (size_t i = 0; i < V * 6 * gauge_site_size; i++) oprod[i] = 0;
}

void TMCloverForce_reference(void *h_mom, void **h_x, double *coeff, int nvector, std::array<void *, 4> gauge,
                             std::vector<char> clover, std::vector<char> clover_inv, QudaGaugeParam *gauge_param,
                             QudaInvertParam *inv_param)
{
  nvector++;
  nvector--;
  quda::ColorSpinorParam qParam;
  // constructWilsonTestSpinorParam(&qParam, inv_param, gauge_param);
  ///

  qParam.nColor = 3;
  qParam.nSpin = 4;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.setPrecision(gauge_param->cpu_prec);
  qParam.pad = 0;
  qParam.twistFlavor = inv_param->twist_flavor;
  qParam.pc_type = QUDA_4D_PC;
  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) { qParam.pc_type = QUDA_5D_PC; }
  for (int dir = 0; dir < 4; ++dir) qParam.x[dir] = gauge_param->X[dir];

  qParam.location = QUDA_CPU_FIELD_LOCATION;
  qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  // qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  qParam.create = QUDA_ZERO_FIELD_CREATE;

  quda::ColorSpinorField x(qParam);
  quda::ColorSpinorField p(qParam);
  qParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  qParam.x[0] /= 2;
  quda::ColorSpinorField tmp(qParam);

  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  qParam.v = h_x[0];
  quda::ColorSpinorField load_half(qParam);
  x.Odd() = load_half;
  qParam.create = QUDA_NULL_FIELD_CREATE;

  Gamma5_host((double *)tmp.V(), (double *)x.Odd().V(), x.Odd().VolumeCB());

  int parity = 0;
  QudaMatPCType myMatPCType = inv_param->matpc_type;
  QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  if (myMatPCType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || myMatPCType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      tmc_dslash(x.Even().V(), gauge.data(), tmp.V(), clover.data(), clover_inv.data(), inv_param->kappa, inv_param->mu,
                 inv_param->twist_flavor, parity, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
    } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      clover_dslash(x.Even().V(), gauge.data(), clover_inv.data(), tmp.V(), parity, QUDA_DAG_YES, inv_param->cpu_prec,
                    *gauge_param);
    } else {
      errorQuda("TMCloverForce_reference: dslash_type not supported\n");
    }
    Gamma5_host((double *)x.Even().V(), (double *)x.Even().V(), x.Even().VolumeCB());

    if (inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      tmc_matpc(p.Odd().V(), gauge.data(), tmp.V(), clover.data(), clover_inv.data(), inv_param->kappa, inv_param->mu,
                inv_param->twist_flavor, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
      tmc_dslash(p.Even().V(), gauge.data(), p.Odd().V(), clover.data(), clover_inv.data(), inv_param->kappa,
                 inv_param->mu, inv_param->twist_flavor, parity, myMatPCType, QUDA_DAG_NO, inv_param->cpu_prec,
                 *gauge_param);
    } else if (inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      clover_matpc(p.Odd().V(), gauge.data(), clover.data(), clover_inv.data(), tmp.V(), inv_param->kappa, myMatPCType,
                   QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
      clover_dslash(p.Even().V(), gauge.data(), clover_inv.data(), p.Odd().V(), parity, QUDA_DAG_NO,
                    inv_param->cpu_prec, *gauge_param);
    } else {
      errorQuda("TMCloverForce_reference: dslash_type not supported\n");
    }
  }
  else {
    errorQuda("TMCloverForce_reference: MATPC type not supported\n");
  }

  Gamma5_host((double *)p.Even().V(), (double *)p.Even().V(), p.Even().VolumeCB());
  Gamma5_host((double *)p.Odd().V(), (double *)p.Odd().V(), p.Odd().VolumeCB());

  double force_coeff = coeff[0];
  quda::GaugeFieldParam momparam(*gauge_param);
  // momparam.order = QUDA_QDP_GAUGE_ORDER;
  momparam.location = QUDA_CPU_FIELD_LOCATION;
  momparam.order = QUDA_MILC_GAUGE_ORDER;
  momparam.reconstruct = QUDA_RECONSTRUCT_10;
  momparam.link_type = QUDA_ASQTAD_MOM_LINKS;
  momparam.create = QUDA_ZERO_FIELD_CREATE;
  quda::cpuGaugeField mom(momparam);
  createMomCPU(mom.Gauge_p(), gauge_param->cpu_prec, 0.0);
  void *refmom = mom.Gauge_p();

  // derivative of the wilson operator it correspond to deriv_Sb(OE,...) plus  deriv_Sb(EO,...) in tmLQCD
  CloverForce_reference(refmom, gauge, x, p, force_coeff);

  // create oprod and trace field
  void *oprod;
  std::vector<char> oprod_;
  oprod_.resize(V * 6 * gauge_site_size * host_gauge_data_type_size);
  oprod = oprod_.data();

  if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION)
    set_to_zero<double>(oprod);
  else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION)
    set_to_zero<float>(oprod);
  else
    errorQuda("precision not valid\n");

  double k_csw_ov_8 = inv_param->kappa * inv_param->clover_csw / 8.0;
  size_t twist_flavor = inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? inv_param->twist_flavor : QUDA_TWIST_NO;
  double mu2
    = twist_flavor != QUDA_TWIST_NO ? 4. * inv_param->kappa * inv_param->kappa * inv_param->mu * inv_param->mu : 0.0;
  double eps2 = twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ?
    4.0 * inv_param->kappa * inv_param->kappa * inv_param->epsilon * inv_param->epsilon :
    0.0;

  // derivative of the determinant of the sw term, second term of (A12) in hep-lat/0112051,  sw_deriv(EE, mnl->mu) in tmLQCD
  computeCloverSigmaTrace_reference(oprod, clover.data(), k_csw_ov_8 * 32.0, 0, mu2, eps2, twist_flavor);

  std::vector<std::vector<double>> ferm_epsilon(nvector);
  for (int i = 0; i < nvector; i++) {
    ferm_epsilon[i].reserve(2);
    ferm_epsilon[i][0] = k_csw_ov_8 * coeff[i];
    ferm_epsilon[i][1] = k_csw_ov_8 * coeff[i] / (inv_param->kappa * inv_param->kappa);
  }
  // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus
  // sw_spinor_eo(OO,..)  in tmLQCD
  computeCloverSigmaOprod_reference(oprod, p, x, ferm_epsilon, *gauge_param);

  // create extended field
  quda::GaugeFieldParam gParamMom(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  gParamMom.link_type = QUDA_GENERAL_LINKS;
  gParamMom.create = QUDA_ZERO_FIELD_CREATE;
  gParamMom.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParamMom.reconstruct = QUDA_RECONSTRUCT_NO;
  gParamMom.geometry = QUDA_TENSOR_GEOMETRY;
  quda::cudaGaugeField cudaOprod(gParamMom);
  cudaOprod.copy_from_buffer(oprod);

  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  quda::TimeProfile profile_host("profile_host");
  quda::cudaGaugeField *cudaOprodEx = createExtendedGauge(cudaOprod, R, profile_host);

  int ghostFace[4];
  int ghost_size = 0;
  for (int i = 0; i < 4; i++) {
    ghostFace[i] = 0;
    if (quda::comm_dim_partitioned(i)) {
      ghostFace[i] = 1;
      for (int j = 0; j < 4; j++) {
        if (i == j)
          continue;
        else if (j == 0)
          ghostFace[i] *= qParam.x[j] * 2;
        else
          ghostFace[i] *= qParam.x[j];
      }
    }
    ghost_size += 2 * R[i] * ghostFace[i];
  }
  std::vector<char> oprod_ex_;
  oprod_ex_.resize((V + ghost_size) * 6 * gauge_site_size * host_gauge_data_type_size);
  void *oprod_ex = oprod_ex_.data();
  cudaOprodEx->copy_to_buffer(oprod_ex);

  // oprod = (A12) of hep-lat/0112051
  // compute the insertion of oprod in Fig.27 of hep-lat/0112051
  cloverDerivative_reference(refmom, gauge.data(), oprod_ex, QUDA_ODD_PARITY, *gauge_param);
  cloverDerivative_reference(refmom, gauge.data(), oprod_ex, QUDA_EVEN_PARITY, *gauge_param);

  add_mom((double *)h_mom, (double *)mom.Gauge_p(), 4 * V * mom_site_size, -1.0);
}