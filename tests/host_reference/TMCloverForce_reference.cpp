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

  // Gamma5_host_UKQCD((double *)tmp.V(), (double *)x.Odd().V(), (qParam.x[0] * qParam.x[1] * qParam.x[2] * qParam.x[3]) );
  Gamma5_host((double *)tmp.V(), (double *)x.Odd().V(), x.Odd().VolumeCB());

  // dirac->dslash
  int parity = 0;
  // QudaMatPCType myMatPCType = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
  QudaMatPCType myMatPCType = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  // QudaMatPCType myMatPCType = QUDA_MATPC_ODD_ODD;
  // QudaMatPCType myMatPCType = QUDA_MATPC_EVEN_EVEN;

  printf("kappa=%g\n", inv_param->kappa);
  printf("mu=%g\n", inv_param->mu);
  printf("twist_flavour=%d\n", inv_param->twist_flavor);
  printf("matpc=%d\n", myMatPCType);
  tmc_dslash(x.Even().V(), gauge.data(), tmp.V(), clover.data(), clover_inv.data(), inv_param->kappa, inv_param->mu,
             inv_param->twist_flavor, parity, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);

  Gamma5_host((double *)x.Even().V(), (double *)x.Even().V(), x.Even().VolumeCB());

  tmc_matpc(p.Odd().V(), gauge.data(), tmp.V(), clover.data(), clover_inv.data(), inv_param->kappa, inv_param->mu,
            inv_param->twist_flavor, myMatPCType, QUDA_DAG_YES, inv_param->cpu_prec, *gauge_param);
  parity = 0;
  tmc_dslash(p.Even().V(), gauge.data(), p.Odd().V(), clover.data(), clover_inv.data(), inv_param->kappa, inv_param->mu,
             inv_param->twist_flavor, parity, myMatPCType, QUDA_DAG_NO, inv_param->cpu_prec, *gauge_param);

  Gamma5_host((double *)p.Even().V(), (double *)p.Even().V(), p.Even().VolumeCB());
  Gamma5_host((double *)p.Odd().V(), (double *)p.Odd().V(), p.Odd().VolumeCB());

  // check
  // int T = qParam.x[3];
  // int LX = qParam.x[0] * 2;
  // int LY = qParam.x[1];
  // int LZ = qParam.x[2];
  // load_half = p.Even();
  // printf("reference  (%d %d %d %d)\n",T,LX,LY,LZ);
  // for (int x0 = 0; x0 < T; x0++) {
  //   for (int x1 = 0; x1 < LX; x1++) {
  //     for (int x2 = 0; x2 < LY; x2++) {
  //       for (int x3 = 0; x3 < LZ; x3++) {
  //         const int q_eo_idx = (x1 + LX * x2 + LY * LX * x3 + LZ * LY * LX * x0) / 2;
  //         const int oddBit = (x0 + x1 + x2 + x3) & 1;
  //         if (oddBit == 0) {
  //           for (int q_spin = 0; q_spin < 4; q_spin++) {
  //             for (int col = 0; col < 3; col++) {
  //               if(getRankVerbosity()){
  //               printf("MARCOreference  (%d %d %d %d),  %d %d,    %g  %g\n", x0, x1, x2, x3, q_spin, col,
  //                      ((double *)load_half.V())[24 * q_eo_idx + 6 * q_spin + 2 * col + 0],
  //                      ((double *)load_half.V())[24 * q_eo_idx + 6 * q_spin + 2 * col + 1]);
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
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

  // FIXME: invert x,p here and in the device version
  CloverForce_reference(refmom, gauge, p, x, force_coeff);

  // create oprod and trace field

  // momparam.link_type = QUDA_GENERAL_LINKS;
  momparam.order = QUDA_QDP_GAUGE_ORDER;
  momparam.geometry = QUDA_TENSOR_GEOMETRY;

  // quda::cudaGaugeField oprod(gParamMom);
  // quda::cpuGaugeField oprod(momparam);
  // std::array<void *, 6> oprod; // like a gauge field
  // std::vector<char> oprod_;
  // for (int i = 0; i < 6; i++) {
  //   oprod_.resize(sizeof(double) * (V * 6 * gauge_site_size * host_gauge_data_type_size));
  //   oprod[i] = oprod_.data() + i * V * gauge_site_size * host_gauge_data_type_size;
  // }
  void *oprod;
  std::vector<char> oprod_;
  oprod_.resize(sizeof(double) * (V * 6 * gauge_site_size * host_gauge_data_type_size));
  oprod = oprod_.data();

  double k_csw_ov_8 = inv_param->kappa * inv_param->clover_csw / 8.0;
  size_t twist_flavor = inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? inv_param->twist_flavor : QUDA_TWIST_NO;
  double mu2
    = twist_flavor != QUDA_TWIST_NO ? 4. * inv_param->kappa * inv_param->kappa * inv_param->mu * inv_param->mu : 0.0;
  double eps2 = twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ?
    4.0 * inv_param->kappa * inv_param->kappa * inv_param->epsilon * inv_param->epsilon :
    0.0;

  // computeCloverSigmaTrace_reference(oprod.Gauge_p(), clover_inv.data(), k_csw_ov_8 * 32.0, 0, mu2, eps2);
  computeCloverSigmaTrace_reference(oprod, clover.data(), k_csw_ov_8 * 32.0, 0, mu2, eps2);

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
  oprod_ex_.resize(sizeof(double) * ((V + ghost_size) * 6 * gauge_site_size * host_gauge_data_type_size));
  void *oprod_ex = oprod_ex_.data();
  cudaOprodEx->copy_to_buffer(oprod_ex);

  cloverDerivative_reference(refmom, gauge.data(), oprod_ex, 1.0, QUDA_ODD_PARITY, *gauge_param);
  cloverDerivative_reference(refmom, gauge.data(), oprod_ex, 1.0, QUDA_EVEN_PARITY, *gauge_param);

  add_mom((double *)h_mom, (double *)mom.Gauge_p(), 4 * V * mom_site_size, -1.0);
}