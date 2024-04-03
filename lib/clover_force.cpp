#include "clover_field.h"
#include "gauge_field.h"
#include "color_spinor_field.h"
#include "momentum.h"
#include "blas_quda.h"
#include "dirac_quda.h"

namespace quda
{

  void computeCloverForce(GaugeField &mom, const GaugeField &gaugeEx, const GaugeField &gauge,
                          const CloverField &clover, cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &x0,
                          const std::vector<double> &coeff, const std::vector<array<double, 2>> &epsilon,
                          double sigma_coeff, bool detratio, QudaInvertParam &inv_param)
  {
    if (inv_param.matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC && inv_param.matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("MatPC type %d not supported", inv_param.matpc_type);

    QudaParity parity = inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
    QudaParity other_parity = static_cast<QudaParity>(1 - parity);
    bool dagger = inv_param.dagger;
    bool not_dagger = static_cast<QudaDagType>(1 - inv_param.dagger);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, true);
    Dirac *dirac = Dirac::create(diracParam);

    ColorSpinorParam csParam(x[0]);
    csParam.create = QUDA_NULL_FIELD_CREATE;
    std::vector<ColorSpinorField> p(x.size());
    for (auto i = 0u; i < p.size(); i++) p[i] = ColorSpinorField(csParam);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    for (auto i = 0u; i < x.size(); i++) {
      gamma5(p[i][parity], x[i][parity]);
      if (dagger) dirac->Dagger(QUDA_DAG_YES);
      dirac->Dslash(x[i][other_parity], p[i][parity], other_parity);
      // want to apply \hat Q_{-} = \hat M_{+}^\dagger \gamma_5 to get Y_o
      dirac->M(p[i][parity], p[i][parity]); // this is the odd part of Y
      if (dagger) dirac->Dagger(QUDA_DAG_NO);

      if(inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET){
        // FIXME: here we used the already existing ca_lambda_max member of inv_param. Maybe it's better to create a new parameter for this purpose
        blas::ax(1.0/inv_param.ca_lambda_max, p[i][parity]);
        ApplyTau(x[i][other_parity], x[i][other_parity], 1);
        ApplyTau(p[i][parity], p[i][parity], 1);
        Complex a(0.0,-inv_param.offset[i] );
        blas::caxpy( a, x[i][parity], p[i][parity]);

      }

      gamma5(x[i][other_parity], x[i][other_parity]);
      if (detratio) blas::xpy(x0[i][parity], p[i][parity]);
      
      if (not_dagger || inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET){
        dirac->Dagger(QUDA_DAG_YES);
        gamma5(p[i][parity], p[i][parity]);
      }
      dirac->Dslash(p[i][other_parity], p[i][parity], other_parity); // and now the even part of Y
      if (not_dagger || inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET){
        dirac->Dagger(QUDA_DAG_NO);
        gamma5(p[i][other_parity], p[i][other_parity]);
        gamma5(p[i][parity], p[i][parity]);
        ApplyTau(p[i][other_parity], p[i][other_parity], 1);
      }
      // up to here x.odd match X.odd in tmLQCD and p.odd=-Y.odd of tmLQCD
      // x.Even= X.Even.tmLQCD/kappa and p.Even=-Y.Even.tmLQCD/kappa
//////////////////////////////////////////////
 /// print from QUDA
 //////////////////////////////////////////////
 csParam.x[0]/=2;
  int T = csParam.x[3];
  int LX = csParam.x[0]*2;
  int LY = csParam.x[1];
  int LZ = csParam.x[2];
  
  ColorSpinorParam pParam(csParam);
 csParam.x[0]*=2;
  pParam.create = QUDA_NULL_FIELD_CREATE;
  pParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  // QudaSiteSubset_s  save_site_sub = qParam.siteSubset;
  pParam.siteSubset=QUDA_PARITY_SITE_SUBSET;
  // qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  ColorSpinorField tmp_quark(pParam);
  double *h_tmp=(double*) malloc(sizeof(double)*(T*LX*LY*LZ*24*2));
  printf("quda  (%d %d %d %d)\n",T,LX,LY,LZ);
  printf("MARCO: twist_flavor  %d\n", inv_param.twist_flavor);
  int Vh=LX*LY*LZ*T/2;

  // tmp_quark=x[i][parity];
  // tmp_quark=p[i][parity];
  // tmp_quark=x[i][other_parity];
  tmp_quark=p[i][other_parity];
  printf("quda copy end  (%d %d %d %d)\n",T,LX,LY,LZ);
  double kappa = inv_param.kappa;
  printf("check kappa = %g\n",kappa);
  tmp_quark.copy_to_buffer(h_tmp);
  // x[i][parity].copy_to_buffer(h_tmp);
  for(int ud = 0; ud < 2; ud++){
    // double *sp;
    // if (ud==0) sp=h_tmp+24*Vh;
    // if (ud==1) sp=h_tmp;
  for (int x0 = 0; x0 < T; x0++){
    for (int x1 = 0; x1 < LX; x1++){
      for (int x2 = 0; x2 < LY; x2++){
        for (int x3 = 0; x3 < LZ; x3++) {
          const int q_eo_idx = (x1 + LX * x2 + LY * LX * x3 + LZ * LY * LX * x0) / 2;
          const int oddBit = (x0 + x1 + x2 + x3) & 1;
          const int change_sign[4] = {-1, 1, 1, -1};
          const int change_spin[4] = {3, 2, 1, 0};
          if (oddBit == 0) {   // parity=1  other_parity=0
            double c=(oddBit==0) ?kappa: 1;
            
            for (int q_spin = 0; q_spin < 4; q_spin++) {
              for (int col = 0; col < 3; col++) {
                
                printf("MARCOfrom QUDA (%-3d%-3d%-3d%-3d),  %-3d%-3d, %-3d%-3d  %-20g   %-20g\n", x0, x1, x2, x3, q_spin, col, i, ud,
                       change_sign[q_spin]*h_tmp[0 + 2* (q_eo_idx + Vh*(ud+2 * ( col+3*change_spin[q_spin] )) )]*c,
                       change_sign[q_spin]*h_tmp[1 + 2* (q_eo_idx + Vh*(ud+2 * ( col+3*change_spin[q_spin] )) )]*c
                       );
              }
            }
          }
        }
      }
    }
  }
  }
  free(h_tmp);
  printf("end printing\n");
//////////////////////////////////////

      // the gamma5 application in tmLQCD is done inside deriv_Sb
      gamma5(p[i], p[i]);
    }

    delete dirac;

    // create oprod and trace field
    GaugeFieldParam param(mom);
    param.link_type = QUDA_GENERAL_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    GaugeField force(param);
    param.geometry = QUDA_TENSOR_GEOMETRY;
    GaugeField oprod(param);

    // derivative of the wilson operator it correspond to deriv_Sb(OE,...) plus  deriv_Sb(EO,...) in tmLQCD
    computeCloverForce(force, gauge, x, p, coeff);
    // derivative of the determinant of the sw term, second term of (A12) in hep-lat/0112051,  sw_deriv(EE, mnl->mu) in tmLQCD
    // if (!detratio) computeCloverSigmaTrace(oprod, clover, sigma_coeff, other_parity);

    // derivative of pseudofermion sw term, first term term of (A12) in hep-lat/0112051,  sw_spinor_eo(EE,..) plus
    // sw_spinor_eo(OO,..)  in tmLQCD
    if ( inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET){
      for (auto i = 0u; i < x.size(); i++) {
        ApplyTau(p[i][parity],p[i][parity],1);
        ApplyTau(p[i][other_parity],p[i][other_parity],1);
      }
    }
    // computeCloverSigmaOprod(oprod, inv_param.dagger == QUDA_DAG_YES ? p : x, inv_param.dagger == QUDA_DAG_YES ? x : p,
    //                         epsilon);

    // oprod = (A12) of hep-lat/0112051
    // compute the insertion of oprod in Fig.27 of hep-lat/0112051
    // cloverDerivative(force, gaugeEx, oprod, 1.0);

    updateMomentum(mom, -1.0, force, "clover");

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
