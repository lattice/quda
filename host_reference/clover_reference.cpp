#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>

#include <util_quda.h>
#include <host_utils.h>
#include <wilson_dslash_reference.h>

/**
   @brief Apply the clover matrix field
   @param[out] out Result field (single parity)
   @param[in] clover Clover-matrix field (full field)
   @param[in] in Input field (single parity)
   @param[in] parity Parity to which we are applying the clover field
 */
template <typename sFloat, typename cFloat>
void cloverReference(sFloat *out, cFloat *clover, sFloat *in, int parity) {
  int nSpin = 4;
  int nColor = 3;
  int N = nColor * nSpin / 2;
  int chiralBlock = N + 2*(N-1)*N/2;

  for (int i=0; i<Vh; i++) {
    std::complex<sFloat> *In = reinterpret_cast<std::complex<sFloat>*>(&in[i*nSpin*nColor*2]);
    std::complex<sFloat> *Out = reinterpret_cast<std::complex<sFloat>*>(&out[i*nSpin*nColor*2]);
  
    for (int chi=0; chi<nSpin/2; chi++) {
      cFloat *D = &clover[((parity*Vh + i)*2 + chi)*chiralBlock];
      std::complex<cFloat> *L = reinterpret_cast<std::complex<cFloat>*>(&D[N]);

      for (int s_col=0; s_col<nSpin/2; s_col++) { // 2 spins per chiral block
	for (int c_col=0; c_col<nColor; c_col++) {
	  const int col = s_col * nColor + c_col;
	  const int Col = chi*N + col;
	  Out[Col] = 0.0;

	  for (int s_row=0; s_row<nSpin/2; s_row++) { // 2 spins per chiral block
	    for (int c_row=0; c_row<nColor; c_row++) {
	      const int row = s_row * nColor + c_row;
	      const int Row = chi*N + row;

	      if (row == col) {
		Out[Col] += D[row] * In[Row];
	      } else if (col < row) {
		int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
		Out[Col] += conj(L[k]) * In[Row];
	      } else if (row < col) {
		int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;		
		Out[Col] += L[k] * In[Row];
	      }
	    }
	  }

	}
      }
      
    }

  }

}

void apply_clover(void *out, void *clover, void *in, int parity, QudaPrecision precision) {

  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
    cloverReference(static_cast<double*>(out), static_cast<double*>(clover), static_cast<double*>(in), parity);
    break;
  case QUDA_SINGLE_PRECISION:
    cloverReference(static_cast<float*>(out), static_cast<float*>(clover), static_cast<float*>(in), parity);
    break;
  default:
    errorQuda("Unsupported precision %d", precision);
  }

}

void clover_dslash(void *out, void **gauge, void *clover, void *in, int parity,
		   int dagger, QudaPrecision precision, QudaGaugeParam &param) {
  void *tmp = malloc(Vh * spinor_site_size * precision);

  wil_dslash(tmp, gauge, in, parity, dagger, precision, param);
  apply_clover(out, clover, tmp, parity, precision);

  free(tmp);
}

// Apply the even-odd preconditioned Wilson-clover operator
void clover_matpc(void *out, void **gauge, void *clover, void *clover_inv, void *in, double kappa, 
		  QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  double kappa2 = -kappa*kappa;
  void *tmp = malloc(Vh * spinor_site_size * precision);

  switch(matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(tmp, gauge, in, 1, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 1, precision);
      wil_dslash(tmp, gauge, out, 0, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 0, precision);
    } else {
      apply_clover(tmp, clover_inv, in, 0, precision);
      wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
      apply_clover(tmp, clover_inv, out, 1, precision);
      wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 1, precision);
    wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 0, precision);
    xpay(tmp, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(tmp, gauge, in, 0, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 0, precision);
      wil_dslash(tmp, gauge, out, 1, dagger, precision, gauge_param);
      apply_clover(out, clover_inv, tmp, 1, precision);
    } else {
      apply_clover(tmp, clover_inv, in, 1, precision);
      wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
      apply_clover(tmp, clover_inv, out, 0, precision);
      wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 0, precision);
    wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 1, precision);
    xpay(tmp, kappa2, out, Vh * spinor_site_size, precision);
    break;
  default:
    errorQuda("Unsupoorted matpc=%d", matpc_type);
  }

  free(tmp);
}

// Apply the full Wilson-clover operator
void clover_mat(void *out, void **gauge, void *clover, void *in, double kappa, 
		int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  void *tmp = malloc(V * spinor_site_size * precision);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmpEven = tmp;
  void *tmpOdd = (char *)tmp + Vh * spinor_site_size * precision;

  // Odd part
  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  apply_clover(tmpOdd, clover, inOdd, 1, precision);

  // Even part
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);
  apply_clover(tmpEven, clover, inEven, 0, precision);

  // lastly apply the kappa term
  xpay(tmp, -kappa, out, V * spinor_site_size, precision);

  free(tmp);
}

void applyTwist(void *out, void *in, void *tmpH, double a, QudaPrecision precision) {
  switch (precision) {
  case QUDA_DOUBLE_PRECISION:
    for(int i = 0; i < Vh; i++)
      for(int s = 0; s < 4; s++) {
        double a5 = ((s / 2) ? -1.0 : +1.0) * a;
        for(int c = 0; c < 3; c++) {
          ((double *) out)[i * 24 + s * 6 + c * 2 + 0] = ((double *) tmpH)[i * 24 + s * 6 + c * 2 + 0] - a5*((double *) in)[i * 24 + s * 6 + c * 2 + 1];
          ((double *) out)[i * 24 + s * 6 + c * 2 + 1] = ((double *) tmpH)[i * 24 + s * 6 + c * 2 + 1] + a5*((double *) in)[i * 24 + s * 6 + c * 2 + 0];
        }
      }
    break;
  case QUDA_SINGLE_PRECISION:
    for(int i = 0; i < Vh; i++)
      for(int s = 0; s < 4; s++) {
        float a5 = ((s / 2) ? -1.0 : +1.0) * a;
        for(int c = 0; c < 3; c++) {
          ((float *) out)[i * 24 + s * 6 + c * 2 + 0] = ((float *) tmpH)[i * 24 + s * 6 + c * 2 + 0] - a5*((float *) in)[i * 24 + s * 6 + c * 2 + 1];
          ((float *) out)[i * 24 + s * 6 + c * 2 + 1] = ((float *) tmpH)[i * 24 + s * 6 + c * 2 + 1] + a5*((float *) in)[i * 24 + s * 6 + c * 2 + 0];
        }
      }
    break;
  default:
    errorQuda("Unsupported precision %d", precision);
  }
}

// out = x - i*a*gamma_5 Clov *in  =
void twistClover(void *out, void *in, void *x, void *clover, const double a, int dagger, int parity,
                 QudaPrecision precision)
{
  void *tmp = malloc(Vh * spinor_site_size * precision);

  // tmp1 = Clov in
  apply_clover(tmp, clover, in, parity, precision);
  applyTwist(out, tmp, x, (dagger ? -a : a), precision);
  free(tmp);
}

// Apply (C + i*a*gamma_5)/(C^2 + a^2)
void twistCloverGamma5(void *out, void *in, void *clover, void *cInv, const int dagger, const double kappa, const double mu,
		       const QudaTwistFlavorType flavor, const int parity, QudaTwistGamma5Type twist, QudaPrecision precision) {
  void *tmp1 = malloc(Vh * spinor_site_size * precision);
  void *tmp2 = malloc(Vh * spinor_site_size * precision);

  double a = 0.0;

  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu * flavor;

    if (dagger) a *= -1.0;

    apply_clover(tmp1, clover, in, parity, precision);
    applyTwist(out, in, tmp1, a, precision);
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    a = -2.0 * kappa * mu * flavor;

    if (dagger) a *= -1.0;

    apply_clover(tmp1, clover, in, parity, precision);
    applyTwist(tmp2, in, tmp1, a, precision);
    apply_clover(out, cInv, tmp2, parity, precision);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  free(tmp2);
  free(tmp1);
}

void tmc_dslash(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu, QudaTwistFlavorType flavor,
		int parity, QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &param) {
  void *tmp1 = malloc(Vh * spinor_site_size * precision);
  void *tmp2 = malloc(Vh * spinor_site_size * precision);

  if (dagger) {
    twistCloverGamma5(tmp1, in, clover, cInv, dagger, kappa, mu, flavor, 1-parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      wil_dslash(tmp2, gauge, tmp1, parity, dagger, precision, param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      wil_dslash(out, gauge, tmp1, parity, dagger, precision, param);
    } 
  } else {
    wil_dslash(tmp1, gauge, in, parity, dagger, precision, param);
    twistCloverGamma5(out, tmp1, clover, cInv, dagger, kappa, mu, flavor, parity, QUDA_TWIST_GAMMA5_INVERSE, precision);
  }

  free(tmp2);
  free(tmp1);
}

// Apply the full twisted-clover operator
void tmc_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu,
	     QudaTwistFlavorType flavor, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  void *tmp = malloc(V * spinor_site_size * precision);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmpEven = tmp;
  void *tmpOdd = (char *)tmp + Vh * spinor_site_size * precision;

  // Odd part
  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  twistCloverGamma5(tmpOdd, inOdd, clover, NULL, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // Even part
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);
  twistCloverGamma5(tmpEven, inEven, clover, NULL, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // lastly apply the kappa term
  xpay(tmp, -kappa, out, V * spinor_site_size, precision);

  free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void tmc_matpc(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu, QudaTwistFlavorType flavor,
              QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  double kappa2 = -kappa*kappa;

  void *tmp1 = malloc(Vh * spinor_site_size * precision);
  void *tmp2 = malloc(Vh * spinor_site_size * precision);

  switch(matpc_type) {
  case QUDA_MATPC_EVEN_EVEN:
    if (!dagger) {
      wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp2, gauge, tmp1, 0, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      twistCloverGamma5(out, in, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out, 1, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out, gauge, tmp2, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, in, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp2, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD:
    if (!dagger) {
      wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp1, out, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp2, gauge, tmp1, 1, dagger, precision, gauge_param);
      twistCloverGamma5(out, tmp2, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else {
      twistCloverGamma5(out, in, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp1, gauge, out, 0, dagger, precision, gauge_param);
      twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    }
    xpay(in, kappa2, out, Vh * spinor_site_size, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(tmp1, gauge, in, 0, dagger, precision, gauge_param);
    twistCloverGamma5(tmp2, tmp1, clover, cInv, dagger, kappa, mu, flavor, 0, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(out, gauge, tmp2, 1, dagger, precision, gauge_param);
    twistCloverGamma5(tmp1, in, clover, cInv, dagger, kappa, mu, flavor, 1, QUDA_TWIST_GAMMA5_DIRECT, precision);
    xpay(tmp1, kappa2, out, Vh * spinor_site_size, precision);
    break;
  default:
    errorQuda("Unsupported matpc=%d", matpc_type);
  }

  free(tmp2);
  free(tmp1);
}

// Apply the full twisted-clover operator
//   for now   [  A             -k D            ]
//             [ -k D    A(1 - i mu gamma_5 A)  ]

void cloverHasenbuchTwist_mat(void *out, void **gauge, void *clover, void *in, double kappa, double mu, int dagger,
                              QudaPrecision precision, QudaGaugeParam &gauge_param, QudaMatPCType matpc_type)
{

  // out = CloverMat in
  clover_mat(out, gauge, clover, in, kappa, dagger, precision, gauge_param);

  bool asymmetric = (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;

  if (asymmetric) {
    // Unprec op for asymmetric prec op:
    // apply a simple twist

    // out_parity = out_parity -/+ i mu gamma_5
    if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // out_e = out_e  -/+ i mu gamma5 in_e
      applyTwist(outEven, inEven, outEven, (dagger ? -mu : mu), precision);

    } else {
      // out_o = out_o  -/+ i mu gamma5 in_o
      applyTwist(outOdd, inOdd, outOdd, (dagger ? -mu : mu), precision);
    }
  } else {

    // Symmetric case:  - i mu gamma_5 A^2 psi_in
    void *tmp = malloc(Vh * spinor_site_size * precision);

    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {

      // tmp = A_ee in_e
      apply_clover(tmp, clover, inEven, 0, precision);

      // two factors of 2 for two clover applications => (1/4) mu
      // out_e = out_e -/+ i gamma_5 mu A_ee (A_ee) in_ee
      twistClover(outEven, tmp, outEven, clover, 0.25 * mu, dagger, 0, precision);

    } else {
      apply_clover(tmp, clover, inOdd, 1, precision);

      // two factors of 2 for two clover applications => (1/4) mu
      // out_e = out_e -/+ i gamma_5 mu A (A_ee)
      twistClover(outOdd, tmp, outOdd, clover, 0.25 * mu, dagger, 1, precision);
    }
    free(tmp);
  }
}

// Apply the even-odd preconditioned Dirac operator
void cloverHasenbuschTwist_matpc(void *out, void **gauge, void *in, void *clover, void *cInv, double kappa, double mu,
                                 QudaMatPCType matpc_type, int dagger, QudaPrecision precision,
                                 QudaGaugeParam &gauge_param)
{

  clover_matpc(out, gauge, clover, cInv, in, kappa, matpc_type, dagger, precision, gauge_param);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    twistClover(out, in, out, clover, 0.5 * mu, dagger, (matpc_type == QUDA_MATPC_EVEN_EVEN ? 0 : 1), precision);
  } else {
    applyTwist(out, in, out, (dagger ? -mu : mu), precision);
  }
}
