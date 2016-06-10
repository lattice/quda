#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>

#include <util_quda.h>
#include <test_util.h>
#include <wilson_dslash_reference.h>
#include <blas_reference.h>


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
  void *tmp = malloc(Vh*spinorSiteSize*precision);

  wil_dslash(tmp, gauge, in, parity, dagger, precision, param);
  apply_clover(out, clover, tmp, parity, precision);

  free(tmp);
}

// Apply the even-odd preconditioned Wilson-clover operator
void clover_matpc(void *out, void **gauge, void *clover, void *clover_inv, void *in, double kappa, 
		  QudaMatPCType matpc_type, int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  double kappa2 = -kappa*kappa;
  void *tmp = malloc(Vh*spinorSiteSize*precision);
    
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
    xpay(in, kappa2, out, Vh*spinorSiteSize, precision);
    break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
    wil_dslash(out, gauge, in, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 1, precision);
    wil_dslash(out, gauge, tmp, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 0, precision);
    xpay(tmp, kappa2, out, Vh*spinorSiteSize, precision);
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
    xpay(in, kappa2, out, Vh*spinorSiteSize, precision);
    break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC:
    wil_dslash(out, gauge, in, 0, dagger, precision, gauge_param);
    apply_clover(tmp, clover_inv, out, 0, precision);
    wil_dslash(out, gauge, tmp, 1, dagger, precision, gauge_param);
    apply_clover(tmp, clover, in, 1, precision);
    xpay(tmp, kappa2, out, Vh*spinorSiteSize, precision);
    break;
  default:
    errorQuda("Unsupoorted matpc=%d", matpc_type);
  }

  free(tmp);
}

// Apply the full Wilson-clover operator
void clover_mat(void *out, void **gauge, void *clover, void *in, double kappa, 
		int dagger, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  void *tmp = malloc(V*spinorSiteSize*precision);

  void *inEven = in;
  void *inOdd  = (char*)in + Vh*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + Vh*spinorSiteSize*precision;
  void *tmpEven = tmp;
  void *tmpOdd = (char*)tmp + Vh*spinorSiteSize*precision;

  // Odd part
  wil_dslash(outOdd, gauge, inEven, 1, dagger, precision, gauge_param);
  apply_clover(tmpOdd, clover, inOdd, 1, precision);

  // Even part
  wil_dslash(outEven, gauge, inOdd, 0, dagger, precision, gauge_param);
  apply_clover(tmpEven, clover, inEven, 0, precision);

  // lastly apply the kappa term
  xpay(tmp, -kappa, out, V*spinorSiteSize, precision);

  free(tmp);
}
