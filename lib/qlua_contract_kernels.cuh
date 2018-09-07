#ifndef QLUA_CONTRACT_KERNELS_H__
#define QLUA_CONTRACT_KERNELS_H__

#ifdef __CUDACC__
#  define DEVFUNC_  __device__ __host__
#  define CONSTVAR_ constexpr
#  define INFUNC_ inline
#else
#  define DEVFUNC_
#  define CONSTVAR_ const
#  if __STDC_VERSION__ >= 199901L
#    define INFUNC_ inline static
#  else
#    define INFUNC_
#  endif
#endif

#include <complex_quda.h>
#include <qlua_contract.h>

#ifndef LONG_T
#define LONG_T long long
#endif


/* C.K. By (re)defining the macros REAL, Ns, Nc with QC_ prefix, the conventions are that these macros
 * will be used ONLY in the contraction kernels. That's why they are defined here.
 * Original definitions with QUDA_ prefix are in interface_qlua.h.
 */
#define QC_REAL QUDA_REAL
#define QC_Ns   QUDA_Ns
#define QC_Nc   QUDA_Nc


#define QC_QUDA_CPLX complex<QC_REAL>
#define QC_QUDA_CONJ conj
  
#ifndef QC_CPLX
#define QC_CPLX QC_QUDA_CPLX
#endif
  
#define QC_CONJ QC_QUDA_CONJ
  
#define I  complex<QC_REAL>{0.0,1.0}
  
#define QC_LEN_G  (QC_Ns*QC_Ns)
#define QC_LEN_GG (QC_LEN_G*QC_LEN_G)
#define QC_LEN_V  (QC_Nc)
#define QC_LEN_M  (QC_Nc*QC_Nc)
#define QC_LEN_D  (QC_Nc*QC_Ns)
#define QC_LEN_P  (QC_LEN_D*QC_LEN_D)

/* indexing macros : depend on QC_Nc,QC_Ns */
#define QC_QUDA_LIDX_V(ic)           (ic)
#define QC_QUDA_LIDX_G(is,js)        ((js) + QC_Ns*(is))
#define QC_QUDA_LIDX_M(ic,jc)        ((jc) + QC_Nc*(ic))
#define QC_QUDA_LIDX_D(ic,is)        ((is) + QC_Ns*(ic))
#define QC_QUDA_LIDX_P(ic,is,jc,js)  ((js) + QC_Ns*(jc) + QC_Ns*QC_Nc*(is) + QC_Ns*QC_Nc*QC_Ns*(ic))

#define QC_LIDX_V(ic)          QC_QUDA_LIDX_V(ic)
#define QC_LIDX_G(is,js)       QC_QUDA_LIDX_G(is,js)
#define QC_LIDX_M(ic,jc)       QC_QUDA_LIDX_M(ic,jc)
#define QC_LIDX_D(ic,is)       QC_QUDA_LIDX_D(ic,is)
#define QC_LIDX_P(ic,is,jc,js) QC_QUDA_LIDX_P(ic,is,jc,js)

#define QC(x) qc_quda_##x

// hard-coded DeGrand-Rossi basis
// g{x,y,z,t} = g{1,2,3,4}
// gamma matrices are specified as either single linear element in the algebra
//    G(n) = g1^n0 . g2^n1 . g3^n2 . g4^n3, with n=n0*2^0+n1*2^1+n2*2^2+n3*2^3
// or a linear combination of them
//    G(v) = \sum_{n=0..15} v(n) G(n)

// parameterization of gamma matrices for efficient unrolling
// [G(n)]_{ij} = gamma_left_coeff[n][i] * (gamma_left_ind[n][i]==j ? 1 : 0)
//             = C(n)_i * delta(j==J(n)_i)
// Tr[A.G(n)] = \sum_j C(n)_j * A_{J(n)_j, j}
// [G(n)   . A]_{i,      j}       = C(n)_i * A_{J(n)_i,j}
// [G(n)^T . A]_{J(n)_k, j} = C(n)_k * A_{k,j}
// [A . G(n)  ]_{i, J(n)_k} = A_{i,k} * C(n)_k
// [A . G(n)^T]_{i,      j} = A_{i,J(n)_j} * C(n)_j
INFUNC_ DEVFUNC_ QC_REAL gamma_left_coeff_Re(int m, int n, int c){

  CONSTVAR_ QC_REAL gamma_left_coeff_Re_[QC_LEN_G][QC_Ns][2] = {
    { {1,0}, {1,0}, {1,0}, {1,0} },             /* G0 = 1 */
    { {0,1}, {0,1},{0,-1},{0,-1} },             /* G1 = g1 */
    {{-1,0}, {1,0}, {1,0},{-1,0} },             /* G2 = g2 */
    {{0,-1}, {0,1},{0,-1}, {0,1} },             /* G3 = g1 g2 */
    { {0,1},{0,-1},{0,-1}, {0,1} },             /* G4 = g3 */
    {{-1,0}, {1,0},{-1,0}, {1,0} },             /* G5 = g1 g3 */
    {{0,-1},{0,-1},{0,-1},{0,-1} },             /* G6 = g2 g3 */
    { {1,0}, {1,0},{-1,0},{-1,0} },             /* G7 = g1 g2 g3 */
    { {1,0}, {1,0}, {1,0}, {1,0} },             /* G8 = g4 */
    { {0,1}, {0,1},{0,-1},{0,-1} },             /* G9 = g1 g4 */
    {{-1,0}, {1,0}, {1,0},{-1,0} },             /* G10= g2 g4 */
    {{0,-1}, {0,1},{0,-1}, {0,1} },             /* G11= g1 g2 g4 */
    { {0,1},{0,-1},{0,-1}, {0,1} },             /* G12= g3 g4 */
    {{-1,0}, {1,0},{-1,0}, {1,0} },             /* G13= g1 g3 g4 */
    {{0,-1},{0,-1},{0,-1},{0,-1} },             /* G14= g2 g3 g4 */
    { {1,0}, {1,0},{-1,0},{-1,0} },             /* G15= g1 g2 g3 g4 */
  };

  return gamma_left_coeff_Re_[m][n][c];
}

INFUNC_ DEVFUNC_ int gamma_left_ind(int m, int n){

  CONSTVAR_ int gamma_left_ind_[QC_LEN_G][QC_Ns] = {
    { 0, 1, 2, 3 },             /* G0 = 1 */
    { 3, 2, 1, 0 },             /* G1 = g1 */
    { 3, 2, 1, 0 },             /* G2 = g2 */
    { 0, 1, 2, 3 },             /* G3 = g1 g2 */
    { 2, 3, 0, 1 },             /* G4 = g3 */
    { 1, 0, 3, 2 },             /* G5 = g1 g3 */
    { 1, 0, 3, 2 },             /* G6 = g2 g3 */
    { 2, 3, 0, 1 },             /* G7 = g1 g2 g3 */
    { 2, 3, 0, 1 },             /* G8 = g4 */
    { 1, 0, 3, 2 },             /* G9 = g1 g4 */
    { 1, 0, 3, 2 },             /* G10= g2 g4 */
    { 2, 3, 0, 1 },             /* G11= g1 g2 g4 */
    { 0, 1, 2, 3 },             /* G12= g3 g4 */
    { 3, 2, 1, 0 },             /* G13= g1 g3 g4 */
    { 3, 2, 1, 0 },             /* G14= g2 g3 g4 */
    { 0, 1, 2, 3 },             /* G15= g1 g2 g3 g4 */
  };

  return gamma_left_ind_[m][n];
}


/* bits (gammas) in 0..15 (in G0..G15) */
INFUNC_ DEVFUNC_ int qc_bitcount16(int n){

  CONSTVAR_ int qc_bitcount16_[QC_LEN_G] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

  return qc_bitcount16_[n];
}

/*  G(n)^\dag = (-1)**gamma_adj_parity[n] * G(n);
    G(n).G(n) = (-1)**gamma_adj_parity[n]  (since G(n)^\dag.G(n)=G(n).G(n)^\dag=1) */
INFUNC_ DEVFUNC_ int qc_gamma_adj_parity(int n){

  CONSTVAR_ int qc_gamma_adj_parity_[QC_LEN_G] = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0 };

  return qc_gamma_adj_parity_[n];
}

/*  G(n)^* = (-1)**gamma_conj_parity[n] * G(n) */
INFUNC_ DEVFUNC_ int qc_gamma_conj_parity(int n){

  CONSTVAR_ int qc_gamma_conj_parity_[QC_LEN_G] = { 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0 };

  return qc_gamma_conj_parity_[n];
}

/*  G(n)^T = (-1)**gamma_transp_parity[n] * G(n) */
INFUNC_ DEVFUNC_ int qc_gamma_transp_parity(int n){

  CONSTVAR_ int qc_gamma_transp_parity_[QC_LEN_G]={ 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 };

  return qc_gamma_transp_parity_[n];
}

/* G(n)^\dag . G(m) . G(n) = (-1)**gamma_uni_parity(m,n) * G(m) */
INFUNC_ DEVFUNC_ int qc_gamma_uni_parity(int m,int n){

  return ( qc_bitcount16(m) * qc_bitcount16(n) - qc_bitcount16(m&n) ) % 2;
}

/* G(n)^T . G(m) . G(n)    = (-1)**gamma_sim_parity(m,n) * G(m) */
INFUNC_ DEVFUNC_ int qc_gamma_sim_parity(int m, int n){

  return ( qc_gamma_uni_parity(m,n) + qc_gamma_conj_parity(n) ) % 2;
}


#define gamma_left_coeff(m,n) (complex<QC_REAL>{gamma_left_coeff_Re(m,n,0),gamma_left_coeff_Re(m,n,1)})


namespace quda { 

  DEVFUNC_ void QC(contract_tr_g_P_P)(
				      QC_CPLX *gres, int gres_stride,
				      const QC_CPLX *F, int F_stride,
				      const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(contract_tr_g_P_aP)(
				       QC_CPLX *gres, int gres_stride,
				       const QC_CPLX *F, int F_stride,
				       const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(contract_tr_g_P_hP)(
				       QC_CPLX *gres, int gres_stride,
				       const QC_CPLX *F, int F_stride,
				       const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(contract_tr_g_P_mgbar_P)(
					    QC_CPLX *gres, int gres_stride,
					    const QC_CPLX *F, int F_stride,
					    const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(contract_tr_g_P_mgbar_aP)(
					     QC_CPLX *gres, int gres_stride,
					     const QC_CPLX *F, int F_stride,
					     const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(contract_tr_g_P_mgbar_hP)(
					     QC_CPLX *gres, int gres_stride,
					     const QC_CPLX *F, int F_stride,
					     const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(baryon_sigma_seqsource_u)(
					     QC_CPLX *r, int r_stride,
					     const QC_CPLX *Fu, int Fu_stride, 
					     const QC_CPLX *Fd, int Fd_stride,
					     const QC_CPLX *T_gvec);
  DEVFUNC_ void QC(contract_tr_g_U_P_P)(
					QC_CPLX *gres, int gres_stride,
					const QC_CPLX *U, int U_stride,
					const QC_CPLX *F, int F_stride,
					const QC_CPLX *B, int B_stride);
  DEVFUNC_ void QC(baryon_sigma_seqsource_d)(
					     QC_CPLX *r, int r_stride,
					     const QC_CPLX *Fu1, int Fu1_stride,
					     const QC_CPLX *Fu2, int Fu2_stride,
					     const QC_CPLX *T_gvec);
  DEVFUNC_ void QC(baryon_sigma_twopt_asymsrc_gvec)(
						    QC_CPLX *r, int r_stride,
						    const QC_CPLX *Fu1, int Fu1_stride,
						    const QC_CPLX *Fu2, int Fu2_stride,
						    const QC_CPLX *Fd,  int Fd_stride);
  
  /* ----------------------------------------------------------------------------------------------- */
  /* ----------------------------------------------------------------------------------------------- */
  //-- Forward declarations for contraction wrappers

  void copySmatricesToSymbol(complex<QC_REAL> *S2, complex<QC_REAL> *S1);

  __device__ void prepareDevicePropSite(complex<QC_REAL> *devProp, Vector *vec, bool preserveBasis);
  __device__ void prepareDeviceLinkSite(complex<QC_REAL> *devLink, Link U);

  __global__ void baryon_sigma_twopt_asymsrc_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_aP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_hP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_B_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_aB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_hB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void tmd_g_U_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);
  /* ----------------------------------------------------------------------------------------------- */

} //- namespace quda

#endif/*QLUA_CONTRACT_KERNELS_H__*/
