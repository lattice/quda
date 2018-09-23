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
#define QC_QUDA_LIDX_D_TR(ic,is)     ((ic) + QC_Nc*(is))
#define QC_QUDA_LIDX_P(ic,is,jc,js)  ((js) + QC_Ns*(jc) + QC_Ns*QC_Nc*(is) + QC_Ns*QC_Nc*QC_Ns*(ic))

#define QC_LIDX_V(ic)          QC_QUDA_LIDX_V(ic)
#define QC_LIDX_G(is,js)       QC_QUDA_LIDX_G(is,js)
#define QC_LIDX_M(ic,jc)       QC_QUDA_LIDX_M(ic,jc)
#define QC_LIDX_D(ic,is)       QC_QUDA_LIDX_D(ic,is)
#define QC_LIDX_D_TR(ic, is)   QC_QUDA_LIDX_D_TR(ic,is)
#define QC_LIDX_P(ic,is,jc,js) QC_QUDA_LIDX_P(ic,is,jc,js)

#define QC_UDD_THREADS_PER_SITE   (QC_Ns*QC_Ns)
#define QC_UDD_SITE_CPLXBUF       (3*QC_LEN_D + QC_LEN_M + QC_LEN_G)
#define QC_UDD_SHMEM_PER_SITE     (QC_UDD_SITE_CPLXBUF*sizeof(complex<QC_REAL>))

#define QC(x) qc_quda_##x

constexpr int cSize = 4096;

namespace quda { 

  //- C.K. Constant variable declarations
  __constant__ QC_CPLX cS1_gvec[QC_LEN_G];
  __constant__ QC_CPLX cS2_gvec[QC_LEN_G];
  __constant__ char cGamma[cSize]; // constant buffer for gamma matrices on GPU


  QC_REAL gamma_left_coeff_Re_cMem(int m, int n, int c);
  int gamma_left_ind_cMem(int m, int n);


  //-- Forward declarations for contraction wrappers
  void copySmatricesToSymbol(complex<QC_REAL> *S2, complex<QC_REAL> *S1);
  void qcCopyGammaToSymbol(qcTMD_gamma gamma_h);

  __global__ void baryon_sigma_twopt_asymsrc_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_aP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_aP_gvec_kernel_vecByVec_preserveBasisTrue(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void qbarq_g_P_hP_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_B_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_aB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void meson_F_hB_gvec_kernel(complex<QC_REAL> *Corr_dev, QluaContractArg *arg);
  __global__ void tmd_g_U_D_aD_gvec_kernel(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);
  /* ------------------------------------------------------------------------------------------- */

} //- namespace quda

#endif/*QLUA_CONTRACT_KERNELS_H__*/


//- Deprecated kernel definitions, will be removed
  // __global__ void tmd_g_U_P_P_gvec_kernel(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);
  // __global__ void tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);
  // __global__ void tmd_g_U_P_P_gvec_kernel_gaugeExt(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);
  // __global__ void tmd_g_U_P_aP_gvec_kernel_vecByVec_preserveBasisTrue_gaugeExt(complex<QC_REAL> *Corr_dev, qcTMD_Arg *arg);  
