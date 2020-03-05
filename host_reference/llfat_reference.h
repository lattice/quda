#ifndef __LLFAT_REFERENCE_H__
#define __LLFAT_REFERENCE_H__

#ifdef __cplusplus
extern "C"{
#endif
  
  void llfat_reference(void** fatlink, void** sitelink, QudaPrecision prec, void* act_path_coeff);
  void llfat_reference_mg(void** fatlink, void** sitelink, void** ghost_sitelink, 
			  void** ghost_sitelink_diag, QudaPrecision prec, void* act_path_coeff);

  void computeLongLinkCPU(void** longlink, void **sitelink, QudaPrecision prec, void* act_path_coeff);  

  // CPU-style BLAS routines
  void cpu_axy(QudaPrecision prec, double a, void* x, void* y, int size);
  void cpu_xpy(QudaPrecision prec, void* x, void* y, int size);

  void computeHISQLinksCPU(void** fatlink, void** longlink, void** fatlink_eps, void** longlink_eps,
        void** sitelink, void* qudaGaugeParamPtr, 
        double** act_path_coeffs, double eps_naik);

    // data reordering routines
  void reorderQDPtoMILC(void* milc_out, void** qdp_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision);
  void reorderMILCtoQDP(void** qdp_out, void* milc_in, int V, int siteSize, QudaPrecision out_precision, QudaPrecision in_precision);

#ifdef __cplusplus
}
#endif

#endif

