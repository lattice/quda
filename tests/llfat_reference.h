#ifndef __LLFAT_REFERENCE_H__
#define __LLFAT_REFERENCE_H__

#ifdef __cplusplus
extern "C"{
#endif
  
  void llfat_reference(void** fatlink, void** sitelink, QudaPrecision prec, void* act_path_coeff);
  void llfat_reference_mg(void** fatlink, void** sitelink, void** ghost_sitelink, 
			  void** ghost_sitelink_diag, QudaPrecision prec, void* act_path_coeff);
  
#ifdef __cplusplus
}
#endif

#endif

