#ifndef __MISC_H__
#define __MISC_H__

#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif
    
  void display_spinor(void* spinor, int len, int precision);
  void display_link(void* link, int len, int precision);
  int link_sanity_check(void* link, int len, int precision, int dir, QudaGaugeParam* gaugeParam);
  int site_link_sanity_check(void* link, int len, int precision, QudaGaugeParam* gaugeParam);
  
  QudaReconstructType get_recon(char* s);
  const char* get_recon_str(QudaReconstructType recon);

  QudaPrecision   get_prec(char* s);
  const char* get_prec_str(QudaPrecision prec);

  const char* get_gauge_order_str(QudaGaugeFieldOrder order);
  const char* get_test_type(int t);
  const char* get_staggered_test_type(int t);
  const char* get_unitarization_str(bool svd_only);

  QudaMassNormalization get_mass_normalization_type(char* s);
  const char* get_mass_normalization_str(QudaMassNormalization);

  QudaVerbosity get_verbosity_type(char* s);
  const char* get_verbosity_str(QudaVerbosity);

  QudaMatPCType get_matpc_type(char* s);
  const char* get_matpc_str(QudaMatPCType);

  QudaSolveType get_solve_type(char* s);
  const char* get_solve_str(QudaSolveType);

  QudaSolutionType get_solution_type(char *s);

  QudaSchwarzType get_schwarz_type(char* s);

  QudaTwistFlavorType get_flavor_type(char* s);

  int get_rank_order(char* s);

  QudaDslashType get_dslash_type(char* s);
  const char* get_dslash_str(QudaDslashType type);

  QudaInverterType get_solver_type(char* s);
  const char* get_solver_str(QudaInverterType type);

  QudaEigSpectrumType get_eig_spectrum_type(char *spec);
  const char *get_eig_spectrum_str(QudaEigSpectrumType type);

  QudaEigType get_eig_type(char *s);
  const char *get_eig_type_str(QudaEigType type);

  const char* get_quda_ver_str();

  QudaExtLibType get_solve_ext_lib_type(char* s);

  QudaFieldLocation get_location(char* s);

  const char *get_ritz_location_str(QudaFieldLocation type);

  QudaMemoryType get_df_mem_type_ritz(char* s);

  const char *get_memory_type_str(QudaMemoryType type);

  QudaContractType get_contract_type(char *s);
  const char *get_contract_str(QudaContractType type);

#ifdef __cplusplus
}
#endif

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir)    (7-(dir))
#define GOES_FORWARDS(dir) (dir<=3)
#define GOES_BACKWARDS(dir) (dir>3)


#endif


