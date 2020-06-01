#pragma once
#include <quda.h>

const char *get_quda_ver_str();
const char *get_recon_str(QudaReconstructType recon);
const char *get_prec_str(QudaPrecision prec);
const char *get_gauge_order_str(QudaGaugeFieldOrder order);
const char *get_test_type(int t);
const char *get_staggered_test_type(int t);
const char *get_unitarization_str(bool svd_only);
const char *get_mass_normalization_str(QudaMassNormalization);
const char *get_verbosity_str(QudaVerbosity);
const char *get_matpc_str(QudaMatPCType);
const char *get_solution_str(QudaSolutionType);
const char *get_solve_str(QudaSolveType);
const char *get_dslash_str(QudaDslashType type);
const char *get_flavor_str(QudaTwistFlavorType type);
const char *get_solver_str(QudaInverterType type);
const char *get_eig_spectrum_str(QudaEigSpectrumType type);
const char *get_eig_type_str(QudaEigType type);
const char *get_ritz_location_str(QudaFieldLocation type);
const char *get_memory_type_str(QudaMemoryType type);
const char *get_contract_str(QudaContractType type);

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir) (7 - (dir))
#define GOES_FORWARDS(dir) (dir <= 3)
#define GOES_BACKWARDS(dir) (dir > 3)
