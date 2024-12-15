#include <stdio.h>
#include <stdlib.h>
#include "invert_quda.h"
#include "misc.h"
#include <assert.h>
#include "util_quda.h"
#include <host_utils.h>

const char *get_verbosity_str(QudaVerbosity type)
{
  const char *ret;

  switch (type) {
  case QUDA_SILENT: ret = "silent"; break;
  case QUDA_SUMMARIZE: ret = "summarize"; break;
  case QUDA_VERBOSE: ret = "verbose"; break;
  case QUDA_DEBUG_VERBOSE: ret = "debug"; break;
  default: fprintf(stderr, "Error: invalid verbosity type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_prec_str(QudaPrecision prec)
{
  const char *ret;

  switch (prec) {
  case QUDA_DOUBLE_PRECISION: ret = "double"; break;
  case QUDA_SINGLE_PRECISION: ret = "single"; break;
  case QUDA_HALF_PRECISION: ret = "half"; break;
  case QUDA_QUARTER_PRECISION: ret = "quarter"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_unitarization_str(bool svd_only)
{
  const char *ret;

  if (svd_only) {
    ret = "SVD";
  } else {
    ret = "CayleyHamilton_SVD";
  }

  return ret;
}

const char *get_gauge_order_str(QudaGaugeFieldOrder order)
{
  const char *ret;

  switch (order) {
  case QUDA_QDP_GAUGE_ORDER: ret = "qdp"; break;
  case QUDA_MILC_GAUGE_ORDER: ret = "milc"; break;
  case QUDA_CPS_WILSON_GAUGE_ORDER: ret = "cps_wilson"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_recon_str(QudaReconstructType recon)
{
  const char *ret;

  switch (recon) {
  case QUDA_RECONSTRUCT_13: ret = "13"; break;
  case QUDA_RECONSTRUCT_12: ret = "12"; break;
  case QUDA_RECONSTRUCT_9: ret = "9"; break;
  case QUDA_RECONSTRUCT_8: ret = "8"; break;
  case QUDA_RECONSTRUCT_NO: ret = "18"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_test_type(int t)
{
  const char *ret;

  switch (t) {
  case 0: ret = "even"; break;
  case 1: ret = "odd"; break;
  case 2: ret = "full"; break;
  case 3: ret = "mcg_even"; break;
  case 4: ret = "mcg_odd"; break;
  case 5: ret = "mcg_full"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_dslash_str(QudaDslashType type)
{
  const char *ret;

  switch (type) {
  case QUDA_WILSON_DSLASH: ret = "wilson"; break;
  case QUDA_CLOVER_WILSON_DSLASH: ret = "clover"; break;
  case QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH: ret = "clover_hasenbusch_twist"; break;
  case QUDA_TWISTED_MASS_DSLASH: ret = "twisted_mass"; break;
  case QUDA_TWISTED_CLOVER_DSLASH: ret = "twisted_clover"; break;
  case QUDA_STAGGERED_DSLASH: ret = "staggered"; break;
  case QUDA_ASQTAD_DSLASH: ret = "asqtad"; break;
  case QUDA_DOMAIN_WALL_DSLASH: ret = "domain_wall"; break;
  case QUDA_DOMAIN_WALL_4D_DSLASH: ret = "domain_wall_4d"; break;
  case QUDA_MOBIUS_DWF_DSLASH: ret = "mobius"; break;
  case QUDA_MOBIUS_DWF_EOFA_DSLASH: ret = "mobius_eofa"; break;
  case QUDA_LAPLACE_DSLASH: ret = "laplace"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

std::vector<const char *> get_dslash_str_list()
{
  static std::vector<const char *> dslash_str_list;
  bool populated = false;

  if (!populated) {
    dslash_str_list.push_back("wilson");
    dslash_str_list.push_back("clover");
    dslash_str_list.push_back("clover-hasenbusch-twist");
    dslash_str_list.push_back("twisted-mass");
    dslash_str_list.push_back("twisted-clover");
    dslash_str_list.push_back("staggered");
    dslash_str_list.push_back("asqtad");
    dslash_str_list.push_back("hisq");
    dslash_str_list.push_back("domain-wall");
    dslash_str_list.push_back("domain-wall-4d");
    dslash_str_list.push_back("mobius");
    dslash_str_list.push_back("mobius-eofa");
    dslash_str_list.push_back("laplace");
    populated = true;
  }
  return dslash_str_list;
}

QudaDslashType get_dslash_from_str(const char *str)
{
  std::string d_type(str);
  if (!d_type.compare("wilson")) return QUDA_WILSON_DSLASH;
  if (!d_type.compare("clover")) return QUDA_CLOVER_WILSON_DSLASH;
  if (!d_type.compare("clover_hasenbusch_twist") || !d_type.compare("clover-hasenbusch-twist"))
    return QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH;
  if (!d_type.compare("twisted_mass") || !d_type.compare("twisted-mass")) return QUDA_TWISTED_MASS_DSLASH;
  if (!d_type.compare("twisted_clover") || !d_type.compare("twisted-clover")) return QUDA_TWISTED_CLOVER_DSLASH;
  if (!d_type.compare("staggered")) return QUDA_STAGGERED_DSLASH;
  if (!d_type.compare("asqtad")) return QUDA_ASQTAD_DSLASH;
  if (!d_type.compare("hisq")) return QUDA_ASQTAD_DSLASH;
  if (!d_type.compare("domain_wall") || !d_type.compare("domain-wall")) return QUDA_DOMAIN_WALL_DSLASH;
  if (!d_type.compare("domain_wall_4d") || !d_type.compare("domain-wall-4d")) return QUDA_DOMAIN_WALL_4D_DSLASH;
  if (!d_type.compare("mobius")) return QUDA_MOBIUS_DWF_DSLASH;
  if (!d_type.compare("mobius_eofa") || !d_type.compare("mobius-eofa")) return QUDA_MOBIUS_DWF_EOFA_DSLASH;
  if (!d_type.compare("laplace")) return QUDA_LAPLACE_DSLASH;
  return QUDA_INVALID_DSLASH;
}

const char *get_contract_str(QudaContractType type)
{
  const char *ret;

  switch (type) {
  case QUDA_CONTRACT_TYPE_OPEN: ret = "open"; break;
  case QUDA_CONTRACT_TYPE_OPEN_SUM_T: ret = "open_sum_t"; break;
  case QUDA_CONTRACT_TYPE_OPEN_SUM_Z: ret = "open_sum_z"; break;
  case QUDA_CONTRACT_TYPE_OPEN_FT_T: ret = "open_ft_t"; break;
  case QUDA_CONTRACT_TYPE_OPEN_FT_Z: ret = "open_ft_z"; break;
  case QUDA_CONTRACT_TYPE_DR: ret = "dr"; break;
  case QUDA_CONTRACT_TYPE_DR_FT_T: ret = "dr_ft_t"; break;
  case QUDA_CONTRACT_TYPE_DR_FT_Z: ret = "dr_ft_z"; break;
  case QUDA_CONTRACT_TYPE_STAGGERED: ret = "stag"; break;
  case QUDA_CONTRACT_TYPE_STAGGERED_FT_T: ret = "stag_ft_t"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_dag_str(QudaDagType type)
{
  const char *ret;

  switch (type) {
  case QUDA_DAG_YES: ret = "dag"; break;
  case QUDA_DAG_NO: ret = "nodag"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_eig_spectrum_str(QudaEigSpectrumType type)
{
  const char *ret;

  switch (type) {
  case QUDA_SPECTRUM_SR_EIG: ret = "SR"; break;
  case QUDA_SPECTRUM_LR_EIG: ret = "LR"; break;
  case QUDA_SPECTRUM_SM_EIG: ret = "SM"; break;
  case QUDA_SPECTRUM_LM_EIG: ret = "LM"; break;
  case QUDA_SPECTRUM_SI_EIG: ret = "SI"; break;
  case QUDA_SPECTRUM_LI_EIG: ret = "LI"; break;
  default: ret = "unknown eigenspectrum"; break;
  }

  return ret;
}

const char *get_eig_type_str(QudaEigType type)
{
  const char *ret;

  switch (type) {
  case QUDA_EIG_TR_LANCZOS: ret = "trlm"; break;
  case QUDA_EIG_BLK_TR_LANCZOS: ret = "blktrlm"; break;
  case QUDA_EIG_TR_LANCZOS_3D: ret = "trlm_3d"; break;
  case QUDA_EIG_IR_ARNOLDI: ret = "iram"; break;
  case QUDA_EIG_BLK_IR_ARNOLDI: ret = "blkiram"; break;
  default: ret = "unknown eigensolver"; break;
  }

  return ret;
}

const char *get_gauge_smear_str(QudaGaugeSmearType type)
{
  const char *ret;

  switch (type) {
  case QUDA_GAUGE_SMEAR_APE: ret = "APE"; break;
  case QUDA_GAUGE_SMEAR_STOUT: ret = "Stout"; break;
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT: ret = "Over Improved"; break;
  case QUDA_GAUGE_SMEAR_HYP: ret = "HYP"; break;
  case QUDA_GAUGE_SMEAR_WILSON_FLOW: ret = "Wilson Flow"; break;
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: ret = "Symanzik Flow"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_mass_normalization_str(QudaMassNormalization type)
{
  const char *s;

  switch (type) {
  case QUDA_KAPPA_NORMALIZATION: s = "kappa"; break;
  case QUDA_MASS_NORMALIZATION: s = "mass"; break;
  case QUDA_ASYMMETRIC_MASS_NORMALIZATION: s = "asym_mass"; break;
  default: fprintf(stderr, "Error: invalid mass normalization\n"); exit(1);
  }

  return s;
}

const char *get_matpc_str(QudaMatPCType type)
{
  const char *ret;

  switch (type) {
  case QUDA_MATPC_EVEN_EVEN: ret = "even_even"; break;
  case QUDA_MATPC_ODD_ODD: ret = "odd_odd"; break;
  case QUDA_MATPC_EVEN_EVEN_ASYMMETRIC: ret = "even_even_asym"; break;
  case QUDA_MATPC_ODD_ODD_ASYMMETRIC: ret = "odd_odd_asym"; break;
  default: fprintf(stderr, "Error: invalid matpc type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_solution_str(QudaSolutionType type)
{
  const char *ret;

  switch (type) {
  case QUDA_MAT_SOLUTION: ret = "mat"; break;
  case QUDA_MATDAG_MAT_SOLUTION: ret = "mat_dag_mat"; break;
  case QUDA_MATPC_SOLUTION: ret = "mat_pc"; break;
  case QUDA_MATPCDAG_MATPC_SOLUTION: ret = "mat_pc_dag_mat_pc"; break;
  case QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION: ret = "mat_pc_dag_mat_pc_shift"; break;
  default: fprintf(stderr, "Error: invalid solution type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_solve_str(QudaSolveType type)
{
  const char *ret;

  switch (type) {
  case QUDA_DIRECT_SOLVE: ret = "direct"; break;
  case QUDA_DIRECT_PC_SOLVE: ret = "direct_pc"; break;
  case QUDA_NORMOP_SOLVE: ret = "normop"; break;
  case QUDA_NORMOP_PC_SOLVE: ret = "normop_pc"; break;
  case QUDA_NORMERR_SOLVE: ret = "normerr"; break;
  case QUDA_NORMERR_PC_SOLVE: ret = "normerr_pc"; break;
  default: fprintf(stderr, "Error: invalid solve type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_schwarz_str(QudaSchwarzType type)
{
  const char *ret;

  switch (type) {
  case QUDA_ADDITIVE_SCHWARZ: ret = "additive_schwarz"; break;
  case QUDA_MULTIPLICATIVE_SCHWARZ: ret = "multiplicative_schwarz"; break;
  default: fprintf(stderr, "Error: invalid schwarz type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_flavor_str(QudaTwistFlavorType type)
{
  const char *ret;

  switch (type) {
  case QUDA_TWIST_SINGLET: ret = "singlet"; break;
  case QUDA_TWIST_NONDEG_DOUBLET: ret = "nondeg_doublet"; break;
  case QUDA_TWIST_NO: ret = "no"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_solver_str(QudaInverterType type)
{
  const char *ret;

  switch (type) {
  case QUDA_CG_INVERTER: ret = "cg"; break;
  case QUDA_BICGSTAB_INVERTER: ret = "bicgstab"; break;
  case QUDA_GCR_INVERTER: ret = "gcr"; break;
  case QUDA_PCG_INVERTER: ret = "pcg"; break;
  case QUDA_MR_INVERTER: ret = "mr"; break;
  case QUDA_SD_INVERTER: ret = "sd"; break;
  case QUDA_EIGCG_INVERTER: ret = "eigcg"; break;
  case QUDA_INC_EIGCG_INVERTER: ret = "inc_eigcg"; break;
  case QUDA_GMRESDR_INVERTER: ret = "gmresdr"; break;
  case QUDA_GMRESDR_PROJ_INVERTER: ret = "gmresdr_proj"; break;
  case QUDA_GMRESDR_SH_INVERTER: ret = "gmresdr_sh"; break;
  case QUDA_FGMRESDR_INVERTER: ret = "fgmresdr"; break;
  case QUDA_MG_INVERTER: ret = "mg"; break;
  case QUDA_BICGSTABL_INVERTER: ret = "bicgstab_l"; break;
  case QUDA_CGNE_INVERTER: ret = "cgne"; break;
  case QUDA_CGNR_INVERTER: ret = "cgnr"; break;
  case QUDA_CG3_INVERTER: ret = "cg3"; break;
  case QUDA_CG3NE_INVERTER: ret = "cg3ne"; break;
  case QUDA_CG3NR_INVERTER: ret = "cg3nr"; break;
  case QUDA_CA_CG_INVERTER: ret = "ca_cg"; break;
  case QUDA_CA_CGNE_INVERTER: ret = "ca_cgne"; break;
  case QUDA_CA_CGNR_INVERTER: ret = "ca_cgnr"; break;
  case QUDA_CA_GCR_INVERTER: ret = "ca_gcr"; break;
  default:
    ret = "unknown";
    errorQuda("Error: invalid solver type %d\n", type);
    break;
  }

  return ret;
}

const char *get_quda_ver_str()
{
  static char vstr[32];
  int major_num = QUDA_VERSION_MAJOR;
  int minor_num = QUDA_VERSION_MINOR;
  int ext_num = QUDA_VERSION_SUBMINOR;
  sprintf(vstr, "%1d.%1d.%1d", major_num, minor_num, ext_num);
  return vstr;
}

const char *get_ritz_location_str(QudaFieldLocation type)
{
  const char *s;

  switch (type) {
  case QUDA_CPU_FIELD_LOCATION: s = "cpu"; break;
  case QUDA_CUDA_FIELD_LOCATION: s = "cuda"; break;
  default: fprintf(stderr, "Error: invalid location\n"); exit(1);
  }

  return s;
}

const char *get_memory_type_str(QudaMemoryType type)
{
  const char *s;

  switch (type) {
  case QUDA_MEMORY_DEVICE: s = "device"; break;
  case QUDA_MEMORY_DEVICE_PINNED: s = "device_pinned"; break;
  case QUDA_MEMORY_HOST: s = "host"; break;
  case QUDA_MEMORY_HOST_PINNED: s = "host_pinned"; break;
  case QUDA_MEMORY_MAPPED: s = "mapped"; break;
  default: fprintf(stderr, "Error: invalid memory type\n"); exit(1);
  }

  return s;
}

std::string get_dilution_type_str(QudaDilutionType type)
{
  std::string s;

  switch (type) {
  case QUDA_DILUTION_SPIN: s = std::string("spin"); break;
  case QUDA_DILUTION_COLOR: s = std::string("color"); break;
  case QUDA_DILUTION_SPIN_COLOR: s = std::string("spin_color"); break;
  case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: s = std::string("spin_color_even_odd"); break;
  case QUDA_DILUTION_BLOCK: s = std::string("block"); break;
  default: fprintf(stderr, "Error: invalid dilution type\n"); exit(1);
  }
  return s;
}

const char *get_blas_type_str(QudaBLASType type)
{
  const char *s;

  switch (type) {
  case QUDA_BLAS_GEMM: s = "gemm"; break;
  case QUDA_BLAS_LU_INV: s = "lu-inv"; break;
  default: fprintf(stderr, "Error: invalid BLAS type\n"); exit(1);
  }
  return s;
}

const char *get_TwistFlavor_str(QudaTwistFlavorType type)
{
  const char *ret;

  switch (type) {
  case QUDA_TWIST_SINGLET: ret = "singlet_"; break;
  case QUDA_TWIST_NONDEG_DOUBLET: ret = "ndeg_doublet_"; break;
  case QUDA_TWIST_NO: ret = ""; break;
  default: ret = ""; break;
  }

  return ret;
}
