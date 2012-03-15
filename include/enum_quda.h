#ifndef _ENUM_QUDA_H
#define _ENUM_QUDA_H

#include <limits.h>
#define QUDA_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

  //
  // Types used in QudaGaugeParam
  //

  typedef enum QudaLinkType_s {
    QUDA_WILSON_LINKS, // used by wilson, clover, twisted mass, and domain wall
    QUDA_ASQTAD_FAT_LINKS,
    QUDA_ASQTAD_LONG_LINKS,
    QUDA_ASQTAD_MOM_LINKS,
    QUDA_INVALID_LINKS = QUDA_INVALID_ENUM
  } QudaLinkType;

  

  typedef enum QudaGaugeFieldOrder_s {
    QUDA_FLOAT_GAUGE_ORDER = 1,
    QUDA_FLOAT2_GAUGE_ORDER = 2, // no reconstruct and double precision
    QUDA_FLOAT4_GAUGE_ORDER = 4, // 8 and 12 reconstruct half and single
    QUDA_QDP_GAUGE_ORDER, // expect *gauge[4], even-odd, row-column color
    QUDA_CPS_WILSON_GAUGE_ORDER, // expect *gauge, even-odd, mu inside, column-row color
    QUDA_MILC_GAUGE_ORDER, // expect *gauge, even-odd, mu inside, row-column order
    QUDA_INVALID_GAUGE_ORDER = QUDA_INVALID_ENUM
  } QudaGaugeFieldOrder;

  typedef enum QudaTboundary_s {
    QUDA_ANTI_PERIODIC_T = -1,
    QUDA_PERIODIC_T = 1,
    QUDA_INVALID_T_BOUNDARY = QUDA_INVALID_ENUM
  } QudaTboundary;

  typedef enum QudaPrecision_s {
    QUDA_HALF_PRECISION = 2,
    QUDA_SINGLE_PRECISION = 4,
    QUDA_DOUBLE_PRECISION = 8,
    QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM
  } QudaPrecision;

  typedef enum QudaReconstructType_s {
    QUDA_RECONSTRUCT_NO = 18, // store all 18 real numbers explicitly
    QUDA_RECONSTRUCT_12 = 12, // reconstruct from 12 real numbers
    QUDA_RECONSTRUCT_8 = 8,  // reconstruct from 8 real numbers
    QUDA_RECONSTRUCT_10 = 10, // 10-number parameterization used for storing the momentum field
    QUDA_RECONSTRUCT_INVALID = QUDA_INVALID_ENUM
  } QudaReconstructType;

  typedef enum QudaGaugeFixed_s {
    QUDA_GAUGE_FIXED_NO,  // no gauge fixing
    QUDA_GAUGE_FIXED_YES, // gauge field stored in temporal gauge
    QUDA_GAUGE_FIXED_INVALID = QUDA_INVALID_ENUM
  } QudaGaugeFixed;

  //
  // Types used in QudaInvertParam
  //

  typedef enum QudaDslashType_s {
    QUDA_WILSON_DSLASH,
    QUDA_CLOVER_WILSON_DSLASH,
    QUDA_DOMAIN_WALL_DSLASH,
    QUDA_ASQTAD_DSLASH,
    QUDA_TWISTED_MASS_DSLASH,
    QUDA_INVALID_DSLASH = QUDA_INVALID_ENUM
  } QudaDslashType;

  typedef enum QudaInverterType_s {
    QUDA_CG_INVERTER,
    QUDA_BICGSTAB_INVERTER,
    QUDA_GCR_INVERTER,
    QUDA_MR_INVERTER,
    QUDA_INVALID_INVERTER = QUDA_INVALID_ENUM
  } QudaInverterType;

  typedef enum QudaSolutionType_s {
    QUDA_MAT_SOLUTION,
    QUDA_MATDAG_MAT_SOLUTION,
    QUDA_MATPC_SOLUTION,
    QUDA_MATPCDAG_MATPC_SOLUTION,
    QUDA_INVALID_SOLUTION = QUDA_INVALID_ENUM
  } QudaSolutionType;

  typedef enum QudaSolveType_s {
    QUDA_DIRECT_SOLVE,
    QUDA_NORMEQ_SOLVE,
    QUDA_DIRECT_PC_SOLVE,
    QUDA_NORMEQ_PC_SOLVE,
    QUDA_INVALID_SOLVE = QUDA_INVALID_ENUM
  } QudaSolveType;

  // Whether the preconditioned matrix is (1-k^2 Deo Doe) or (1-k^2 Doe Deo)
  //
  // For the clover-improved Wilson Dirac operator, QUDA_MATPC_EVEN_EVEN
  // defaults to the "symmetric" form, (1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe),
  // and likewise for QUDA_MATPC_ODD_ODD.
  //
  // For the "asymmetric" form, (A_ee - k^2 D_eo A_oo^-1 D_oe), select
  // QUDA_MATPC_EVEN_EVEN_ASYMMETRIC.
  //
  typedef enum QudaMatPCType_s {
    QUDA_MATPC_EVEN_EVEN,
    QUDA_MATPC_ODD_ODD,
    QUDA_MATPC_EVEN_EVEN_ASYMMETRIC,
    QUDA_MATPC_ODD_ODD_ASYMMETRIC,
    QUDA_MATPC_INVALID = QUDA_INVALID_ENUM
  } QudaMatPCType;

  typedef enum QudaDagType_s {
    QUDA_DAG_NO,
    QUDA_DAG_YES,
    QUDA_DAG_INVALID = QUDA_INVALID_ENUM
  } QudaDagType;
  
  typedef enum QudaMassNormalization_s {
    QUDA_KAPPA_NORMALIZATION,
    QUDA_MASS_NORMALIZATION,
    QUDA_ASYMMETRIC_MASS_NORMALIZATION,
    QUDA_INVALID_NORMALIZATION = QUDA_INVALID_ENUM
  } QudaMassNormalization;

  typedef enum QudaPreserveSource_s {
    QUDA_PRESERVE_SOURCE_NO,  // use the source for the residual
    QUDA_PRESERVE_SOURCE_YES, // keep the source intact
    QUDA_PRESERVE_SOURCE_INVALID = QUDA_INVALID_ENUM
  } QudaPreserveSource;

  typedef enum QudaDiracFieldOrder_s {
    QUDA_DIRAC_ORDER,            // even-odd, color inside spin
    QUDA_QDP_DIRAC_ORDER,        // even-odd, spin inside color
    QUDA_CPS_WILSON_DIRAC_ORDER, // odd-even, color inside spin
    QUDA_LEX_DIRAC_ORDER,        // lexicographical order, color inside spin
    QUDA_INVALID_DIRAC_ORDER = QUDA_INVALID_ENUM
  } QudaDiracFieldOrder;  

  typedef enum QudaCloverFieldOrder_s {
    QUDA_PACKED_CLOVER_ORDER,     // even-odd, packed
    QUDA_LEX_PACKED_CLOVER_ORDER, // lexicographical order, packed
    QUDA_INVALID_CLOVER_ORDER = QUDA_INVALID_ENUM
  } QudaCloverFieldOrder;

  typedef enum QudaVerbosity_s {
    QUDA_SILENT,
    QUDA_SUMMARIZE,
    QUDA_VERBOSE,
    QUDA_DEBUG_VERBOSE,
    QUDA_INVALID_VERBOSITY = QUDA_INVALID_ENUM
  } QudaVerbosity;

  typedef enum QudaTune_s {
    QUDA_TUNE_NO,
    QUDA_TUNE_YES,
    QUDA_TUNE_INVALID = QUDA_INVALID_ENUM
  } QudaTune;

  typedef enum QudaPreserveDirac_s {
    QUDA_PRESERVE_DIRAC_NO,
    QUDA_PRESERVE_DIRAC_YES,
    QUDA_PRESERVE_DIRAC_INVALID = QUDA_INVALID_ENUM
  } QudaPreserveDirac;

  //
  // Type used for "parity" argument to dslashQuda()
  //

  typedef enum QudaParity_s {
    QUDA_EVEN_PARITY = 0,
    QUDA_ODD_PARITY,
    QUDA_INVALID_PARITY = QUDA_INVALID_ENUM
  } QudaParity;

  //  
  // Types used only internally
  //

  typedef enum QudaDiracType_s {
    QUDA_WILSON_DIRAC,
    QUDA_WILSONPC_DIRAC,
    QUDA_CLOVER_DIRAC,
    QUDA_CLOVERPC_DIRAC,
    QUDA_DOMAIN_WALL_DIRAC,
    QUDA_DOMAIN_WALLPC_DIRAC,
    QUDA_ASQTAD_DIRAC,
    QUDA_ASQTADPC_DIRAC,
    QUDA_TWISTED_MASS_DIRAC,
    QUDA_TWISTED_MASSPC_DIRAC,
    QUDA_INVALID_DIRAC = QUDA_INVALID_ENUM
  } QudaDiracType;

  // Where the field is stored
  typedef enum QudaFieldLocation_s {
    QUDA_CPU_FIELD_LOCATION = 1,
    QUDA_CUDA_FIELD_LOCATION = 2,
    QUDA_INVALID_FIELD_LOCATION = QUDA_INVALID_ENUM
  } QudaFieldLocation;
  
  // Which sites are included
  typedef enum QudaSiteSubset_s {
    QUDA_FULL_SITE_SUBSET,
    QUDA_PARITY_SITE_SUBSET,
    QUDA_INVALID_SITE_SUBSET = QUDA_INVALID_ENUM
  } QudaSiteSubset;
  
  // Site ordering (always t-z-y-x, with rightmost varying fastest)
  typedef enum QudaSiteOrder_s {
    QUDA_LEXICOGRAPHIC_SITE_ORDER, // lexicographic ordering
    QUDA_EVEN_ODD_SITE_ORDER, // QUDA and QDP use this
    QUDA_ODD_EVEN_SITE_ORDER, // CPS uses this
    QUDA_INVALID_SITE_ORDER = QUDA_INVALID_ENUM
  } QudaSiteOrder;
  
  // Degree of freedom ordering
  typedef enum QudaFieldOrder_s {
    QUDA_FLOAT_FIELD_ORDER = 1, // spin-color-complex-space
    QUDA_FLOAT2_FIELD_ORDER = 2, // (spin-color-complex)/2-space-(spin-color-complex)%2
    QUDA_FLOAT4_FIELD_ORDER = 4, // (spin-color-complex)/4-space-(spin-color-complex)%4
    QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, // CPS/QDP++ ordering
    QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, // QLA ordering (spin inside color)
    QUDA_QOP_DOMAIN_WALL_FIELD_ORDER, // QOP domain-wall ordering
    QUDA_INVALID_FIELD_ORDER = QUDA_INVALID_ENUM
  } QudaFieldOrder;
  
  typedef enum QudaFieldCreate_s {
    QUDA_NULL_FIELD_CREATE, // create new field
    QUDA_ZERO_FIELD_CREATE, // create new field and zero it
    QUDA_COPY_FIELD_CREATE, // create copy to field
    QUDA_REFERENCE_FIELD_CREATE, // create reference to field
    QUDA_INVALID_FIELD_CREATE = QUDA_INVALID_ENUM
  } QudaFieldCreate;

  typedef enum QudaGammaBasis_s {
    QUDA_DEGRAND_ROSSI_GAMMA_BASIS,
    QUDA_UKQCD_GAMMA_BASIS,
    QUDA_INVALID_GAMMA_BASIS = QUDA_INVALID_ENUM
  } QudaGammaBasis;

  typedef enum QudaSourceType_s {
    QUDA_POINT_SOURCE,
    QUDA_RANDOM_SOURCE,
    QUDA_INVALID_SOURCE = QUDA_INVALID_ENUM
  } QudaSourceType;
  
  typedef enum QudaTwistFlavorType_s {
    QUDA_TWIST_MINUS = -1,
    QUDA_TWIST_PLUS = +1,
    QUDA_TWIST_NO  = 0,
    QUDA_TWIST_INVALID = QUDA_INVALID_ENUM
  } QudaTwistFlavorType;  

  typedef enum QudaTwistGamma5Type_s {
    QUDA_TWIST_GAMMA5_DIRECT,
    QUDA_TWIST_GAMMA5_INVERSE,
    QUDA_TWIST_GAMMA5_INVALID = QUDA_INVALID_ENUM
  } QudaTwistGamma5Type;

  typedef enum QudaUseInitGuess_s {
    QUDA_USE_INIT_GUESS_YES,
    QUDA_USE_INIT_GUESS_NO,    
    QUDA_USE_INIT_GUESS_INVALID = QUDA_INVALID_ENUM
  } QudaUseInitGuess;

  typedef enum QudaDirection_s {
    QUDA_BACKWARDS,
    QUDA_FORWARDS
  } QudaDirection;
  
  typedef enum QudaComputeFatMethod_s{
    QUDA_COMPUTE_FAT_STANDARD,
    QUDA_COMPUTE_FAT_EXTENDED_VOLUME,
    QUDA_COMPUTE_FAT_INVALID=  QUDA_INVALID_ENUM
  }QudaComputeFatMethod;

  typedef enum QudaFatLinkFlag_s{
    QUDA_FAT_PRESERVE_CPU_GAUGE=1,
    QUDA_FAT_PRESERVE_GPU_GAUGE=2,

    QUDA_FAT_PRESERVE_COMM_MEM=4,
  }QudaFatLinkFlag;

#ifdef __cplusplus
}
#endif

#endif // _ENUM_QUDA_H
