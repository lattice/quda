#ifndef _ENUM_QUDA_H
#define _ENUM_QUDA_H

#include <limits.h>
#define QUDA_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum QudaGaugeFieldOrder_s {
    QUDA_QDP_GAUGE_ORDER, // expect *gauge[4], even-odd, row-column colour
    QUDA_CPS_WILSON_GAUGE_ORDER, // expect *gauge, even-odd, mu inside, column-row colour
    QUDA_INVALID_GAUGE_ORDER = QUDA_INVALID_ENUM
  } QudaGaugeFieldOrder;

  typedef enum QudaDiracFieldOrder_s {
    QUDA_DIRAC_ORDER, // even-odd, colour inside spin
    QUDA_QDP_DIRAC_ORDER, // even-odd, spin inside colour
    QUDA_CPS_WILSON_DIRAC_ORDER, // odd-even, colour inside spin
    QUDA_LEX_DIRAC_ORDER, // lexicographical order, colour inside spin
    QUDA_INVALID_DIRAC_ORDER = QUDA_INVALID_ENUM
  } QudaDiracFieldOrder;  

  typedef enum QudaCloverFieldOrder_s {
    QUDA_PACKED_CLOVER_ORDER, // even-odd, packed
    QUDA_LEX_PACKED_CLOVER_ORDER, // lexicographical order, packed
    QUDA_INVALID_CLOVER_ORDER = QUDA_INVALID_ENUM
  } QudaCloverFieldOrder;

  typedef enum QudaDslashType_s {
    QUDA_WILSON_DSLASH,
    QUDA_CLOVER_WILSON_DSLASH,
    QUDA_STAGGERED_DSLASH,
    QUDA_INVALID_DSLASH = QUDA_INVALID_ENUM
  } QudaDslashType;

  typedef enum QudaDiracType_s {
    QUDA_WILSON_DIRAC,
    QUDA_WILSONPC_DIRAC,
    QUDA_CLOVER_DIRAC,
    QUDA_CLOVERPC_DIRAC,
    QUDA_STAGGERED_DIRAC,
    QUDA_INVALID_DIRAC
  } QudaDiracType;

  typedef enum QudaInverterType_s {
    QUDA_CG_INVERTER,
    QUDA_BICGSTAB_INVERTER,
    QUDA_INVALID_INVERTER = QUDA_INVALID_ENUM
  } QudaInverterType;

  typedef enum QudaPrecision_s {
    QUDA_HALF_PRECISION = 2,
    QUDA_SINGLE_PRECISION = 4,
    QUDA_DOUBLE_PRECISION = 8,
    QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM
  } QudaPrecision;

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

  // The different solutions supported
  typedef enum QudaSolutionType_s {
    QUDA_MAT_SOLUTION,
    QUDA_MATPC_SOLUTION,
    QUDA_MATPCDAG_SOLUTION, // not implemented
    QUDA_MATPCDAG_MATPC_SOLUTION,
    QUDA_INVALID_SOLUTION = QUDA_INVALID_ENUM
  } QudaSolutionType;

  typedef enum QudaMassNormalization_s {
    QUDA_KAPPA_NORMALIZATION,
    QUDA_MASS_NORMALIZATION,
    QUDA_ASYMMETRIC_MASS_NORMALIZATION,
    QUDA_INVALID_NORMALIZATION = QUDA_INVALID_ENUM
  } QudaMassNormalization;

  typedef enum QudaPreserveSource_s {
    QUDA_PRESERVE_SOURCE_NO, // use the source for the residual
    QUDA_PRESERVE_SOURCE_YES, // keep the source intact
    QUDA_PRESERVE_SOURCE_INVALID = QUDA_INVALID_ENUM
  } QudaPreserveSource;

  typedef enum QudaReconstructType_s {
    QUDA_RECONSTRUCT_NO, // store all 18 real numbers explicitly
    QUDA_RECONSTRUCT_8, // reconstruct from 8 real numbers
    QUDA_RECONSTRUCT_12, // reconstruct from 12 real numbers
    QUDA_RECONSTRUCT_INVALID = QUDA_INVALID_ENUM
  } QudaReconstructType;

  typedef enum QudaGaugeFixed_s {
    QUDA_GAUGE_FIXED_NO, // No gauge fixing
    QUDA_GAUGE_FIXED_YES, // Gauge field stored in temporal gauge
    QUDA_GAUGE_FIXED_INVALID = QUDA_INVALID_ENUM
  } QudaGaugeFixed;

  typedef enum QudaDagType_s {
    QUDA_DAG_NO,
    QUDA_DAG_YES,
    QUDA_DAG_INVALID = QUDA_INVALID_ENUM
  } QudaDagType;
  
  typedef enum QudaTboundary_s {
    QUDA_ANTI_PERIODIC_T = -1,
    QUDA_PERIODIC_T = 1,
    QUDA_INVALID_T_BOUNDARY = QUDA_INVALID_ENUM
  } QudaTboundary;

  typedef enum QudaVerbosity_s {
    QUDA_SILENT,
    QUDA_SUMMARIZE,
    QUDA_VERBOSE,
    QUDA_INVALID_VERBOSITY = QUDA_INVALID_ENUM
  } QudaVerbosity;

  // These have been added along with ColorSpinorFields
typedef enum FieldType_s {
  QUDA_CPU_FIELD,
  QUDA_CUDA_FIELD,
  QUDA_INVALID_FIELD = QUDA_INVALID_ENUM
} FieldType;

typedef enum FieldSubset_s {
  QUDA_FULL_FIELD_SUBSET,
  QUDA_PARITY_FIELD_SUBSET,
  QUDA_INVALID_SUBSET = QUDA_INVALID_ENUM
} FieldSubset;

typedef enum SubsetOrder_s {
  QUDA_LEXICOGRAPHIC_SUBSET_ORDER, // lexicographic ordering
  QUDA_EVEN_ODD_SUBSET_ORDER, // QUDA and qdp use this
  QUDA_ODD_EVEN_SUBSET_ORDER, // cps uses this
  QUDA_INVALID_SUBSET_ORDER = QUDA_INVALID_ENUM
} SubsetOrder;

// Unless otherwise stated, color runs faster than spin
typedef enum QudaColorSpinorOrder_s {
  QUDA_FLOAT_ORDER, // spin-color-complex-space
  QUDA_FLOAT2_ORDER, // (spin-color-complex)/2-space-(spin-color-complex)%2
  QUDA_FLOAT4_ORDER, // (spin-color-complex)/4-space-(spin-color-complex)%4
  QUDA_SPACE_SPIN_COLOR_ORDER,
  QUDA_SPACE_COLOR_SPIN_ORDER, // QLA ordering (spin inside color)
  QUDA_INVALID_ORDER = QUDA_INVALID_ENUM
} QudaColorSpinorOrder;

typedef enum FieldCreate_s {
  QUDA_NULL_CREATE, // create new field
  QUDA_ZERO_CREATE, // create new field and zero it
  QUDA_COPY_CREATE, // create copy to field
  QUDA_REFERENCE_CREATE, // create reference to field
  QUDA_INVALID_CREATE = QUDA_INVALID_ENUM
} FieldCreate;

typedef enum GammaBasis_s {
  QUDA_DEGRAND_ROSSI_BASIS,
  QUDA_UKQCD_BASIS,
  QUDA_INVALID_BASIS = QUDA_INVALID_ENUM
} GammaBasis;

typedef enum QudaParity_s{
  QUDA_EVEN_PARITY = 0,
  QUDA_ODD_PARITY,
  QUDA_FULL_PARITY,
  QUDA_INVALID_PARITY = QUDA_INVALID_ENUM
}QudaParity;
  
typedef enum QudaSourceType_s {
  QUDA_POINT_SOURCE,
  QUDA_RANDOM_SOURCE,
  QUDA_INVALID_SOURCE = QUDA_INVALID_ENUM
} QudaSourceType;

typedef enum QudaGaugeType_s{
    QUDA_WILSON_GAUGE = 0,
    QUDA_STAGGERED_FAT_GAUGE,
    QUDA_STAGGERED_LONG_GAUGE,
    QUDA_STAGGERED_GAUGE,
    QUDA_INVALID_GAUGE = QUDA_INVALID_ENUM,
}QudaGaugeType;

#ifdef __cplusplus
}
#endif

#endif // _ENUM_QUDA_H
