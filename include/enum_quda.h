#ifndef _ENUM_QUDA_H
#define _ENUM_QUDA_H

#include <limits.h>
#define QUDA_INVALID_ENUM INT_MIN

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum QudaMemoryType_s {
    QUDA_MEMORY_DEVICE,
    QUDA_MEMORY_PINNED,
    QUDA_MEMORY_MAPPED,
    QUDA_MEMORY_INVALID = QUDA_INVALID_ENUM
  } QudaMemoryType;

  //
  // Types used in QudaGaugeParam
  //

  typedef enum QudaLinkType_s {
    QUDA_SU3_LINKS,
    QUDA_GENERAL_LINKS,
    QUDA_THREE_LINKS,
    QUDA_MOMENTUM_LINKS,
    QUDA_COARSE_LINKS, // used for coarse-gauge field with multigrid
    QUDA_SMEARED_LINKS, // used for loading and saving gaugeSmeared in the interface
    QUDA_WILSON_LINKS = QUDA_SU3_LINKS, // used by wilson, clover, twisted mass, and domain wall
    QUDA_ASQTAD_FAT_LINKS = QUDA_GENERAL_LINKS,
    QUDA_ASQTAD_LONG_LINKS = QUDA_THREE_LINKS,
    QUDA_ASQTAD_MOM_LINKS  = QUDA_MOMENTUM_LINKS,
    QUDA_ASQTAD_GENERAL_LINKS = QUDA_GENERAL_LINKS,
    QUDA_INVALID_LINKS = QUDA_INVALID_ENUM
  } QudaLinkType;

  typedef enum QudaGaugeFieldOrder_s {
    QUDA_FLOAT_GAUGE_ORDER = 1,
    QUDA_FLOAT2_GAUGE_ORDER = 2,  // no reconstruct and double precision
    QUDA_FLOAT4_GAUGE_ORDER = 4,  // 8 reconstruct single, and 12 reconstruct single, half, quarter
    QUDA_FLOAT8_GAUGE_ORDER = 8,  // 8 reconstruct half and quarter
    QUDA_NATIVE_GAUGE_ORDER,      // used to denote one of the above types in a trait, not used directly
    QUDA_QDP_GAUGE_ORDER,         // expect *gauge[mu], even-odd, spacetime, row-column color
    QUDA_QDPJIT_GAUGE_ORDER,      // expect *gauge[mu], even-odd, complex-column-row-spacetime
    QUDA_CPS_WILSON_GAUGE_ORDER,  // expect *gauge, even-odd, mu, spacetime, column-row color
    QUDA_MILC_GAUGE_ORDER,        // expect *gauge, even-odd, mu, spacetime, row-column order
    QUDA_MILC_SITE_GAUGE_ORDER,   // packed into MILC site AoS [even-odd][spacetime] array, and [dir][row][col] inside
    QUDA_BQCD_GAUGE_ORDER,        // expect *gauge, mu, even-odd, spacetime+halos, column-row order
    QUDA_TIFR_GAUGE_ORDER,        // expect *gauge, mu, even-odd, spacetime, column-row order
    QUDA_TIFR_PADDED_GAUGE_ORDER, // expect *gauge, mu, parity, t, z+halo, y, x/2, column-row order
    QUDA_INVALID_GAUGE_ORDER = QUDA_INVALID_ENUM
  } QudaGaugeFieldOrder;

  typedef enum QudaTboundary_s {
    QUDA_ANTI_PERIODIC_T = -1,
    QUDA_PERIODIC_T = 1,
    QUDA_INVALID_T_BOUNDARY = QUDA_INVALID_ENUM
  } QudaTboundary;

  typedef enum QudaPrecision_s {
    QUDA_QUARTER_PRECISION = 1,
    QUDA_HALF_PRECISION = 2,
    QUDA_SINGLE_PRECISION = 4,
    QUDA_DOUBLE_PRECISION = 8,
    QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM
  } QudaPrecision;

  typedef enum QudaReconstructType_s {
    QUDA_RECONSTRUCT_NO = 18, // store all 18 real numbers explicitly
    QUDA_RECONSTRUCT_12 = 12, // reconstruct from 12 real numbers
    QUDA_RECONSTRUCT_8 = 8,  // reconstruct from 8 real numbers
    QUDA_RECONSTRUCT_9 = 9,   // used for storing HISQ long-link variables
    QUDA_RECONSTRUCT_13 = 13, // used for storing HISQ long-link variables
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
    QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH,
    QUDA_DOMAIN_WALL_DSLASH,
    QUDA_DOMAIN_WALL_4D_DSLASH,
    QUDA_MOBIUS_DWF_DSLASH,
    QUDA_MOBIUS_DWF_EOFA_DSLASH,
    QUDA_STAGGERED_DSLASH,
    QUDA_ASQTAD_DSLASH,
    QUDA_TWISTED_MASS_DSLASH,
    QUDA_TWISTED_CLOVER_DSLASH,
    QUDA_LAPLACE_DSLASH,
    QUDA_COVDEV_DSLASH,
    QUDA_INVALID_DSLASH = QUDA_INVALID_ENUM
  } QudaDslashType;

  typedef enum QudaInverterType_s {
    QUDA_CG_INVERTER,
    QUDA_BICGSTAB_INVERTER,
    QUDA_GCR_INVERTER,
    QUDA_MR_INVERTER,
    QUDA_MPBICGSTAB_INVERTER,
    QUDA_SD_INVERTER,
    QUDA_XSD_INVERTER,
    QUDA_PCG_INVERTER,
    QUDA_MPCG_INVERTER,
    QUDA_EIGCG_INVERTER,
    QUDA_INC_EIGCG_INVERTER,
    QUDA_GMRESDR_INVERTER,
    QUDA_GMRESDR_PROJ_INVERTER,
    QUDA_GMRESDR_SH_INVERTER,
    QUDA_FGMRESDR_INVERTER,
    QUDA_MG_INVERTER,
    QUDA_BICGSTABL_INVERTER,
    QUDA_CGNE_INVERTER,
    QUDA_CGNR_INVERTER,
    QUDA_CG3_INVERTER,
    QUDA_CG3NE_INVERTER,
    QUDA_CG3NR_INVERTER,
    QUDA_CA_CG_INVERTER,
    QUDA_CA_CGNE_INVERTER,
    QUDA_CA_CGNR_INVERTER,
    QUDA_CA_GCR_INVERTER,
    QUDA_INVALID_INVERTER = QUDA_INVALID_ENUM
  } QudaInverterType;

  typedef enum QudaEigType_s {
    QUDA_EIG_TR_LANCZOS,     // Thick restarted lanczos solver
    QUDA_EIG_BLK_TR_LANCZOS, // Block Thick restarted lanczos solver
    QUDA_EIG_IR_LANCZOS,     // Implicitly Restarted Lanczos solver (not implemented)
    QUDA_EIG_IR_ARNOLDI,     // Implicitly Restarted Arnoldi solver (not implemented)
    QUDA_EIG_INVALID = QUDA_INVALID_ENUM
  } QudaEigType;

  /** S=smallest L=largest
      R=real M=modulus I=imaniary **/
  typedef enum QudaEigSpectrumType_s {
    QUDA_SPECTRUM_SR_EIG,
    QUDA_SPECTRUM_LR_EIG,
    QUDA_SPECTRUM_SM_EIG,
    QUDA_SPECTRUM_LM_EIG,
    QUDA_SPECTRUM_SI_EIG,
    QUDA_SPECTRUM_LI_EIG,
    QUDA_SPECTRUM_INVALID = QUDA_INVALID_ENUM
  } QudaEigSpectrumType;

  typedef enum QudaSolutionType_s {
    QUDA_MAT_SOLUTION,
    QUDA_MATDAG_MAT_SOLUTION,
    QUDA_MATPC_SOLUTION,
    QUDA_MATPC_DAG_SOLUTION,
    QUDA_MATPCDAG_MATPC_SOLUTION,
    QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION,
    QUDA_INVALID_SOLUTION = QUDA_INVALID_ENUM
  } QudaSolutionType;

  typedef enum QudaSolveType_s {
    QUDA_DIRECT_SOLVE,
    QUDA_NORMOP_SOLVE,
    QUDA_DIRECT_PC_SOLVE,
    QUDA_NORMOP_PC_SOLVE,
    QUDA_NORMERR_SOLVE,
    QUDA_NORMERR_PC_SOLVE,
    QUDA_NORMEQ_SOLVE = QUDA_NORMOP_SOLVE, // deprecated
    QUDA_NORMEQ_PC_SOLVE = QUDA_NORMOP_PC_SOLVE, // deprecated
    QUDA_INVALID_SOLVE = QUDA_INVALID_ENUM
  } QudaSolveType;

  typedef enum QudaMultigridCycleType_s {
    QUDA_MG_CYCLE_VCYCLE,
    QUDA_MG_CYCLE_FCYCLE,
    QUDA_MG_CYCLE_WCYCLE,
    QUDA_MG_CYCLE_RECURSIVE,
    QUDA_MG_CYCLE_INVALID = QUDA_INVALID_ENUM
  } QudaMultigridCycleType;

  typedef enum QudaSchwarzType_s {
    QUDA_ADDITIVE_SCHWARZ,
    QUDA_MULTIPLICATIVE_SCHWARZ,
    QUDA_INVALID_SCHWARZ = QUDA_INVALID_ENUM
  } QudaSchwarzType;

  typedef enum QudaResidualType_s {
    QUDA_L2_RELATIVE_RESIDUAL = 1, // L2 relative residual (default)
    QUDA_L2_ABSOLUTE_RESIDUAL = 2, // L2 absolute residual
    QUDA_HEAVY_QUARK_RESIDUAL = 4, // Fermilab heavy quark residual
    QUDA_INVALID_RESIDUAL = QUDA_INVALID_ENUM
  } QudaResidualType;

  // Which basis to use for CA algorithms
  typedef enum QudaCABasis_s {
    QUDA_POWER_BASIS,
    QUDA_CHEBYSHEV_BASIS,
    QUDA_INVALID_BASIS = QUDA_INVALID_ENUM
  } QudaCABasis;

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

  typedef enum QudaSolverNormalization_s {
    QUDA_DEFAULT_NORMALIZATION, // leave source and solution untouched
    QUDA_SOURCE_NORMALIZATION  // normalize such that || src || = 1
  } QudaSolverNormalization;

  typedef enum QudaPreserveSource_s {
    QUDA_PRESERVE_SOURCE_NO,  // use the source for the residual
    QUDA_PRESERVE_SOURCE_YES, // keep the source intact
    QUDA_PRESERVE_SOURCE_INVALID = QUDA_INVALID_ENUM
  } QudaPreserveSource;

  typedef enum QudaDiracFieldOrder_s {
    QUDA_INTERNAL_DIRAC_ORDER,    // internal dirac order used, varies on precision and dslash type
    QUDA_DIRAC_ORDER,             // even-odd, color inside spin
    QUDA_QDP_DIRAC_ORDER,         // even-odd, spin inside color
    QUDA_QDPJIT_DIRAC_ORDER,      // even-odd, complex-color-spin-spacetime
    QUDA_CPS_WILSON_DIRAC_ORDER,  // odd-even, color inside spin
    QUDA_LEX_DIRAC_ORDER,         // lexicographical order, color inside spin
    QUDA_TIFR_PADDED_DIRAC_ORDER, // padded z dimension for TIFR RHMC code
    QUDA_INVALID_DIRAC_ORDER = QUDA_INVALID_ENUM
  } QudaDiracFieldOrder;

  typedef enum QudaCloverFieldOrder_s {
    QUDA_FLOAT_CLOVER_ORDER = 1,  // even-odd float ordering
    QUDA_FLOAT2_CLOVER_ORDER = 2, // even-odd float2 ordering
    QUDA_FLOAT4_CLOVER_ORDER = 4, // even-odd float4 ordering
    QUDA_PACKED_CLOVER_ORDER,     // even-odd, QDP packed
    QUDA_QDPJIT_CLOVER_ORDER,     // (diagonal / off-diagonal)-chirality-spacetime
    QUDA_BQCD_CLOVER_ORDER,       // even-odd, super-diagonal packed and reordered
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
    QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC,
    QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC,
    QUDA_CLOVERPC_DIRAC,
    QUDA_DOMAIN_WALL_DIRAC,
    QUDA_DOMAIN_WALLPC_DIRAC,
    QUDA_DOMAIN_WALL_4D_DIRAC,
    QUDA_DOMAIN_WALL_4DPC_DIRAC,
    QUDA_MOBIUS_DOMAIN_WALL_DIRAC,
    QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC,
    QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC,
    QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC,
    QUDA_STAGGERED_DIRAC,
    QUDA_STAGGEREDPC_DIRAC,
    QUDA_ASQTAD_DIRAC,
    QUDA_ASQTADPC_DIRAC,
    QUDA_TWISTED_MASS_DIRAC,
    QUDA_TWISTED_MASSPC_DIRAC,
    QUDA_TWISTED_CLOVER_DIRAC,
    QUDA_TWISTED_CLOVERPC_DIRAC,
    QUDA_COARSE_DIRAC,
    QUDA_COARSEPC_DIRAC,
    QUDA_GAUGE_LAPLACE_DIRAC,
    QUDA_GAUGE_LAPLACEPC_DIRAC,
    QUDA_GAUGE_COVDEV_DIRAC,
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
    QUDA_PARITY_SITE_SUBSET = 1,
    QUDA_FULL_SITE_SUBSET = 2,
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
    QUDA_FLOAT_FIELD_ORDER = 1,               // spin-color-complex-space
    QUDA_FLOAT2_FIELD_ORDER = 2,              // (spin-color-complex)/2-space-(spin-color-complex)%2
    QUDA_FLOAT4_FIELD_ORDER = 4,              // (spin-color-complex)/4-space-(spin-color-complex)%4
    QUDA_FLOAT8_FIELD_ORDER = 8,              // (spin-color-complex)/8-space-(spin-color-complex)%8
    QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,        // CPS/QDP++ ordering
    QUDA_SPACE_COLOR_SPIN_FIELD_ORDER,        // QLA ordering (spin inside color)
    QUDA_QDPJIT_FIELD_ORDER,                  // QDP field ordering (complex-color-spin-spacetime)
    QUDA_QOP_DOMAIN_WALL_FIELD_ORDER,         // QOP domain-wall ordering
    QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER, // TIFR RHMC ordering
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
    QUDA_CHIRAL_GAMMA_BASIS,
    QUDA_INVALID_GAMMA_BASIS = QUDA_INVALID_ENUM
  } QudaGammaBasis;

  typedef enum QudaSourceType_s {
    QUDA_POINT_SOURCE,
    QUDA_RANDOM_SOURCE,
    QUDA_CONSTANT_SOURCE,
    QUDA_SINUSOIDAL_SOURCE,
    QUDA_CORNER_SOURCE,
    QUDA_INVALID_SOURCE = QUDA_INVALID_ENUM
  } QudaSourceType;

  typedef enum QudaNoiseType_s {
    QUDA_NOISE_GAUSS,
    QUDA_NOISE_UNIFORM,
    QUDA_NOISE_INVALID = QUDA_INVALID_ENUM
  } QudaNoiseType;

  // used to select projection method for deflated solvers
  typedef enum QudaProjectionType_s {
      QUDA_MINRES_PROJECTION,
      QUDA_GALERKIN_PROJECTION,
      QUDA_INVALID_PROJECTION = QUDA_INVALID_ENUM
  } QudaProjectionType;

  // used to select checkerboard preconditioning method
  typedef enum QudaPCType_s { QUDA_4D_PC = 4, QUDA_5D_PC = 5, QUDA_PC_INVALID = QUDA_INVALID_ENUM } QudaPCType;

  typedef enum QudaTwistFlavorType_s {
    QUDA_TWIST_SINGLET = 1,
    QUDA_TWIST_NONDEG_DOUBLET = +2,
    QUDA_TWIST_DEG_DOUBLET = -2,
    QUDA_TWIST_NO = 0,
    QUDA_TWIST_INVALID = QUDA_INVALID_ENUM
  } QudaTwistFlavorType;

  typedef enum QudaTwistDslashType_s {
    QUDA_DEG_TWIST_INV_DSLASH,
    QUDA_DEG_DSLASH_TWIST_INV,
    QUDA_DEG_DSLASH_TWIST_XPAY,
    QUDA_NONDEG_DSLASH,
    QUDA_DSLASH_INVALID = QUDA_INVALID_ENUM
  } QudaTwistDslashType;

  typedef enum QudaTwistCloverDslashType_s {
    QUDA_DEG_CLOVER_TWIST_INV_DSLASH,
    QUDA_DEG_DSLASH_CLOVER_TWIST_INV,
    QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY,
    QUDA_TC_DSLASH_INVALID = QUDA_INVALID_ENUM
  } QudaTwistCloverDslashType;

  typedef enum QudaTwistGamma5Type_s {
    QUDA_TWIST_GAMMA5_DIRECT,
    QUDA_TWIST_GAMMA5_INVERSE,
    QUDA_TWIST_GAMMA5_INVALID = QUDA_INVALID_ENUM
  } QudaTwistGamma5Type;

  typedef enum QudaUseInitGuess_s {
    QUDA_USE_INIT_GUESS_NO,
    QUDA_USE_INIT_GUESS_YES,
    QUDA_USE_INIT_GUESS_INVALID = QUDA_INVALID_ENUM
  } QudaUseInitGuess;

  typedef enum QudaDeflatedGuess_s {
    QUDA_DEFLATED_GUESS_NO,
    QUDA_DEFLATED_GUESS_YES,
    QUDA_DEFLATED_GUESS_INVALID = QUDA_INVALID_ENUM
  } QudaDeflatedGuess;

  typedef enum QudaComputeNullVector_s {
    QUDA_COMPUTE_NULL_VECTOR_NO,
    QUDA_COMPUTE_NULL_VECTOR_YES,
    QUDA_COMPUTE_NULL_VECTOR_INVALID = QUDA_INVALID_ENUM
  } QudaComputeNullVector;

  typedef enum QudaSetupType_s {
    QUDA_NULL_VECTOR_SETUP,
    QUDA_TEST_VECTOR_SETUP,
    QUDA_INVALID_SETUP_TYPE = QUDA_INVALID_ENUM
  } QudaSetupType;

  typedef enum QudaBoolean_s {
    QUDA_BOOLEAN_FALSE = 0,
    QUDA_BOOLEAN_TRUE = 1,
    QUDA_BOOLEAN_INVALID = QUDA_INVALID_ENUM
  } QudaBoolean;

  // define these for backwards compatibility
#define QUDA_BOOLEAN_NO QUDA_BOOLEAN_FALSE
#define QUDA_BOOLEAN_YES QUDA_BOOLEAN_TRUE

  typedef enum QudaDirection_s {
    QUDA_BACKWARDS = -1,
    QUDA_FORWARDS = +1,
    QUDA_BOTH_DIRS = 2
  } QudaDirection;

  typedef enum QudaLinkDirection_s {
    QUDA_LINK_BACKWARDS,
    QUDA_LINK_FORWARDS,
    QUDA_LINK_BIDIRECTIONAL
  } QudaLinkDirection;

  typedef enum QudaFieldGeometry_s {
    QUDA_SCALAR_GEOMETRY = 1,
    QUDA_VECTOR_GEOMETRY = 4,
    QUDA_TENSOR_GEOMETRY = 6,
    QUDA_COARSE_GEOMETRY = 8,
    QUDA_INVALID_GEOMETRY = QUDA_INVALID_ENUM
  } QudaFieldGeometry;

  typedef enum QudaGhostExchange_s {
    QUDA_GHOST_EXCHANGE_NO,
    QUDA_GHOST_EXCHANGE_PAD,
    QUDA_GHOST_EXCHANGE_EXTENDED,
    QUDA_GHOST_EXCHANGE_INVALID = QUDA_INVALID_ENUM
  } QudaGhostExchange;

  typedef enum QudaStaggeredPhase_s {
    QUDA_STAGGERED_PHASE_NO = 0,
    QUDA_STAGGERED_PHASE_MILC = 1,
    QUDA_STAGGERED_PHASE_CPS = 2,
    QUDA_STAGGERED_PHASE_TIFR = 3,
    QUDA_STAGGERED_PHASE_INVALID = QUDA_INVALID_ENUM
  } QudaStaggeredPhase;

  typedef enum QudaContractType_s {
    QUDA_CONTRACT_TYPE_OPEN, // Open spin elementals
    QUDA_CONTRACT_TYPE_DR,   // DegrandRossi
    QUDA_CONTRACT_TYPE_INVALID = QUDA_INVALID_ENUM
  } QudaContractType;

  typedef enum QudaContractGamma_s {
    QUDA_CONTRACT_GAMMA_I = 0,
    QUDA_CONTRACT_GAMMA_G1 = 1,
    QUDA_CONTRACT_GAMMA_G2 = 2,
    QUDA_CONTRACT_GAMMA_G3 = 3,
    QUDA_CONTRACT_GAMMA_G4 = 4,
    QUDA_CONTRACT_GAMMA_G5 = 5,
    QUDA_CONTRACT_GAMMA_G1G5 = 6,
    QUDA_CONTRACT_GAMMA_G2G5 = 7,
    QUDA_CONTRACT_GAMMA_G3G5 = 8,
    QUDA_CONTRACT_GAMMA_G4G5 = 9,
    QUDA_CONTRACT_GAMMA_S12 = 10,
    QUDA_CONTRACT_GAMMA_S13 = 11,
    QUDA_CONTRACT_GAMMA_S14 = 12,
    QUDA_CONTRACT_GAMMA_S21 = 13,
    QUDA_CONTRACT_GAMMA_S23 = 14,
    QUDA_CONTRACT_GAMMA_S34 = 15,
    QUDA_CONTRACT_GAMMA_INVALID = QUDA_INVALID_ENUM
  } QudaContractGamma;

  typedef enum QudaWFlowType_s {
    QUDA_WFLOW_TYPE_WILSON,
    QUDA_WFLOW_TYPE_SYMANZIK,
    QUDA_WFLOW_TYPE_INVALID = QUDA_INVALID_ENUM
  } QudaWFlowType;

  // Allows to choose an appropriate external library
  typedef enum QudaExtLibType_s {
    QUDA_CUSOLVE_EXTLIB,
    QUDA_EIGEN_EXTLIB,
    QUDA_MAGMA_EXTLIB,
    QUDA_EXTLIB_INVALID = QUDA_INVALID_ENUM
  } QudaExtLibType;

#ifdef __cplusplus
}
#endif

#endif // _ENUM_QUDA_H
