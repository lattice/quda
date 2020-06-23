/**
   @file quda_define.h
   @brief Macros defined set by the cmake build system.  This file
   should not be edited manually.
 */

/**
 * @def   __COMPUTE_CAPABILITY__
 * @brief This macro sets the target GPU architecture.  Unlike
 * __CUDA_ARCH__, this is defined on host and device.
 */
#define __COMPUTE_CAPABILITY__ 750

/**
 * @def   MAX_MULTI_BLAS_N
 * @brief This macro sets the limit of blas fusion in the multi-blas
 * and multi-reduce kernels
 */
#define MAX_MULTI_BLAS_N 4

/* #undef QUDA_TEX */
#ifdef QUDA_TEX
/**
 * @def   USE_TEXTURE_OBJECTS
 * @brief This macro sets whether we are compiling QUDA with texture
 * support enabled or not
 */
#define USE_TEXTURE_OBJECTS
#undef QUDA_TEX
#endif

/* #undef QUDA_DYNAMIC_CLOVER */
#ifdef QUDA_DYNAMIC_CLOVER
/**
 * @def   DYNAMIC_CLOVER
 * @brief This macro sets whether we are compiling QUDA with dynamic
 * clover inversion support enabled or not
 */
#define DYNAMIC_CLOVER
#undef QUDA_DYNAMIC_CLOVER
#endif

/* #undef QUDA_FLOAT8 */
#ifdef QUDA_FLOAT8
/**
 * @def FLOAT8
 * @brief This macro set whether float8-ordered fields are enabled or
 * not
 */
#define FLOAT8
#undef QUDA_FLOAT8
#endif

#if defined(USE_TEXTURE_OBJECTS) && defined(FLOAT8)
#error "Cannot simultanteously enable QUDA_TEX and QUDA_FLOAT8"
#endif
