#include <quda_internal.h>

#pragma once

namespace quda {
  namespace cublas {

    /**
       @brief Create the CUBLAS context
    */
    void init();

    /**
       @brief Destroy the CUBLAS context
    */
    void destroy();

    /**
       Batch inversion the matrix field using an LU decomposition method.
       @param[out] Ainv Matrix field containing the inverse matrices
       @param[in] A Matrix field containing the input matrices
       @param[in] n Dimension each matrix
       @param[in] batch Problem batch size
       @param[in] precision Precision of the input/output data
       @param[in] Location of the input/output data
       @return Number of flops done in this computation
    */
    long long BatchInvertMatrix(void *Ainv, void *A, const int n, const uint64_t batch, QudaPrecision precision,
                                QudaFieldLocation location);

    /**
       Batch GEMM. This function performs N GEMM type operations in a batched
       fashion. It constructs arrays of pointers to the data for the Nth operation
       before calling <T>gemmBatched(). Pointers may not alias data.
       @param[in] A Matrix field containing the A input matrices
       @param[in] B Matrix field containing the A input matrices
       @param[in/out] C Matrix field containing the result, and matrix to be added
       @param[in] cublas_param Parameter structure defining the batched GEMM type
       @param[in] Location of the input/output data
       @return Number of flops done in this computation
    */
    long long BatchGEMM(void *A, void *B, void *C, QudaCublasParam cublas_param, QudaFieldLocation location);

    /**
       Strided Batch GEMM. This function performs N GEMM type operations in a 
       strided batched fashion. If the user passes 

       stride<A,B,C> = -1 

       it deduces the strides for the A, B, and C arrays from the matrix dimensions, 
       leading dims, etc, and will behave identically to the batched GEMM.
       If any of the stride<A,B,C> values passed in the parameter structure are 
       greater than or equal to 0, the routine accepts the user's values instead. 

       Example 1: If the user passes 

       strideA = 0 

       the routine will use only the first matrix in the A array and compute

       a * A_{0} * B_{n} + b * C_{n}

       where n is the batch index.

       @param[in] A Matrix field containing the A input matrices
       @param[in] B Matrix field containing the A input matrices
       @param[in/out] C Matrix field containing the result, and matrix to be added
       @param[in] cublas_param Parameter structure defining the GEMM type
       @param[in] Location of the input/output data
       @return Number of flops done in this computation
    */
    long long stridedBatchGEMM(void *A, void *B, void *C, QudaCublasParam cublas_param, QudaFieldLocation location);

  } // namespace cublas

} // namespace quda
