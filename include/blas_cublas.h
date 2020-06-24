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
       Batch GEMM
       @param[in] A Matrix field containing the A input matrices
       @param[in] B Matrix field containing the A input matrices
       @param[in/out] C Matrix field containing the result, and matrix to be added
       @param[in] cublas_param Parameter structure defining the GEMM type
       @param[in] Location of the input/output data
       @return Number of flops done in this computation
    */
    long long BatchGEMM(void *A, void *B, void *C, QudaCublasParam cublas_param, QudaFieldLocation location);

  } // namespace cublas

} // namespace quda
