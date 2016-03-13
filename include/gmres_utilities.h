#ifndef _GMRES_UTILITIES
#define _GMRES_UTILITIES

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <complex>
#include <cuComplex.h>
#include <stdio.h>

#include <stdio.h>
#include <color_spinor_field.h>     // cudaColorSpinorField, ColorSpinorParam
#include <invert_quda.h>            // SolverParam, 
#include <util_quda.h>



// This file contains some utilities and auxiliary functions needed for
// F-GMRES-DR which is implemented in "/lib/inv_gmres_quda.cpp".
//-----------------------------------------------------------------------------
namespace utilGMRES {

// Creates neccessary parameters for the inner and outer solver
void fillInnerSolveParam(quda::SolverParam &inner, const quda::SolverParam &outer);

// Solve a general linear system A.x = b. Size of the problem is order.ldA,
// whereas ldA is the leading dimension of A (A(ij) = A[i+j*ldA])
void linearSolve(quda::Complex *x, quda::Complex *b, quda::Complex *A, int order, int ldA);

// Get the eigenvectors (sorted) of a quadratic matrix A. On entry A is the
// problems matrix, on exit its columns are the eigenvectors.
void getRitzVectors(quda::Complex *A, int dim);

// QR factorize the matrix A. On exit A is Q.
// "columns" is also ldA.
void qrFactorize(quda::Complex *A, int rows, int columns);

// Perform deflation of the Arnoldi vectors V and Z. Update all variables.
void deflate(quda::Complex *H, quda::cudaColorSpinorField **V, quda::cudaColorSpinorField **Z,
             quda::Complex *c, quda::Complex *rho, int dim, int ldH, int kAug,
             quda::ColorSpinorParam csParam);
              
// Check if augmented subspace is in bad shape and if clean restart is required.
// If so, return 1 otherwise 0.
// NOTE: For a explanation of the underlying algorithm see Masters Thesis
int checkRestartCriterion(int currentIter, int *deflCount, int *badSubspCount, int maxDefl,
                          int maxBadSubsp, int kAug, double *detGref, quda::cudaColorSpinorField **V);

// Contract a spinor-matrix with an ordinary c-type matrix: spinorOut <-- spinorIn.cMatrixIn
// *in  has rows(cIn) columns, out has columns(cIn) columns
void contractSpinorAndCMatrix(quda::cudaColorSpinorField **out, quda::cudaColorSpinorField **in,
                              quda::Complex *cIn, int rowsCin, int columnsCin, int ldCin);

// Solve the least squares problem min||c - H.y||. Input values are c and H,
// while y is returned. dim(H) = (row, col) with ldH. dim(c) = row, dim(y) = col
void solveLeastSquares(quda::Complex *H, quda::Complex *c, quda::Complex *y,
		       int row, int col, int ldH);
                              
}  // end of namespace utilGMRES            
//-----------------------------------------------------------------------------
#endif
