#include <gmres_utilities.h>

#include <stdio.h>               // standard C++ includes
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm>             // std::swap() and std::max()

#include <quda_internal.h>       // quda includes
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <face_quda.h>

#ifdef MAGMA_LIB
#include <magma.h>
#include <magma_operators.h>     // manipulating complex magma numbers
#endif


// This file implements some utilities and auxiliary functions needed for
// F-GMRES-DR which is implemented in "/lib/inv_gmres_quda.cpp".
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
namespace utilGMRES {

   // Creates necessary parameters for the inner and outer solver
   //--------------------------------------------------------------------------
   void fillInnerSolveParam(quda::SolverParam &inner, const quda::SolverParam &outer) {

      // Define inner solvers precision
      inner.tol      = outer.tol_precondition;
      inner.maxiter  = outer.maxiter_precondition;
      inner.delta    = 1e-20;    // no reliable updates within the inner solver

      // Preconditioners are uni-precision solvers
      inner.precision        = outer.precision_precondition;
      inner.precision_sloppy = outer.precision_precondition;
            
      // Set the inner flops counter
      inner.iter = 0;
      inner.gflops = 0;
      inner.secs = 0;
      
      // Tell the inner solver it is an inner solver (see e.g., line 26 of "inv_mr_quda.ccp")
      inner.inv_type_precondition = QUDA_GCR_INVERTER;
            
      // Preserve the source depending on precision
      if (outer.precision_sloppy != outer.precision_precondition)
         inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
      else
         inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
   }
   //--------------------------------------------------------------------------


   // Solve a general linear system A.x = b. Size of the problem is order.ldA,
   // whereas ldA is the leading dimension of A (A(ij) = A[i+j*ldA])
   //--------------------------------------------------------------------------
   void linearSolve(quda::Complex *x, quda::Complex *b, quda::Complex *A, int order, int ldA) {
#ifdef MAGMA_LIB
      // define variables required by the magma solver
      magmaDoubleComplex *tmpA, *tmpB;
      magma_int_t *ipiv;
      magma_int_t err, info = 0;
      
      // allocate pinned memory for magma
      magma_zmalloc_pinned( &tmpA, order*ldA );
      magma_zmalloc_pinned( &tmpB, order );
      magma_imalloc_pinned( &ipiv, order );
      
      // duplicate the data (this is necessary, since A and b get destroyed during solve)
      memcpy( tmpA, A, order*ldA*sizeof(magmaDoubleComplex) );
      memcpy( tmpB, b, order*sizeof(magmaDoubleComplex) );
                
      // perform the actual solve, catch possibly occuring error
      err = magma_zgesv(order, 1, tmpA, ldA, ipiv, tmpB, order, &info);
      if (err != 0) {
         printf("\nError in linearSolveGMRES, with INFO(magma_zgesv) = %d, exit ...\n", info);
         exit(-1);   
      }
                      
      // copy back solution (b is now the solution vector x)
      memcpy( x, tmpB, order*sizeof(magmaDoubleComplex) );
               
      // delete all temporary data
      magma_free_pinned(tmpA);
      magma_free_pinned(tmpB);
      magma_free_pinned(ipiv);
#endif
      return;
   }
   //--------------------------------------------------------------------------


   // Get the eigenvectors (sorted) of a quadratic matrix A. On entry A is the
   // problems matrix, on exit its columns are the eigenvectors.
   //--------------------------------------------------------------------------
   void getRitzVectors(quda::Complex *A, int dim) {
#ifdef MAGMA_LIB
      // define variables required by the magma solver
      magmaDoubleComplex *leftVecs, *rightVecs;    // for the eigenvectors
      magmaDoubleComplex *lambda;                  // for the eigenvalues
      magmaDoubleComplex *work;                    // workspace
      double *rwork;                               // workspace
      magma_int_t nb, lwork, err, info = 0;
      
      magma_vec_t jobvl, jobvr;                    // calculate left or right eigenpairs
      jobvl = MagmaNoVec;
      jobvr = MagmaVec;
   
      nb = magma_get_zgehrd_nb(dim);               // get ideal workspace blocksize
      lwork = (1+2*nb)*dim;
      
      // allocate memory
      magma_zmalloc_pinned( &rightVecs, dim*dim );
      magma_zmalloc_pinned( &lambda, dim );
      magma_zmalloc_pinned( &work, lwork );
      magma_dmalloc_pinned( &rwork, 2*dim );
      
      // perform the actual solve, catch possibly occurring error
      err = magma_zgeev(jobvl, jobvr, dim, (magmaDoubleComplex*)A, dim, lambda, leftVecs,
                        dim, rightVecs, dim, work, lwork, rwork, &info);
      if (err != 0) {
         printf("\nError in getRitzVectorsGMRES, with INFO(magma_zgeev) = %d, exit ...\n", info);
         exit(-1);   
      }
                      
      // now sort the eigenvalues/vectors in ascending order (by magnitude)
      // this uses insertion-sort with an permutation matrix to avoid repeated copying
      magmaDoubleComplex tmp;
      int i, tmp2;
      int *permTable = new int[dim];
      for (int i=0; i<dim; i++)
	 permTable[i] = i;
      
      for (int j=1; j<dim; j++) {
	 tmp  = lambda[j];
	 tmp2 = permTable[j];
	 i = j;
	 while ( i>0 && fabs(lambda[i-1]) > fabs(tmp) ) {
	    lambda[i] = lambda[i-1];
	    permTable[i] = permTable[i-1];
	    i--;
	 }
	 lambda[i] = tmp;
	 permTable[i] = tmp2;
      }
                      
      // rearrange the eigenvectors in A according to the permutation matrix
      for (int i=0; i<dim; i++)
	 memcpy( &A[i*dim], &rightVecs[permTable[i]*dim], dim*sizeof(magmaDoubleComplex) );
                      
      // delete all temporary data
      magma_free_pinned(rightVecs);
      magma_free_pinned(lambda);
      magma_free_pinned(work);
      magma_free_pinned(rwork);
      delete[] permTable;
#endif
      return;
   }
   //--------------------------------------------------------------------------


   // QR factorize the matrix A. On exit A is Q.
   // "columns" is also ldA.
   //--------------------------------------------------------------------------
   void qrFactorize(quda::Complex *A, int rows, int columns) {
#ifdef MAGMA_LIB
      // define variables required by the magma solver
      magmaDoubleComplex *tau, *work;
      magma_int_t nb, lwork, err, info = 0;
      
      nb = magma_get_zgeqrf_nb( std::max(rows, columns) );  // get ideal workspace blocksize
      lwork = std::max( columns*nb, 2*nb*nb );
      
      // allocate memory
      magma_zmalloc_pinned( &tau, std::min(rows, columns) );
      magma_zmalloc_pinned( &work, lwork );
      
      // generate the raw data for A = QR
      err = magma_zgeqrf( rows, columns, (magmaDoubleComplex*)A,
                          rows, tau, work, lwork, &info );
      if (err != 0) {
         printf("\nError in qrFactorizeGMRES, with INFO(magma_zgeqrf) = %d, exit ...\n", info);
         exit(-1);   
      }
                      
      // extract the Q-matrix out of this data
      err = magma_zungqr2( rows, columns, std::min(rows, columns),
                           (magmaDoubleComplex*)A, rows, tau, &info );
      if (err != 0) {
         printf("\nError in qrFactorizeGMRES, with INFO(magma_zungqr2) = %d, exit ...\n", info);
         exit(-1);   
      }
      
      // delete all temporary data
      magma_free_pinned(tau);
      magma_free_pinned(work);
#endif      
      return;
   }
   //--------------------------------------------------------------------------


   // Perform deflation of the Arnoldi vectors V and Z. Update all variables.
   //--------------------------------------------------------------------------
   void deflate(quda::Complex *H, quda::cudaColorSpinorField **V, quda::cudaColorSpinorField **Z,
                quda::Complex *c, quda::Complex *rho, int dim, int ldH, int kAug,
                quda::ColorSpinorParam csParam) {
            
      // step a): calculate Hm + fm.adj(hm), where Hm !def= Haux
      //-----------------------------------------------------------------------
      // get some space for auxiliary matrices
      quda::Complex *Haux = new quda::Complex[dim*dim];
      quda::Complex *hm   = new quda::Complex[dim];
      quda::Complex *fm   = new quda::Complex[dim];
               
      // build Haux = adj(Hm)
      for (int i=0; i<dim; i++) {
         for (int j=0; j<dim; j++) {
            Haux[i+j*dim] = conj(H[j+i*ldH]);
         }
         hm[i] = conj(H[(ldH-1)+i*ldH]);
      }
               
      // calculate the vector fm
      linearSolve(fm, hm, Haux, dim, dim);
               
      // compute Hm + fm.adj(hm) -- Haux above cant be used, cause its adjoint,
      // but it can be recycled however
      for (int i=0; i<dim; i++)
         for (int j=0; j<dim; j++) {
            Haux[i+j*dim] = H[i+j*ldH] + fm[i]*conj(hm[j]);
         }
      //---------------------------------------------------------------------
            
      // step b) and c) together:
      // get the kAug smallest eigenpairs (lambdaI, gI) of Hm + fm.adj(hm)
      // and calculate the matrix Gk1
      //---------------------------------------------------------------------
      // get space for another matrix
      quda::Complex *Gk1 = new quda::Complex[ldH*(kAug+1)]();
            
      // get ALL Ritz vectors
      getRitzVectors(Haux, dim);
               
      // fill the matrix Gk1 with numbers -- first part
      for (int i=0; i<dim; i++)
         for (int j=0; j<kAug; j++) {
            Gk1[i+j*ldH] = Haux[i+j*dim];
         }
               
      // and now the last column of Gk1
      for (int i=0; i<ldH; i++) {
         Gk1[i+kAug*ldH] = c[i];
         for (int j=0; j<dim; j++) {
            Gk1[i+kAug*ldH] -= H[i+j*ldH]*rho[j];
         }
      }
      //-----------------------------------------------------------------------
            
      // step d) and e):
      // QR-factorize Gk1 and update H, V and Z
      //-----------------------------------------------------------------------
      // QR-factorization: dim(Gk1) = (ldH, kAug+1)   
      qrFactorize(Gk1, ldH, (kAug+1)); // ATTENTION: Gk1 is now Qk1
               
      // calculate Hk = adj(Qk1).Hm.Qk
      // first step: Haux2 = Hm.Qk
      quda::Complex *Haux2 = new quda::Complex[ldH*kAug]();
      for (int i=0; i<ldH; i++)  // Haux = Hm.Qk
         for (int j=0; j<kAug; j++) {
            for (int m=0; m<dim; m++) {
               Haux2[i+j*ldH] += H[i+m*ldH] * Gk1[m+j*ldH];
            }
         }
      
      // second step: Hk = adj(Qk1).Haux2, i.e., Hk = adj(Qk1).Hm.Qk
      std::memset( H, 0.0, dim*ldH*sizeof(quda::Complex) );
      for (int i=0; i<(kAug+1); i++)   // Hk = adj(Qk1).Haux2
         for (int j=0; j<kAug; j++) {
            for (int m=0; m<ldH; m++) {
               H[i+j*ldH] += conj(Gk1[m+i*ldH]) * Haux2[m+j*ldH];
            }
         }
               
      // calculate Vk1 = Vm1.Qk1 and Zk = Zm.Qk
      //-----------------------------------------------------------------------
      // allocate space for the auxiliary fields
      csParam.create = QUDA_ZERO_FIELD_CREATE;  // defensive measure
      quda::cudaColorSpinorField **tmpField = new quda::cudaColorSpinorField*[kAug+1];
      for (int i=0; i<(kAug+1); i++)
         tmpField[i] = new quda::cudaColorSpinorField(*V[0], csParam);
                  
      // calculate the matrix product tmpField = Vm1.Qk1
      contractSpinorAndCMatrix(tmpField, V, Gk1, ldH, (kAug+1), ldH);
               
      // copy Vk1 = tmpField
      for (int i=0; i<(kAug+1); i++)
         quda::copyCuda( *V[i], *tmpField[i] );
         
      // calculate the matrix product tmpField = Zm.Qk
      contractSpinorAndCMatrix(tmpField, Z, Gk1, (ldH-1), kAug, ldH);
               
      // copy Zm = tmpField
      for (int i=0; i<(kAug+1); i++)
         quda::copyCuda( *Z[i], *tmpField[i] );
                  
      // delete temporary data
      for (int i=0; i<(kAug+1); i++)
         delete tmpField[i];
      delete[] tmpField;
      //-----------------------------------------------------------------------
                  
      // clear all memory that is not needed anymore
      //-----------------------------------------------------------------------
      delete[] Haux;
      delete[] Haux2;
      delete[] hm;
      delete[] fm;
      delete[] Gk1;
      //-----------------------------------------------------------------------
      return;
   }
   //--------------------------------------------------------------------------


   // Check if augmented subspace is in bad shape and if clean restart is required.
   // If so, return 1 otherwise 0.
   //--------------------------------------------------------------------------
   int checkRestartCriterion(int currentIter, int *deflCount, int *badSubspCount, int maxDefl,
                             int maxBadSubsp, int kAug, double *detGref, quda::cudaColorSpinorField **V) {
#ifdef MAGMA_LIB
      // check if we have a condition where restart-checking is mandatory
      //-----------------------------------------------------------------------
      if (currentIter == 1 || *deflCount >= maxDefl) {
	 *badSubspCount = 0;
	 *deflCount     = 0;
	 return 1;
      }
      
      // allocate memory and calculate the Gram-Matrix of the first
      // kAug+1 vectors of V[i]
      //-----------------------------------------------------------------------
      int ldG = (kAug+1);
      quda::Complex *G = new quda::Complex[ldG*ldG];
      magma_int_t err, *ipiv, info = 0;
      magma_imalloc_pinned( &ipiv, ldG );
      
      // Gram-Matrix is symmetric, so we just have to get the upper triangle
      // the rest can be copied
      for (int j=0; j<ldG; j++)
         for (int i=0; i<=j; i++) {
            G[i+j*ldG] = quda::cDotProductCuda(*V[i], *V[j]);
            if (i != j)
               G[j+i*ldG] = conj(G[i+j*ldG]);
         }
            
      // calculate |det(G)| which is a criterion for the linear dependence
      // of the vectors V[i]
      // MAGMA LU-factorization will be used to achieve this, det(G) is then
      // proportional to the product of all diagonal coefficients
      //-----------------------------------------------------------------------
      err = magma_zgetrf( ldG, ldG, (magmaDoubleComplex*)G, ldG, ipiv, &info );
      if (err != 0) {
         printf("\nError in checkRestartCriterionGMRES, with INFO(magma_zgetrf) = %d, exit ...\n", info);
         exit(-1);   
      }
                      
      quda::Complex detGram = 1.0;
      for (int i=0; i<ldG; i++) {
         detGram *= G[i+i*ldG];
      }
                      
      // clean memory which is not used anymore
      //-----------------------------------------------------------------------
      delete [] G;
      magma_free_pinned(ipiv);
      
      // if we have done the 2nd iteration, |det(G)| will be stored and used as
      // a reference for future restart checks
      //-----------------------------------------------------------------------
      if (currentIter == 2)
	 *detGref = abs(detGram);   // we are only interested in the absolute value
      
      // check if linear independence of subspace vectors has become worse and
      // count how often this happened
      // if in between one subspace should be ok, reset the counter
      //-----------------------------------------------------------------------
      if (abs(detGram) < *detGref)
	 *badSubspCount += 1;
      else
	 *badSubspCount = 0;
                   
      // finally check if one of our restart criteria got violated
      //-----------------------------------------------------------------------
      if (*badSubspCount >= maxBadSubsp) {
	 *badSubspCount = 0;   // reset all counters
	 *deflCount     = 0;
	 return 1;
      }
      else {
	 *deflCount += 1;      // increment deflation counter
	 return 0;
      }
#endif
      return 1;   // in case MAGMA isnt available         
   }
   //--------------------------------------------------------------------------
   
   
   // Contract a spinor-matrix with an ordinary c-type matrix
   // spinorOut <-- spinorIn.cMatrixIn
   // *in  has rows(cIn) columns 
   // *out has columns(cIn) columns
   //--------------------------------------------------------------------------
   void contractSpinorAndCMatrix(quda::cudaColorSpinorField **out, quda::cudaColorSpinorField **in,
                                 quda::Complex *cIn, int rowsCin, int columnsCin, int ldCin) {
                                 
      // calculate the matrix product
      for (int i=0; i<columnsCin; i++) {
      
         // as a defensive measure zero all rows of *out, before doing the contraction
         quda::zeroCuda(*out[i]);
         
         // do the contraction
         for (int j=0; j<rowsCin; j++)
            quda::caxpyCuda( cIn[j+i*ldCin], *in[j], *out[i] );
      }                          
   }
   //--------------------------------------------------------------------------
   
   
   // Solve the least squares problem min||c - H.y||. Input values are c and H,
   // while y is returned. dim(H) = (row, col) with ldH. dim(c) = row, dim(y) = col
   //--------------------------------------------------------------------------
   void solveLeastSquares(quda::Complex *H, quda::Complex *c, quda::Complex *y,
		          int row, int col, int ldH) {
#ifdef MAGMA_LIB
      // define the required variables
      magmaDoubleComplex *tmpA, *tmpB, *hwork;
      magma_int_t lwork, err, nb, info;
      
      // get ideal workspace blocksize
      nb = magma_get_zgeqrf_nb( row );
      lwork = std::max( col*nb, 2*nb*nb );
      
      // allocate GPU and CPU memory for the hybrid algorithm
      magma_zmalloc( &tmpA, row*col );
      magma_zmalloc( &tmpB, row );
      magma_zmalloc_cpu( &hwork, std::max(1, lwork) );
      
      // copy H and c onto the device
      magma_zsetmatrix( row, col, (magmaDoubleComplex*)H, ldH,  tmpA, ldH);
      magma_zsetmatrix( row, 1, (magmaDoubleComplex*)c, ldH, tmpB, ldH );
      
      // solve the problem
      // NOTE: Newer versions of Magma support magma_zgels_cpu, which might be more handy
      err = magma_zgels_gpu( MagmaNoTrans, row, col, 1, tmpA, ldH, tmpB,
			     ldH, hwork, lwork, &info );
      if (err != 0) {
	 printf("\nError in solveLeastSquaresGMRES, with INFO(magma_zgels) = %d, exit ...\n", info);
	 exit(-1);   
      }
      
      // get back the solution
      magma_zgetmatrix( col, 1, tmpB, ldH, (magmaDoubleComplex*)y, ldH );
      
      // free the memory
      magma_free(tmpA);
      magma_free(tmpB);
      magma_free_cpu(hwork);
#endif
   }
   //--------------------------------------------------------------------------

}  // end of namespace utilGMRES
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
