// define the necessary includes
//-----------------------------------------------------------------------------
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

#include <gmres_utilities.h>
//-----------------------------------------------------------------------------


// envelope everything in the corresponding namespace
//-----------------------------------------------------------------------------
namespace quda {
   
   
   // define the solver factories call
   //--------------------------------------------------------------------------
   GMRES::GMRES( DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
                 SolverParam &param, TimeProfile &profile ) :
      Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param) {
         
   // derive parameters for inner solver
   utilGMRES::fillInnerSolveParam(Kparam, param);

   if (param.inv_type_precondition == QUDA_CG_INVERTER)                    // inner CG preconditioner
      K = new CG(matPrecon, matPrecon, Kparam, profile);
   else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER)         // inner BiCGstab preconditioner
      K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
   else if (param.inv_type_precondition == QUDA_MR_INVERTER)               // inner MR preconditioner
      K = new MR(matPrecon, Kparam, profile);
   else if (param.inv_type_precondition == QUDA_SD_INVERTER)               // inner SD preconditioner
      K = new SD(matPrecon, Kparam, profile);
   else if (param.inv_type_precondition != QUDA_INVALID_INVERTER)          // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
   }

   GMRES::~GMRES() {
      profile.Start(QUDA_PROFILE_FREE);   // time profiling
      // delete inner solver structure, if it should already exist
      if (K) delete K;
      profile.Stop(QUDA_PROFILE_FREE); // stop profiling
   }
   //--------------------------------------------------------------------------
   

   //--------------------------------------------------------------------------
   //--------------------------------------------------------------------------
   // implement the solver
   //--------------------------------------------------------------------------
   //--------------------------------------------------------------------------
   void GMRES::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) {
      
      // start the time-profiler --> FOR INITIALIZATION
      //-----------------------------------------------------------------------
      profile.Start(QUDA_PROFILE_INIT);
      //-----------------------------------------------------------------------
   
      // Check to see that we're not trying to invert on a zero-field source
      //-----------------------------------------------------------------------
      const double b2 = normCuda(b);   // norm sq of source vector
      if (b2==0) {
         profile.Stop(QUDA_PROFILE_INIT);
         printfQuda("Warning: inverting on zero-field source\n");
         x=b;
         param.true_res = 0.0;
         param.true_res_hq = 0.0;
         return;
      }
      //-----------------------------------------------------------------------
           
      // set up usage of heavy quark residual
      //-----------------------------------------------------------------------
      const bool use_heavy_quark_res = 
         (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
      double heavy_quark_res = 0.0; // true value calculated later
      //-----------------------------------------------------------------------
              
      // get the spinors properties
      //-----------------------------------------------------------------------
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      //-----------------------------------------------------------------------
               
      // get the vital parameters of the solver
      //-----------------------------------------------------------------------
      double stop = stopping(param.tol, b2, param.residual_type);
      int maxIter = param.maxiter;
      int maxInnerIter = param.Nkrylov;
      int kAug    = param.deflation_grid; // dim. of augmented subspace
      int startArnoldi = 0;               // begin of arnoldi loop
           
      double detGref;                     // measure for quality of augmented subspace
      int deflCount     = 0;              // counts the number of consecutive deflations
      int badSubspCount = 0;              // counts the number of rejected subspaces
      int subspaceShape  = 1;             // 1 -> GMRES-R, 0 -> GMRES-DR
      //-----------------------------------------------------------------------
           
      // allocate memory for the required matrices and spinor-matrices
      //-----------------------------------------------------------------------
      // for the arnoldi spinors
      csParam.setPrecision(param.precision_sloppy);         // change precision
      cudaColorSpinorField **V = new cudaColorSpinorField*[maxInnerIter+1];
      for (int i=0; i<(maxInnerIter+1); i++)
         V[i] = new cudaColorSpinorField(b, csParam);
           
      // auxiliary spinor for arnoldi process
      cudaColorSpinorField *w;
      w = new cudaColorSpinorField(b, csParam);
           
      // preconditioned spinors in main solver precision
      cudaColorSpinorField **Z = new cudaColorSpinorField*[maxInnerIter];
      for (int i=0; i<maxInnerIter; i++)
         Z[i] = new cudaColorSpinorField(b, csParam);
      csParam.setPrecision(param.precision);                // change back precision
           
      // auxiliary precon. spinors in preconditioner-precision
      // allocate memory only if necessary (might throw a compiler warning, but this is ok)
      cudaColorSpinorField *Z_precon, *V_precon;
      if ( param.precision_precondition!=param.precision_sloppy ) {
         csParam.setPrecision(param.precision_precondition);   // change precision
         Z_precon = new cudaColorSpinorField(b, csParam);
         V_precon = new cudaColorSpinorField(b, csParam);
         csParam.setPrecision(param.precision);                // change back precision
      }
           
      // temporary spinors for the dirac operators in both precisions
      csParam.setPrecision(param.precision);
      cudaColorSpinorField temp(b, csParam);
      csParam.setPrecision(param.precision_sloppy);            // change precision
      cudaColorSpinorField temp_sloppy(b, csParam);
      csParam.setPrecision(param.precision);                   // change back precision
           
      // residual spinors
      cudaColorSpinorField r(b);
      cudaColorSpinorField *r_sloppy;
      if ( param.precision_sloppy!=param.precision ) {
         csParam.setPrecision(param.precision_sloppy);         // change precision
         r_sloppy = new cudaColorSpinorField(b, csParam);
         csParam.setPrecision(param.precision);                // change back precision
      }
      else
         r_sloppy = &r;
            
      // for the arnoldi H-matrix (initialize to zero)
      int ldH = maxInnerIter+1;
      Complex *H = new Complex[maxInnerIter*ldH]();
           
      // vectors for the approximate solve (the paper calls y = rho)
      Complex *c = new Complex[maxInnerIter+1]();
      Complex *y = new Complex[maxInnerIter]();
            
      // value stored in H[k+1]
      double normV;
      //-----------------------------------------------------------------------   
           
      // compute parity of the node (needed for the Schwarz solver)
      //-----------------------------------------------------------------------
      int parity = 0;
      for (int i=0; i<4; i++)
         parity += commCoords(i);
      parity = parity % 2;
      //-----------------------------------------------------------------------
           
      // switch time-profilers --> FROM INITIALIZATION TO PREAMBLE
      //-----------------------------------------------------------------------
      profile.Stop(QUDA_PROFILE_INIT);
      profile.Start(QUDA_PROFILE_PREAMBLE);
      //-----------------------------------------------------------------------
           
      // calculate values of the 0th iteration
      //-----------------------------------------------------------------------
      // reset blas flop-counter
      blas_flops = 0;
           
      // exact initial residual depending on the values for x and b
      mat(r, x, temp);              // r = Ax
      axpbyCuda(1.0, b, -1.0, r);   // r = b - r = b - Ax
      double r2 = normCuda(r);      // r2 = |r|^2
           
      // first arnoldi vector (low precision)
      Complex beta = 1.0/sqrt(r2);        // caxpy needs compl. constants
      copyCuda(*r_sloppy, r);             // get r also in low precision
      caxpyCuda(beta, *r_sloppy, *V[0]);  // V[0] = beta*r_sloppy + V[0]
      
      // calculate heavy quark residual                                       FIX ME: is this line necessary
      if ( use_heavy_quark_res )
         heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
            
      // print stats of the current iteration
      PrintStats("GMRES", 0, r2, b2, heavy_quark_res);
      //-----------------------------------------------------------------------
           
      // switch time-profilers --> FROM PREAMBLE TO CALCULATION
      //-----------------------------------------------------------------------
      profile.Stop(QUDA_PROFILE_PREAMBLE);
      profile.Start(QUDA_PROFILE_COMPUTE);
      //-----------------------------------------------------------------------
           
      // start the outer iteration
      //-----------------------------------------------------------------------
      int k=0;         
      while( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && k<maxIter ) {
      
	 // run the deflation subroutine, with check if deflation makes sense or not
	 //--------------------------------------------------------------------
	 // check shape of last subspace and modify matrices for the next iteration,
	 // i.e., deflate or zero them
	 if (kAug != 0) {
	    // we accept one bad subspace, but if the next one should also be
	    // corrupted we force a clean restart
	    subspaceShape = utilGMRES::checkRestartCriterion(k+1, &deflCount,
				            &badSubspCount, 20, 1, kAug, &detGref, V);
	 }
	 
	 if ( subspaceShape == 1 || kAug == 0 ) {
	    // perform clean restart and print warning message
	    printfQuda("GMRES: performing clean restart after iteration %d\n", k);
	    std::memset( H, 0.0, maxInnerIter*ldH*sizeof(Complex) );
	 
	    beta = 1.0/sqrt(r2);                                     // sqrt() since r2 = res^2
	    zeroCuda(*V[0]);                                         // V[0] = 0 so that next line works
	    if ( param.precision_sloppy!=param.precision ) {
	       copyCuda(*r_sloppy, r);                               // convert precision of r
	       caxpyCuda(beta, *r_sloppy, *V[0]);                    // V[0] = beta*r + V[0]
	    }
	    else
	       caxpyCuda(beta, r, *V[0]);                            // V[0] = beta*r + V[0]
	    
	    // in the next iteration perform clean arnoldi procedure
	    startArnoldi = 0;
	 }
	 else {   // deflate the system
	    csParam.setPrecision(param.precision_sloppy);
	    utilGMRES::deflate( H, V, Z, c, y, maxInnerIter, ldH, kAug, csParam);
	    csParam.setPrecision(param.precision);
	 
	    // start Arnoldi procedure at a later stage
	    startArnoldi = kAug;
	 }
         //--------------------------------------------------------------------

         // start the inner iteration, i.e., flexible Arnoldi and Gram Schmidt
         //--------------------------------------------------------------------
         for (int l=startArnoldi; l<maxInnerIter; l++) {
               
            // preconditioning for m consecutive steps (compare with inv_gcr_quda.cpp)
            //-----------------------------------------------------------------
            for (int m=0; m<param.precondition_cycle; m++) {
               // if a valid inverter was set, do the following
               if (param.inv_type_precondition != QUDA_INVALID_INVERTER) {
                  // perform one approximate iteration
                  if ((parity+m)%2 == 0 || param.schwarz_type == QUDA_ADDITIVE_SCHWARZ) {
                     if ( param.precision_precondition!=param.precision_sloppy ) {
                        copyCuda(*V_precon, *V[l]);   // V_precon = V[l]
                        (*K)(*Z_precon, *V_precon);   // Z_precon = inv(A)*V_precon
                        copyCuda(*Z[l], *Z_precon);   // Z[l] = Z_precon
                     }
                     else
                        (*K)(*Z[l], *V[l]);           // Z_precon = inv(A)*V_precon
                  }
               }
               else  // no preconditioner was set --> copy V into Z
                  copyCuda(*Z[l], *V[l]);             // Z[l] = V[l]
            }
            //-----------------------------------------------------------------
               
            // do flexible arnoldi iteration (in low precision)
            if ( param.precision_sloppy!=param.precision )
               matSloppy(*w, *Z[l], temp_sloppy);     // w = A*Z[l]
            else
               mat(*w, *Z[l], temp);                  // copy w <-- Z[l]
               
            for (int j=0; j<=l; j++) {
               H[j+ldH*l] = cDotProductCuda(*w, *V[j]);
               caxpyCuda(-H[j+ldH*l], *V[j], *w);     // w = w - H[j][l]*V[j]
            }
            
            // fill in the missing elements of H and V
            normV          = sqrt( normCuda(*w) );    // normCuda(x) returns |x|^2
            H[(l+1)+ldH*l] = normV;
            axCuda(1.0/normV, *w);                    // V[l+1] = V[l+1]/H[l+1][l]
            copyCuda(*V[l+1], *w);  
         }
         //--------------------------------------------------------------------
         
         // form the approximate solution
         if ( k==0 || subspaceShape == 1 || kAug == 0 ) {            // in case of clean (re-) start
            std::memset( c, 0.0, (maxInnerIter+1)*sizeof(Complex) ); // defensive measure
            c[0] = sqrt(r2);                                         // sqrt() since r2 = res^2
         }
         else {                                                      // in case of deflated restart
            std::memset( c, 0.0, (maxInnerIter+1)*sizeof(Complex) ); // defensive measure
            copyCuda(*r_sloppy, r);                                  // convert r into low prec. (just to be shure)
            for (int i=0; i<(kAug+1); i++) {
               c[i] = cDotProductCuda(*V[i], *r_sloppy);             // update the c-vector
            }
         }
         utilGMRES::solveLeastSquares(H, c, y, ldH, maxInnerIter, ldH); // y = argmin||c-H.y||
         
         // calculate the new solution by x = x + Z.y
         if ( param.precision_sloppy!=param.precision ) {
            for (int i=0; i<maxInnerIter; i++) {      
               copyCuda(temp, *Z[i]);                                // convert Z[i] into high precision
               caxpyCuda(y[i], temp, x);                             // x = x + Z.y
            }
         }
         else {
            for (int i=0; i<maxInnerIter; i++)  
               caxpyCuda(y[i], *Z[i], x);                            // x = x + Z.y
         }
            
         //...and mathematically correct resNorm (in high precision)
         mat(r, x, temp);                                            // r = A.x
         axpbyCuda(1.0, b, -1.0, r);                                 // r = b - r = b - A.x
         r2 = normCuda(r);
            
         // calculate heavy quark residual
         if ( use_heavy_quark_res )
               heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
            
         // print stats of the current iteration
         PrintStats("GMRES", (k+1), r2, b2, heavy_quark_res);
                  
         // increment k
         k++;
      }
      //-----------------------------------------------------------------------
            
      // switch time-profilers --> FROM CALCULATION TO EPILOGUE
      //-----------------------------------------------------------------------
      profile.Stop(QUDA_PROFILE_COMPUTE);
      profile.Start(QUDA_PROFILE_EPILOGUE);
      //-----------------------------------------------------------------------
           
      // calculate some additional informations to finish the solve
      //-----------------------------------------------------------------------
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
           
      // calculate the flops number (same as for GCR solver)
      double gflops;
      gflops = (blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
      reduceDouble(gflops);
         
      // set the flop-counters
      param.gflops += gflops;
      param.iter   += k;      // No. of total outer iterations
      blas_flops    = 0;
      mat.flops();
      matSloppy.flops();
      matPrecon.flops();
         
      // tell calling function the true residual
      param.true_res = sqrt( r2/b2 );     // sqrt() since r2 = res^2
#if (__COMPUTE_CAPABILITY__ >= 200)
      param.true_res_hq = heavy_quark_res;
#else
      param.true_res_hq = 0.0;
#endif   
      //-----------------------------------------------------------------------
           
      // switch time-profilers --> FROM EPILOGUE TO FREE
      //-----------------------------------------------------------------------
      profile.Stop(QUDA_PROFILE_EPILOGUE);
      profile.Start(QUDA_PROFILE_FREE);
      //-----------------------------------------------------------------------
                 
      // free allocated memory
      //-----------------------------------------------------------------------
      for (int i=0; i<(maxInnerIter+1); i++)
         delete V[i];
           
      for (int i=0; i<maxInnerIter; i++)
         delete Z[i];
           
      delete[] V;
      delete[] Z;
      delete w;
              
      if ( param.precision_sloppy!=param.precision )
         delete r_sloppy;

      if ( param.precision_precondition!=param.precision_sloppy ) {
         delete V_precon;
         delete Z_precon;
      }
      delete[] H;
      delete[] y;
      delete[] c;
      //-----------------------------------------------------------------------
           
      // kill the last time-profiler
      //-----------------------------------------------------------------------
      profile.Stop(QUDA_PROFILE_FREE);
      //-----------------------------------------------------------------------
           
   }
   //--------------------------------------------------------------------------
   //--------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
