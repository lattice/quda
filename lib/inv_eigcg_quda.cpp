#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <string.h>

#include <face_quda.h>

#include <iostream>

#ifdef REAL
#undef REAL //due to conflicts with magma
#endif

#include "magma.h"

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

/*
Base eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
//Warning: magma matrices in coloumn-major format...
WARNING: for coalescing m must be multiple of 16, use pad for other choises
*/

namespace quda {

  EigCG::EigCG(DiracMatrix &mat, DiracMatrix &matSloppy, cudaColorSpinorField *vm, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), Vm(vm) //eigenvector set from a given object?
  {

  }

  EigCG::~EigCG() {

    //Vm->destroy();

    if(init) delete v0;
  }


void compute_2nev_Ritz_vectors(cudaColorSpinorField Vm, magmaDoubleComplex *dQ, const int m, const int _2nev){
       magmaDoubleComplex cone     =  MAGMA_Z_MAKE(1.0, 0.0);
       magmaDoubleComplex czero    =  MAGMA_Z_MAKE(0.0, 0.0);
   
       magma_trans_t transV   = 'N';
       magma_trans_t transQ   = 'N';
 
       magma_int_t ldV       = Vm.Eigenvec(0).Length();
       magma_int_t ldQ       = m;//not vsize (= 2*nev) 
       
       magmaDoubleComplex *V = (magmaDoubleComplex*)Vm.V(); 
       magmaDoubleComplex *Tmp;
       magma_malloc((void**)&Tmp, ldV*m*sizeof(magmaDoubleComplex)); 

       cudaMemset(Tmp, 0, ldV*m*sizeof(magmaDoubleComplex)); 
       magmablas_zgemm(transV, transQ, ldV, _2nev, m, (magmaDoubleComplex)cone, V, ldV, dQ, ldQ, (magmaDoubleComplex)czero, Tmp, ldV);//in colour-major format
       cudaMemcpy(V, Tmp, ldV*(_2nev)*sizeof(magmaDoubleComplex), cudaMemcpyDefault); 

       magma_free(Tmp);
  }

  void EigCG::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
    profile.Start(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    cudaColorSpinorField r(b);

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, csParam);

    mat(r, x, y);
    double r2 = xmyNormCuda(b, r);//compute residual
  
    csParam.setPrecision(param.precision_sloppy);
    cudaColorSpinorField Ap(x, csParam);
    cudaColorSpinorField tmp(x, csParam);

    cudaColorSpinorField *tmp2_p = &tmp;
    // tmp only needed for multi-gpu Wilson-like kernels
    if (mat.Type() != typeid(DiracStaggeredPC).name() && 
	mat.Type() != typeid(DiracStaggered).name()) {
      tmp2_p = new cudaColorSpinorField(x, csParam);
    }
    cudaColorSpinorField &tmp2 = *tmp2_p;

    cudaColorSpinorField *x_sloppy, *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      x_sloppy = &x;
      r_sloppy = &r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x, csParam);
      r_sloppy = new cudaColorSpinorField(r, csParam);
    }

    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;
    cudaColorSpinorField p(rSloppy);

    if(&x != &xSloppy){
      copyCuda(y,x);
      zeroCuda(xSloppy);
    }else{
      zeroCuda(y);
    }
    
    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    
    profile.Stop(QUDA_PROFILE_INIT);
    profile.Start(QUDA_PROFILE_PREAMBLE);

    double r2_old;
    double stop = b2*param.tol*param.tol; // stopping condition of solver

    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double alpha=1.0, beta=0.0;
    double aux=0.0; 
 
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    int maxResIncrease = 0; // 0 means we have no tolerance 

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);
    blas_flops = 0;

//EigCG specific code:
    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      //csParam.setPrecision(param.precision_sloppy);
      //Ap0  = new cudaColorSpinorField(x, csParam);
      csParam.setPrecision(Vm->Precision());//must be the same precision as Vm
      v0 = new cudaColorSpinorField(x, csParam); 
      init = true;
    }

    cudaColorSpinorField &Ap0 = *tmp2_p;

    const int m   = param.m;
    const int nev = param.nev;

    //init (set to zero) host Lanczos matrice, and its eigenvalue/vector arrays:
    Complex *hTm     = new Complex[m*m];//VH A V
    Complex *hTvecm0 = new Complex[m*m];//eigenvectors of T[m,  m  ]
    Complex *hTvecm1 = new Complex[m*m];//eigenvectors of T[m-1,m-1]
    double  *hTvalm0 = new double[m];   //eigenvalues of T[m,  m  ]
    double  *hTvalm1 = new double[m];   //eigenvalues of T[m-1,m-1]

    //init device Lanczos matrix, and its eigenvalue/vector arrays:
    cuDoubleComplex *dTm;     //VH A V
    cuDoubleComplex *dTvecm0; //eigenvectors of T[m,  m  ]
    cuDoubleComplex *dTvecm1; //eigenvectors of T[m-1,m-1]

    //allocate dT etc. buffers on GPU:
    cudaMalloc(&dTm, m*m*sizeof(cuDoubleComplex));//  
    cudaMalloc(&dTvecm0, m*m*sizeof(cuDoubleComplex));  
    cudaMalloc(&dTvecm1, m*m*sizeof(cuDoubleComplex));  

    //set everything to zero:
    cudaMemset(dTm, 0, m*m*sizeof(cuDoubleComplex));
    cudaMemset(dTvecm0, 0, m*m*sizeof(cuDoubleComplex));
    cudaMemset(dTvecm1, 0, m*m*sizeof(cuDoubleComplex));

    //magma initialization:
    magma_init();
    magma_int_t info = -1;

    //magma params/objects:
    magma_int_t lddTm      = m;//dTm (device)ld (may include padding)
    magma_int_t ldhTm      = m;//hTm (host)ld    (may include padding)

    magma_int_t nb    = magma_get_zhetrd_nb(m);
    magma_int_t m_1   = m-1;

    magma_int_t llwork = MAX(m + m*nb, 2*m + m*m); 
    magma_int_t lrwork = 1 + 5*m + 2*m*m;
    magma_int_t liwork = 3 + 5*m;

    magma_int_t htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    magma_int_t dtsize   = ( 4*nev + ((2*nev + 31)/32)*32 )*nb;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    int sideLR = (m - 2*nev + nb)*(m + nb) + m*nb;
    magma_int_t lwork_max = sideLR; 

    magmaDoubleComplex *W;
    magma_malloc_pinned((void**)&W, lwork_max*sizeof(magmaDoubleComplex));

    magmaDoubleComplex *hTauL;
    magma_malloc_pinned((void**)&hTauL, htsize*sizeof(magmaDoubleComplex));

    magmaDoubleComplex *dTauL;
    magma_malloc((void**)&dTauL, dtsize*sizeof(magmaDoubleComplex));

    magmaDoubleComplex *hTauR;
    magma_malloc_pinned((void**)&hTauR, htsize*sizeof(magmaDoubleComplex));

    magmaDoubleComplex *dTauR;
    magma_malloc((void**)&dTauR, dtsize*sizeof(magmaDoubleComplex));


    magmaDoubleComplex *lwork;
    double             *rwork;
    magma_int_t        *iwork;

    magma_malloc_pinned((void**)&lwork, llwork*sizeof(magmaDoubleComplex));
    magma_malloc_cpu((void**)&rwork, lrwork*sizeof(double));
    magma_malloc_cpu((void**)&iwork, liwork*sizeof(magma_int_t));

    double alpha0, beta0;//EigCG additional parameters

//Begin CG iterations:
    int k=0, l=0;
    
    PrintStats("eigCG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;

    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    k < param.maxiter) {
      //save previous mat-vec result 
      if (l == m) copyCuda(Ap0,Ap);

      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp
      alpha0 = alpha;//??
      pAp    = reDotProductCuda(p, Ap);
      alpha  = r2 / pAp; 
      
//begin eigCG part here:
      if(nev > 0){
         if (k > 0) hTm[(l-1)*m+(l-1)] = Complex(1/alpha + beta0/alpha0, 0.0);
         
         //Start Rayleigh-Ritz procedure here:
         if (l == m){
           //solve m-dim eigenproblem:
           llwork = MAX(m + m*nb, 2*m + m*m); 
           lrwork = 1 + 5*m + 2*m*m;
           liwork = 3 + 5*m;

           cudaMemcpy(dTm, hTm, m*m*sizeof(cuDoubleComplex), cudaMemcpyDefault);
           cudaMemcpy(dTvecm0, dTm, m*m*sizeof(cuDoubleComplex), cudaMemcpyDefault);//how to refine it with streams??
           //simplify this:
           magma_zheevd_gpu('V', 'U', m, 
                            (magmaDoubleComplex*)dTvecm0, lddTm, 
                             hTvalm0, (magmaDoubleComplex*)hTvecm0, ldhTm, 
                             lwork, llwork, rwork, lrwork, iwork, liwork, &info);
           if(info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

           //solve (m-1)-dim eigenproblem:
           
           llwork = MAX(m_1 + m_1*nb, 2*m_1 + m_1*m_1); 
           lrwork = 1 + 5*m_1 + 2*m_1*m_1;
           liwork = 3 + 5*m_1;
           cudaMemcpy(dTvecm1, dTm, m*m*sizeof(cuDoubleComplex), cudaMemcpyDefault);
           magma_zheevd_gpu('V', 'U', m_1, 
                            (magmaDoubleComplex*)dTvecm1, lddTm, 
                            hTvalm1, (magmaDoubleComplex*)hTvecm1, ldhTm, 
                            lwork, llwork, rwork, lrwork, iwork, liwork, &info);
           if(info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

           //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
           cudaMemset2D(&dTvecm1[m_1], m*sizeof(cuDoubleComplex), 0, sizeof(cuDoubleComplex),  m_1);

           //attach nev old vectors to nev new vectors (note 2*nev < m):
           cudaMemcpy(&dTvecm0[nev*m], dTvecm1, nev*m*sizeof(cuDoubleComplex), cudaMemcpyDefault);

           //Orthogonalize 2*nev vectors:
           l = 2 * nev;
           magma_zgeqrf_gpu(m, l, dTvecm0, lddTm, hTauR, dTauR, &info);
           if(info != 0) printf("\nError in magma_zgeqrf_gpu, exit ...\n"), exit(-1);

           //compute dTevecm1=QHTQ
           cudaMemcpy(dTauL, dTauR, dtsize*sizeof(magmaDoubleComplex), cudaMemcpyDefault); 
           memcpy    (hTauL, hTauR, htsize*sizeof(magmaDoubleComplex));            

           //compute TQ product:
           magma_zunmqr_gpu( 'R', 'N', m, m, l, dTvecm0, lddTm, hTauR, dTm, lddTm, W, sideLR, dTauR, nb, &info); 
           if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);
              	
           //compute QHT product:
           magma_zunmqr_gpu( 'L', 'C', m, m, l, dTvecm0, lddTm, hTauL, dTm, lddTm, W, sideLR, dTauL, nb, &info);
           if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);                 	

           //solve l=2*nev-dim eigenproblem:
           llwork = MAX(l + l*nb, 2*l + l*l); 
           lrwork = 1 + 5*l + 2*l*l;
           liwork = 3 + 5*l;
           magma_zheevd_gpu('V', 'U', l, 
                            (magmaDoubleComplex*)dTm, lddTm, 
                             hTvalm0, (magmaDoubleComplex*)hTvecm0, ldhTm, 
                             lwork, llwork, rwork, lrwork, iwork, liwork, &info);
           if(info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

           //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
           cudaMemset2D(&dTm[l], m*sizeof(cuDoubleComplex), 0, (m-l)*sizeof(cuDoubleComplex),  l);//check..
        
           //Compute dTm=dTevecm0*dTm (Q * Z):
           //(compute QT product):
           magma_zunmqr_gpu('L', 'N', m, m, l, dTvecm0, lddTm, hTauL, dTm, lddTm, W, sideLR, dTauL, nb, &info);
           if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1); 

           //Compute Ritz vectors : V=V(n, m)*dTm(m, l)
           compute_2nev_Ritz_vectors(*Vm, dTm, m, (2*nev));
           
           //Fill-up diagonal elements of the matrix T
           memset(hTm, 0, m*m*sizeof(Complex));
    	   for (int i = 0; i < l; i++) hTm[i*m+i]= hTvalm0[i];//fill-up diagonal

           //Compute Ap0 = Ap - beta*Ap0:
           axpyCuda(-beta, Ap0, Ap);//mind precision...
           
           copyCuda(*v0, Ap0);//make full precision here.
	   for (int i = 0; i < l; i++){
	     Complex s = cDotProductCuda(Vm->Eigenvec(i), *v0);
	     hTm[i+l*m] = s/sqrt(r2);
	     hTm[l+i*m] = conj(s)/sqrt(r2);
	   }
         } else{ //no-RR branch:
            hTm[(l+1)*m+l] = Complex(-sqrt(beta)/alpha0, 0.0);//'U' (!)
            hTm[l*m+(l+1)] = Complex(-sqrt(beta)/alpha0, 0.0);//'L' (!)
         }//end of nev
         l += 1;
         double scale = 1.0 / sqrt(r2);
         copyCuda(Vm->Eigenvec(l), *r_sloppy);//convert arrays!
         axCuda(scale, Vm->Eigenvec(l));//
      }
//end of eigCG part 

      double sigma;
      bool breakdown = false;
      r2_old = r2;

      // here we are deploying the alternative beta computation 
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy);
      r2 = real(cg_norm); // (r_new, r_new)
      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k+1-r_k) breaks


      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && delta >= param.tol) updateX = 1;

      if ( !(updateR || updateX)) {
	//beta = r2 / r2_old;
        beta0  = beta; //used for the Lanczos matrix construction
	beta   = sigma / r2_old; // use the alternative beta computation

	axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);//update of xSloppy and then p

	if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	  copyCuda(tmp,y);
	  heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
	}

	steps_since_reliable++;
      } else {//reliable update
	axpyCuda(alpha, p, xSloppy);
	if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
	xpyCuda(x, y); // swap these around?
	mat(r, y, x); // here we can use x as tmp
	r2 = xmyNormCuda(b, r);

	if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
	zeroCuda(xSloppy);

	// break-out check if we have reached the limit of the precision
	static int resIncrease = 0;
	if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
	  warningQuda("EigCG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
	  k++;
	  rUpdate++;
	  if (++resIncrease > maxResIncrease) break; 
	} else {
	  resIncrease = 0;
	}

	rNorm = sqrt(r2);
	maxrr = rNorm;
	maxrx = rNorm;
	r0Norm = rNorm;      
	rUpdate++;

	// explicitly restore the orthogonality of the gradient vector
	double rp = reDotProductCuda(rSloppy, p) / (r2);
	axpyCuda(-rp, rSloppy, p);
        
        beta0 = beta;
	beta = r2 / r2_old; 
	xpayCuda(rSloppy, beta, p);

	if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(y,r).z);
	
	steps_since_reliable = 0;
      }//end of the reliable update

      breakdown = false;
      k++;

      PrintStats("EigCG", k, r2, b2, heavy_quark_res);
    }

//Shutdown magma:
    magma_free(dTauL);
    magma_free(dTauR);

    magma_free_cpu(hTauL);
    magma_free_cpu(hTauR);

    magma_free_pinned(W);
    magma_free_pinned(lwork);

    magma_free_cpu(rwork);
    magma_free_cpu(iwork);

    magma_finalize();


    if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
    xpyCuda(y, x);

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
      param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("EigCG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, x, y);
    param.true_res = sqrt(xmyNormCuda(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    param.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
    param.true_res_hq = 0.0;
#endif      
    PrintSummary("EigCG", k, r2, b2);

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp) delete tmp2_p;

    if (param.precision_sloppy != x.Precision()) {
      delete r_sloppy;
      delete x_sloppy;
    }
//Clean EigCG resources:
    delete[] hTm;
    delete[] hTvecm0;
    delete[] hTvecm1;

    delete[] hTvalm0;
    delete[] hTvalm1;

    cudaFree(dTm);
    cudaFree(dTvecm0);
    cudaFree(dTvecm1);

    profile.Stop(QUDA_PROFILE_FREE);

    return;
  }
//temporal hack redefine REAL from quda_intenal.h
#define REAL(a) (*((double*)&a))
} // namespace quda
