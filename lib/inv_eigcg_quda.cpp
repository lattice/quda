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

#include <blas_magma.h>

#define MAX_EIGENVEC_WINDOW 16

#define SINGLE_PRECISION_EPSILON 1e-7

/*
Based on  eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
*/

namespace quda {

   static DeflationParam *defl_param = 0;
   static double eigcg_global_stop         = 0.0;

   struct DeflationParam {
     //host   projection matrix:
     Complex *proj_matrix; //VH A V

     QudaPrecision           ritz_prec;             //keep it right now.    
     cudaColorSpinorField    *cudaRitzVectors;      //device buffer for Ritz vectors
     double *ritz_values;

     int ld;                 //projection matrix leading dimension
     int tot_dim;            //projection matrix full (maximum) dimension (nev*deflation_grid)
     int cur_dim;            //current dimension (must match rhs_idx: if(rhs_idx < deflation_grid) curr_nevs <= nev * rhs_idx) 
     int rtz_dim;            //number of ritz values contained in ritz_values array
     int added_nevs;

     bool cuda_ritz_alloc;
     bool in_incremental_stage;

     DeflationParam(ColorSpinorParam &eigv_param, SolverParam &param) : cur_dim(0), rtz_dim(0), added_nevs(0), cuda_ritz_alloc(true), in_incremental_stage(true){

        if(param.nev == 0 || param.deflation_grid == 0) errorQuda("\nIncorrect deflation space parameters...\n");
       
        tot_dim      = param.deflation_grid*param.nev;

        ld           = ((tot_dim+15) / 16) * tot_dim;

        //allocate deflation resources:
        proj_matrix  = new Complex[ld*tot_dim];
        ritz_values  = (double*)calloc(tot_dim, sizeof(double));
       
        ritz_prec = param.precision_ritz;

        eigv_param.setPrecision(ritz_prec);//the same as for full precision iterations (see diracDeflateParam)
        eigv_param.create   = QUDA_ZERO_FIELD_CREATE;
        eigv_param.eigv_dim = tot_dim;
        eigv_param.eigv_id  = -1;
  
        //if(eigv_param.siteSubset == QUDA_FULL_SITE_SUBSET) eigv_param.siteSubset = QUDA_PARITY_SITE_SUBSET;
        cudaRitzVectors = new cudaColorSpinorField(eigv_param);

        return;
     }

     ~DeflationParam(){
        if(proj_matrix)        delete[] proj_matrix;

        if(cuda_ritz_alloc)    delete cudaRitzVectors;

        if(ritz_values)        free(ritz_values);
     }

     //reset current dimension:
     void ResetDeflationCurrentDim(const int addedvecs){

       if(addedvecs == 0) return; //nothing to do

       if((cur_dim+addedvecs) > tot_dim) errorQuda("\nCannot reset projection matrix dimension.\n");

       added_nevs = addedvecs;
       cur_dim   += added_nevs;

       return;
     }   

     //print information about the deflation space:
     void PrintInfo(){

        printfQuda("\nProjection matrix information:\n");
        printfQuda("Leading dimension %d\n", ld);
        printfQuda("Total dimension %d\n", tot_dim);
        printfQuda("Current dimension %d\n", cur_dim);
        printfQuda("Host pointer: %p\n", proj_matrix);

        return;

     }

     void CleanDeviceRitzVectors()
     {
        if( cuda_ritz_alloc ){

          delete cudaRitzVectors;

          cuda_ritz_alloc = false;
        }

        return;
     }

     void ReshapeDeviceRitzVectorsSet(const int nev, QudaPrecision new_ritz_prec = QUDA_INVALID_PRECISION)//reset param.ritz_prec?
     {
        if(nev > tot_dim || (nev == tot_dim && new_ritz_prec == QUDA_INVALID_PRECISION)) return;//nothing to do

        if(!cuda_ritz_alloc) errorQuda("\nCannot reshape Ritz vectors set.\n");
        //
        ColorSpinorParam cudaEigvParam(cudaRitzVectors->Eigenvec(0));

        cudaEigvParam.create   = QUDA_ZERO_FIELD_CREATE;
        cudaEigvParam.eigv_dim = nev;
        cudaEigvParam.eigv_id  = -1;

        if(new_ritz_prec != QUDA_INVALID_PRECISION)
        {
	  ritz_prec = new_ritz_prec;

          cudaEigvParam.setPrecision(ritz_prec);
        }

        CleanDeviceRitzVectors();

        cudaRitzVectors = new cudaColorSpinorField(cudaEigvParam);

        cur_dim = nev;

        cuda_ritz_alloc = true;

        return;
     }

   };


   template<typename Float, typename CudaComplex>
   class EigCGArgs{
     
      private:
      BlasMagmaArgs *eigcg_magma_args;
    
      //host Lanczos matrice, and its eigenvalue/vector arrays:
      std::complex<Float> *hTm;//VH A V
      Float  *hTvalm;   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

      //device Lanczos matrix, and its eigenvalue/vector arrays:
      CudaComplex *dTm;     //VH A V
      CudaComplex *dTvecm; //eigenvectors of T[m,  m  ]
      CudaComplex *dTvecm1; //eigenvectors of T[m-1,m-1]

      int m;
      int nev;
      int ldm;
      
      public:

      EigCGArgs(int m, int nev);
      ~EigCGArgs();
      
      //methods for constructing Lanczos matrix:
      void LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0);
      void LoadLanczosOffDiag(int idx, double alpha0, double beta0);
      
      //methods for Rayleigh Ritz procedure: 
      int RestartVm(void* vm, const int cld, const int clen, const int vprec);//complex length

      //methods 
      void FillLanczosDiag(const int _2nev);
      void FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2);
      //
      void CheckEigenvalues(const cudaColorSpinorField *Vm, const DiracMatrix &matDefl, const int restart_num);//this method is designed to monitor eigcg effeciency, not for production runs
   };

   template<typename Float, typename CudaComplex>
   EigCGArgs<Float, CudaComplex>::EigCGArgs(int m, int nev): m(m), nev(nev){
    //include pad?
    ldm    = ((m+15)/16)*16;//too naive
       
    //magma initialization:
    const int prec = sizeof(Float);
    eigcg_magma_args = new BlasMagmaArgs(m, 2*nev, ldm, prec);

    hTm     = new std::complex<Float>[ldm*m];//VH A V
    hTvalm  = (Float*)safe_malloc(m*sizeof(Float));//eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

    //allocate dTm etc. buffers on GPU:
    dTm     = (CudaComplex*)device_malloc(ldm*m*sizeof(CudaComplex));
    dTvecm  = (CudaComplex*)device_malloc(ldm*m*sizeof(CudaComplex));
    dTvecm1 = (CudaComplex*)device_malloc(ldm*m*sizeof(CudaComplex));

    //set everything to zero:
    cudaMemset(dTm, 0, ldm*m*sizeof(CudaComplex));//?
    cudaMemset(dTvecm, 0, ldm*m*sizeof(CudaComplex));
    cudaMemset(dTvecm1, 0, ldm*m*sizeof(CudaComplex));

    //Error check...
    checkCudaError();

    return;
  }

  template<typename Float, typename CudaComplex>
  EigCGArgs<Float, CudaComplex>::~EigCGArgs() {
    delete[] hTm;

    host_free(hTvalm);

    device_free(dTm);
    device_free(dTvecm);
    device_free(dTvecm1);

    delete eigcg_magma_args;

    return;
  }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0)
  {
    hTm[idx*ldm+idx] = std::complex<Float>((Float)(1.0/alpha + beta0/alpha0), 0.0);
    return;
  } 

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::LoadLanczosOffDiag(int idx, double alpha, double beta)
  {
    hTm[(idx+1)*ldm+idx] = std::complex<Float>((Float)(-sqrt(beta)/alpha), 0.0f);//'U' 
    hTm[idx*ldm+(idx+1)] = hTm[(idx+1)*ldm+idx];//'L'
    return;
  }

  template<typename Float, typename CudaComplex>
  int EigCGArgs<Float, CudaComplex>::RestartVm(void* v, const int cld, const int clen, const int vprec) 
  {
    //Create device version of the Lanczos matrix:
    cudaMemcpy(dTm, hTm, ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);//!

    //Solve m-dimensional eigenproblem:
    cudaMemcpy(dTvecm, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    eigcg_magma_args->MagmaHEEVD((void*)dTvecm, (void*)hTvalm, m);

    //Solve (m-1)-dimensional eigenproblem:
    cudaMemcpy(dTvecm1, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    eigcg_magma_args->MagmaHEEVD((void*)dTvecm1, (void*)hTvalm, m-1);

    //Zero the last row (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
    cudaMemset2D(&dTvecm1[(m-1)], ldm*sizeof(CudaComplex), 0, sizeof(CudaComplex),  (m-1));

    //Attach nev old vectors to nev new vectors (note 2*nev << m):
    cudaMemcpy(&dTvecm[ldm*nev], dTvecm1, ldm*nev*sizeof(CudaComplex), cudaMemcpyDefault);

    //Perform QR-factorization and compute QH*Tm*Q:
    int i = eigcg_magma_args->MagmaORTH_2nev((void*)dTvecm, (void*)dTm);

    //Solve 2nev-dimensional eigenproblem:
    eigcg_magma_args->MagmaHEEVD((void*)dTm, (void*)hTvalm, i);

    //solve zero unused part of the eigenvectors in dTm:
    cudaMemset2D(&(dTm[i]), ldm*sizeof(CudaComplex), 0, (m-i)*sizeof(CudaComplex), i);//check..

    //Restart V:
    eigcg_magma_args->RestartV(v, cld, clen, vprec, (void*)dTvecm, (void*)dTm);

    return i;
  }


  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosDiag(const int _2nev)
 {
    memset(hTm, 0, ldm*m*sizeof(std::complex<Float>));
    for (int i = 0; i < _2nev; i++) hTm[i*ldm+i]= hTvalm[i];//fill-up diagonal

    return;
 }

  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2)
  {
    if(v->Precision() != u->Precision()) errorQuda("\nIncorrect precision...\n");
    for (int i = 0; i < _2nev; i++){
       std::complex<double> s = cDotProductCuda(*v, u->Eigenvec(i));
       s *= inv_sqrt_r2;
       hTm[_2nev*ldm+i] = std::complex<Float>((Float)s.real(), (Float)s.imag());
       hTm[i*ldm+_2nev] = conj(hTm[_2nev*ldm+i]);
    }
  }

  //not implemented 
  template<typename Float, typename CudaComplex>
  void EigCGArgs<Float, CudaComplex>::CheckEigenvalues(const cudaColorSpinorField *Vm, const DiracMatrix &matDefl, const int restart_num)
  {
    printfQuda("\nPrint eigenvalue accuracy after %d restart.\n", restart_num);

    Complex *hproj = (Complex*)mapped_malloc(nev*nev*sizeof(Complex));
    memset(hproj, 0, nev*nev*sizeof(Complex));

    ColorSpinorParam csParam(Vm->Eigenvec(0));
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    csParam.eigv_dim  = 0;
    csParam.eigv_id   = -1;

    cudaColorSpinorField *W   = new cudaColorSpinorField(csParam); 
    cudaColorSpinorField *W2  = new cudaColorSpinorField(csParam);

    cudaColorSpinorField tmp (*W, csParam);

    cudaColorSpinorField *tmp2_p = !matDefl.isStaggered() ? new cudaColorSpinorField(*W, csParam) : &tmp;
    cudaColorSpinorField &tmp2   = *tmp2_p;

    Complex alpha;

    for (int j = 0; j < nev; j++)//
    {
       matDefl(*W, Vm->Eigenvec(j), tmp, tmp2);

       //off-diagonal:
       for (int i = 0; i < j; i++)//row id
       {
          alpha  =  cDotProductCuda(Vm->Eigenvec(i), *W);
          //
          hproj[j*nev+i] = alpha;
          hproj[i*nev+j] = conj(alpha);//conj
       }

       //diagonal:
       alpha  =  cDotProductCuda(Vm->Eigenvec(j), *W);
       //
       hproj[j*nev+j] = alpha;
    }

    double *evals   = (double*)calloc(nev, sizeof(double));

    BlasMagmaArgs magma_args2(nev, nev, sizeof(double));//change precision..

    magma_args2.MagmaHEEVD(hproj, evals, nev, true);

    for(int i = 0; i < nev; i++)//newnev
    {
      for(int j = 0; j < nev; j++) caxpyCuda(hproj[i*nev+j], Vm->Eigenvec(j), *W);

      double  norm2W = normCuda(*W);            

      matDefl(*W2, *W, tmp, tmp2);
 
      Complex dotWW2 = cDotProductCuda(*W, *W2);

      evals[i] = dotWW2.real() / norm2W;

      axCuda(evals[i], *W);

      mxpyCuda(*W2, *W);

      zeroCuda(*W);
            
      double relerr = sqrt( normCuda(*W) / norm2W );
            
      printfQuda("Eigenvalue %d: %1.12e Res.: %1.12e\n", i+1, evals[i], relerr);

    }

    if (&tmp2 != &tmp) delete tmp2_p;

    delete W;
    //
    delete W2;
    //
    host_free(hproj);

    delete evals;

    return;
  }

  // set the required parameters for the initCG solver
  void fillInitCGSolveParam(SolverParam &initCGparam) {
    initCGparam.iter   = 0;
    initCGparam.gflops = 0;
    initCGparam.secs   = 0;

    initCGparam.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    initCGparam.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...
  }

  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matCGSloppy, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matCGSloppy(matCGSloppy), matDefl(matDefl), search_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), initCGparam(param), profile(profile), eigcg_alloc(false)
  {
    if((param.rhs_idx < param.deflation_grid) || (param.inv_type == QUDA_EIGCG_INVERTER))
    {
       if(param.nev > MAX_EIGENVEC_WINDOW )
       { 
          warningQuda("\nWarning: the eigenvector window is too big, using default value %d.\n", MAX_EIGENVEC_WINDOW);
          param.nev = MAX_EIGENVEC_WINDOW;
       }

       search_space_prec = param.precision_ritz;//note that the search vectors and accumulation Ritz array are of the same precision
       //
       use_eigcg = true;
       //
       printfQuda("\nInitialize eigCG(m=%d, nev=%d) solver.\n", param.m, param.nev);
    }
    else
    {
       fillInitCGSolveParam(initCGparam);
       //
       use_eigcg = false;
       //
       printfQuda("\nIncEigCG will deploy initCG solver.\n");
    }

    //hack (think about this!): sloppy precision for the initCG is now always half precision
    initCGparam.precision_sloppy = param.precision_precondition;

    return;
  }

  IncEigCG::~IncEigCG() {

    if(eigcg_alloc)   delete Vm;

  }

/*
 * This is a solo precision solver.
*/

  int IncEigCG::EigCG(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    if (eigcg_precision != x.Precision()) errorQuda("\nInput/output field precision is incorrect (solver precision: %u spinor precision: %u).\n", eigcg_precision, x.Precision());

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);

    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return 0;
    }

    cudaColorSpinorField r(b);

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, csParam);
    //mat(r, x, y);
    //double r2 = xmyNormCuda(b, r);//compute residual
  
    csParam.setPrecision(eigcg_precision);

    cudaColorSpinorField Ap(x, csParam);

    cudaColorSpinorField tmp(x, csParam);
    //matSloppy(r, x, tmp, tmp2);
    //double r2 = xmyNormCuda(b, r);//compute residual

    // tmp2 only needed for multi-gpu Wilson-like kernels
    cudaColorSpinorField *tmp2_p = (!mat.isStaggered()) ? new cudaColorSpinorField(x, csParam) : &tmp;
    cudaColorSpinorField &tmp2 = *tmp2_p;

    matSloppy(r, x, tmp, tmp2);

    double r2 = xmyNormCuda(b, r);//compute residual

    cudaColorSpinorField p(r);

    zeroCuda(y);

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    
    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double r2_old;
    double stop = b2*param.tol*param.tol; // stopping condition of solver

    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double alpha=1.0, beta=0.0;
 
    double pAp;

    int eigvRestart = 0;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas_flops = 0;

//eigCG specific code:
    if(eigcg_alloc == false){

       printfQuda("\nAllocating resources for the EigCG solver...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;
       csParam.setPrecision(search_space_prec);//eigCG internal search space precision: must be adjustable (this coinsides with the accumulation array precision).
       csParam.eigv_dim = param.m;

       Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors 

       checkCudaError();
       printfQuda("\n..done.\n");
       
       eigcg_alloc = true;
    }

    ColorSpinorParam eigParam(Vm->Eigenvec(0));
    eigParam.create = QUDA_ZERO_FIELD_CREATE;

    cudaColorSpinorField  *v0   = NULL; //temporary field.

    cudaColorSpinorField Ap0(Ap);

    if(search_space_prec != param.precision_sloppy)
    {
       v0 = new cudaColorSpinorField(Vm->Eigenvec(0), eigParam); //temporary field. 
    }
    else
    {
       v0 = &Ap0;//just an alias pointer
    }

    //create EigCG objects:
    EigCGArgs<double, cuDoubleComplex> *eigcg_args = new EigCGArgs<double, cuDoubleComplex>(param.m, param.nev); //must be adjustable..
    
    //EigCG additional parameters:
    double alpha0 = 1.0, beta0 = 0.0;

//Begin CG iterations:
    int k=0, l=0;
    
    PrintStats("EigCG", k, r2, b2, heavy_quark_res);

    double sigma = 0.0;

    while ( (!convergence(r2, heavy_quark_res, stop, param.tol_hq) && !convergence(r2, heavy_quark_res, eigcg_global_stop, param.tol_hq)) && k < param.maxiter) {

      if(k > 0)
      {
        beta0 = beta;

        beta = sigma / r2_old;
        axpyZpbxCuda(alpha, p, x, r, beta);

        if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	     heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(x, y, r).z);//note:y is a zero array here.
        }
      }

      //save previous mat-vec result 
      if (l == param.m) copyCuda(Ap0, Ap);

      //mat(Ap, p, tmp, tmp2); // tmp as tmp
      matSloppy(Ap, p, tmp, tmp2);  

      //construct the Lanczos matrix:
      if(l > 0){
        eigcg_args->LoadLanczosDiag(l-1, alpha, alpha0, beta0);
      }

      //Begin Rayleigh-Ritz procedure:
      if (l == param.m){

         eigvRestart++;

         //Restart search space : 
         int cldn = Vm->EigvTotalLength() >> 1; //complex leading dimension
         int clen = Vm->EigvLength()      >> 1; //complex vector length
         //
         int _2nev = eigcg_args->RestartVm(Vm->V(), cldn, clen, Vm->Precision()); 

         if(getVerbosity() >= QUDA_DEBUG_VERBOSE) eigcg_args->CheckEigenvalues(Vm, matDefl, eigvRestart);          

         //Fill-up diagonal elements of the matrix T
         eigcg_args->FillLanczosDiag(_2nev);

         //Compute Ap0 = Ap - beta*Ap0:
         xpayCuda(Ap, -beta, Ap0);//mind precision...
           
         if(search_space_prec != param.precision_sloppy) copyCuda(*v0, Ap0);//convert arrays here:

         eigcg_args->FillLanczosOffDiag(_2nev, v0, Vm, 1.0 / sqrt(r2));

         l = _2nev;

      } else{ //no-RR branch:

         if(l > 0){
            eigcg_args->LoadLanczosOffDiag(l-1, alpha, beta);
         }
      }

      //construct Lanczos basis:
      copyCuda(Vm->Eigenvec(l), r);//convert arrays

      //rescale the vector
      axCuda(1.0 / sqrt(r2), Vm->Eigenvec(l));

      //update search space index
      l += 1;

      //end of RR-procedure
      alpha0 = alpha;


      pAp    = reDotProductCuda(p, Ap);
      alpha  = r2 / pAp; 

      // here we are deploying the alternative beta computation 

      r2_old = r2;
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, r);
      r2 = real(cg_norm); // (r_new, r_new)
      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k+1-r_k) breaks

      k++;

      PrintStats("EigCG", k, r2, b2, heavy_quark_res);
    }

//Free eigcg resources:
    delete eigcg_args;

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops())*1e-9;
    reduceDouble(gflops);
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE){
      printfQuda("EigCG: Eigenspace restarts = %d\n", eigvRestart);
    }

    // compute the true residuals
    //mat(r, x, y);
    matSloppy(r, x, tmp, tmp2);

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

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp) delete tmp2_p;

//Clean EigCG resources:
    if(search_space_prec != param.precision_sloppy)  delete v0;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return eigvRestart;
  }

//END of eigcg solver.

//Deflation space management:
  void IncEigCG::CreateDeflationSpace(cudaColorSpinorField &eigcgSpinor, DeflationParam *&dpar)
  {
    printfQuda("\nCreate deflation space...\n");

    if(eigcgSpinor.SiteSubset() != QUDA_PARITY_SITE_SUBSET) errorQuda("\nRitz spinors must be parity spinors\n");//or adjust it

    ColorSpinorParam cudaEigvParam(eigcgSpinor);

    dpar = new DeflationParam(cudaEigvParam, param);

    printfQuda("\n...done.\n");

    //dpar->PrintInfo();

    return;
  }

  void IncEigCG::DeleteDeflationSpace(DeflationParam *&dpar)
  {
    if(dpar != 0) 
    {
      delete dpar;

      dpar = 0;
    }

    return;
  }

  void IncEigCG::DeleteEigCGSearchSpace()
  {
    if(eigcg_alloc)
    {
       delete Vm;

       Vm = 0;

       eigcg_alloc = false;
    }

    return;
  }


  void IncEigCG::ExpandDeflationSpace(DeflationParam *dpar, const int newnevs)
  {
     if(!use_eigcg || (newnevs == 0)) return; //nothing to do

     if(!eigcg_alloc) errorQuda("\nError: cannot expand deflation spase (eigcg ritz vectors were cleaned).\n"); 
     
     printfQuda("\nConstruct projection matrix..\n");

     int addednev = 0;

     if((newnevs + dpar->cur_dim) > dpar->ld) errorQuda("\nIncorrect deflation space...\n"); //nothing to do...

     //GS orthogonalization

     Complex alpha;

     for(int i = dpar->cur_dim; i < (dpar->cur_dim + newnevs); i++)
     {
       for(int j = 0; j < i; j++)
       {
         alpha = cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(j), dpar->cudaRitzVectors->Eigenvec(i));//<j,i>
         Complex scale = Complex(-alpha.real(), -alpha.imag());
         caxpyCuda(scale, dpar->cudaRitzVectors->Eigenvec(j), dpar->cudaRitzVectors->Eigenvec(i)); //i-<j,i>j
       }
         
       alpha = norm2(dpar->cudaRitzVectors->Eigenvec(i));
       if(alpha.real() > 1e-16)
       {
          axCuda(1.0 /sqrt(alpha.real()), dpar->cudaRitzVectors->Eigenvec(i));  
          addednev += 1;
       }
       else
       {
          errorQuda("\nCannot orthogonalize %dth vector\n", i);
       }
     }

     ColorSpinorParam csParam(dpar->cudaRitzVectors->Eigenvec(0));
     csParam.create = QUDA_ZERO_FIELD_CREATE;

     csParam.eigv_dim  = 0;
     csParam.eigv_id   = -1;

     cudaColorSpinorField *W   = new cudaColorSpinorField(csParam); 
     cudaColorSpinorField *W2  = new cudaColorSpinorField(csParam);

     cudaColorSpinorField tmp (*W, csParam);

     cudaColorSpinorField *tmp2_p = !matDefl.isStaggered() ? new cudaColorSpinorField(*W, csParam) : &tmp;
     cudaColorSpinorField &tmp2   = *tmp2_p;

     for (int j = dpar->cur_dim; j < (dpar->cur_dim+addednev); j++)//
     {
       matDefl(*W, dpar->cudaRitzVectors->Eigenvec(j), tmp, tmp2);//precision must match!

       //off-diagonal:
       for (int i = 0; i < j; i++)//row id
       {
          alpha  =  cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(i), *W);
          //
          dpar->proj_matrix[j*dpar->ld+i] = alpha;
          dpar->proj_matrix[i*dpar->ld+j] = conj(alpha);//conj
       }

       //diagonal:
       alpha  =  cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(j), *W);
       //
       dpar->proj_matrix[j*dpar->ld+j] = alpha;
     }

     dpar->ResetDeflationCurrentDim(addednev);

     printfQuda("\n.. done.\n");

     if (&tmp2 != &tmp) delete tmp2_p;

     delete W;
     delete W2;

     return;
  }

//new:
  void IncEigCG::ReportEigenvalueAccuracy(DeflationParam *dpar, int nevs_to_print)
  {
     int curr_evals = dpar->cur_dim;

     double *evals   = (double*)calloc(curr_evals,sizeof(double));

     Complex *projm  = (Complex*)mapped_malloc(dpar->ld*dpar->tot_dim*sizeof(Complex));
     memset(projm, 0, dpar->ld*dpar->tot_dim*sizeof(Complex));

     memcpy(projm, dpar->proj_matrix, dpar->ld*curr_evals*sizeof(Complex));

     BlasMagmaArgs magma_args(dpar->tot_dim, dpar->ld, sizeof(double));//change precision..
     
     magma_args.MagmaHEEVD(projm, evals, curr_evals, true);

     ColorSpinorParam csParam(dpar->cudaRitzVectors->Eigenvec(0));

     csParam.create = QUDA_ZERO_FIELD_CREATE;

     csParam.eigv_dim  = 0;
     csParam.eigv_id   = -1;

     cudaColorSpinorField *W   = new cudaColorSpinorField(csParam); 
     cudaColorSpinorField *W2  = new cudaColorSpinorField(csParam);

     cudaColorSpinorField tmp (*W, csParam);

     cudaColorSpinorField *tmp2_p = !matDefl.isStaggered() ? new cudaColorSpinorField(*W, csParam) : &tmp;
     cudaColorSpinorField &tmp2   = *tmp2_p;

     for(int i = 0; i < nevs_to_print; i++)//newnev
     {
         for(int j = 0; j < curr_evals; j++) caxpyCuda(projm[i*dpar->ld+j], dpar->cudaRitzVectors->Eigenvec(j), *W);

         double  norm2W = normCuda(*W);            

         matDefl(*W2, *W, tmp, tmp2);

         Complex dotWW2 = cDotProductCuda(*W, *W2);

         evals[i] = dotWW2.real() / norm2W;

         axCuda(evals[i], *W);

         mxpyCuda(*W2, *W);

         double relerr = sqrt( norm2(*W) / norm2W );

         zeroCuda(*W);
            
         printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", i+1, evals[i], relerr);

     }

     if (&tmp2 != &tmp) delete tmp2_p;

     delete W;
    
     delete W2;

     free(evals);

     host_free(projm);

     return;
  }


  void IncEigCG::LoadEigenvectors(DeflationParam *dpar, int max_nevs, double tol /*requested tolerance for the eigenvalues*/)
  {
     if(dpar->cur_dim < max_nevs) 
     {
        printf("\nToo big number of eigenvectors was requested, switched to maximum available number %d\n", dpar->cur_dim);
        max_nevs = dpar->cur_dim; 
     }

     double *evals   = (double*)calloc(dpar->cur_dim, sizeof(double));//WARNING: Ritz values always in double.

     Complex *projm  = (Complex*)mapped_malloc(dpar->ld*dpar->tot_dim * sizeof(Complex));
     memset( projm, 0, dpar->ld*dpar->tot_dim * sizeof(Complex));

     memcpy(projm, dpar->proj_matrix, dpar->ld*dpar->cur_dim*sizeof(Complex));

     BlasMagmaArgs magma_args(dpar->tot_dim, dpar->ld, sizeof(double));//change precision..

     magma_args.MagmaHEEVD(projm, evals, dpar->cur_dim, true);

     //reset projection matrix:
     for(int i = 0; i < dpar->cur_dim; i++) 
     {
       if(fabs(evals[i]) > 1e-16) 
       {
         dpar->ritz_values[i] = 1.0 / evals[i];
       }
       else
       {
          errorQuda("\nCannot invert Ritz value.\n");
       }
     }

     ColorSpinorParam csParam(dpar->cudaRitzVectors->Eigenvec(0));
     csParam.create = QUDA_ZERO_FIELD_CREATE;

     csParam.eigv_dim  = 0;
     csParam.eigv_id   = -1;

     cudaColorSpinorField *W   = new cudaColorSpinorField(csParam); 
     cudaColorSpinorField *W2  = new cudaColorSpinorField(csParam);

     cudaColorSpinorField tmp (*W, csParam);

     cudaColorSpinorField *tmp2_p = !matDefl.isStaggered() ? new cudaColorSpinorField(*W, csParam) : &tmp;
     cudaColorSpinorField &tmp2   = *tmp2_p;

     if(eigcg_alloc == false){//or : search_space_prec != ritz_precision

       printfQuda("\nAllocating resources for the eigenvectors...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;
       //csParam.setPrecision(search_space_prec);//eigCG internal search space precision: must be adjustable.
       csParam.eigv_dim = max_nevs;

       Vm = new cudaColorSpinorField(csParam); //search space for Ritz vectors

       checkCudaError();
       printfQuda("\n..done.\n");
       
       eigcg_alloc = true;
     }

     int idx       = 0;

     double relerr = 0.0;

     while ((relerr < tol) && (idx < max_nevs))//newnev
     {
         for(int j = 0; j < dpar->cur_dim; j++) caxpyCuda(projm[idx*dpar->ld+j], dpar->cudaRitzVectors->Eigenvec(j), *W);
         //load aigenvector into temporary buffer:
         copyCuda(Vm->Eigenvec(idx), *W);
       
         if(getVerbosity() >= QUDA_VERBOSE)
         {
             double  norm2W = normCuda(*W);            

             matDefl(*W2, *W, tmp, tmp2);
 
	     Complex dotWW2 = cDotProductCuda(*W, *W2);

             evals[idx] = dotWW2.real() / norm2W;

             axCuda(evals[idx], *W);

             mxpyCuda(*W2, *W);
            
             relerr = sqrt( normCuda(*W) / norm2W );
             //
             printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", (idx+1), evals[idx], relerr);
         }

         zeroCuda(*W);
            
         idx += 1;
     }

     dpar->ReshapeDeviceRitzVectorsSet(idx);//

     //copy all the stuff to cudaRitzVectors set:
     for(int i = 0; i < idx; i++) copyCuda(dpar->cudaRitzVectors->Eigenvec(i), Vm->Eigenvec(i));
 
     //reset current dimension:
     printfQuda("\nUsed eigenvectors: %d\n", idx);

     dpar->rtz_dim = idx;//idx never exceeds cur_dim.

     if (&tmp2 != &tmp) delete tmp2_p;

     delete W;
    
     delete W2;

     free(evals);

     host_free(projm);

     return;
  }

  void IncEigCG::DeflateSpinor(cudaColorSpinorField &x, cudaColorSpinorField &b, DeflationParam *dpar, bool set2zero)
  {
    if(set2zero) zeroCuda(x);
    if(dpar->cur_dim == 0) return;//nothing to do

    BlasMagmaArgs magma_args(sizeof(double));//change precision..

    Complex  *vec   = new Complex[dpar->ld];

    double check_nrm2 = norm2(b);
    printfQuda("\nSource norm (gpu): %1.15e\n", sqrt(check_nrm2));

    cudaColorSpinorField *in  = NULL;
    cudaColorSpinorField *out = NULL;

    if(dpar->cudaRitzVectors->Precision() != x.Precision())
    {

      ColorSpinorParam csParam(dpar->cudaRitzVectors->Eigenvec(0));
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      //
      csParam.eigv_dim  = 0;
      csParam.eigv_id   = -1;
      //Create an eigenvector set:
      csParam.create   = QUDA_ZERO_FIELD_CREATE;
      //
      in   = new cudaColorSpinorField(csParam); 
      out  = new cudaColorSpinorField(csParam);

      copyCuda(*out, x);
      copyCuda(*in, b);

    }
    else
    {
       in  = &b;
       out = &x;
    }

    for(int i = 0; i < dpar->cur_dim; i++)
    {
      vec[i] = cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(i), *in);//<i, b>
    }    

    magma_args.SolveProjMatrix((void*)vec, dpar->ld,  dpar->cur_dim, (void*)dpar->proj_matrix, dpar->ld);

    for(int i = 0; i < dpar->cur_dim; i++)
    {
      caxpyCuda(vec[i], dpar->cudaRitzVectors->Eigenvec(i), *out); //a*i+x
    }

    if(dpar->cudaRitzVectors->Precision() != x.Precision())
    {
      copyCuda(x, *out);

      delete in;
      delete out;
    }

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));


    delete [] vec;

    return;
  }

  void IncEigCG::DeflateSpinorReduced(cudaColorSpinorField &x, cudaColorSpinorField &b, DeflationParam *dpar, bool set2zero)
  {
    if(set2zero) zeroCuda(x);

    if(dpar->rtz_dim == 0) return;//nothing to do

    double check_nrm2 = norm2(b);
    printfQuda("\nSource norm (gpu): %1.15e\n", sqrt(check_nrm2));

    cudaColorSpinorField *in  = NULL;
    cudaColorSpinorField *out = NULL;

    if(dpar->cudaRitzVectors->Precision() != x.Precision())
    {

      ColorSpinorParam csParam(dpar->cudaRitzVectors->Eigenvec(0));
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      //
      csParam.eigv_dim  = 0;
      csParam.eigv_id   = -1;
      //Create an eigenvector set:
      csParam.create   = QUDA_ZERO_FIELD_CREATE;
      //
      in   = new cudaColorSpinorField(csParam); 
      out  = new cudaColorSpinorField(csParam);

      copyCuda(*out, x);
      copyCuda(*in, b);

    }
    else
    {
       in  = &b;
       out = &x;
    }

    for(int i = 0; i < dpar->rtz_dim; i++)
    {
      Complex tmp = cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(i), *in);//<i, b>

      tmp = tmp * dpar->ritz_values[i];

      caxpyCuda(tmp, dpar->cudaRitzVectors->Eigenvec(i), *out); //a*i+x
    } 

    if(dpar->cudaRitzVectors->Precision() != x.Precision())
    {
      copyCuda(x, *out);

      delete in;
      delete out;
    }   

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));

    return;
  }

//copy EigCG ritz vectors.
  void IncEigCG::SaveEigCGRitzVecs(DeflationParam *dpar, bool cleanEigCGResources)
  {
     const int first_idx = dpar->cur_dim; 

     if(dpar->cudaRitzVectors->EigvDim() < (first_idx+param.nev)) errorQuda("\nNot enough space to copy %d vectors..\n", param.nev); 

     else if(!eigcg_alloc || !dpar->cuda_ritz_alloc) errorQuda("\nEigCG resources were cleaned.\n"); 

     for(int i = 0; i < param.nev; i++) copyCuda(dpar->cudaRitzVectors->Eigenvec(first_idx+i), Vm->Eigenvec(i));
     
     if(cleanEigCGResources)
     {
       DeleteEigCGSearchSpace();
     }
     else//just call zeroCuda..
     {
       zeroCuda(*Vm);
     }

     return;
  }

//not optimal : temorary hack! (let's keep it as is)
  void IncEigCG::StoreRitzVecs(void *hu, double *inv_eigenvals, const int *X, QudaInvertParam *inv_par, const int nev, bool cleanResources)
  {
      const int spinorSize = 24;
      size_t h_size   = spinorSize*defl_param->ritz_prec*defl_param->cudaRitzVectors->EigvVolume();//WARNING: might be brocken when padding is set!
      
      int nev_to_copy = nev > defl_param->cur_dim ? defl_param->cur_dim : nev;
      if(nev > defl_param->cur_dim) warningQuda("\nWill copy %d eigenvectors (requested %d)\n", nev_to_copy, nev);
      
      for(int i = 0; i < nev_to_copy; i++)
      {
          
          size_t offset = i*h_size;
          void *hptr    = (void*)((char*)hu+offset);
          
          ColorSpinorParam cpuParam(hptr, *inv_par, X, true);
          
          cpuColorSpinorField *tmp = new cpuColorSpinorField(cpuParam);
          
          *tmp = defl_param->cudaRitzVectors->Eigenvec(i);//copy gpu field
          
          delete tmp;
      }

      if(inv_eigenvals) memcpy(inv_eigenvals, defl_param->ritz_values, nev_to_copy*sizeof(double));
      
      if(cleanResources) CleanResources();
      
      return;
  }

  void IncEigCG::CleanResources()
  {
    DeleteEigCGSearchSpace();

    if(defl_param != 0)
    {
       DeleteDeflationSpace(defl_param);

       defl_param = 0;
    }

    return;
  } 

  void IncEigCG::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
  {
     const bool use_reduced_vector_set = true;

     const bool use_cg_updates         = false; 

     const int eigcg_min_restarts      = 3;

     int eigcg_restarts = 0;
/*
 * Set internal (iterative refinement) tolerance for the cg solver
 * In general, this is a tunable parameters, e.g.: 
 * 24^3x48 : 5e-3
 * 48^3x96 : 5e-2, works but not perfect, 1e-1 seems to be better...
 */  
     const double cg_iterref_tol = 5e-2;//this must be external for the end-user tuning!
 
     if(defl_param == 0)
     {
       CreateDeflationSpace(*in, defl_param);
     }
     else if(use_eigcg && !defl_param->in_incremental_stage)
     {
       use_eigcg = false;

       printfQuda("\nWarning: IncEigCG will deploy initCG solver.\n");

       DeleteEigCGSearchSpace();
       //
       initCGparam.use_sloppy_partial_accumulator=0;

       fillInitCGSolveParam(initCGparam);

     }

     //if this operator applied during the first stage of the incremental eigCG (to construct deflation space):
     //then: call eigCG inverter 
     if(use_eigcg){

        const bool use_mixed_prec = (eigcg_precision != param.precision); 

        //deflate initial guess ('out'-field):
        DeflateSpinor(*out, *in, defl_param);

        if(!use_mixed_prec) //ok, just run full precision eigcg.
        {
           printf("\nRunning solo precision EigCG.\n");

           eigcg_restarts = EigCG(*out, *in);

           //store computed Ritz vectors: 
           SaveEigCGRitzVecs(defl_param);
           //Construct(extend) projection matrix:
           ExpandDeflationSpace(defl_param, param.nev);
        }
        else //this is mixed precision eigcg
        {

           printf("\nRunning mixed precision EigCG.\n");

           const double ext_tol = param.tol;

           const double stop    = norm2(*in)*ext_tol*ext_tol;

           double tot_time  = 0.0;
           //
           ColorSpinorParam cudaParam(*out);
           //
           cudaParam.create = QUDA_ZERO_FIELD_CREATE;
           //
           cudaParam.setPrecision(eigcg_precision);

           cudaColorSpinorField *outSloppy = new cudaColorSpinorField(cudaParam);
           cudaColorSpinorField *inSloppy  = new cudaColorSpinorField(cudaParam);

           copyCuda(*inSloppy, *in);//input is outer residual
           copyCuda(*outSloppy, *out);

           if (ext_tol < SINGLE_PRECISION_EPSILON) param.tol = SINGLE_PRECISION_EPSILON;//single precision eigcg tolerance

           //the first eigcg cycle:
           eigcg_restarts = EigCG(*outSloppy, *inSloppy);

           tot_time  += param.secs;

           //store computed Ritz vectors: 
           SaveEigCGRitzVecs(defl_param);
        
           //Construct(extend) projection matrix:
           ExpandDeflationSpace(defl_param, param.nev);

           cudaColorSpinorField y(*in);//full precision accumulator
           cudaColorSpinorField r(*in);//full precision residual

           //launch again eigcg:
           copyCuda(*out, *outSloppy);
           //
           mat(r, *out, y);  //here we can use y as tmp
           //
           double r2 = xmyNormCuda(*in, r);//new residual (and RHS)
           //
           double stop_div_r2 = stop / r2;
           //
           eigcg_global_stop = stop;//for the eigcg stuff only

           Solver *initCG = 0;

           initCGparam.tol       = cg_iterref_tol;
           initCGparam.precision = eigcg_precision;//the same as eigcg
           //
           initCGparam.precision_sloppy = QUDA_HALF_PRECISION; //don't need it ...   
           initCGparam.use_sloppy_partial_accumulator=0;   //more stable single-half solver

           fillInitCGSolveParam(initCGparam);

           //too messy..
           bool cg_updates    = (use_cg_updates || (eigcg_restarts < eigcg_min_restarts) || (defl_param->cur_dim == defl_param->tot_dim) || (stop_div_r2 > cg_iterref_tol));

           bool eigcg_updates = !cg_updates;

           if(cg_updates) initCG = new CG(matSloppy, matCGSloppy, initCGparam, profile);

           //start the main loop:
           while(r2 > stop)
           {
              zeroCuda(y);
              //deflate initial guess:
              DeflateSpinor(y, r, defl_param);
              //
              copyCuda(*inSloppy, r);

              //
              copyCuda(*outSloppy, y);
              // 
              if(eigcg_updates) //call low precision eigcg solver
              {
                eigcg_restarts = EigCG(*outSloppy, *inSloppy);
              }
              else //if(initCG) call low precision initCG solver
              {
                (*initCG)(*outSloppy, *inSloppy);  
              }

              copyCuda(y, *outSloppy);
              //
              xpyCuda(y, *out);
              //
              mat(r, *out, y);
              //
              r2 = xmyNormCuda(*in, r);
              //
              stop_div_r2 = stop / r2;

              tot_time  += (eigcg_updates ? param.secs : initCGparam.secs);

              if(eigcg_updates) 
              {
                 if(((eigcg_restarts >= eigcg_min_restarts) || (stop_div_r2 < cg_iterref_tol)) && (defl_param->cur_dim < defl_param->tot_dim))
                 {
                   SaveEigCGRitzVecs(defl_param);//accumulate
                   //
                   ExpandDeflationSpace(defl_param, param.nev);
                   //
                 }
                 else 
                 {
                   if(!initCG && (r2 > stop)) initCG = new CG(matSloppy, matCGSloppy, initCGparam, profile);
                   //param.tol     = cg_iterref_tol;
                   cg_updates    = true;

                   eigcg_updates = false;
                 }
              }
           }//endof while loop
           
           if(cg_updates && initCG)
           {
             delete initCG;

             initCG = 0;
           }

           delete outSloppy;

           delete inSloppy;

           //copy solver statistics:
           param.iter   += initCGparam.iter;
           //
           param.gflops += initCGparam.gflops;
           //
           param.secs   = tot_time;

        }//end of the mixed precision branch

        if(getVerbosity() >= QUDA_VERBOSE)
        {
            printfQuda("\neigCG  stat: %i iter / %g secs = %g Gflops. \n", param.iter, param.secs, param.gflops);

            printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", initCGparam.iter, initCGparam.secs, initCGparam.gflops);

            DeleteEigCGSearchSpace();

            ReportEigenvalueAccuracy(defl_param, param.nev);
        }
     }
     //else: use deflated CG solver with proper restarting. 
     else{
        double full_tol    = initCGparam.tol;

        double restart_tol = initCGparam.tol_restart;

        ColorSpinorParam cudaParam(*out);

        cudaParam.create = QUDA_ZERO_FIELD_CREATE;

        cudaParam.eigv_dim  = 0;

        cudaParam.eigv_id   = -1;

        cudaColorSpinorField *W   = new cudaColorSpinorField(cudaParam);

        cudaColorSpinorField tmp (*W, cudaParam);

        Solver *initCG = 0;

        if(use_reduced_vector_set)
       	    DeflateSpinorReduced(*out, *in, defl_param, true);                
        else
	    DeflateSpinor(*out, *in, defl_param, true);                

        //initCGparam.precision_sloppy = QUDA_HALF_PRECISION;

        const int max_restart_num = 3;//must be external for end-user tuning

        int restart_idx  = 0;

        const double inc_tol = 1e-2;//must be external for the end-user tuning 

        //In many cases, there is no need to use full precision accumulator, since low-mode deflation stabilizes 
        //the mixed precision solver. Moreover, full precision acummulation results in worse performance of the deflated solver, upto 15% in my experiments
        //However, this parameter should be exposed to the enduser for the performance tuning, just in some rare cases when low-mode deflation will be insufficient 
        //for stable double-half mixed precision CG.
        initCGparam.use_sloppy_partial_accumulator = 0;   
 
        initCGparam.delta = 1e-2; // might be a bit better than the default value 1e-1 (think about this)

        //launch initCG:
        while((restart_tol > full_tol) && (restart_idx < max_restart_num))//currently just one restart, think about better algorithm for the restarts. 
        {
          initCGparam.tol = restart_tol; 

          initCG = new CG(mat, matCGSloppy, initCGparam, profile);

          (*initCG)(*out, *in);           

          delete initCG;

          mat(*W, *out, tmp);

          xpayCuda(*in, -1, *W); 

          if(use_reduced_vector_set)

          	DeflateSpinorReduced(*out, *W, defl_param, false);                
          else

		DeflateSpinor(*out, *W, defl_param, false);                

          if(getVerbosity() >= QUDA_VERBOSE)
          {
            printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", initCGparam.iter, initCGparam.secs, initCGparam.gflops);
          }

          double new_restart_tol = restart_tol*inc_tol;

          restart_tol = (new_restart_tol > full_tol) ? new_restart_tol : full_tol;                               

          restart_idx += 1;

          param.secs   += initCGparam.secs;
        }

        initCGparam.tol = full_tol; 

        initCG = new CG(mat, matCGSloppy, initCGparam, profile);

        (*initCG)(*out, *in);           

        delete initCG;

        if(getVerbosity() >= QUDA_VERBOSE)
        {
            printfQuda("\ninitCG total stat (%d restarts): %i iter / %g secs = %g Gflops. \n", restart_idx, initCGparam.iter, initCGparam.secs, initCGparam.gflops);
        }

        //copy solver statistics:
        param.iter   += initCGparam.iter;
        //
        param.secs   += initCGparam.secs;
        //
        param.gflops += initCGparam.gflops;

        delete W;
     } 

     if( (defl_param->cur_dim == defl_param->tot_dim) && use_eigcg )
     {
        defl_param->in_incremental_stage = false;//stop the incremental stage now.

        DeleteEigCGSearchSpace();

        if(use_reduced_vector_set){

          const int max_nev = defl_param->cur_dim;//param.m;

          double eigenval_tol = 1e-1;

          LoadEigenvectors(defl_param, max_nev, eigenval_tol);

          printfQuda("\n...done. \n");
        }
     }

     //compute true residual: 
     ColorSpinorParam cudaParam(*out);
     //
     cudaParam.create = QUDA_ZERO_FIELD_CREATE;
     //
     cudaColorSpinorField   *final_r = new cudaColorSpinorField(cudaParam);
     cudaColorSpinorField   *tmp2    = new cudaColorSpinorField(cudaParam);
           
     
     mat(*final_r, *out, *tmp2);

     param.true_res = sqrt(xmyNormCuda(*in, *final_r) / norm2(*in));
    
     delete final_r;

     delete tmp2;

     param.rhs_idx += 1;

     return;
  }


} // namespace quda
