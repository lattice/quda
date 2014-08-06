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

#define DEBUG_MODE

#define MAX_EIGENVEC_WINDOW 16

/*
Based on  GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting"
*/

namespace quda {

   static DeflationParam *defl_param = 0;

   template<typename Float, typename CudaComplex>
   class GmresDRArgs{
     
      private:
      BlasMagmaArgs *gmresdr_magma_args;
    
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

      GmresDRArgs(int m, int nev);
      ~GmresDRArgs();
      
      //methods for constructing Lanczos matrix:
      void LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0);
      void LoadLanczosOffDiag(int idx, double alpha0, double beta0);
      
      //methods for Rayleigh Ritz procedure: 
      int RestartVm(void* vm, const int cld, const int clen, const int vprec);//complex length

      //methods 
      void FillLanczosDiag(const int _2nev);
      void FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2);
   };

   template<typename Float, typename CudaComplex>
   GmresDRArgs<Float, CudaComplex>::GmresDRArgs(int m, int nev): m(m), nev(nev){
    //include pad?
    ldm    = ((m+15)/16)*16;//too naive
       
    //magma initialization:
    const int prec = sizeof(Float);
    gmresdr_magma_args = new BlasMagmaArgs(m, nev, ldm, prec);

    hTm     = new std::complex<Float>[ldm*m];//VH A V
    hTvalm  = (Float*)malloc(m*sizeof(Float));   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

    //allocate dTm etc. buffers on GPU:
    cudaMalloc(&dTm, ldm*m*sizeof(CudaComplex));//  
    cudaMalloc(&dTvecm, ldm*m*sizeof(CudaComplex));  
    cudaMalloc(&dTvecm1, ldm*m*sizeof(CudaComplex));  

    //set everything to zero:
    cudaMemset(dTm, 0, ldm*m*sizeof(CudaComplex));//?
    cudaMemset(dTvecm, 0, ldm*m*sizeof(CudaComplex));
    cudaMemset(dTvecm1, 0, ldm*m*sizeof(CudaComplex));

    //Error check...
    checkCudaError();

    return;
  }

  template<typename Float, typename CudaComplex>
  GmresDRArgs<Float, CudaComplex>::~GmresDRArgs() {
    delete[] hTm;

    free(hTvalm);

    cudaFree(dTm);
    cudaFree(dTvecm);
    cudaFree(dTvecm1);

    delete gmresdr_magma_args;

    return;
  }

  template<typename Float, typename CudaComplex>
  void GmresDRArgs<Float, CudaComplex>::LoadLanczosDiag(int idx, double alpha, double alpha0, double beta0)
  {
    hTm[idx*ldm+idx] = std::complex<Float>((Float)(1.0/alpha + beta0/alpha0), 0.0);
    return;
  } 

  template<typename Float, typename CudaComplex>
  void GmresDRArgs<Float, CudaComplex>::LoadLanczosOffDiag(int idx, double alpha, double beta)
  {
    hTm[(idx+1)*ldm+idx] = std::complex<Float>((Float)(-sqrt(beta)/alpha), 0.0f);//'U' 
    hTm[idx*ldm+(idx+1)] = hTm[(idx+1)*ldm+idx];//'L'
    return;
  }

  template<typename Float, typename CudaComplex>
  int GmresDRArgs<Float, CudaComplex>::RestartVm(void* v, const int cld, const int clen, const int vprec) 
  {
    //Create device version of the Lanczos matrix:
    cudaMemcpy(dTm, hTm, ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);//!

    //Solve m-dimensional eigenproblem:
    cudaMemcpy(dTvecm, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    gmresdr_magma_args->MagmaHEEVD((void*)dTvecm, (void*)hTvalm, m);

    //Solve (m-1)-dimensional eigenproblem:
    cudaMemcpy(dTvecm1, dTm,   ldm*m*sizeof(CudaComplex), cudaMemcpyDefault);
    gmresdr_magma_args->MagmaHEEVD((void*)dTvecm1, (void*)hTvalm, m-1);

    //Zero the last row (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
    cudaMemset2D(&dTvecm1[(m-1)], ldm*sizeof(CudaComplex), 0, sizeof(CudaComplex),  (m-1));

    //Attach nev old vectors to nev new vectors (note 2*nev << m):
    cudaMemcpy(&dTvecm[ldm*nev], dTvecm1, ldm*nev*sizeof(CudaComplex), cudaMemcpyDefault);

    //Perform QR-factorization and compute QH*Tm*Q:
    int i = eigcg_magma_args->MagmaORTH_2nev((void*)dTvecm, (void*)dTm);

    //Solve 2nev-dimensional eigenproblem:
    gmresdr_magma_args->MagmaHEEVD((void*)dTm, (void*)hTvalm, i);

    //solve zero unused part of the eigenvectors in dTm:
    cudaMemset2D(&(dTm[i]), ldm*sizeof(CudaComplex), 0, (m-i)*sizeof(CudaComplex), i);//check..

    //Restart V:
    gmresdr_magma_args->RestartV(v, cld, clen, vprec, (void*)dTvecm, (void*)dTm);

    return i;
  }


  template<typename Float, typename CudaComplex>
  void GmresDRArgs<Float, CudaComplex>::FillLanczosDiag(const int _2nev)
 {
    memset(hTm, 0, ldm*m*sizeof(std::complex<Float>));
    for (int i = 0; i < _2nev; i++) hTm[i*ldm+i]= hTvalm[i];//fill-up diagonal

    return;
 }

  template<typename Float, typename CudaComplex>
  void GmresDRArgs<Float, CudaComplex>::FillLanczosOffDiag(const int _2nev, cudaColorSpinorField *v, cudaColorSpinorField *u, double inv_sqrt_r2)
  {
    if(v->Precision() != u->Precision()) errorQuda("\nIncorrect precision...\n");
    for (int i = 0; i < _2nev; i++){
       std::complex<double> s = cDotProductCuda(*v, u->Eigenvec(i));
       s *= inv_sqrt_r2;
       hTm[_2nev*ldm+i] = std::complex<Float>((Float)s.real(), (Float)s.imag());
       hTm[i*ldm+_2nev] = conj(hTm[_2nev*ldm+i]);
    }
  }

  GmresDR::GmresDR(DiracMatrix &mat, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile) :
    DeflatedSolver(param, profile), mat(mat), matDefl(matDefl), search_space_prec(QUDA_INVALID_PRECISION), 
    Vm(0), initCGparam(param), profile(profile), gmres_alloc(false)
  {
    if((param.rhs_idx < param.deflation_grid) || (param.inv_type == QUDA_EIGCG_INVERTER))
    {
       if(param.nev > MAX_EIGENVEC_WINDOW )
       { 
          warningQuda("\nWarning: the eigenvector window is too big, using default value %d.\n", MAX_EIGENVEC_WINDOW);
          param.nev = MAX_EIGENVEC_WINDOW;
       }

       search_space_prec = param.precision_ritz;
       //
       use_gmres = true;
       //
       printfQuda("\nIncEigCG will deploy eigCG(m=%d, nev=%d) solver.\n", param.m, param.nev);
    }
    else
    {
       fillInitCGSolveParam(initCGparam);
       //
       use_gmres = false;
       //
       printfQuda("\nIncEigCG will deploy initCG solver.\n");
    }

    return;
  }

  GmresDR::~GmresDR() {

    if(gmres_alloc)   delete Vm;

  }

  void GmresDR::GmresDRFirstCycle(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    if (eigcg_precision != x.Precision()) errorQuda("\nInput/output field precision is incorrect (solver precision: %u spinor precision: %u).\n", eigcg_precision, x.Precision());

    profile.Start(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);

    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      return;
    }

    cudaColorSpinorField r(b);

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    cudaColorSpinorField y(b, csParam);
    //
    cudaColorSpinorField Av(x, csParam);
    //
    cudaColorSpinorField tmp(x, csParam);


    mat(r, x, tmp);
    double r2 = xmyNormCuda(b, r);//compute residual

    zeroCuda(y);
    
    if(gmres_alloc == false){

       printfQuda("\nAllocating resources for the EigCG solver...\n");

       //Create an eigenvector set:
       csParam.create   = QUDA_ZERO_FIELD_CREATE;
       csParam.setPrecision(x.Precision());//precision for the Arnoldi vectors.
       csParam.eigv_dim = param.m;

       Vm = new cudaColorSpinorField(csParam);

       checkCudaError();
       printfQuda("\n..done.\n");
       
       gmres_alloc = true;
    }

    double beta= norm2(r);

    //load the first vector:
    copyCuda(Vm->Eigenvec(0), r);//convert arrays?

    //rescale the vector
    axCuda(1.0 / beta, Vm->Eigenvec(0));

 
    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);
    blas_flops = 0;

    PrintStats("GMRES DR (first cycle)", k, r2, b2);

    double sigma = 0.0;

    const int m = param.m;

    while ( !convergence(r2, stop) && k < m) {

      mat(Av, Vm->Eigenvec(k), tmp);
      
      for(int l  = 0; l <= k; l++)
      {
         H[l*m+k]=cDotProductCuda(Vm->Eigenvec(l), Av);
         caxpbyCuda(H[l*m+k], Vm->Eigenvec(l), -1.0, Av);
      } 

      H[(k+1)*m+k] = sqrt(norm2(Av));

      //perform rotation (principal part):

      for(int l = 0; l < k; l++)
      {
         Complex tmp1 = H[l*m+k]*s[l] - c[i]*H[(l+1)*m+k];
         Complex tmp2 = H[(l+1)*m+k]*s[l] + conj(c[i])*H[l*m+k];

	 H[l*m+k]     = tmp1;
	 H[(l+1)*m+k] = tmp2;
      } 
      
      beta     = sqrt(norm(H[k*m+k])+norm(H[(k+1)*m+k]));
      s[k]     = H[(k+1)*m+k].real()/beta;
      c[k]     = H[k*m+k]/beta;
      H[k*m+k] = beta; 
 
      //update vector g:
      g[k+1] = g[k] * s[k];
      g[k]   = conj(c[k])*g[k];

      //get residual norm:
      r2 = g[l+1].real()*g[l+1].real();

      axCuda(1/H[(l+1)*m+l].real(), Av);
      copyCuda(Vm->Eigenvec(k+1), Av);

      k++;

      PrintStats("GMRES DR (first cycle)", k, r2, b2);
    }

    if(convergence(r2, stop))//we done , get solution
    {
       g[k] /= H[k*m+k].real(); 
       caxpyCuda(g[k], Vm->Eigenvec(k), x);
    
       //solve triangular system and update the solution here...
    }

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops())*1e-9;
    reduceDouble(gflops);
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE){
      printfQuda("GMRES: Eigenspace restarts = %d\n", eigvRestart);
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

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp) delete tmp2_p;

//Clean EigCG resources:
    delete v0;

    profile.Stop(QUDA_PROFILE_FREE);

    return;
  }

//END of eigcg solver.

//Deflation space management:
  void GmresDR::CreateDeflationSpace(cudaColorSpinorField &eigcgSpinor, DeflationParam *&dpar)
  {
    printfQuda("\nCreate deflation space...\n");

    if(eigcgSpinor.SiteSubset() != QUDA_PARITY_SITE_SUBSET) errorQuda("\nRitz spinors must be parity spinors\n");//or adjust it

    ColorSpinorParam cudaEigvParam(eigcgSpinor);

    dpar = new DeflationParam(cudaEigvParam, param);

    printfQuda("\n...done.\n");

    //dpar->PrintInfo();

    return;
  }

  void GmresDR::DeleteDeflationSpace(DeflationParam *&dpar)
  {
    if(dpar != 0) 
    {
      delete dpar;
      dpar = 0;
    }

    return;
  }


  void GmresDR::ExpandDeflationSpace(DeflationParam *dpar, const int newnevs)
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

     for (int j = dpar->cur_dim; j < (dpar->cur_dim+addednev); j++)//
     {
       matDefl(*W, dpar->cudaRitzVectors->Eigenvec(j), tmp);

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

     delete W;
     delete W2;

     return;
  }


  void GmresDR::DeflateSpinor(cudaColorSpinorField &x, cudaColorSpinorField &b, DeflationParam *dpar, bool set2zero)
  {
    if(set2zero) zeroCuda(x);
    if(dpar->cur_dim == 0) return;//nothing to do

    BlasMagmaArgs *magma_args = new BlasMagmaArgs(sizeof(double));//change precision..

    Complex  *vec   = new Complex[dpar->ld];

    double check_nrm2 = norm2(b);
    printfQuda("\nSource norm (gpu): %1.15e\n", sqrt(check_nrm2));


    for(int i = 0; i < dpar->cur_dim; i++)
    {
      vec[i] = cDotProductCuda(dpar->cudaRitzVectors->Eigenvec(i), b);//<i, b>
    }    

    magma_args->SolveProjMatrix((void*)vec, dpar->ld,  dpar->cur_dim, (void*)dpar->proj_matrix, dpar->ld);

    for(int i = 0; i < dpar->cur_dim; i++)
    {
      caxpyCuda(vec[i], dpar->cudaRitzVectors->Eigenvec(i), x); //a*i+x
    }

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));


    delete magma_args;

    delete [] vec;

    return;
  }

//!!!!
//copy EigCG ritz vectors.
  void GmresDR::SaveEigCGRitzVecs(DeflationParam *dpar, bool cleanEigCGResources)
  {
     const int first_idx = dpar->cur_dim; 

     if(dpar->cudaRitzVectors->EigvDim() < (first_idx+param.nev)) errorQuda("\nNot enough space to copy %d vectors..\n", param.nev); 

     else if(!eigcg_alloc || !dpar->cuda_ritz_alloc) errorQuda("\nEigCG resources were cleaned.\n"); 

     for(int i = 0; i < param.nev; i++) copyCuda(dpar->cudaRitzVectors->Eigenvec(first_idx+i), Vm->Eigenvec(i));
     
     if(cleanEigCGResources)
     {
       delete Vm;
       Vm = 0;
       eigcg_alloc = false;
     }
     else//just call zeroCuda..
     {
       zeroCuda(*Vm);
     }

     return;
  }

  void GmresDR::CleanResources()
  {
    if(eigcg_alloc)
    {
       delete Vm;
       Vm = 0;
       eigcg_alloc = false;
    }
    if(defl_param != 0)
    {
       DeleteDeflationSpace(defl_param);
       defl_param = 0;
    }

    return;
  } 

  void GmresDR::operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) 
  {
     if(defl_param == 0) CreateDeflationSpace(*in, defl_param);

     //if this operator applied during the first stage of the incremental eigCG (to construct deflation space):
     //then: call eigCG inverter 
     if(use_eigcg){

        const bool use_mixed_prec = (eigcg_precision != param.precision); 

        //deflate initial guess:
        DeflateSpinor(*out, *in, defl_param);

        cudaColorSpinorField *outSloppy = 0;
        cudaColorSpinorField *inSloppy  = 0;

        double ext_tol = param.tol;

        if(use_mixed_prec)
        {
           ColorSpinorParam cudaParam(*out);
           //
           cudaParam.create = QUDA_ZERO_FIELD_CREATE;
           //
           cudaParam.setPrecision(eigcg_precision);

           outSloppy = new cudaColorSpinorField(cudaParam);
           inSloppy  = new cudaColorSpinorField(cudaParam);

           copyCuda(*inSloppy, *in);//input is outer residual
           copyCuda(*outSloppy, *out);

           param.tol = 1e-7;//single precision eigcg tolerance
        }
        else//full precision solver:
        {
           outSloppy = out;
           inSloppy  = in;
        }

        EigCG(*outSloppy, *inSloppy);

        if(use_mixed_prec)
        {
           double b2   = norm2(*in);
           double stop = b2*ext_tol*ext_tol;
      
           param.tol   = 5e-3;//initcg sloppy precision tolerance

           cudaColorSpinorField y(*in);//full precision accumulator
           cudaColorSpinorField r(*in);//full precision residual

           Solver *initCG = 0;

           initCGparam.tol       = param.tol;
           initCGparam.precision = eigcg_precision;//the same as eigcg
           //
           initCGparam.precision_sloppy = QUDA_HALF_PRECISION; //may not be half, in general?    
           initCGparam.use_sloppy_partial_accumulator=false;   //more stable single-half solver
     
           //no reliable updates?

           initCG = new CG(matSloppy, matCGSloppy, initCGparam, profile);

           //
           copyCuda(*out, *outSloppy);
           //
           mat(r, *out, y); // here we can use y as tmp
           //
           double r2 = xmyNormCuda(*in, r);//new residual (and RHS)
          
           while(r2 > stop)
           {
              zeroCuda(y);//deflate initial guess:
              //
              DeflateSpinor(y, r, defl_param);
              //
              copyCuda(*inSloppy, r);
              //
              copyCuda(*outSloppy, y);
              // 
              (*initCG)(*outSloppy, *inSloppy);

              copyCuda(y, *outSloppy);
              //
              xpyCuda(y, *out); //accumulate solution
              //
              mat(r, *out, y);  //here we can use y as tmp
              //
              r2 = xmyNormCuda(*in, r);//new residual (and RHS)
              //
              param.secs   += initCGparam.secs;

           }

           //clean objects:
           //
           delete initCG;
           //
           delete outSloppy;
           //
           delete inSloppy;
        }

	//store computed Ritz vectors:
        SaveEigCGRitzVecs(defl_param);

        //Construct(extend) projection matrix:
        ExpandDeflationSpace(defl_param, param.nev);

        //copy solver statistics:
        param.iter   += initCGparam.iter;
        //
        //param.secs   += initCGparam.secs;
        //
        param.gflops += initCGparam.gflops;

        if(getVerbosity() >= QUDA_VERBOSE)
        {
              printfQuda("\neigCG  stat: %i iter / %g secs = %g Gflops. \n", param.iter, param.secs, param.gflops);
              printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", initCGparam.iter, initCGparam.secs, initCGparam.gflops);
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

        DeflateSpinor(*out, *in, defl_param);

        //initCGparam.precision_sloppy = QUDA_HALF_PRECISION;

        int max_restart_num = 3;

        int restart_idx  = 0;

        double inc_tol = 1e-2; 

        //launch initCG:
        while((restart_tol > full_tol) && (restart_idx < max_restart_num))//currently just one restart, think about better algorithm for the restarts. 
        {
          initCGparam.tol = restart_tol; 

          initCG = new CG(mat, matCGSloppy, initCGparam, profile);

          (*initCG)(*out, *in);           

          delete initCG;

          matDefl(*W, *out, tmp);

          xpayCuda(*in, -1, *W); 

          DeflateSpinor(*out, *W, defl_param, false);                

          if(getVerbosity() >= QUDA_VERBOSE)
          {
            printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", initCGparam.iter, initCGparam.secs, initCGparam.gflops);
          }

          double new_restart_tol = restart_tol*inc_tol;

          restart_tol = (new_restart_tol > full_tol) ? new_restart_tol : full_tol;                               

          restart_idx += 1;

          //param.iter   += initCGparam.iter;
          //
          param.secs   += initCGparam.secs;
          //
          //param.gflops += initCGparam.gflops;              
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
