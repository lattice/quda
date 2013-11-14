#include "blas_magma.h"

#ifdef MAGMA_LIB
#include "magma.h"
#endif

BlasMagmaArgs::BlasMagmaArgs(const int m, const int nev) : m(m), nev(nev), info(-1)
{
#ifdef MAGMA_LIB
    magma_init();

    //magma params/objects:
    ldTm  = m;//hTm (host/device)ld (may include padding)

    nb    = magma_get_chetrd_nb(m);

    llwork = MAX(m + m*nb, 2*m + m*m); 
    lrwork = 1 + 5*m + 2*m*m;
    liwork = 3 + 5*m;

    htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    dtsize   = ( 4*nev + ((2*nev + 31)/32)*32 )*nb;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    sideLR = (m - 2*nev + nb)*(m + nb) + m*nb;
    lwork_max = sideLR; 

    magma_malloc_pinned((void**)&W, lwork_max*sizeof(magmaFloatComplex));
    magma_malloc_pinned((void**)&hTau, htsize*sizeof(magmaFloatComplex));//fixed!
    magma_malloc((void**)&dTau, dtsize*sizeof(magmaFloatComplex));

    magma_malloc_pinned((void**)&lwork, llwork*sizeof(magmaFloatComplex));
    magma_malloc_cpu((void**)&rwork, lrwork*sizeof(double));
    magma_malloc_cpu((void**)&iwork, liwork*sizeof(magma_int_t));

    init = true;
#else
    printf("\nError: Magma library was not compiled, check your compilation options...\n");
    exit(-1);
    //init = false;
#endif    
    return;
}

BlasMagmaArgs::~BlasMagmaArgs()
{
#ifdef MAGMA_LIB
   if(!init) printf("\n\nError: Magma was not initialized..\n"), exit(-1);
   magma_free(dTau);
   magma_free_pinned(hTau);

   magma_free_pinned(W);
   magma_free_pinned(lwork);

   magma_free_cpu(rwork);
   magma_free_cpu(iwork);

   magma_finalize();
   init = false;
#endif
   return;
}

int BlasMagmaArgs::RayleighRitz(cuComplex *dTm, cuComplex *dTvecm0, cuComplex *dTvecm1, std::complex<float> *hTvecm, float *hTvalm)
{
     int l = 0;
#ifdef MAGMA_LIB
     //solve m-dim eigenproblem:
     cudaMemcpy(dTvecm0, dTm, ldTm*m*sizeof(cuComplex), cudaMemcpyDefault);
     magma_cheevd_gpu('V', 'U', m, 
                      (magmaFloatComplex*)dTvecm0, ldTm, 
                       hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                       (magmaFloatComplex *)lwork, llwork, rwork, lrwork, (magma_int_t*)iwork, liwork, (magma_int_t*)&info);
     if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

     //solve (m-1)-dim eigenproblem:
     cudaMemcpy(dTvecm1, dTm, ldTm*m*sizeof(cuComplex), cudaMemcpyDefault);
     magma_cheevd_gpu('V', 'U', (m-1), 
                       (magmaFloatComplex*)dTvecm1, ldTm, 
                        hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                        (magmaFloatComplex *)lwork, llwork, rwork, lrwork, iwork, liwork, (magma_int_t*)&info);
     if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);
      //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
     cudaMemset2D(&dTvecm1[(m-1)], ldTm*sizeof(cuComplex), 0, sizeof(cuComplex),  m-1);

     //attach nev old vectors to nev new vectors (note 2*nev < m):
     cudaMemcpy(&dTvecm0[nev*m], dTvecm1, nev*m*sizeof(cuComplex), cudaMemcpyDefault);

      //Orthogonalize 2*nev vectors:
      l = 2 * nev;
      magma_cgeqrf_gpu(m, l, dTvecm0, ldTm, hTau, dTau, (magma_int_t*)&info);
      if(info != 0) printf("\nError in magma_cgeqrf_gpu, exit ...\n"), exit(-1);

      //compute dTevecm0=QHTmQ
      //get TQ product:
      magma_cunmqr_gpu( 'R', 'N', m, m, l, dTvecm0, ldTm, hTau, dTm, ldTm, W, sideLR, dTau, nb, (magma_int_t*)&info); 
      if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);
             	
      //get QHT product:
      magma_cunmqr_gpu( 'L', 'C', m, m, l, dTvecm0, ldTm, hTau, dTm, ldTm, W, sideLR, dTau, nb, (magma_int_t*)&info);
      if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);                 	

      //solve l=2*nev-dim eigenproblem:
//dTm
      magma_cheevd_gpu('V', 'U', l, 
                      (magmaFloatComplex*)dTm, ldTm, 
                       hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                       lwork, llwork, rwork, lrwork, iwork, liwork, (magma_int_t*)&info);
      if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

      //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
      cudaMemset2D(&dTm[l], ldTm*sizeof(cuComplex), 0, (m-l)*sizeof(cuComplex),  l);//check..
        
      //Compute dTm=dTevecm0*dTm (Q * Z):
      //(compute QT product):
      magma_cunmqr_gpu('L', 'N', m, m, l, dTvecm0, ldTm, hTau, dTm, ldTm, W, sideLR, dTau, nb, (magma_int_t*)&info);
      if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1); 
#endif
      return l;
}


void BlasMagmaArgs::Restart_2nev_vectors(cuComplex *dVm, cuComplex *dQ, const int len)
{
#ifdef MAGMA_LIB
       int _2nev = 2*nev;
 
       magma_int_t ldV       = (magma_int_t) (len/2);//len includes complex!
       magma_int_t ldQ       = m;//not vsize (= 2*nev) 

       magmaFloatComplex *V = (magmaFloatComplex*)dVm; 
       magmaFloatComplex *Tmp;
       magma_malloc((void**)&Tmp, ldV*_2nev*sizeof(magmaFloatComplex)); 

       cudaMemset(Tmp, 0, ldV*_2nev*sizeof(magmaFloatComplex)); 
       magmablas_cgemm('N', 'N', ldV, _2nev, m, MAGMA_C_ONE, V, ldV, dQ, ldQ, MAGMA_C_ZERO, Tmp, ldV);//in colour-major format
       cudaMemcpy(V, Tmp, ldV*(_2nev)*sizeof(cuComplex), cudaMemcpyDefault); 

       magma_free(Tmp);
#endif
       return;
}

void BlasMagmaArgs::SolveProjMatrix(void* rhs, const int n, void* H, const int ldH, const int prec)
{
#ifdef MAGMA_LIB
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;//??

       magma_malloc_pinned((void**)&tmp, ldH*n*(2*prec));
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       if(prec == sizeof(double))
       {
          magma_zcopymatrix(n, n, (magmaDoubleComplex*)tmp, ldH, (magmaDoubleComplex*)H, ldH);
          err = magma_zgesv(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, n, &info);
          if(err != 0) printf("\nError in magma_zgesv, exit ...\n"), exit(-1);
       }
       else if (prec == sizeof(float))
       {
          magma_ccopymatrix(n, n, (magmaFloatComplex*)tmp, ldH, (magmaFloatComplex*)H, ldH);
          err = magma_cgesv(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, n, &info);
          if(err != 0) printf("\nError in magma_cgesv, exit ...\n"), exit(-1);
       }
       else
       {
          printf("\nError: unsupported precision.\n");
          exit(-1);   
       }
       magma_free_pinned(tmp);
       magma_free_pinned(ipiv);
#endif
       return;
}

void BlasMagmaArgs::SolveGPUProjMatrix(void* rhs, const int n, void* H, const int ldH, const int prec)
{
#ifdef MAGMA_LIB
       if(prec == sizeof(double))
       {
       }
       else if (prec == sizeof(float))
       {
       }
       else
       {
          printf("\nError: unsupported precision.\n");
          exit(-1);
       }
#endif
       return;
}


