#include "blas_magma.h"
#include <string.h>

#ifdef MAGMA_LIB
#include "magma.h"
#endif

BlasMagmaArgs::BlasMagmaArgs(const int prec) : m(0), nev(0), prec(prec), ldTm(0), nb(0), llwork(0), lrwork(0), liwork(0), htsize(0), dtsize(0),  sideLR(0), lwork_max(0), W(0), hTau(0), dTau(0), lwork(0), rwork(0), iwork(0),  info(-1)
{
#ifdef MAGMA_LIB
    magma_init();
    complex_prec = 2*prec;
    alloc = false;
    init  = true;
#else
    printf("\nError: Magma library was not compiled, check your compilation options...\n");
    exit(-1);
    //init = false;
#endif    
    return;
}


BlasMagmaArgs::BlasMagmaArgs(const int m, const int nev, const int prec) : m(m), nev(nev), prec(prec), info(-1)
{
#ifdef MAGMA_LIB
    magma_init();

    complex_prec = 2*prec;

    //magma params/objects:
    ldTm  = m;//hTm (host/device)ld (may include padding)

    nb    = prec == 4 ? magma_get_chetrd_nb(m) : magma_get_zhetrd_nb(m);

    llwork = MAX(m + m*nb, 2*m + m*m); 
    lrwork = 1 + 5*m + 2*m*m;
    liwork = 3 + 5*m;

    htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    dtsize   = ( 4*nev + ((2*nev + 31)/32)*32 )*nb;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    sideLR = (m - 2*nev + nb)*(m + nb) + m*nb;
    lwork_max = sideLR; 

    magma_malloc_pinned((void**)&W, lwork_max*complex_prec);
    magma_malloc_pinned((void**)&hTau, htsize*complex_prec);//fixed!
    magma_malloc((void**)&dTau, dtsize*complex_prec);

    magma_malloc_pinned((void**)&lwork, llwork*complex_prec);
    magma_malloc_cpu((void**)&rwork, lrwork*prec);
    magma_malloc_cpu((void**)&iwork, liwork*sizeof(magma_int_t));

    init  = true;
    alloc = true;
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
   if(alloc == true)
   {
     magma_free(dTau);
     magma_free_pinned(hTau);

     magma_free_pinned(W);
     magma_free_pinned(lwork);

     magma_free_cpu(rwork);
     magma_free_cpu(iwork);
     alloc = false;
   }
   magma_finalize();
   init  = false;
#endif
   return;
}

int BlasMagmaArgs::RayleighRitz(void *dTm, void *dTvecm0, void *dTvecm1,  void *hTvecm,  void *hTvalm)
{
     int l = 0;
#ifdef MAGMA_LIB
     if(prec == 4)
     {
        //solve m-dim eigenproblem:
        cudaMemcpy(dTvecm0, dTm, ldTm*m*complex_prec, cudaMemcpyDefault);
        magma_cheevd_gpu('V', 'U', m, (magmaFloatComplex*)dTvecm0, ldTm, (float*)hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                       (magmaFloatComplex*)lwork, llwork, (float*)rwork, lrwork, (magma_int_t*)iwork, liwork, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

        //solve (m-1)-dim eigenproblem:
        cudaMemcpy(dTvecm1, dTm, ldTm*m*complex_prec, cudaMemcpyDefault);
        magma_cheevd_gpu('V', 'U', (m-1), (magmaFloatComplex*)dTvecm1, ldTm, (float*)hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                        (magmaFloatComplex *)lwork, llwork, (float*)rwork, lrwork, iwork, liwork, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);
        
        //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
        cudaMemset2D(&(((magmaFloatComplex *)dTvecm1)[(m-1)]), ldTm*complex_prec, 0, complex_prec,  m-1);

        //attach nev old vectors to nev new vectors (note 2*nev < m):
        cudaMemcpy(&(((magmaFloatComplex *)dTvecm0)[nev*m]), dTvecm1, nev*m*complex_prec, cudaMemcpyDefault);

        //Orthogonalize 2*nev vectors:
        l = 2 * nev;
        magma_cgeqrf_gpu(m, l, (magmaFloatComplex *)dTvecm0, ldTm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTau, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_cgeqrf_gpu, exit ...\n"), exit(-1);

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_cunmqr_gpu( 'R', 'N', m, m, l, (magmaFloatComplex *)dTvecm0, ldTm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldTm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, (magma_int_t*)&info); 
        if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);
             	
        //get QHT product:
        magma_cunmqr_gpu( 'L', 'C', m, m, l, (magmaFloatComplex *)dTvecm0, ldTm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldTm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);                 	

        //solve l=2*nev-dim eigenproblem:
//dTm
        magma_cheevd_gpu('V', 'U', l, (magmaFloatComplex*)dTm, ldTm, (float*)hTvalm, (magmaFloatComplex*)hTvecm, ldTm, 
                       (magmaFloatComplex *)lwork, llwork, (float*)rwork, lrwork, iwork, liwork, (magma_int_t*)&info);

        if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

        //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
        cudaMemset2D(&(((magmaFloatComplex *)dTm)[l]), ldTm*complex_prec, 0, (m-l)*complex_prec, l);//check..
        
        //Compute dTm=dTevecm0*dTm (Q * Z):
        //(compute QT product):
        magma_cunmqr_gpu('L', 'N', m, m, l, (magmaFloatComplex *)dTvecm0, ldTm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldTm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, (magma_int_t*)&info);

        if(info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1); 
     }
     else if(prec == 8)
     {
        //solve m-dim eigenproblem:
        cudaMemcpy(dTvecm0, dTm, ldTm*m*complex_prec, cudaMemcpyDefault);
        magma_zheevd_gpu('V', 'U', m, (magmaDoubleComplex*)dTvecm0, ldTm, (double*)hTvalm, (magmaDoubleComplex*)hTvecm, ldTm, 
                       (magmaDoubleComplex*)lwork, llwork, (double*)rwork, lrwork, (magma_int_t*)iwork, liwork, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

        //solve (m-1)-dim eigenproblem:
        cudaMemcpy(dTvecm1, dTm, ldTm*m*complex_prec, cudaMemcpyDefault);
        magma_zheevd_gpu('V', 'U', (m-1), (magmaDoubleComplex*)dTvecm1, ldTm, (double*)hTvalm, (magmaDoubleComplex*)hTvecm, ldTm, 
                        (magmaDoubleComplex *)lwork, llwork, (double*)rwork, lrwork, iwork, liwork, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

        //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
        cudaMemset2D(&(((magmaDoubleComplex *)dTvecm1)[(m-1)]), ldTm*complex_prec, 0, complex_prec,  m-1);

        //attach nev old vectors to nev new vectors (note 2*nev < m):
        cudaMemcpy(&(((magmaDoubleComplex *)dTvecm0)[nev*m]), dTvecm1, nev*m*complex_prec, cudaMemcpyDefault);

        //Orthogonalize 2*nev vectors:
        l = 2 * nev;
        magma_zgeqrf_gpu(m, l, (magmaDoubleComplex *)dTvecm0, ldTm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTau, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_zgeqrf_gpu, exit ...\n"), exit(-1);

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_zunmqr_gpu( 'R', 'N', m, m, l, (magmaDoubleComplex *)dTvecm0, ldTm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldTm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, (magma_int_t*)&info); 
        if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);
             	
        //get QHT product:
        magma_zunmqr_gpu( 'L', 'C', m, m, l, (magmaDoubleComplex *)dTvecm0, ldTm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldTm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);                 	

        //solve l=2*nev-dim eigenproblem:
//dTm
        magma_zheevd_gpu('V', 'U', l, (magmaDoubleComplex*)dTm, ldTm, (double*)hTvalm, (magmaDoubleComplex*)hTvecm, ldTm, 
                       (magmaDoubleComplex *)lwork, llwork, (double*)rwork, lrwork, iwork, liwork, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

        //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
        cudaMemset2D(&(((magmaDoubleComplex *)dTm)[l]), ldTm*complex_prec, 0, (m-l)*complex_prec, l);//check..
        
        //Compute dTm=dTevecm0*dTm (Q * Z):
        //(compute QT product):
        magma_zunmqr_gpu('L', 'N', m, m, l, (magmaDoubleComplex *)dTvecm0, ldTm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldTm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, (magma_int_t*)&info);
        if(info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1); 
     }
     else
     {
        printf("\nError:unsupported precision...\n"), exit(-1);
     } 
#endif
      return l;
}


void BlasMagmaArgs::Restart_2nev_vectors(void *dVm, void *dQ, const int len)
{
#ifdef MAGMA_LIB
       int _2nev = 2*nev;
 
       magma_int_t ldV       = (magma_int_t) len;//complex length
       magma_int_t ldQ       = m;//not vsize (= 2*nev) 

       if(prec == 4)
       {
         magmaFloatComplex *Tmp;
         magma_malloc((void**)&Tmp, ldV*_2nev*complex_prec); 

         cudaMemset(Tmp, 0, ldV*_2nev*complex_prec); 
         magmablas_cgemm('N', 'N', ldV, _2nev, m, MAGMA_C_ONE, (magmaFloatComplex*)dVm, ldV, (magmaFloatComplex*)dQ, ldQ, MAGMA_C_ZERO, Tmp, ldV);//in colour-major format
         cudaMemcpy(dVm, Tmp, ldV*(_2nev)*complex_prec, cudaMemcpyDefault); 

         magma_free(Tmp);
       }
       else if(prec == 8)
       {
         magmaDoubleComplex *Tmp;
         magma_malloc((void**)&Tmp, ldV*_2nev*complex_prec); 

         cudaMemset(Tmp, 0, ldV*_2nev*complex_prec); 
         magmablas_zgemm('N', 'N', ldV, _2nev, m, MAGMA_Z_ONE, (magmaDoubleComplex*)dVm, ldV, (magmaDoubleComplex*)dQ, ldQ, MAGMA_Z_ZERO, Tmp, ldV);//in colour-major format
         cudaMemcpy(dVm, Tmp, ldV*(_2nev)*complex_prec, cudaMemcpyDefault); 

         magma_free(Tmp);
       }
       else
       {
         printf("\nError: unsupported precision\n"), exit(-1);
       }
#endif
       return;
}

void BlasMagmaArgs::SolveProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
#ifdef MAGMA_LIB
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc_pinned((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       memcpy(tmp, H, ldH*n*complex_prec);
       if(prec == 8)
       {
          err = magma_zgesv(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in magma_zgesv, exit ...\n"), exit(-1);
       }
       else if (prec == 4)
       {
          err = magma_cgesv(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
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

void BlasMagmaArgs::SolveGPUProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
#ifdef MAGMA_LIB
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       cudaMemcpy(tmp, H, ldH*n*complex_prec, cudaMemcpyDefault);

       if(complex_prec == 16)
       {
          err = magma_zgesv_gpu(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in magma_zgesv, exit ...\n"), exit(-1);
       }
       else if (complex_prec == 8)
       {
          err = magma_cgesv_gpu(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in magma_cgesv, exit ...\n"), exit(-1);
       }
       else
       {
          printf("\nError: unsupported precision.\n");
          exit(-1);   
       }
       magma_free(tmp);
       magma_free_pinned(ipiv);
#endif
       return;
}

void BlasMagmaArgs::SpinorMatVec
(void *spinorOut, const void *spinorSetIn, const void *vec, const int slen, const int vlen)
{
#ifdef MAGMA_LIB
       magma_int_t m       = (magma_int_t)slen;
       magma_int_t n       = (magma_int_t)vlen;
       magma_int_t lds     = m;

       if(prec == 8)
       {
           magmaDoubleComplex *spmat = (magmaDoubleComplex*)spinorSetIn; 
           magmaDoubleComplex *spout = (magmaDoubleComplex*)spinorOut; 

           magmablas_zgemv('N', m, n, MAGMA_Z_ONE, spmat, lds, (magmaDoubleComplex*)vec, 1, MAGMA_Z_ZERO, spout, 1);//in colour-major format
       }
       else if (prec == 4)
       {
           magmaFloatComplex *spmat = (magmaFloatComplex*)spinorSetIn; 
           magmaFloatComplex *spout = (magmaFloatComplex*)spinorOut; 

           magmablas_cgemv('N', m, n, MAGMA_C_ONE, spmat, lds, (magmaFloatComplex*)vec, 1, MAGMA_C_ZERO, spout, 1);//in colour-major format
       }
       else
       {
          printf("\nError: unsupported precision.\n");
          exit(-1);   
       }
#endif
       return;
}


