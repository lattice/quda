#include "blas_magma.h"
#include <string.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

#ifdef MAGMA_LIB
#include "magma.h"
#endif

const char cV = 'V';
const char cU = 'U';
const char cR = 'R';
const char cL = 'L';
const char cN = 'N';
const char cC = 'C';

void BlasMagmaArgs::OpenMagma(){ 
 
    magma_err_t err;
    err = magma_init(); 
    if(err != MAGMA_SUCCESS) printf("\nError: cannot initialize MAGMA library\n");

    int major, minor, micro;
    magma_version( &major, &minor, &micro);
    printf("\nMAGMA library version: %d.%d\n\n", major,  minor);

}

void BlasMagmaArgs::CloseMagma(){  

    magma_err_t err; 
    err = magma_finalize();
    if(err != MAGMA_SUCCESS) printf("\nError: cannot close MAGMA library\n");
}

BlasMagmaArgs::BlasMagmaArgs(const int prec) : m(0), nev(0), prec(prec), ldm(0), llwork(0), lrwork(0), liwork(0), htsize(0), dtsize(0),  sideLR(0), lwork_max(0), W(0), W2(0), hTau(0), dTau(0), lwork(0), rwork(0), iwork(0),  info(-1)
{

#ifdef MAGMA_LIB
    //magma_init();
    magma_int_t dev_info;
    dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...
    if(dev_info == 0)  exit(-1);

    printf("\nMAGMA will use device architecture %d.\n", dev_info);

    alloc = false;
    init  = true;
#else
    printf("\nError: MAGMA library was not compiled, check your compilation options...\n");
    exit(-1);
    //init = false;
#endif    
    return;
}


BlasMagmaArgs::BlasMagmaArgs(const int m, const int nev, const int ldm, const int prec) : m(m), nev(nev), ldm(ldm), prec(prec), info(-1)
{
#ifdef MAGMA_LIB
    //magma_init();
    magma_int_t dev_info;
    dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...
    if(dev_info == 0)  exit(-1);

    printf("\nMAGMA will use device architecture %d.\n", dev_info);

    const int complex_prec = 2*prec;

    magma_int_t nbtrd = prec == 4 ? magma_get_chetrd_nb(ldm) : magma_get_zhetrd_nb(ldm);//ldm
    magma_int_t nbqrf = prec == 4 ? magma_get_cgeqrf_nb(ldm) : magma_get_zgeqrf_nb(ldm);//ldm

    llwork = MAX(ldm + ldm*nbtrd, 2*ldm + ldm*ldm);//ldm 
    lrwork = 1 + 5*ldm + 2*ldm*ldm;//ldm
    liwork = 3 + 5*ldm;//ldm

    htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    dtsize   = ( 2*htsize + ((htsize + 31)/32)*32 )*nbqrf;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    sideLR = (ldm - 2*nev + nbqrf)*(ldm + nbqrf) + ldm*nbqrf;//ldm

    magma_malloc_pinned((void**)&W,    sideLR*complex_prec);
    magma_malloc_pinned((void**)&W2,   ldm*m*complex_prec);
    magma_malloc_pinned((void**)&hTau, htsize*complex_prec);
    magma_malloc((void**)&dTau,        dtsize*complex_prec);

    magma_malloc_pinned((void**)&lwork, llwork*complex_prec);
    magma_malloc_cpu((void**)&rwork,    lrwork*prec);
    magma_malloc_cpu((void**)&iwork,    liwork*sizeof(magma_int_t));

    init  = true;
    alloc = true;
#else
    printf("\nError: MAGMA library was not compiled, check your compilation options...\n");
    exit(-1);
    //init = false;
#endif    
    return;
}

BlasMagmaArgs::~BlasMagmaArgs()
{
#ifdef MAGMA_LIB
   //if(!init) printf("\n\nError: MAGMA was not initialized..\n"), exit(-1);
   if(alloc == true)
   {
     magma_free(dTau);
     magma_free_pinned(hTau);

     magma_free_pinned(W);
     magma_free_pinned(W2);
     magma_free_pinned(lwork);

     magma_free_cpu(rwork);
     magma_free_cpu(iwork);
     alloc = false;
   }
   //magma_finalize();
   init  = false;
#endif
   return;
}

void BlasMagmaArgs::MagmaHEEVD(void *dTvecm, void *hTvalm, const int prob_size)
{
     if(prob_size > m) printf("\nError in MagmaHEEVD (problem size cannot exceed given search space %d), exit ...\n", m), exit(-1);
#ifdef MAGMA_LIB
     if(prec == 4)
     {
        magma_cheevd_gpu(cV, cU, prob_size, (magmaFloatComplex*)dTvecm, ldm, (float*)hTvalm, (magmaFloatComplex*)W2, ldm, (magmaFloatComplex*)lwork, llwork, (float*)rwork, lrwork, iwork, liwork, &info);
        if(info != 0) printf("\nError in MagmaHEEVD (magma_cheevd_gpu), exit ...\n"), exit(-1);
     }
     else
     {
        magma_zheevd_gpu(cV, cU, prob_size, (magmaDoubleComplex*)dTvecm, ldm, (double*)hTvalm, (magmaDoubleComplex*)W2, ldm, (magmaDoubleComplex*)lwork, llwork, (double*)rwork, lrwork, iwork, liwork, &info);
        if(info != 0) printf("\nError in MagmaHEEVD (magma_zheevd_gpu), exit ...\n"), exit(-1);
     } 
#endif
  return;
}  

int BlasMagmaArgs::MagmaORTH_2nev(void *dTvecm, void *dTm)
{
     const int l = 2*nev;

#ifdef MAGMA_LIB
     if(prec == 4)
     {
        magma_int_t nb = magma_get_cgeqrf_nb(ldm);//ldm

        magma_cgeqrf_gpu(m, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTau, &info);
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_cgeqrf_gpu), exit ...\n"), exit(-1);

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_cunmqr_gpu(cR,cN, m, m, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, &info); 
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_cunmqr_gpu), exit ...\n"), exit(-1);
             	
        //get QHT product:
        magma_cunmqr_gpu(cL,cC, m, l, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, &info);
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_cunmqr_gpu), exit ...\n"), exit(-1);  
     }
     else
     {
        magma_int_t nb = magma_get_zgeqrf_nb(ldm);//ldm

        magma_zgeqrf_gpu(m, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTau, &info);
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_zgeqrf_gpu), exit ...\n"), exit(-1);

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_zunmqr_gpu(cR,cN, m, m, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, &info); 
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n"), exit(-1);
             	
        //get QHT product:
        magma_zunmqr_gpu(cL,cC, m, l, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, &info);
        if(info != 0) printf("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n"), exit(-1);  

     }
#endif

  return l;
}

void BlasMagmaArgs::RestartV(void *dV, int ldn, void *dTevecm, void *dTm)
{
       const int complex_prec = 2*prec;
       int l                  = 2*nev;
#ifdef MAGMA_LIB 
       void *Tmp = 0;
       magma_malloc((void**)&Tmp, ldn*l*complex_prec);     

       cudaMemset(Tmp, 0, ldn*l*complex_prec);   

       if(prec == 4)
       {
         magma_int_t nb = magma_get_cgeqrf_nb(ldm);//ldm
         magma_cunmqr_gpu(cL, cN, m, l, l, (magmaFloatComplex*)dTevecm, ldm, (magmaFloatComplex*)hTau, (magmaFloatComplex*)dTm, ldm, (magmaFloatComplex*)W, sideLR, (magmaFloatComplex*)dTau, nb, &info);
        
         if(info != 0) printf("\nError in RestartV (magma_cunmqr_gpu), exit ...\n"), exit(-1); 

         magmablas_cgemm(cN, cN, ldn, l, m, MAGMA_C_ONE, (magmaFloatComplex*)dV, ldn, (magmaFloatComplex*)dTm, ldm, MAGMA_C_ZERO, (magmaFloatComplex*)Tmp, ldn);

       }
       else
       {
         magma_int_t nb = magma_get_zgeqrf_nb(ldm);//ldm
         magma_zunmqr_gpu(cL, cN, m, l, l, (magmaDoubleComplex*)dTevecm, ldm, (magmaDoubleComplex*)hTau, (magmaDoubleComplex*)dTm, ldm, (magmaDoubleComplex*)W, sideLR, (magmaDoubleComplex*)dTau, nb, &info);

         if(info != 0) printf("\nError in RestartV (magma_zunmqr_gpu), exit ...\n"), exit(-1);

         magmablas_zgemm(cN, cN, ldn, l, m, MAGMA_Z_ONE, (magmaDoubleComplex*)dV, ldn, (magmaDoubleComplex*)dTm, ldm, MAGMA_Z_ZERO, (magmaDoubleComplex*)Tmp, ldn);
       }

       cudaMemcpy(dV, Tmp, ldn*l*complex_prec, cudaMemcpyDefault); 

       magma_free(Tmp);
#endif
       return;
}


void BlasMagmaArgs::SolveProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
       const int complex_prec = 2*prec;
#ifdef MAGMA_LIB
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc_pinned((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       memcpy(tmp, H, ldH*n*complex_prec);

       if (prec == 4)
       {
          err = magma_cgesv(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in SolveProjMatrix (magma_cgesv), exit ...\n"), exit(-1);
       }
       else
       {
          err = magma_zgesv(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in SolveProjMatrix (magma_zgesv), exit ...\n"), exit(-1);
       }

       magma_free_pinned(tmp);
       magma_free_pinned(ipiv);
#endif
       return;
}

void BlasMagmaArgs::SolveGPUProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
       const int complex_prec = 2*prec;
#ifdef MAGMA_LIB
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       cudaMemcpy(tmp, H, ldH*n*complex_prec, cudaMemcpyDefault);

       if (prec == 4)
       {
          err = magma_cgesv_gpu(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in SolveGPUProjMatrix (magma_cgesv), exit ...\n"), exit(-1);
       }
       else
       {
          err = magma_zgesv_gpu(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) printf("\nError in SolveGPUProjMatrix (magma_zgesv), exit ...\n"), exit(-1);
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

       if (prec == 4)
       {
           magmaFloatComplex *spmat = (magmaFloatComplex*)spinorSetIn; 
           magmaFloatComplex *spout = (magmaFloatComplex*)spinorOut; 

           magmablas_cgemv(cN, m, n, MAGMA_C_ONE, spmat, lds, (magmaFloatComplex*)vec, 1, MAGMA_C_ZERO, spout, 1);//in colour-major format
       }
       else
       {
           magmaDoubleComplex *spmat = (magmaDoubleComplex*)spinorSetIn; 
           magmaDoubleComplex *spout = (magmaDoubleComplex*)spinorOut; 

           magmablas_zgemv(cN, m, n, MAGMA_Z_ONE, spmat, lds, (magmaDoubleComplex*)vec, 1, MAGMA_Z_ZERO, spout, 1);//in colour-major format
       }
#endif
       return;
}


