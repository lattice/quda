#include "blas_magma.h"
#include <string.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

#ifdef MAGMA_LIB
#include "magma.h"
#endif

char cV = 'V';
char cU = 'U';
char cR = 'R';
char cL = 'L';
char cN = 'N';
char cC = 'C';

void BlasMagmaArgs::OpenMagma(){ 
 
    magma_err_t err = magma_init(); 

    if(err != MAGMA_SUCCESS) printf("\nError: cannot initialize MAGMA library\n");

    int major, minor, micro;

    magma_version( &major, &minor, &micro);
    printf("\nMAGMA library version: %d.%d\n\n", major,  minor);

    return;
}

void BlasMagmaArgs::CloseMagma(){  

    magma_err_t err = magma_finalize();

    if(magma_finalize() != MAGMA_SUCCESS) printf("\nError: cannot close MAGMA library\n");

    return;
}

BlasMagmaArgs::BlasMagmaArgs(const int prec) : m(0), nev(0), prec(prec), ldm(0), llwork(0), lrwork(0), liwork(0), htsize(0), dtsize(0),  sideLR(0), lwork_max(0), W(0), W2(0), hTau(0), dTau(0), lwork(0), rwork(0), iwork(0),  info(-1)
{

#ifdef MAGMA_LIB

    magma_int_t dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...
    if(dev_info == 0)  exit(-1);

    printf("\nMAGMA will use device architecture %d.\n", dev_info);

    alloc = false;
    init  = true;

#else

    printf("\nError: MAGMA library was not compiled, check your compilation options...\n");
    exit(-1);

#endif    

    return;
}


BlasMagmaArgs::BlasMagmaArgs(const int m, const int nev, const int ldm, const int prec) : m(m), nev(nev), ldm(ldm), prec(prec), info(-1)
{

#ifdef MAGMA_LIB

    magma_int_t dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...

    if(dev_info == 0)  exit(-1);

    printf("\nMAGMA will use device architecture %d.\n", dev_info);

    const int complex_prec = 2*prec;

    magma_int_t nbtrd = prec == 4 ? magma_get_chetrd_nb(m) : magma_get_zhetrd_nb(m);//ldm
    magma_int_t nbqrf = prec == 4 ? magma_get_cgeqrf_nb(m) : magma_get_zgeqrf_nb(m);//ldm

    llwork = MAX(m + m*nbtrd, 2*m + m*m);//ldm 
    lrwork = 1 + 5*m + 2*m*m;//ldm
    liwork = 3 + 5*m;//ldm

    htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    dtsize   = ( 2*htsize + ((htsize + 31)/32)*32 )*nbqrf;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    sideLR = (m - 2*nev + nbqrf)*(m + nbqrf) + m*nbqrf;//ldm

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

#endif    

    return;
}

BlasMagmaArgs::~BlasMagmaArgs()
{
#ifdef MAGMA_LIB

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
        magma_int_t nb = magma_get_cgeqrf_nb(m);//ldm

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
        magma_int_t nb = magma_get_zgeqrf_nb(m);//ldm

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

#define __min(a, b) (a < b ? a : b)

void BlasMagmaArgs::RestartV(void *dV, const int vld, const int vlen, const int vprec, void *dTevecm, void *dTm)
{
       const int complex_prec = 2*prec;
       int l                  = 2*nev;
#ifdef MAGMA_LIB 
//
       //void *Tmp = 0;
       //magma_malloc((void**)&Tmp, vld*l*complex_prec);     

       //cudaMemset(Tmp, 0, vld*l*complex_prec);   
//
       const int rworkSize = 2*vld+l*l;
      
       int AvailRows = __min(rworkSize / l, vlen);

       void  *rwork;
       magma_malloc(&rwork, rworkSize*complex_prec);
       cudaMemset(rwork, 0, rworkSize*complex_prec);


       if(prec == 4)
       {
         magma_int_t nb = magma_get_cgeqrf_nb(m);//ldm
         magma_cunmqr_gpu(cL, cN, m, l, l, (magmaFloatComplex*)dTevecm, ldm, (magmaFloatComplex*)hTau, (magmaFloatComplex*)dTm, ldm, (magmaFloatComplex*)W, sideLR, (magmaFloatComplex*)dTau, nb, &info);
        
         if(info != 0) printf("\nError in RestartV (magma_cunmqr_gpu), exit ...\n"), exit(-1); 
       }
       else
       {
         magma_int_t nb = magma_get_zgeqrf_nb(m);//ldm
         magma_zunmqr_gpu(cL, cN, m, l, l, (magmaDoubleComplex*)dTevecm, ldm, (magmaDoubleComplex*)hTau, (magmaDoubleComplex*)dTm, ldm, (magmaDoubleComplex*)W, sideLR, (magmaDoubleComplex*)dTau, nb, &info);

         if(info != 0) printf("\nError in RestartV (magma_zunmqr_gpu), exit ...\n"), exit(-1); 
       }

       if(vprec == 4)
       {
         magmaFloatComplex *dtm;

         if(prec == 8)
         {
            magma_malloc((void**)&dtm, ldm*l*prec);

            double *hbuff1;
            float  *hbuff2;

            magma_malloc_pinned((void**)&hbuff1, ldm*l*complex_prec );
            magma_malloc_pinned((void**)&hbuff2, ldm*l*prec);

            cudaMemcpy(hbuff1, dTm, ldm*l*complex_prec, cudaMemcpyDefault); 
            for(int i = 0; i < ldm*l; i++) hbuff2[i] = (float)hbuff1[i];

            cudaMemcpy(dtm, hbuff2, ldm*l*prec, cudaMemcpyDefault); 

            magma_free_pinned(hbuff1);
            magma_free_pinned(hbuff2);
         }
         else 
         {
            dtm = (magmaFloatComplex *)dTm;
         }

         //magmablas_cgemm(cN, cN, vlen, l, m, MAGMA_C_ONE, (magmaFloatComplex*)dV, vld, dtm, ldm, MAGMA_C_ZERO, (magmaFloatComplex*)Tmp, vld);

         int i = 0;
         while (i < vlen) 
         {
           magmaFloatComplex *ptrV = &(((magmaFloatComplex*)dV)[i]);
           magmablas_cgemm(cN, cN, AvailRows, l, m, MAGMA_C_ONE, ptrV, vld, dtm, ldm, MAGMA_C_ZERO, (magmaFloatComplex*)rwork, AvailRows);

           for (int k = 0; k < l; k++) {

             magmaFloatComplex *ptrV = &(((magmaFloatComplex*)dV)[i + vld * k]);
             cudaMemcpy(ptrV, &((magmaFloatComplex*)rwork)[AvailRows*k], AvailRows*sizeof(magmaFloatComplex), cudaMemcpyDefault);
           }

           i += AvailRows;
           AvailRows = __min(AvailRows, (vlen-i));
         }

         if(prec == 8) magma_free(dtm);

       }
       else
       {
         //magmablas_zgemm(cN, cN, vlen, l, m, MAGMA_Z_ONE, (magmaDoubleComplex*)dV, vld, (magmaDoubleComplex*)dTm, ldm, MAGMA_Z_ZERO, (magmaDoubleComplex*)Tmp, vld);

         int i = 0;
         while (i < vlen) 
         {
           magmaDoubleComplex *ptrV = &(((magmaDoubleComplex*)dV)[i]);
           magmablas_zgemm(cN, cN, AvailRows, l, m, MAGMA_Z_ONE, ptrV, vld, (magmaDoubleComplex*)dTm, ldm, MAGMA_Z_ZERO, (magmaDoubleComplex*)rwork, AvailRows);

           for (int k = 0; k < l; k++) {

             magmaDoubleComplex *ptrV = &(((magmaDoubleComplex*)dV)[i + vld * k]);
             cudaMemcpy(ptrV, &((magmaDoubleComplex*)rwork)[AvailRows*k], AvailRows*sizeof(magmaDoubleComplex), cudaMemcpyDefault);
           }

           i += AvailRows;
           AvailRows = __min(AvailRows, (vlen-i));
         }

       }

       //cudaMemcpy(dV, Tmp, vld*l*complex_prec, cudaMemcpyDefault); 

       //magma_free(Tmp);
       magma_free(rwork);
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
(void *spinorOut, const void *spinorSetIn, const int sld, const int slen, const void *vec, const int vlen)
{
#ifdef MAGMA_LIB
       if (prec == 4)
       {
           magmaFloatComplex *spmat = (magmaFloatComplex*)spinorSetIn; 
           magmaFloatComplex *spout = (magmaFloatComplex*)spinorOut; 

           magmablas_cgemv(cN, slen, vlen, MAGMA_C_ONE, spmat, sld, (magmaFloatComplex*)vec, 1, MAGMA_C_ZERO, spout, 1);//in colour-major format
       }
       else
       {
           magmaDoubleComplex *spmat = (magmaDoubleComplex*)spinorSetIn; 
           magmaDoubleComplex *spout = (magmaDoubleComplex*)spinorOut; 

           magmablas_zgemv(cN, slen, vlen, MAGMA_Z_ONE, spmat, sld, (magmaDoubleComplex*)vec, 1, MAGMA_Z_ZERO, spout, 1);//in colour-major format
       }
#endif
       return;
}


