#include <blas_magma.h>
#include <string.h>

#include <vector>
#include <algorithm>

#include <util_quda.h>

#ifndef MAX
#define MAX(a, b) (a > b) ? a : b;
#endif

#define MAGMA_17 //default version version of the MAGMA library

#ifdef MAGMA_LIB
#include <magma.h>

#ifdef MAGMA_14

#define _cV 'V' 
#define _cU 'U'

#define _cR 'R'
#define _cL 'L'

#define _cC 'C'
#define _cN 'N'

#define _cNV 'N'

#else

#define _cV MagmaVec 
#define _cU MagmaUpper

#define _cR MagmaRight
#define _cL MagmaLeft

#define _cC MagmaConjTrans
#define _cN MagmaNoTrans

#define _cNV MagmaNoVec

#endif

#endif

//Column major format: Big matrix times Little matrix.

#ifdef MAGMA_LIB

//Simplified version for the above:
#define BLOCK_SIZE 16

__global__ void SMatCMatCuda_16x16(cuFloatComplex *outBuff, const int bldm, cuFloatComplex *sMat, const int sldm, cuDoubleComplex *cMat, const int cldm, const int scols)
{
    //block coords:
    int by = blockIdx.x;
    int bx = blockIdx.y;

    //local coords:
    int ty = threadIdx.x;
    int tx = threadIdx.y;

    int sBegin = BLOCK_SIZE * by;//global offset in Y-direction for the Big matrix

    int sEnd   = sBegin + sldm*scols - 1;//loop limit in X-direction for the Big matrix

    int sStep  = sldm * BLOCK_SIZE;//step in X-direction for the Big matrix

    int cBegin = cldm * BLOCK_SIZE * bx;//global offset in X-direction for the Little matrix

    int cStep  = BLOCK_SIZE;//step in Y-direction for the Little matrix

    cuDoubleComplex accum = make_cuDoubleComplex (0.0, 0.0);

    cuFloatComplex  ftmp;
    cuDoubleComplex dtmp;


    for (int s = sBegin, c = cBegin; s <= sEnd; s += sStep, c += cStep)
    {

        __shared__ float reSmat[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float imSmat[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ double reCmat[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double imCmat[BLOCK_SIZE][BLOCK_SIZE];


        ftmp = sMat[s + sldm * tx + ty];
        reSmat[ty][tx] = cuCrealf(ftmp);
        imSmat[ty][tx] = cuCimagf(ftmp);

        dtmp = cMat[c + cldm * tx + ty];
        reCmat[ty][tx] = cuCreal(dtmp);
        imCmat[ty][tx] = cuCimag(dtmp);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            ftmp = make_cuFloatComplex(reSmat[ty][k], imSmat[ty][k]);

            dtmp = make_cuDoubleComplex(reCmat[k][tx], imCmat[k][tx]);

            cuDoubleComplex dtmp2 = cuComplexFloatToDouble( ftmp );

            accum = cuCfma(dtmp2, dtmp, accum);
        }

        __syncthreads();
    }

    int idx = BLOCK_SIZE * by + bldm * BLOCK_SIZE * bx;

    outBuff[idx + bldm * tx + ty] = cuComplexDoubleToFloat( accum );

    return;
}
#endif

void sMM_v2(void *outBuff, const int bldm,  void *sMat, const int srows, const int scols, const int sldm, void *cMat, const int crows, const int ccols, const int cldm)
{
#ifdef MAGMA_LIB
    // for test only:
    if(scols != crows) errorQuda("\nError: wrong dimensions\n");

    const int block_size = 16;

    if (ccols % block_size != 0) errorQuda("\nError: wrong dimensions\n");

    // Setup execution parameters (column-major format):
    dim3 threads(block_size, block_size);
    dim3 grid((srows+15) / threads.x, ccols / threads.y);//both ccols and srows must be multiple of block_size...

    cudaFuncSetCacheConfig( SMatCMatCuda_16x16, cudaFuncCachePreferShared );

    SMatCMatCuda_16x16<<< grid, threads >>>((cuFloatComplex*)outBuff, bldm, (cuFloatComplex*)sMat, sldm, (cuDoubleComplex*)cMat, cldm, scols);
#endif
}

#undef BLOCK_SIZE


void BlasMagmaArgs::OpenMagma(){ 

#ifdef MAGMA_LIB
    magma_int_t err = magma_init(); 

    if(err != MAGMA_SUCCESS) errorQuda("\nError: cannot initialize MAGMA library\n");

    int major, minor, micro;

    magma_version( &major, &minor, &micro);
    printfQuda("\nMAGMA library version: %d.%d\n\n", major,  minor);
#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif    

    return;
}

void BlasMagmaArgs::CloseMagma(){  

#ifdef MAGMA_LIB
    if(magma_finalize() != MAGMA_SUCCESS) errorQuda("\nError: cannot close MAGMA library\n");
#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif    

    return;
}

 BlasMagmaArgs::BlasMagmaArgs(const int prec) : m(0), max_nev(0), prec(prec), ldm(0), info(-1), llwork(0), 
  lrwork(0), liwork(0), sideLR(0), htsize(0), dtsize(0), lwork_max(0), W(0), W2(0), 
  hTau(0), dTau(0), lwork(0), rwork(0), iwork(0)
{

#ifdef MAGMA_LIB
    magma_int_t dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...
    if(dev_info == 0)  exit(-1);

    printfQuda("\nMAGMA will use device architecture %d.\n", dev_info);

    alloc = false;
    init  = true;
#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif    

    return;
}


BlasMagmaArgs::BlasMagmaArgs(const int m, const int ldm, const int prec) 
  : m(m), max_nev(0),  prec(prec), ldm(ldm), info(-1), sideLR(0), htsize(0), dtsize(0), 
  W(0), hTau(0), dTau(0)
{

#ifdef MAGMA_LIB

    magma_int_t dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...

    if(dev_info == 0)  exit(-1);

    printfQuda("\nMAGMA will use device architecture %d.\n", dev_info);

    const int complex_prec = 2*prec;

    magma_int_t nbtrd = prec == 4 ? magma_get_chetrd_nb(m) : magma_get_zhetrd_nb(m);//ldm

    llwork = MAX(m + m*nbtrd, 2*m + m*m);//ldm 
    lrwork = 1 + 5*m + 2*m*m;//ldm
    liwork = 3 + 5*m;//ldm

    magma_malloc_pinned((void**)&W2,   ldm*m*complex_prec);
    magma_malloc_pinned((void**)&lwork, llwork*complex_prec);
    magma_malloc_cpu((void**)&rwork,    lrwork*prec);
    magma_malloc_cpu((void**)&iwork,    liwork*sizeof(magma_int_t));

    init  = true;
    alloc = true;

#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif    

    return;
}



BlasMagmaArgs::BlasMagmaArgs(const int m, const int max_nev, const int ldm, const int prec) 
  : m(m), max_nev(max_nev),  prec(prec), ldm(ldm), info(-1)
{

#ifdef MAGMA_LIB

    magma_int_t dev_info = magma_getdevice_arch();//mostly to check whether magma is intialized...

    if(dev_info == 0)  exit(-1);

    printfQuda("\nMAGMA will use device architecture %d.\n", dev_info);

    const int complex_prec = 2*prec;

    magma_int_t nbtrd = prec == 4 ? magma_get_chetrd_nb(ldm) : magma_get_zhetrd_nb(ldm);//ldm<-m
    magma_int_t nbqrf = prec == 4 ? magma_get_cgeqrf_nb(ldm) : magma_get_zgeqrf_nb(ldm);//ldm

    htsize   = max_nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    dtsize   = ( 2*htsize + ((htsize + 31)/32)*32 )*nbqrf;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    magma_malloc_pinned((void**)&hTau, htsize*complex_prec);
    magma_malloc((void**)&dTau,        dtsize*complex_prec);

//these are needed for the eigCG solver only.
    sideLR = (m - max_nev + nbqrf)*(m + nbqrf) + m*nbqrf;//ldm

    magma_malloc_pinned((void**)&W,    sideLR*complex_prec);
    magma_malloc_pinned((void**)&W2,   ldm*m*complex_prec);

    llwork = MAX(m + m*nbtrd, 2*m + m*m);//ldm 
    lrwork = 1 + 5*m + 2*m*m;//ldm
    liwork = 3 + 5*m;//ldm

    magma_malloc_pinned((void**)&lwork, llwork*complex_prec);
    magma_malloc_cpu((void**)&rwork,    lrwork*prec);
    magma_malloc_cpu((void**)&iwork,    liwork*sizeof(magma_int_t));

    init  = true;
    alloc = true;

#else
    errorQuda("\nError: MAGMA library was not compiled, check your compilation options...\n");
#endif    

    return;
}

BlasMagmaArgs::~BlasMagmaArgs()
{
#ifdef MAGMA_LIB

   if(alloc == true)
   {
     if(dTau) magma_free(dTau);
     if(hTau) magma_free_pinned(hTau);

     if(W) magma_free_pinned(W);
     if(W2) magma_free_pinned(W2);
     if(lwork) magma_free_pinned(lwork);

     if(rwork) magma_free_cpu(rwork);
     if(iwork) magma_free_cpu(iwork);

     alloc = false;
   }

   init  = false;

#endif

   return;
}

void BlasMagmaArgs::MagmaHEEVD(void *dTvecm, void *hTvalm, const int prob_size, bool host)
{
#ifdef MAGMA_LIB
     if(prob_size > m) errorQuda("\nError in MagmaHEEVD (problem size cannot exceed given search space %d), exit ...\n", m);

     cudaPointerAttributes ptr_attr;

     if(!host)
     {
       //check if dTvecm is a device pointer..
       cudaPointerGetAttributes(&ptr_attr, dTvecm);

       if(ptr_attr.memoryType != cudaMemoryTypeDevice || ptr_attr.devicePointer == NULL ) errorQuda("Error in MagmaHEEVD, no device pointer found.");

       if(prec == 4)
       {
         magma_cheevd_gpu(_cV, _cU, prob_size, (magmaFloatComplex*)dTvecm, ldm, (float*)hTvalm, (magmaFloatComplex*)W2, ldm, (magmaFloatComplex*)lwork, llwork, (float*)rwork, lrwork, iwork, liwork, &info);
         if(info != 0) errorQuda("\nError in MagmaHEEVD (magma_cheevd_gpu), exit ...\n");
       }
       else
       {
         magma_zheevd_gpu(_cV, _cU, prob_size, (magmaDoubleComplex*)dTvecm, ldm, (double*)hTvalm, (magmaDoubleComplex*)W2, ldm, (magmaDoubleComplex*)lwork, llwork, (double*)rwork, lrwork, iwork, liwork, &info);
         if(info != 0) errorQuda("\nError in MagmaHEEVD (magma_zheevd_gpu), exit ...\n");
       }
     }
     else
     {
       //check if dTvecm is a device pointer..
       cudaPointerGetAttributes(&ptr_attr, dTvecm);

       if(ptr_attr.memoryType != cudaMemoryTypeHost || ptr_attr.hostPointer == NULL ) errorQuda("Error in MagmaHEEVD, no host pointer found.");

       if(prec == 4)
       {
         magma_cheevd(_cV, _cU, prob_size, (magmaFloatComplex*)dTvecm, ldm, (float*)hTvalm, (magmaFloatComplex*)lwork, llwork, (float*)rwork, lrwork, iwork, liwork, &info);
         if(info != 0) errorQuda("\nError in MagmaHEEVD (magma_cheevd_gpu), exit ...\n");
       }
       else
       {
         magma_zheevd(_cV, _cU, prob_size, (magmaDoubleComplex*)dTvecm, ldm, (double*)hTvalm, (magmaDoubleComplex*)lwork, llwork, (double*)rwork, lrwork, iwork, liwork, &info);
         if(info != 0) errorQuda("\nError in MagmaHEEVD (magma_zheevd_gpu), exit ...\n");
       }
     }   
#endif
  return;
}  

int BlasMagmaArgs::MagmaORTH_2nev(void *dTvecm, void *dTm)
{
     const int l = max_nev;

#ifdef MAGMA_LIB
     if(prec == 4)
     {
        magma_int_t nb = magma_get_cgeqrf_nb(m);//ldm

        magma_cgeqrf_gpu(m, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTau, &info);
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_cgeqrf_gpu), exit ...\n");

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_cunmqr_gpu(_cR, _cN, m, m, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, &info); 
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_cunmqr_gpu), exit ...\n");
             	
        //get QHT product:
        magma_cunmqr_gpu(_cL, _cC, m, l, l, (magmaFloatComplex *)dTvecm, ldm, (magmaFloatComplex *)hTau, (magmaFloatComplex *)dTm, ldm, (magmaFloatComplex *)W, sideLR, (magmaFloatComplex *)dTau, nb, &info);
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_cunmqr_gpu), exit ...\n");  
     }
     else
     {
        magma_int_t nb = magma_get_zgeqrf_nb(m);//ldm

        magma_zgeqrf_gpu(m, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTau, &info);
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_zgeqrf_gpu), exit ...\n");

        //compute dTevecm0=QHTmQ
        //get TQ product:
        magma_zunmqr_gpu(_cR, _cN, m, m, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, &info); 
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n");
             	
        //get QHT product:
        magma_zunmqr_gpu(_cL, _cC, m, l, l, (magmaDoubleComplex *)dTvecm, ldm, (magmaDoubleComplex *)hTau, (magmaDoubleComplex *)dTm, ldm, (magmaDoubleComplex *)W, sideLR, (magmaDoubleComplex *)dTau, nb, &info);
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n");  

     }
#endif

  return l;
}

void BlasMagmaArgs::RestartV(void *dV, const int vld, const int vlen, const int vprec, void *dTevecm, void *dTm)
{
#ifdef MAGMA_LIB 
       if( (vld % 32) != 0) errorQuda("\nError: leading dimension must be multiple of the warp size\n");

       const int cvprec = 2*vprec;

       const int l     = max_nev;

       //int bufferSize =  2*vld+l*l;
       //int bufferBlock = bufferSize / l;

       int bufferBlock = (2*vld) / l;
       bufferBlock     = (bufferBlock / 32) * 32;//corrected bufferBlock to be multiple of the warp size
       int bufferSize  = (bufferBlock * l);

       void  *buffer = 0;
       magma_malloc(&buffer, bufferSize*cvprec);
       cudaMemset(buffer, 0, bufferSize*cvprec);


       if(prec == 4)
       {
         magma_int_t nb = magma_get_cgeqrf_nb(m);//ldm
         magma_cunmqr_gpu(_cL, _cN, m, l, l, (magmaFloatComplex*)dTevecm, ldm, (magmaFloatComplex*)hTau, (magmaFloatComplex*)dTm, ldm, (magmaFloatComplex*)W, sideLR, (magmaFloatComplex*)dTau, nb, &info);
        
         if(info != 0) errorQuda("\nError in RestartV (magma_cunmqr_gpu), exit ...\n"); 
       }
       else
       {
         magma_int_t nb = magma_get_zgeqrf_nb(m);//ldm
         magma_zunmqr_gpu(_cL, _cN, m, l, l, (magmaDoubleComplex*)dTevecm, ldm, (magmaDoubleComplex*)hTau, (magmaDoubleComplex*)dTm, ldm, (magmaDoubleComplex*)W, sideLR, (magmaDoubleComplex*)dTau, nb, &info);

         if(info != 0) errorQuda("\nError in RestartV (magma_zunmqr_gpu), exit ...\n"); 
       }

       if(vprec == 4)
       {
         if(prec == vprec) errorQuda("\nError: option is not currently supported, exit ...\n");

         for (int blockOffset = 0; blockOffset < vlen; blockOffset += bufferBlock) 
         {
           if (bufferBlock > (vlen-blockOffset)) bufferBlock = (vlen-blockOffset);

           magmaFloatComplex *ptrV = &(((magmaFloatComplex*)dV)[blockOffset]);

           sMM_v2(buffer, bufferBlock, ptrV, bufferBlock, m, vld, dTm, m, l, ldm);

           cudaMemcpy2D(ptrV, vld*cvprec, buffer, bufferBlock*cvprec,  bufferBlock*cvprec, l, cudaMemcpyDefault);
         }
       }
       else
       {
         for (int blockOffset = 0; blockOffset < vlen; blockOffset += bufferBlock) 
         {
           if (bufferBlock > (vlen-blockOffset)) bufferBlock = (vlen-blockOffset);

           magmaDoubleComplex *ptrV = &(((magmaDoubleComplex*)dV)[blockOffset]);

           magmablas_zgemm(_cN, _cN, bufferBlock, l, m, MAGMA_Z_ONE, ptrV, vld, (magmaDoubleComplex*)dTm, ldm, MAGMA_Z_ZERO, (magmaDoubleComplex*)buffer, bufferBlock);

           cudaMemcpy2D(ptrV, vld*cvprec, buffer, bufferBlock*cvprec,  bufferBlock*cvprec, l, cudaMemcpyDefault);
	 }
       }

       magma_free(buffer);
#endif
       return;
}


void BlasMagmaArgs::SolveProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
#ifdef MAGMA_LIB
       const int complex_prec = 2*prec;
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc_pinned((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       memcpy(tmp, H, ldH*n*complex_prec);

       if (prec == 4)
       {
          err = magma_cgesv(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
          if(err != 0) errorQuda("\nError in SolveProjMatrix (magma_cgesv), exit ...\n");
       }
       else
       {
          err = magma_zgesv(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) errorQuda("\nError in SolveProjMatrix (magma_zgesv), exit ...\n");
       }

       magma_free_pinned(tmp);
       magma_free_pinned(ipiv);
#endif
       return;
}

void BlasMagmaArgs::SolveGPUProjMatrix(void* rhs, const int ldn, const int n, void* H, const int ldH)
{
#ifdef MAGMA_LIB
       const int complex_prec = 2*prec;
       void *tmp; 
       magma_int_t *ipiv;
       magma_int_t err;

       magma_malloc((void**)&tmp, ldH*n*complex_prec);
       magma_malloc_pinned((void**)&ipiv, n*sizeof(magma_int_t));

       cudaMemcpy(tmp, H, ldH*n*complex_prec, cudaMemcpyDefault);

       if (prec == 4)
       {
          err = magma_cgesv_gpu(n, 1, (magmaFloatComplex*)tmp, ldH, ipiv, (magmaFloatComplex*)rhs, ldn, &info);
          if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_cgesv), exit ...\n");
       }
       else
       {
          err = magma_zgesv_gpu(n, 1, (magmaDoubleComplex*)tmp, ldH, ipiv, (magmaDoubleComplex*)rhs, ldn, &info);
          if(err != 0) errorQuda("\nError in SolveGPUProjMatrix (magma_zgesv), exit ...\n");
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

           magmablas_cgemv(_cN, slen, vlen, MAGMA_C_ONE, spmat, sld, (magmaFloatComplex*)vec, 1, MAGMA_C_ZERO, spout, 1);//in colour-major format
       }
       else
       {
           magmaDoubleComplex *spmat = (magmaDoubleComplex*)spinorSetIn; 
           magmaDoubleComplex *spout = (magmaDoubleComplex*)spinorOut; 

           magmablas_zgemv(_cN, slen, vlen, MAGMA_Z_ONE, spmat, sld, (magmaDoubleComplex*)vec, 1, MAGMA_Z_ZERO, spout, 1);//in colour-major format
       }
#endif
       return;
}

void BlasMagmaArgs::MagmaRightNotrUNMQR(const int clen, const int qrlen, const int nrefls, void *QR, const int ldqr, void *Vm, const int cldn)
{
#ifdef MAGMA_LIB
     magma_int_t m = clen; 
     magma_int_t n = qrlen; 
     magma_int_t k = nrefls;

     magma_int_t lwork = -1;

     if(prec == 4)
     {

     }
     else
     {
        magmaDoubleComplex *dQR  = NULL;

        magmaDoubleComplex *dtau = NULL;

        magmaDoubleComplex *htau = NULL;

        magmaDoubleComplex *hW   = NULL;

        magmaDoubleComplex qW;

        magma_malloc_pinned((void**)&dQR, ldqr*k*sizeof(magmaDoubleComplex));

        magma_malloc_pinned((void**)&htau, k*sizeof(magmaDoubleComplex));
        //
        magma_malloc((void**)&dTau,  k*sizeof(magmaDoubleComplex));

        cudaMemcpy(dQR, QR, ldqr*k*sizeof(magmaDoubleComplex), cudaMemcpyDefault);

        magma_int_t nb = magma_get_zgeqrf_nb(m);//ldm
        //
        magma_zgeqrf_gpu(n, k, (magmaDoubleComplex *)dQR, ldqr, (magmaDoubleComplex *)htau, (magmaDoubleComplex *)dtau, &info);//identical to zgeqrf?

        magma_zunmqr_gpu(_cR, _cN, m, n, k, dQR, ldqr, htau, (magmaDoubleComplex *)Vm, cldn, &qW, lwork, dtau, nb, &info); 
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n");

        lwork = (magma_int_t) MAGMA_Z_REAL(qW);

        magma_malloc_cpu((void**)&hW, lwork*sizeof(magmaDoubleComplex));

        //get TQ product:
        magma_zunmqr_gpu(_cR, _cN, m, n, k, dQR, ldqr, htau, (magmaDoubleComplex *)Vm, cldn, hW, lwork, dtau, nb, &info); 
        if(info != 0) errorQuda("\nError in MagmaORTH_2nev (magma_zunmqr_gpu), exit ...\n");

        magma_free_cpu(hW);

        magma_free(dtau);

        magma_free_pinned(htau);

        magma_free_pinned(dQR);

     }
#endif

  return;
}

//STL based version:
//

struct SortedEval{

   double eval_nrm;
   int    eval_idx;

   SortedEval(double val, int idx) : eval_nrm(val), eval_idx(idx) {}; 
};

bool cmp_eigen_nrms (SortedEval v1, SortedEval v2)
{
  return (v1.eval_nrm < v2.eval_nrm);
}

void BlasMagmaArgs::Sort(const int m, const int ldm, void *eVecs, const int nev, void *unsorted_eVecs, void *eVals)
{
  if (prec == 4) errorQuda("\nSingle precision is currently not supported.\n");

  std::vector<SortedEval> sorted_evals_cntr;

  for(int e = 0; e < m; e++) sorted_evals_cntr.push_back( SortedEval( abs(((std::complex<double>*)eVals)[e]), e ));

  std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), cmp_eigen_nrms);
  

  for(int e = 0; e < nev; e++)
  {  
    memcpy(&(((std::complex<double>*)eVecs)[ldm*e]), &(((std::complex<double>*)unsorted_eVecs)[ldm*( sorted_evals_cntr[e].eval_idx)]), (ldm)*sizeof(std::complex<double>));
    //set zero in m+1 element:
    ((std::complex<double>*)eVecs)[ldm*e+m] = std::complex<double>(0.0, 0.0);
  }

  return;
}


///NEW STUFF:

void BlasMagmaArgs::ComputeQR(const int nev, Complex * evmat, const int m, const int ldm, Complex  *tau)
{
#ifdef MAGMA_LIB
  magma_int_t _m   = m;//matrix size

  magma_int_t _nev = nev;//matrix size
 
  magma_int_t _ldm = ldm;

  //Lapack parameters:   
  magma_int_t info  = 0;

  magma_int_t lwork = -1; 

  magmaDoubleComplex *work = NULL;

  magmaDoubleComplex qwork; //parameter to extract optimal size of work

  magma_zgeqrf(_m, _nev, (magmaDoubleComplex *)evmat, _ldm, (magmaDoubleComplex *)tau, &qwork, lwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: MAGMA_ZGEQRF, info %d\n",info);

  lwork = (magma_int_t) MAGMA_Z_REAL(qwork);

  magma_malloc_cpu((void**)&work, lwork*sizeof(magmaDoubleComplex));

  magma_zgeqrf(_m, _nev, (magmaDoubleComplex *)evmat, _ldm, (magmaDoubleComplex *)tau, work, lwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: ZGEQRF, info %d\n",info);

  if(work) magma_free_cpu(work);
#endif
  return;
}


void BlasMagmaArgs::LeftConjZUNMQR(const int k /*number of reflectors*/, const int n /*number of columns of H*/, Complex *H, const int dh /*number of rows*/, 
const int ldh, Complex * QR,  const int ldqr, Complex *tau)//for vectors: n =1
{
#ifdef MAGMA_LIB
//Note: # rows of QR = # rows of H.
  magma_int_t _h   = dh;//matrix size

  magma_int_t _n   = n;//vector size

  magma_int_t _k   = k;
 
  magma_int_t _ldh = ldh;

  magma_int_t _ldqr = ldqr;

  //Lapack parameters:   
  magma_side_t  _s = _cL;//apply QR-matrix from the left

  magma_trans_t _t = _cC;//conjugate 

  magma_int_t info  = 0;

  magma_int_t lwork = -1; 

  magmaDoubleComplex *work = NULL;

  magmaDoubleComplex qwork; //parameter to extract optimal size of work

  //Pdagger_{k+1} PrevRes

  magma_zunmqr(_s, _t, _h, _n, _k, (magmaDoubleComplex *)QR, _ldqr, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldh, &qwork, lwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

  lwork = (magma_int_t) MAGMA_Z_REAL(qwork);

  magma_malloc_cpu((void**)&work, lwork*sizeof(magmaDoubleComplex));

  magma_zunmqr(_s, _t, _h, _n, _k, (magmaDoubleComplex *)QR, _ldqr, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldh, work, lwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

  if(work) magma_free_cpu(work);
#endif
  return;
}


void BlasMagmaArgs::Construct_harmonic_matrix(Complex * const harmH, Complex * const conjH, const double beta2, const int m, const int ldH)
{
#ifdef MAGMA_LIB
  //Lapack parameters:
  magma_int_t _m    = m;
  //
  magma_int_t _ldH  = ldH;
  //
  magma_int_t info  = 0;
  //
  magma_int_t I_ONE = 1;
  //
  magma_int_t *ipiv;
  magma_malloc_cpu((void**)&ipiv, ldH*sizeof(magma_int_t));  
  //
  //Construct H + beta*H^{-H} e_m*e_m^{T}
  // 1. need to solve H^{H}y = e_m;
  Complex *em = new Complex[m];
  
  em[m-1] = beta2;//in fact, we construct beta*em,

  magma_zgesv(_m, I_ONE, (magmaDoubleComplex *)conjH, _ldH, ipiv, (magmaDoubleComplex *)em, _ldH, &info);

  if( (info != 0 ) ) errorQuda( "Error: DGESV, info %d\n",info);

//make this cleaner!
  //check solution:
  for (int j = 0; j < m; j++)
  {
    Complex accum = 0.0;

    for (int i = 0; i < m; i++) accum = (accum + harmH[ldH*j+i]*em[(ipiv[i])-1]);
  } 

  // 2. Construct matrix for harmonic Ritz vectors:
  //    Adjust last column with KroneckerProd((H^{-H}*beta*em)=em, em^{T}=[0,....,1]):

  for(int i = 0; i < m; i++) harmH[ldH*(m-1)+i] += em[i]; 

  magma_free_cpu(ipiv);
  //
  delete [] em; 
#endif

  return;
}

void BlasMagmaArgs::Compute_harmonic_matrix_eigenpairs(Complex *harmH, const int m, const int ldH, Complex *vr, Complex *evalues, const int ldv) 
{
#ifdef MAGMA_LIB
  magma_int_t _m   = m;//matrix size
 
  magma_int_t _ldH = ldH;

  magma_int_t _ldv = ldv;

  //Lapack parameters:   
  magma_int_t info = 0;
  //
  magma_vec_t _r = _cV;

  magma_vec_t _l = _cNV;//no left eigenvectors

  magma_int_t lwork = -1;
 
  magmaDoubleComplex *work = NULL;

  magmaDoubleComplex qwork; //parameter to extract optimal size of work
  
  double *rwork = NULL;
  magma_malloc_cpu((void**)&rwork, 2*_m*sizeof(double)); 

  //Get optimal work:
  magma_zgeev(_l, _r, _m, (magmaDoubleComplex *)harmH, _ldH, (magmaDoubleComplex *)evalues, NULL, _ldv, (magmaDoubleComplex *)vr, _ldv, &qwork, lwork, rwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: ZGEEVX, info %d\n",info);

  lwork = (magma_int_t) MAGMA_Z_REAL(qwork);

  magma_malloc_cpu((void**)&work, lwork*sizeof(magmaDoubleComplex));

  //now get eigenpairs:
  magma_zgeev(_l, _r, _m, (magmaDoubleComplex *)harmH, _ldH, (magmaDoubleComplex *)evalues, NULL, _ldv, (magmaDoubleComplex *)vr, _ldv, work, lwork, rwork, &info);

  if( (info != 0 ) ) errorQuda( "Error: ZGEEVX, info %d\n",info);

  if(rwork)  magma_free_cpu(rwork);
  //
  if(work)   magma_free_cpu(work);
  //
#endif
  return;
}


void BlasMagmaArgs::RestartVH(void *dV, const int vlen, const int vld, const int vprec, void *sortedHarVecs, void *H, const int ldh)
{
#ifdef MAGMA_LIB
    if(prec == 4)
    {
       errorQuda("\nError: single precision is not currently supported\n");
    }

    if( (vld % 32) != 0) errorQuda("\nError: leading dimension must be multiple of the warp size\n");

    int nev  = (max_nev - 1); //(nev+1) - 1 for GMRESDR

    int _m   = m;//matrix size

    int _k   = nev;

    int _kp1 = max_nev;

    int _mp1 = (m+1);
 
    int _ldm = ldh;

    magma_side_t  _s = _cR;//apply P-matrix from the right

    magma_trans_t _t = _cN;//no left eigenvectors

    int info  = 0;

    int lwork = -1; 

    Complex  *work = NULL;
    Complex qwork; //parameter to extract optimal size of work

    const int cprec  = 2*prec; //currently: sizeof(Complex)
    const int cvprec = 2*vprec;

    const int l = max_nev;

    int lbsize           = 2*((nev / 16)*16);

    //const int bufferSize = 2*vld+lbsize*lbsize;
    //int bufferBlock = bufferSize / lbsize;//or: lbsize = (nev+1)

    int bufferBlock = (2*vld) / lbsize;
    bufferBlock     = (bufferBlock / 32) * 32;//corrected bufferBlock to be multiple of the warp size
    int bufferSize  = (bufferBlock * lbsize);

    void  *buffer = NULL;
    void  *dQmat  = NULL;

    magma_malloc(&buffer, bufferSize*cvprec);
    cudaMemset(buffer, 0, bufferSize*cvprec);

    magma_malloc(&dQmat, l*ldh*cprec);

    //GPU code:
    Complex *tau  = new Complex[l];//nev+1 =>max_nev

    Complex *Qmat = new Complex[ldh*_mp1];//need (m+1)x(m+1) matrix on input...

    ComputeQR(l, (Complex*)sortedHarVecs, _mp1, ldh, tau);//lapack version 
   
    //max_nev vectors are stored in Qmat (output):
    //restoreOrthVectors(Qmat, max_nev, (Complex*)sortedHarVecs, (m+1), ldh, tau);
    //Load diagonal units
    for(int d = 0; d < (m+1); d++) Qmat[ldh*d+d] = Complex(1.0, 0.0);
   
    magma_zunmqr(_s, _t, _mp1, _mp1, _kp1, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)Qmat, _ldm, (magmaDoubleComplex *)&qwork, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    lwork = (int) qwork.real();
    work = new Complex[lwork];

    magma_zunmqr(_s, _t, _mp1, _mp1, _kp1, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)Qmat, _ldm, (magmaDoubleComplex *)work, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    //Copy (nev+1) vectors on the device:
    cudaMemcpy(dQmat, Qmat, (max_nev)*ldh*cprec, cudaMemcpyDefault);

    if(cvprec == sizeof(magmaDoubleComplex))
    {
      for (int blockOffset = 0; blockOffset < vlen; blockOffset += bufferBlock) 
      {
        if (bufferBlock > (vlen-blockOffset)) bufferBlock = (vlen-blockOffset);

//printfQuda("\nBuffer block : %d\n", bufferBlock);

        magmaDoubleComplex *ptrV = &(((magmaDoubleComplex*)dV)[blockOffset]);

        magmablas_zgemm(_cN, _cN, bufferBlock, l, _mp1, MAGMA_Z_ONE, ptrV, vld, (magmaDoubleComplex*)dQmat, ldh, MAGMA_Z_ZERO, (magmaDoubleComplex*)buffer, bufferBlock);

        cudaMemcpy2D(ptrV, vld*cvprec, buffer, bufferBlock*cvprec,  bufferBlock*cvprec, l, cudaMemcpyDefault);//make this async!
      }

      cudaMemset(&(((magmaDoubleComplex*)dV)[vld*max_nev]), 0, (m+1-max_nev)*vld*sizeof(magmaDoubleComplex));//= m - nev
    }
    else // low precision field
    {
      for (int blockOffset = 0; blockOffset < vlen; blockOffset += bufferBlock) 
      {
        if (bufferBlock > (vlen-blockOffset)) bufferBlock = (vlen-blockOffset);

        magmaFloatComplex *ptrV = &(((magmaFloatComplex*)dV)[blockOffset]);

        sMM_v2(buffer, bufferBlock, ptrV, bufferBlock, _mp1, vld, dQmat, _mp1, l, ldh);

        cudaMemcpy2D(ptrV, vld*cvprec, buffer, bufferBlock*cvprec,  bufferBlock*cvprec, l, cudaMemcpyDefault);
      }

      cudaMemset(&(((magmaFloatComplex*)dV)[vld*max_nev]), 0, (m+1-max_nev)*vld*sizeof(magmaFloatComplex));//= m - nev
    }

    //Construct H_new = Pdagger_{k+1} \bar{H}_{m} P_{k}  

    //bar{H}_{m} P_{k}

    lwork = -1;

    magma_zunmqr(_s, _t, _mp1, _m, _k, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldm, (magmaDoubleComplex *)&qwork, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    delete[] work;
    lwork = (int) qwork.real();
    work = new Complex[lwork];

    magma_zunmqr(_s, _t, _mp1, _m, _k, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldm, (magmaDoubleComplex *)work, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    //Pdagger_{k+1} PrevRes
    lwork = -1;

    _s = _cL;
    _t = _cC;

    magma_zunmqr(_s, _t, _mp1, _k, _kp1, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldm, (magmaDoubleComplex *)&qwork, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    delete [] work;
    lwork = (int) qwork.real();
    work = new Complex[lwork];

    magma_zunmqr(_s, _t, _mp1, _k, _kp1, (magmaDoubleComplex *)sortedHarVecs, _ldm, (magmaDoubleComplex *)tau, (magmaDoubleComplex *)H, _ldm, (magmaDoubleComplex *)work, lwork, &info);

    if( (info != 0 ) ) errorQuda( "Error: ZUNMQR, info %d\n",info);

    const int len = ldh - nev-1;
    for(int i = 0; i < nev; i++) memset(&(((Complex*)H)[ldh*i+nev+1]), 0, len*sizeof(Complex) );

    //
    memset(&(((Complex*)H)[ldh*(nev)]), 0, (m-nev)*ldh*sizeof(Complex));

    delete [] work;

    magma_free(buffer);
    magma_free(dQmat);

    delete [] Qmat;
    delete [] tau ;
#endif
    return; 
}


#ifdef MAGMA_LIB

#undef _cV
#undef _cU
#undef _cR
#undef _cL
#undef _cC
#undef _cN
#undef _cNV

#endif


