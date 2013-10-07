#include "blas_magma.h"
#include "magma.h"

void init_magma(blasMagmaParam *param, const int m, const int nev)
{
    magma_init();

    param->m   = m;
    param->nev = nev;

    param->info = -1;

    //magma params/objects:
    param->ldTm  = m;//hTm (host/device)ld (may include padding)

    param->nb    = magma_get_zhetrd_nb(m);

    param->llwork = MAX(m + m*param->nb, 2*m + m*m); 
    param->lrwork = 1 + 5*m + 2*m*m;
    param->liwork = 3 + 5*m;

    param->htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    param->dtsize   = ( 4*nev + ((2*nev + 31)/32)*32 )*param->nb;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    param->sideLR = (m - 2*nev + param->nb)*(m + param->nb) + m*param->nb;
    param->lwork_max = param->sideLR; 

    magma_malloc_pinned((void**)&(param->W), param->lwork_max*sizeof(magmaDoubleComplex));
    magma_malloc_pinned((void**)&(param->hTau), param->htsize*sizeof(magmaDoubleComplex));
    magma_malloc((void**)&(param->dTau), param->dtsize*sizeof(magmaDoubleComplex));

    magma_malloc_pinned((void**)&(param->lwork), param->llwork*sizeof(magmaDoubleComplex));
    magma_malloc_cpu((void**)&(param->rwork), param->lrwork*sizeof(double));
    magma_malloc_cpu((void**)&(param->iwork), param->liwork*sizeof(magma_int_t));
    
    param->init = true;

    return;
}

void shutdown_magma(blasMagmaParam *param)
{
   if(!param->init) printf("\n\nError: Magma was not initialized..\n"), exit(-1);
   magma_free(param->dTau);
   magma_free_cpu(param->hTau);

   magma_free_pinned(param->W);
   magma_free_pinned(param->lwork);

   magma_free_cpu(param->rwork);
   magma_free_cpu(param->iwork);

   magma_finalize();
   param->init = false;

   return;
}
int runRayleighRitz(cuDoubleComplex *dTm, 
                    cuDoubleComplex *dTvecm0,  
                    cuDoubleComplex *dTvecm1, 
                    std::complex<double> *hTvecm, 
                    double *hTvalm, 
                    const blasMagmaParam *param, const int i)
{
     int l = i;
     //solve m-dim eigenproblem:
     magma_zheevd_gpu('V', 'U', param->m, 
                      (magmaDoubleComplex*)dTvecm0, param->ldTm, 
                       hTvalm, (magmaDoubleComplex*)hTvecm, param->ldTm, 
                       (magmaDoubleComplex *)param->lwork, param->llwork, param->rwork, param->lrwork, (magma_int_t*)param->iwork, param->liwork, (magma_int_t*)&param->info);
     if(param->info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

     //solve (m-1)-dim eigenproblem:
      cudaMemcpy(dTvecm1, dTm, param->ldTm*param->m*sizeof(cuDoubleComplex), cudaMemcpyDefault);
      magma_zheevd_gpu('V', 'U', (param->m-1), 
                       (magmaDoubleComplex*)dTvecm1, param->ldTm, 
                        hTvalm, (magmaDoubleComplex*)hTvecm, param->ldTm, 
                        (magmaDoubleComplex *)param->lwork, param->llwork, param->rwork, param->lrwork, param->iwork, param->liwork, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);
      //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
      cudaMemset2D(&dTvecm1[(param->m-1)], param->ldTm*sizeof(cuDoubleComplex), 0, sizeof(cuDoubleComplex),  param->m-1);

      //attach nev old vectors to nev new vectors (note 2*nev < m):
      cudaMemcpy(&dTvecm0[param->nev*param->m], dTvecm1, param->nev*param->m*sizeof(cuDoubleComplex), cudaMemcpyDefault);

      //Orthogonalize 2*nev vectors:
      l = 2 * param->nev;
      magma_zgeqrf_gpu(param->m, l, dTvecm0, param->ldTm, param->hTau, param->dTau, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_zgeqrf_gpu, exit ...\n"), exit(-1);

      //compute dTevecm0=QHTmQ
      //get TQ product:
      magma_zunmqr_gpu( 'R', 'N', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info); 
      if(param->info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);
             	
      //get QHT product:
      magma_zunmqr_gpu( 'L', 'C', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1);                 	

      //solve l=2*nev-dim eigenproblem:
//dTm
      magma_zheevd_gpu('V', 'U', l, 
                      (magmaDoubleComplex*)dTm, param->ldTm, 
                       hTvalm, (magmaDoubleComplex*)hTvecm, param->ldTm, 
                       param->lwork, param->llwork, param->rwork, param->lrwork, param->iwork, param->liwork, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_zheevd_gpu, exit ...\n"), exit(-1);

      //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
      cudaMemset2D(&dTm[l], param->ldTm*sizeof(cuDoubleComplex), 0, (param->m-l)*sizeof(cuDoubleComplex),  l);//check..
        
      //Compute dTm=dTevecm0*dTm (Q * Z):
      //(compute QT product):
      magma_zunmqr_gpu('L', 'N', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_zunmqr_gpu, exit ...\n"), exit(-1); 

      return l;
}


void restart_2nev_vectors(cuDoubleComplex *dVm, cuDoubleComplex *dQ, const blasMagmaParam *param, const int len)
{
       int _2nev = 2*param->nev;
       magmaDoubleComplex cone     =  MAGMA_Z_MAKE(1.0, 0.0);
       magmaDoubleComplex czero    =  MAGMA_Z_MAKE(0.0, 0.0);
   
       magma_trans_t transV   = 'N';
       magma_trans_t transQ   = 'N';
 
       magma_int_t ldV       = (magma_int_t)len;
       magma_int_t ldQ       = param->m;//not vsize (= 2*nev) 
       
       magmaDoubleComplex *V = (magmaDoubleComplex*)dVm; 
       magmaDoubleComplex *Tmp;
       magma_malloc((void**)&Tmp, ldV*param->m*sizeof(magmaDoubleComplex)); 

       cudaMemset(Tmp, 0, ldV*param->m*sizeof(magmaDoubleComplex)); 
       magmablas_zgemm(transV, transQ, ldV, _2nev, param->m, (magmaDoubleComplex)cone, V, ldV, dQ, ldQ, (magmaDoubleComplex)czero, Tmp, ldV);//in colour-major format
       cudaMemcpy(V, Tmp, ldV*(_2nev)*sizeof(magmaDoubleComplex), cudaMemcpyDefault); 

       magma_free(Tmp);

       return;
}

