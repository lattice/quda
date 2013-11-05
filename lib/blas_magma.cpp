#include "blas_magma.h"
#include "magma.h"

void init_magma(blasMagmaArgs *param, const int m, const int nev)
{
    magma_init();

    param->m   = m;
    param->nev = nev;

    param->info = -1;

    //magma params/objects:
    param->ldTm  = m;//hTm (host/device)ld (may include padding)

    param->nb    = magma_get_chetrd_nb(m);

    param->llwork = MAX(m + m*param->nb, 2*m + m*m); 
    param->lrwork = 1 + 5*m + 2*m*m;
    param->liwork = 3 + 5*m;

    param->htsize   = 2*nev;//MIN(l,k)-number of Householder vectors, but we always have k <= MIN(m,n)
    param->dtsize   = ( 4*nev + ((2*nev + 31)/32)*32 )*param->nb;//in general: MIN(m,k) for side = 'L' and MIN(n,k) for side = 'R'

    param->sideLR = (m - 2*nev + param->nb)*(m + param->nb) + m*param->nb;
    param->lwork_max = param->sideLR; 

    magma_malloc_pinned((void**)&(param->W), param->lwork_max*sizeof(magmaFloatComplex));
    magma_malloc_pinned((void**)&(param->hTau), param->htsize*sizeof(magmaFloatComplex));//fixed!
    magma_malloc((void**)&(param->dTau), param->dtsize*sizeof(magmaFloatComplex));

    magma_malloc_pinned((void**)&(param->lwork), param->llwork*sizeof(magmaFloatComplex));
    magma_malloc_cpu((void**)&(param->rwork), param->lrwork*sizeof(double));
    magma_malloc_cpu((void**)&(param->iwork), param->liwork*sizeof(magma_int_t));
    
    param->init = true;

    return;
}

void shutdown_magma(blasMagmaArgs *param)
{
   if(!param->init) printf("\n\nError: Magma was not initialized..\n"), exit(-1);
   magma_free(param->dTau);
   magma_free_pinned(param->hTau);

   magma_free_pinned(param->W);
   magma_free_pinned(param->lwork);

   magma_free_cpu(param->rwork);
   magma_free_cpu(param->iwork);

   magma_finalize();
   param->init = false;

   return;
}
int runRayleighRitz(cuFloatComplex *dTm, 
                    cuFloatComplex *dTvecm0,  
                    cuFloatComplex *dTvecm1, 
                    std::complex<float> *hTvecm, 
                    float *hTvalm, 
                    const blasMagmaArgs *param, const int i)
{
     int l = i;
     //solve m-dim eigenproblem:
     magma_cheevd_gpu('V', 'U', param->m, 
                      (magmaFloatComplex*)dTvecm0, param->ldTm, 
                       hTvalm, (magmaFloatComplex*)hTvecm, param->ldTm, 
                       (magmaFloatComplex *)param->lwork, param->llwork, param->rwork, param->lrwork, (magma_int_t*)param->iwork, param->liwork, (magma_int_t*)&param->info);
     if(param->info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

     //solve (m-1)-dim eigenproblem:
      cudaMemcpy(dTvecm1, dTm, param->ldTm*param->m*sizeof(cuFloatComplex), cudaMemcpyDefault);
      magma_cheevd_gpu('V', 'U', (param->m-1), 
                       (magmaFloatComplex*)dTvecm1, param->ldTm, 
                        hTvalm, (magmaFloatComplex*)hTvecm, param->ldTm, 
                        (magmaFloatComplex *)param->lwork, param->llwork, param->rwork, param->lrwork, param->iwork, param->liwork, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);
      //add last row with zeros (coloumn-major format of the matrix re-interpreted as 2D row-major formated):
      cudaMemset2D(&dTvecm1[(param->m-1)], param->ldTm*sizeof(cuFloatComplex), 0, sizeof(cuFloatComplex),  param->m-1);

      //attach nev old vectors to nev new vectors (note 2*nev < m):
      cudaMemcpy(&dTvecm0[param->nev*param->m], dTvecm1, param->nev*param->m*sizeof(cuFloatComplex), cudaMemcpyDefault);

      //Orthogonalize 2*nev vectors:
      l = 2 * param->nev;
      magma_cgeqrf_gpu(param->m, l, dTvecm0, param->ldTm, param->hTau, param->dTau, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_cgeqrf_gpu, exit ...\n"), exit(-1);

      //compute dTevecm0=QHTmQ
      //get TQ product:
      magma_cunmqr_gpu( 'R', 'N', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info); 
      if(param->info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);
             	
      //get QHT product:
      magma_cunmqr_gpu( 'L', 'C', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1);                 	

      //solve l=2*nev-dim eigenproblem:
//dTm
      magma_cheevd_gpu('V', 'U', l, 
                      (magmaFloatComplex*)dTm, param->ldTm, 
                       hTvalm, (magmaFloatComplex*)hTvecm, param->ldTm, 
                       param->lwork, param->llwork, param->rwork, param->lrwork, param->iwork, param->liwork, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_cheevd_gpu, exit ...\n"), exit(-1);

      //solve zero unused part of the eigenvectors in dTm (to complement each coloumn...):
      cudaMemset2D(&dTm[l], param->ldTm*sizeof(cuFloatComplex), 0, (param->m-l)*sizeof(cuFloatComplex),  l);//check..
        
      //Compute dTm=dTevecm0*dTm (Q * Z):
      //(compute QT product):
      magma_cunmqr_gpu('L', 'N', param->m, param->m, l, dTvecm0, param->ldTm, param->hTau, dTm, param->ldTm, param->W, param->sideLR, param->dTau, param->nb, (magma_int_t*)&param->info);
      if(param->info != 0) printf("\nError in magma_cunmqr_gpu, exit ...\n"), exit(-1); 

      return l;
}


void restart_2nev_vectors(cuFloatComplex *dVm, cuFloatComplex *dQ, const blasMagmaArgs *param, const int len)
{
       int _2nev = 2*param->nev;
 
       magma_int_t ldV       = (magma_int_t)len;
       magma_int_t ldQ       = param->m;//not vsize (= 2*nev) 
       
       magmaFloatComplex *V = (magmaFloatComplex*)dVm; 
       magmaFloatComplex *Tmp;
       magma_malloc((void**)&Tmp, ldV*param->m*sizeof(magmaFloatComplex)); 

       cudaMemset(Tmp, 0, ldV*param->m*sizeof(magmaFloatComplex)); 
       magmablas_cgemm('N', 'N', ldV, _2nev, param->m, MAGMA_C_ONE, V, ldV, dQ, ldQ, MAGMA_C_ZERO, Tmp, ldV);//in colour-major format
       cudaMemcpy(V, Tmp, ldV*(_2nev)*sizeof(magmaFloatComplex), cudaMemcpyDefault); 

       magma_free(Tmp);

       return;
}

