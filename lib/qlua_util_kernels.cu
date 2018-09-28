/* C. Kallidonis: Utility kernels required for performing
 * contractions with and/or without propagator and/or gauge shifts
 * September 2018
 */

#include <qlua_util_kernels.cuh>

namespace quda {

  __global__ void phaseMatrix_kernel(complex<QUDA_REAL> *phaseMatrix, int *momMatrix, MomProjArg *arg)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if(tid < arg->V3){ // run through the spatial volume
    
      int lcoord[3]; // Need to hard code
      int gcoord[3]; // this for now
      
      int a1 = tid / arg->localL[0];
      int a2 = a1 / arg->localL[1];
      lcoord[0] = tid - a1 * arg->localL[0];
      lcoord[1] = a1  - a2 * arg->localL[1];
      lcoord[2] = a2;
      
      gcoord[0] = lcoord[0] + arg->commCoord[0] * arg->localL[0] - arg->csrc[0];
      gcoord[1] = lcoord[1] + arg->commCoord[1] * arg->localL[1] - arg->csrc[1];
      gcoord[2] = lcoord[2] + arg->commCoord[2] * arg->localL[2] - arg->csrc[2];
      
      QUDA_REAL f = (QUDA_REAL) arg->expSgn;
      for(int im=0;im<arg->Nmoms;im++){
	QUDA_REAL phase = 0.0;
	for(int id=0;id<arg->momDim;id++)
	  phase += momMatrix[id + arg->momDim*im]*gcoord[id] / (QUDA_REAL)arg->totalL[id];
	
	phaseMatrix[tid + arg->V3*im].x =   cos(2.0*PI*phase);
	phaseMatrix[tid + arg->V3*im].y = f*sin(2.0*PI*phase);
      }

    }//-- tid check

  }//--kernel
  //---------------------------------------------------------------------------

  __global__ void QluaSiteOrderCheck_kernel(QluaUtilArg *utilArg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    if (x_cb >= utilArg->volumeCB) return;
    if (pty >= utilArg->nParity) return;

    int crd[5];
    getCoords(crd, x_cb, utilArg->lL, pty);  //-- Get local coordinates crd[] at given x_cb and pty
    crd[4] = 0;

    int idx_cb = linkIndex(crd, utilArg->lL); //-- Checkerboard index, MUST be equal to x_cb

    int i_rlex = crd[0] + utilArg->lL[0]*(crd[1] + utilArg->lL[1]*(crd[2] + utilArg->lL[2]*(crd[3])));  //-- Full lattice site index
    int i_par = (crd[0] + crd[1] + crd[2] + crd[3]) & 1;

    if( (i_rlex/2 != x_cb) || (pty != i_par) || (idx_cb != x_cb) ){
      d_crdChkVal = -1;
      printf("coordCheck - ERROR: x_cb = %d, pty = %d: Site order mismatch!\n", x_cb, pty);
    }
    else d_crdChkVal = 0;

  }//-- function
  //---------------------------------------------------------------------------

  __global__ void convertSiteOrder_QudaQDP_to_momProj_kernel(void *dst, const void *src, QluaUtilArg *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;

    int crd[5];
    getCoords(crd, x_cb, arg->lL, pty);
    int i_t = crd[arg->t_axis];
    int i_sp = 0;

    for (int i = 0 ; i < 4 ; i++)
      i_sp += arg->sp_stride[i] * crd[i];

    for (int i_f = 0 ; i_f < arg->nFldSrc ; i_f++){
      char *dst_i = (char*)dst + arg->rec_size * (i_sp + arg->sp_locvol *
						  (i_f + arg->nFldDst   * i_t));
      const char *src_i = (char*)src + arg->rec_size * (x_cb + arg->volumeCB *
							(pty + 2 * i_f));
      for (int j = 0 ; j < arg->rec_size ; j++)
        *dst_i++ = *src_i++;
    }//- i_f
    
  }//-- function
  //---------------------------------------------------------------------------

  __global__ void qcCopyCudaLink_kernel(Arg_CopyCudaLink *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    if(arg->extendedGauge){
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];

      Link srcU = arg->Usrc(arg->i_src, linkIndex(c2, arg->dimEx), pty);
      arg->Udst(arg->i_dst, linkIndex(c2, arg->dimEx), pty) = srcU;
    }
    else{
      Link srcU = arg->Usrc(arg->i_src, x_cb, pty);
      arg->Udst(arg->i_dst, x_cb, pty) = srcU;
    }

  }
  //---------------------------------------------------------------------------

  __global__ void qcSetGaugeToUnity_kernel(Arg_SetUnityLink *arg){

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int pty  = blockIdx.y*blockDim.y + threadIdx.y;
    pty = (arg->nParity == 2) ? pty : arg->parity;
    if (x_cb >= arg->volumeCB) return;
    if (pty >= arg->nParity) return;

    Link unitU(arg->unityU); //- That's a 3x3 unity matrix

    if(arg->extendedGauge){
      int crd[5];
      getCoords(crd, x_cb, arg->dim, pty);
      crd[4] = 0;
      int c2[5] = {0,0,0,0,0};
      for(int i=0;i<4;i++) c2[i] = crd[i] + arg->brd[i];

      arg->U(arg->mu, linkIndex(c2, arg->dimEx), pty) = unitU;
    }
    else arg->U(arg->mu, x_cb, pty) = unitU;

  }
  //---------------------------------------------------------------------------

}//-- namespace quda
